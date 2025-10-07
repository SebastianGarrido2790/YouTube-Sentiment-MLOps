### Baseline Model Training Script: src/models/train_model.py

Based on the MLflow metrics from `handling_imbalanced_data.py`, the best imbalance handling method is **RandomUnderSampler** (undersampling), which achieved the highest F1-score for the minority class (negative: 0.605) while maintaining strong performance on neutral (F1: 0.743). This method balances classes by reducing majority samples, improving minority detection without synthetic data artifacts, though it discards information—suitable for this baseline given the dataset size. Alternatives like ADASYN (F1 negative: 0.599) offer higher recall (0.597) but slightly lower precision; iterate in advanced runs.

This script loads pre-engineered TF-IDF features (default from `feature_engineering.py`), applies undersampling to the train set, trains a Logistic Regression baseline, and logs to MLflow (accuracy, macro F1, per-class F1). It evaluates on the test set for final metrics. Add `"imbalanced-learn"` and `"scikit-learn"` to `pyproject.toml` if needed (`uv sync`).

#### Usage and Best Practices
- **Run**: Executes training, logs to MLflow (view UI for comparisons), saves model.
- **Reliability**: Undersampling on train only; stratified splits preserved.
- **Scalability**: Sparse matrices efficient; extend to GPU for LSTM.
- **Maintainability**: Modular; DVC add `models/` post-run.
- **Adaptability**: Swap model (e.g., `from sklearn.ensemble import RandomForestClassifier`); add Optuna for tuning in v2.

#### Prototype Inference Endpoint
For quick testing, add this function to `src/models/predict_model.py` (create if needed). It loads the model and preprocesses new YouTube comments via saved vectorizer.

This endpoint is callable from the Chrome extension (e.g., via AWS Lambda later). Test locally; deploy next.

### Advanced Model Training Script: src/models/advanced_training.py

This script extends the pipeline with advanced models: XGBoost and LightGBM on TF-IDF features (tuned for gradient boosting), and BERT fine-tuning via Hugging Face on raw text (for semantic nuance). Optuna optimizes hyperparameters (e.g., learning rate, depth) over 50 trials, using validation F1 (macro) as objective. Imbalance handled via RandomUnderSampler on train. Results logged to MLflow for comparison; best model saved in `models/advanced/`.

Add to `pyproject.toml`:
```
optuna>=3.6
xgboost>=2.0
lightgbm>=4.3
transformers>=4.40
datasets>=2.20  # For BERT data loading
accelerate>=0.30  # For distributed training
```
Run `uv sync`. For BERT, GPU recommended (set `device='cuda'`).

#### Usage and Best Practices
- **Execution**: Runs tuning sequentially (~30-60 min total; BERT longest). Outputs best params/models.
- **Reliability**: Nested runs in Optuna log trials; undersampling applied consistently.
- **Scalability**: Optuna parallelizable (`n_jobs=-1`); BERT uses accelerate for multi-GPU.
- **Maintainability**: Pruning via `Pruner` in Optuna for efficiency; DVC track `models/advanced/`.
- **Adaptability**: Extend objectives for custom metrics; integrate YouTube data for domain tuning.

For innovation, chain Optuna with Bayesian optimization or ensemble best models. Next: Inference prototype or deployment?

### Troubleshooting BERT training process

The process is failing because the **labels in the dataset are not mapped to the expected range** for the PyTorch cross-entropy loss function.

#### 🐛 Root Cause: Label Encoding Mismatch

The sentiment labels are $\{-1, 0, 1\}$. The **PyTorch cross-entropy loss function** (which is used internally by the Hugging Face `Trainer`) requires classification targets to be non-negative integers starting from **zero** (i.e., $0, 1, 2, \dots, N-1$).

Since The labels include **$-1$**, the loss function attempts to index into its output distribution at position $-1$, which results in the `IndexError: Target -1 is out of bounds.`

#### ✅ Solution: Shift the Labels to $\{0, 1, 2\}$

Shift all the labels so the minimum value is $0$. Since the original labels are $\{-1, 0, 1\}$, adding $1$ to each will correctly map them to $\{0, 1, 2\}$.

**Action:** Modify the `bert_objective` function to shift the labels immediately after loading and renaming.

By shifting the labels, the data will be compatible with the standard PyTorch classification loss, allowing the BERT tuning to proceed. The model will now learn to predict $\{0, 1, 2\}$, corresponding to $\{-1, 0, 1\}$.

---

### Rationale for Focusing on F1-Score in Sentiment Analysis

In this pipeline, the F1-score (harmonic mean of precision and recall) is prioritized as the primary evaluation metric due to the dataset's inherent class imbalance (Negative: 22.22%, Neutral: 35.28%, Positive: 42.50%) and the task's practical demands. Below, I outline the key reasons, structured for clarity.

#### 1. **Handling Imbalance Robustly**
   - **Accuracy Pitfalls**: Simple accuracy favors the majority class (e.g., predicting all samples as Positive yields ~42.5% accuracy, masking poor minority performance). This is unreliable for real-world YouTube sentiment, where negatives (e.g., toxic comments) are underrepresented but critical.
   - **F1's Balance**: F1 penalizes imbalances in precision (TP / (TP + FP)) and recall (TP / (TP + FN)), ensuring models detect rare classes without excessive false alarms. Macro-F1 (unweighted average across classes) further equalizes treatment, amplifying minority class contributions—essential here for equitable evaluation.

#### 2. **Alignment with Task Requirements**
   - **Sentiment Nuances**: In video comment analysis, false negatives (missing a negative comment) could overlook harmful content, while false positives dilute trust in positive signals. F1 directly optimizes this trade-off, unlike precision (ignores missed detections) or recall (ignores false alarms).
   - **Multi-Class Suitability**: For three classes, macro-F1 provides a holistic score, while per-class F1 (logged in MLflow) enables granular insights (e.g., boosting Negative F1 from ~0.37 in baselines to ~0.60 with undersampling).

#### 3. **MLOps and Optimization Fit**
   - **Tuning and Selection**: Optuna uses macro-F1 as the objective for hyperparameter search, as it correlates with deployment KPIs (e.g., Chrome extension reliability). Cross-validation on F1 ensures generalizability.
   - **Comparability**: It standardizes A/B testing across models (Logistic Regression, XGBoost, BERT), facilitating selection of the best (e.g., via MLflow UI).

#### Practical Recommendations
- **Thresholding**: In production, adjust decision thresholds per class (e.g., lower for negatives) to fine-tune F1 components.
- **Innovation Opportunity**: Experiment with weighted F1 (emphasizing negatives) or custom metrics (e.g., incorporating latency for real-time inference). Track via MLflow to iterate empirically.

This focus ensures reliable, balanced performance, directly supporting the pipeline's reliability and adaptability goals. If needed, pivot to AUC-PR for probabilistic outputs in advanced iterations.

---
 
### Models Performance

The **macro F1 score** is the primary metric for comparing the performance of these models, as it handles class imbalance better than simple accuracy.

| Model | Best Macro F1 Score |
| :--- | :--- |
| **LightGBM** | **0.79986** (from `LightGBM_Trial_18`) |
| **Logistic Regression** | **0.78679** (from `macro_f1`) |
| **XGBoost** | **0.78317** (from `XGBoost_Trial_22`) |

***

## Performance Summary

| Model | Best Macro F1 Score | Notes |
| :--- | :--- | :--- |
| **LightGBM** | **0.79986** | Achieved the highest performance during Optuna tuning. |
| **Logistic Regression** | 0.78679 | A strong baseline model, performing better than XGBoost. |
| **XGBoost** | 0.78317 | Achieved a high score, but was slightly outperformed by both LightGBM and the Logistic Regression baseline. |

**LightGBM's best trial achieved a Macro F1 score of 0.79986, making it the top-performing model.**