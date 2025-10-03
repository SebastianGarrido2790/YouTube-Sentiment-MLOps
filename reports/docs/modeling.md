### Baseline Model Training Script: src/models/train_model.py

Based on the MLflow metrics from `handling_imbalanced_data.py`, the best imbalance handling method is **RandomUnderSampler** (undersampling), which achieved the highest F1-score for the minority class (negative: 0.605) while maintaining strong performance on neutral (F1: 0.743). This method balances classes by reducing majority samples, improving minority detection without synthetic data artifacts, though it discards information—suitable for this baseline given the dataset size. Alternatives like ADASYN (F1 negative: 0.599) offer higher recall (0.597) but slightly lower precision; iterate in advanced runs.

This script loads pre-engineered TF-IDF features (default from `feature_engineering.py`), applies undersampling to the train set, trains a Logistic Regression baseline, and logs to MLflow (accuracy, macro F1, per-class F1). It evaluates on the test set for final metrics. Add `"imbalanced-learn"` and `"scikit-learn"` to `pyproject.toml` if needed (`uv sync`).

### Usage and Best Practices
- **Run**: Executes training, logs to MLflow (view UI for comparisons), saves model.
- **Reliability**: Undersampling on train only; stratified splits preserved.
- **Scalability**: Sparse matrices efficient; extend to GPU for LSTM.
- **Maintainability**: Modular; DVC add `models/` post-run.
- **Adaptability**: Swap model (e.g., `from sklearn.ensemble import RandomForestClassifier`); add Optuna for tuning in v2.

### Prototype Inference Endpoint
For quick testing, add this function to `src/models/predict_model.py` (create if needed). It loads the model and preprocesses new YouTube comments via saved vectorizer.

This endpoint is callable from the Chrome extension (e.g., via AWS Lambda later). Test locally; deploy next.
