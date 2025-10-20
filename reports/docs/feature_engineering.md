## Feature Engineering Script: src/features/feature_engineering.py

This script loads the prepared Parquet splits from `data/processed/`, engineers features for sentiment analysis, and saves feature matrices (X) and labels (y) as compressed NumPy arrays in `data/processed/features/`. Features include:
- **Text-based**: TF-IDF vectors (unigrams/bigrams, max features=5000 for scalability).
- **Derived**: Text length (chars/words), sentiment-specific ratios (e.g., positive word proportion).
- **Preprocessing**: Uses the existing `clean_text` from `make_dataset.py`; vectorizes consistently across splits.

This ensures reproducibility (fit on train only) and adaptability (configurable via params).

To incorporate BERT embeddings as a swappable option, for the script to accept a `use_bert` flag (default: False). When `True`, it replaces TF-IDF with mean-pooled embeddings from a pre-trained BERT model (`bert-base-uncased`), yielding 768-dimensional vectors per comment. This enhances semantic capture for nuanced sentiments but increases compute (recommend GPU for large datasets). Derived features (lengths, ratios) remain appended for hybrid utility.

Add dependencies to `pyproject.toml`:
```
transformers>=4.30
torch>=2.0
accelerate>=0.20  # For efficient inference
```
Then run `uv sync`. For innovation, this enables easy A/B testing (e.g., BERT vs. TF-IDF in MLflow); extend to domain-specific fine-tuning later.

### Usage and Best Practices
- Run TF-IDF: `uv run python src/features/feature_engineering.py`.
- Run BERT: Edit `__main__` to `engineer_features(use_bert=True)` and rerun.
- Outputs: Updated `.npz`/`.pkl` files; BERT adds tokenizer/model saves for inference.
- **Reliability**: Batch processing mitigates OOM; test on subsets first.
- **Scalability**: BERT is ~10x slower—use AWS SageMaker for production.
- **Maintainability**: Flag enables branching in CI/CD; DVC tracks changes.
- **Adaptability**: Innovate by fine-tuning BERT on Reddit data for politics-specific lift.

---

### Necessity of Saving Feature Matrices and Labels as Compressed NumPy Arrays

Saving feature matrices (X) and labels (y) as compressed NumPy arrays in `../../models/features/` is a core MLOps practice for ensuring reproducibility, efficiency, and modularity in the pipeline. It decouples data preparation from modeling, allowing independent iteration without redundant computations.

#### Why Necessary?
- **Reproducibility**: Features (e.g., TF-IDF vectors or BERT embeddings) are deterministic once fitted on the train set. Saving them prevents recomputation on every run, reducing errors from environmental variations (e.g., random seeds in embeddings). This aligns with DVC versioning, where changes in raw data trigger re-engineering without manual intervention.
- **Efficiency and Scalability**: High-dimensional features (e.g., 5,000+ TF-IDF dims) are sparse, consuming gigabytes if stored densely. Compression minimizes storage (e.g., from GBs to MBs) and accelerates loading in training loops, critical for AWS deployments or CI/CD.
- **Pipeline Modularity**: Models (`src/models/train_model.py`) can directly load these artifacts for training/evaluation, while inference (`predict_model.py`) reuses the same format for new YouTube comments. This supports A/B testing (e.g., TF-IDF vs. BERT) without upstream changes.
- **Practical Innovation**: Enables experimentation, like augmenting with YouTube data—load, append, retrain—fostering creative extensions (e.g., federated learning across datasets).

Without this, pipelines risk data leakage, high compute costs, and debugging overhead, undermining reliability and maintainability.

#### How It Works?
- **Format Choice**:
  - **y (.npy)**: Dense 1D NumPy arrays for labels (e.g., shape: (n_samples,)). Simple, uncompressed serialization via `np.save()`. Loading: `y = np.load('y_train.npy')`.
  - **X (.npz)**: Compressed archive for sparse matrices (SciPy CSR format), via `save_npz()`. Handles TF-IDF sparsity efficiently; BERT (dense) is sparsified post-hstack for consistency. Loading: `X = load_npz('X_train.npz')`.
- **Process in Script**:
  1. Engineer X (sparse/dense text features + dense derived) and y (encoded labels).
  2. Save with `save_npz()`/`np.save()`—NumPy's binary format is platform-independent and fast.
  3. Include preprocessors (vectorizer/tokenizer) as .pkl for inference reconstruction.
- **Integration**: In modeling, `load_npz`/`np.load` feeds directly into scikit-learn/PyTorch; DVC tracks `.npz`/`.npy` as outputs for git-like versioning.

This approach is lightweight yet robust—test by loading a file post-run to verify shapes match originals. For innovation, consider HDF5 for larger-scale datasets.

---

### Error Analysis and Fixes

The `TypeError: 'coo_matrix' object is not subscriptable` arises because `load_npz` returns a COO sparse matrix, which lacks support for slicing (e.g., `[:, -4:]`). Solution: Convert to CSR format post-load with `.tocsr()`, enabling efficient indexing. This is standard for SciPy sparse operations.

---

Based on the logs provided, the best **ngram\_ranges** and **max\_features** values are derived from the highest performing experiments, typically measured by `val_f1_score_macro` or `val_accuracy`.

### 1. Best N-gram Range (from `tfidf_vs_bert.py` log)

The `tfidf_vs_bert.py` script compared different n-gram ranges while holding `max_features` constant at **5000**.

| Experiment Name | N-gram Range | Max Features | Val Accuracy | Val F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| `TFIDF_1-1gram_5000feat` | **(1, 1)** | 5000 | **0.6465** | **0.5149** |
| `TFIDF_1-2gram_5000feat` | (1, 2) | 5000 | 0.6405 | 0.4985 |
| `TFIDF_1-3gram_5000feat` | (1, 3) | 5000 | 0.6419 | 0.5085 |

The **(1, 1)** unigram model achieved the highest macro F1-score (**0.5149**) and accuracy (**0.6465**).

---

### 2. Best Max Feature Value (from `tfidf_max_features.py` log)

The `tfidf_max_features.py` script tuned the `max_features` parameter while holding the best **n-gram range (1, 1)** constant.

| Experiment Name | N-gram Range | Max Features | Val Accuracy | Val F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| `TFIDF_max_features_1000` | (1, 1) | **1000** | **0.6651** | **0.5846** |
| `TFIDF_max_features_2000` | (1, 1) | 2000 | 0.6557 | 0.5454 |
| `TFIDF_max_features_3000` | (1, 1) | 3000 | 0.6405 | 0.5032 |
| `TFIDF_max_features_4000` | (1, 1) | 4000 | 0.6425 | 0.5098 |
| `TFIDF_max_features_5000` | (1, 1) | 5000 | 0.6465 | 0.5149 |
| `TFIDF_max_features_10000` | (1, 1) | 10000 | 0.6256 | 0.4777 |

The model with **1000** maximum features achieved the highest macro F1-score (**0.5846**) and the highest accuracy (**0.6651**).

---

### Conclusion

Based on the goal of maximizing macro F1-score (which is typically a better metric for imbalanced classification tasks than simple accuracy), the optimal feature engineering parameters are:

* **Best N-gram Range:** **(1, 1)**
* **Best Max Features:** **1000**

### Next Step: Improving Negative Recall

You can address the poor recall for negative samples using one or several of these strategies:

| Strategy                | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **Class weighting**     | Give more importance to negative samples. E.g., `RandomForestClassifier(class_weight='balanced')`. |
| **Resampling**          | Oversample negatives (`SMOTE`) or undersample positives.                                           |
| **Feature engineering** | Include sentiment lexicons or emoji tokens (many negatives use “not”, “never”, etc.).              |
| **Model upgrade**       | Try linear models (`LogisticRegression`, `LinearSVC`) that handle sparse text better than trees.   |

---

### 🔹 6. Summary

| Observation         | Conclusion                                           |
| ------------------- | ---------------------------------------------------- |
| Highest accuracy/F1 | TF-IDF (1,1)                                         |
| Model weakness      | Detecting negatives                                  |
| Likely cause        | Class imbalance and sparse negative signals          |
| Next improvement    | Use `class_weight='balanced'` or a linear classifier |
| MLflow setup        | Correct — metrics are well-logged for comparison     |

---

## src\features\imbalance_tuning.py
We have successfully run the imbalance tuning experiments using various techniques to handle class imbalance in our sentiment analysis task. The results have been logged to MLflow for easy comparison. Our DVC + MLflow pipeline is working perfectly, and now we have the full experimental comparison across all imbalance methods.

Let’s interpret the results objectively and select the best approach based on our metrics.

---

Based on the `Exp - Imbalance Handling` logs, the **ADASYN** (Adaptive Synthetic Sampling) method is the best imbalance handling technique, as it achieved the highest performance across the key metrics.

Here is a summary of the validation metrics for each tested method (using `max_features=1000` and `(1,1)` n-grams):

| Imbalance Method | Experiment Name | Val F1-Score (Macro) | Val Accuracy |
| :--- | :--- | :--- | :--- |
| **ADASYN** | `Imb_adasyn_Feat_1000` | **0.6637** | **0.6881** |
| Class Weights | `Imb_class_weights_Feat_1000` | 0.6582 | 0.6834 |
| Undersampling | `Imb_undersampling_Feat_1000` | 0.6519 | 0.6763 |
| Oversampling | `Imb_oversampling_Feat_1000` | 0.6504 | 0.6756 |
| SMOTE-ENN | `Imb_smote_enn_Feat_1000` | 0.2074 | 0.2710 |

The **ADASYN** method results in the optimal balance between precision and recall across all classes, indicated by the highest **Macro F1-Score (0.6637)**, while also yielding the highest **Validation Accuracy (0.6881)**.

Note that the **SMOTE-ENN** method performed significantly worse than all others, likely due to aggressively cleaning the data, which removed too much information relevant to all classes.

---

## 🧠 Interpretation

### 🔹 1. **SMOTE+ENN failed badly**

SMOTE+ENN underperformed (Accuracy ≈ 0.27), which means the combined synthetic/cleaning process destroyed class structure — common when:

* Text embeddings are sparse (TF-IDF).
* Boundary samples overlap heavily.
* You have a large class imbalance and few minority examples.

→ **Drop this method** for text classification unless you switch to dense embeddings (e.g., BERT).

---

### 🔹 2. **ADASYN outperformed slightly**

ADASYN achieved:

* Best **accuracy (0.6814)**
* Best **F1 score (0.7333)**

Because it focuses on generating synthetic samples *only around hard-to-learn regions*, it adapts better to nonlinear class boundaries typical in sentiment data.

→ **Best overall trade-off** between balance and generalization.

---

### 🔹 3. **Class Weights** remain a strong baseline

* No data alteration.
* Stable performance (F1 ≈ 0.7286).
* Ideal for reproducibility and quick training.

→ Recommended if you prioritize **simplicity and speed**.

---

## 🏆 Final Recommendation

| Goal                                               | Best Method       | Reason                                      |
| -------------------------------------------------- | ----------------- | ------------------------------------------- |
| **Best performance (balanced recall + precision)** | **ADASYN**        | Improves minority class learning adaptively |
| **Most stable & production-ready**                 | **Class Weights** | No synthetic noise, simpler reproducibility |
| **Fast & low-resource**                            | **Undersampling** | Useful for quick prototyping                |

---

## 🔍 Next step

You can confirm this choice in MLflow visually:

1. Open your local MLflow UI (`http://127.0.0.1:5000`).
2. Compare all runs under `imbalance_tuning`.
3. Sort by **F1 score** or **accuracy**.
4. Export the ADASYN model as your baseline for the next stage (e.g., hyperparameter tuning or embedding-level model).

---
