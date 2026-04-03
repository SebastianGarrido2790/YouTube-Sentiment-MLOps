## Feature Engineering Script: src/features/feature_engineering.py

This script loads the prepared Parquet splits from `data/processed/`, engineers features for sentiment analysis, and saves feature matrices (X) and labels (y) as compressed NumPy arrays in `models/features/`.

Features include:
- **Text-based**: TF-IDF vectors (unigrams/bigrams, max features configurable via `feature_comparison` or `feature_tuning`).
- **Optional**: DistilBERT embeddings (if enabled).

This ensures reproducibility (fit on train only) and adaptability (configurable via `params.yaml`).

The script checks `params.yaml` for:
- `feature_engineering.use_distilbert` (boolean or string "True"/"False")
- `imbalance_tuning.best_max_features` (int)
- `imbalance_tuning.best_ngram_range` (list, e.g., `[1, 1]`)

### Usage and Best Practices
- **Run via DVC**: `dvc repro feature_engineering` (uses `params.yaml` configuration).
- **Run directly**: `python -m src.features.feature_engineering`
- **Outputs**: Updated `.npz` (X) and `.npy` (y) files in `models/features/`.
- **Reliability**: Validates data existence and shapes.
- **Scalability**: Handles sparse matrices for TF-IDF efficiently.
- **Maintainability**: DVC tracks changes via `params.yaml` dependencies.
- **Adaptability**: Easy switching between TF-IDF configuration or adding DistilBERT embeddings.

---

### Necessity of Saving Feature Matrices and Labels as Compressed NumPy Arrays

Saving feature matrices (X) and labels (y) as compressed NumPy arrays in `models/features/` is a core MLOps practice for ensuring reproducibility, efficiency, and modularity in the pipeline. It decouples data preparation/engineering from modeling.

#### Why Necessary?
- **Reproducibility**: Features are deterministic once fitted on the train set. Saving them prevents recomputation on every run.
- **Efficiency**: High-dimensional features (e.g., 5,000+ TF-IDF dims) are sparse. Compression minimizes storage (e.g., from GBs to MBs) and accelerates loading.
- **Pipeline Modularity**: Models (`src/models/train_model.py`) can directly load these artifacts.
- **CI/CD**: Faster pipeline runs since models load pre-computed features.

#### How It Works?
- **Format Choice**:
  - **y (.npy)**: Dense 1D NumPy arrays for labels. Simple serialization.
  - **X (.npz)**: Compressed archive for sparse matrices (SciPy CSR format). Handles TF-IDF sparsity efficiently.
- **Process**:
  1. Engineer X (sparse TF-IDF) and y (encoded labels).
  2. Save with `scipy.sparse.save_npz` and `np.save`.
  3. Include `label_encoder.pkl` for inference reconstruction.
- **Integration**: In modeling, `load_npz`/`np.load` feeds directly into scikit-learn/XGBoost.

---

### Error Analysis and Fixes

The pipeline handles sparse matrices carefully. `load_npz` returns a CSR/COO matrix. If slicing is needed, ensure it is in CSR format (`.tocsr()`).

---

### Hyperparameter Selection Logic

Based on the logs provided from earlier tuning stages (`feature_comparison`, `feature_tuning`), the best **ngram_ranges** and **max_features** are derived from the highest performing experiments (typically `val_macro_f1`).

#### 1. Best N-gram Range (from `feature_comparison` stage)

The comparison typically shows unigrams `(1, 1)` or bigrams `(1, 2)` performing best depending on the dataset. Our current configuration defaults to `(1, 1)` based on initial findings.

#### 2. Best Max Feature Value (from `feature_tuning` stage)

The tuning script tests values like `1000`, `5000`, `10000`.
- **1000 features** often yield good generalization with lower dimensionality.
- **5000 features** capture more vocabulary but risk overfitting.

**Conclusion**: The optimal parameters are set in `params.yaml` under `imbalance_tuning`:
- **Best N-gram Range:** `[1, 1]`
- **Best Max Features:** `1000` (or `5000`, depending on experiment results)

### Next Step: Improving Negative Recall

Strategies to improve recall for negative samples (often the minority class or harder to detect):

| Strategy | Description |
| :--- | :--- |
| **Class weighting** | `class_weight='balanced'` in Logistic Regression/Random Forest. |
| **Resampling** | SMOTE/ADASYN during `imbalance_tuning`. |
| **Feature engineering** | Add specific sentiment lexicons or negation handling. |
| **Model upgrade** | Gradient boosting (LightGBM/XGBoost) handles non-linearities better. |

---

### 🔹 Summary

| Observation | Conclusion |
| :--- | :--- |
| **Features** | TF-IDF (1,1) with ~1000-5000 features is efficient. |
| **Weakness** | Detecting negatives/minority classes. |
| **Fix** | ADASYN or Class Weights (implemented in `imbalance_tuning`). |
| **Artifacts** | `models/features/X_train.npz` used by all downstream models. |

---

## Integration with DVC Pipeline

The feature engineering stage is integrated into `dvc.yaml`:

```yaml
feature_engineering:
  cmd: python -m src.features.feature_engineering
  deps:
    - data/processed/train.parquet
    - src/features/feature_engineering.py
  params:
    - feature_engineering.use_distilbert
    - imbalance_tuning.best_max_features
    - imbalance_tuning.best_ngram_range
  outs:
    - models/features/
```

This ensures that if feature parameters change (e.g., increasing `max_features`), the feature engineering stage re-runs, and subsequently all model training stages re-run with the new data.
