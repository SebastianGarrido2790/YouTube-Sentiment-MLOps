### Feature Engineering Script: src/features/feature_engineering.py

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

Based on the MLflow metrics provided, the best choice for the **TF-IDF `max_features`** is **$7000$**. This value yields the highest overall accuracy.

The relevant runs and their final overall `accuracy` metric are extracted from the provided data:

| Run ID | `vectorizer_max_features` (Inferred) | `accuracy` |
| :--- | :--- | :--- |
| `c4bfb747eaa341ef93be37f767e80a30` | 10000 | 0.6345634563456346 |
| `c5d92a5cd68941cfa9c27a833351a0d8` | 9000 | 0.6304230423042304 |
| `633b1c20ef5548b48877f44eaa915678` | 8000 | 0.6334833483348334 |
| **`d9ab47065c7447e896471aaa5c859a15`** | **7000** | **0.6477047704770477** |
| `1cebe189798f4d8682f1953ea8058f88` | 6000 | 0.6343834383438344 |
| `87bee845e6434409bfa5edecec23e4d5` | 5000 | 0.6387038703870387 |
| `dcdde7eac604456f8a2fe63d753fba37` | 4000 | 0.6336633663366337 |
| `6ad69553ca6d45a2a95526e1ad91d0c5` | 3000 | 0.6345634563456346 |
| `ec0159be51284e6f8659d5f918f4c0a9` | 2000 | 0.6441044104410441 |
| (Not fully logged) | 1000 | (Incomplete) |

The run with an **accuracy of $0.6477$** (or $64.77\%$) belongs to the run with `run_id` `6ad69553ca6d45a2a95526e1ad91d0c5`.

### Best `max_features`

The **highest accuracy score** of **$0.6477$** was achieved with the run corresponding to `max_features = 7000`.

| Metric | Value ($\text{max\_features} = 7000$) |
| :--- | :--- |
| **Accuracy** | **$0.6477$** |
| Weighted Avg F1-Score | $0.5861$ |
| Macro Avg F1-Score | $0.5219$ |

This suggests that using $7000$ features (trigrams) strikes the best balance between providing enough information for the model and avoiding noise or overfitting compared to the other tested values.