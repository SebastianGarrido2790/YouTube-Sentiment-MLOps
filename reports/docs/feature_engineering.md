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

