# Technical Implementation Report: Feature Engineering Pipeline

**Stage:** 06 — Feature Engineering  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Scikit-Learn + Pandas + Scipy + Transformers

---

## 1. Overview
The Feature Engineering implementation transforms cleaned YouTube comments into a multi-modal feature set. It serves as the "Production Foundry," combining the research results from Stage 03-05 into a deterministic transformation pipeline that produces final training artifacts.

---

## 2. Technical Stack
- **Vectorization:** `TfidfVectorizer` (Best params).
- **Embeddings:** `DistilBertModel` (Optional).
- **Feature Wrangling:** `scipy.sparse` (hstack, CSR).
- **Serialization:** `pickle` (Transformer objects) + `numpy` (Label arrays).

---

## 3. Implementation Workflow

### 3.1 Domain-Specific Derived Features
The `build_derived_features` static method extracts structural signals that are often lost in pure BOW/Embedding approaches:
- **Length Stats:** Character and word counts to capture verbosity.
- **Lexicon Ratios:** Calculates the density of "Positive" vs "Negative" keywords from a curated domain dictionary.
- **Ratio Calculation:** `len(lexicon_words) / max(len(words), 1)` ensures zero-division protection for empty strings.

### 3.2 Representation Strategies
The component supports two distinct paths via the `use_distilbert` config:

#### Path A: TF-IDF (Production Default)
- **Parameter Extraction:** Safely parses the `best_ngram_range` string (e.g., `"(1, 2)"`) into a numeric tuple using `map(int, ...)`.
- **Fit-Transform:** Strict adherence to the MLOps contract: `fit_transform` on Train, `transform` on Val/Test.
- **Serialization:** Saves the fitted `vectorizer.pkl` for serving.

#### Path B: DistilBERT (Contextual Dense)
- **Model Loading:** Lazy-imports `torch` and utilizes "Mean Pooling" over the final hidden states to generate 768-dim embeddings.
- **Batch Processing:** Implements a generator-style loop with `batch_size` (default 32) to prevent out-of-memory errors on GPU or high-RAM systems.

### 3.3 Sparse-Dense Fusion (Feature Union)
The implementation handles the concatenation of high-dimensional sparse text vectors with low-dimensional dense derived features:
```python
X_train_final = hstack([X_train_text, X_train_derived]) if issparse(X_text) else np.hstack([X_text, X_derived])
```
It then forces a conversion to **Compressed Sparse Row (CSR)** format via `csr_matrix()` to optimize the footprint for the downstream RandomForest trainer.

---

## 4. Label Engineering
- **Label Encoding:** Uses `LabelEncoder` to convert the `category` string labels into integers.
- **Inference Parity:** The `label_encoder.pkl` is persisted to ensure that the FastAPI response (e.g., `0`, `1`, `2`) can be translated back to user-friendly text in the Chrome extension.

---

## 5. Robustness & Optimization

- **Leakage Guard:** The `le.fit_transform` and `vectorizer.fit_transform` calls are strictly limited to the `train_df`, ensuring that the validation and test sets remain "unseen" distributions.
- **Artifact Security:** Every directory in the output path is initialized via `path.mkdir(parents=True, exist_ok=True)` to prevent Permission/FileNotFound errors in containerized environments.
- **Efficient IO:** Standardized on Parquet for input (`TRAIN_PATH`) and optimized binary formats (`.npz`, `.npy`) for output to ensure the training loop starts in seconds rather than minutes.

---

## 6. Execution Logic Summary
1.  **Ingest:** Load Train/Val/Test splits from `artifacts/data/processed/`.
2.  **Derive:** Extract numerical structural features (Length, Sentiment Ratios).
3.  **Vectorize:** Apply the winning text representation (TF-IDF or BERT).
4.  **Unite:** Stack text features and derived metrics into a unified CSR matrix.
5.  **Encode:** Generate integer target labels and verify class mapping.
6.  **Persist:** Save matrices, label arrays, and transformer pickles to `artifacts/feature_engineering/`.
