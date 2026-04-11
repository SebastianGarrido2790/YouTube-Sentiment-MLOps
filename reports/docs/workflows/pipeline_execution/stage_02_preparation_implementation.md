# Technical Implementation Report: Data Preparation

**Stage:** 02 — Data Preparation  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Pandas + NLTK + Sklearn + Parquet

---

## 1. Overview
The Data Preparation implementation converts raw CSV data into high-performance Parquet splits. It ensures that the text features are normalized and the datasets are partitioned using a stratified strategy to guarantee class balance across all training and evaluation experiments.

---

## 2. Technical Stack
- **Text Processing:** `nltk` (Tokenization, Stopwords) + `re` (Regex).
- **Data Engineering:** `Pandas` + `pyarrow` (Engine for Parquet).
- **Statistical Operations:** `scikit-learn` (Stratified Shuffling).
- **Environment:** Container-aware resource management.

---

## 3. Implementation Workflow

### 3.1 Environmental Readiness
The implementation ensures that the processing environment is ready without manual intervention.
- **NLTK Lifecycle:** The Conductor (`stage_02_...py`) uses `nltk.download` inside the execution flow to ensure the `punkt_tab` and `stopwords` resources are cached locally.
- **Directory Persistence:** Uses `os.makedirs` to initialize the `artifacts/data/processed/` landing zone.

### 3.2 Feature Engineering: Text Refinement
Core logic in `DataPreparation.clean_text`:
1.  **Normalization:** `text.lower().strip()`
2.  **Noise Removal:** `re.sub(r"[^a-zA-Z\s]", " ", text)` removes special characters and digits.
3.  **Tokenization:** `word_tokenize` splits the strings into atomic units.
4.  **Stop-word Filtering:** Removes common English noise words and tokens shorter than 3 characters (`len(t) > 2`).
5.  **Reconstruction:** `" ".join(tokens)` returns a clean, space-delimited feature string.

### 3.3 Target Engineering: Label Alignment
To satisfy model requirements (particularly XGBoost), raw labels are mapped to a zero-indexed contiguous range:
```python
df["category_encoded"] = df["category"].map({-1: 0, 0: 1, 1: 2})
```
- `-1 (Negative) -> 0`
- ` 0 (Neutral)  -> 1`
- ` 1 (Positive) -> 2`

### 3.4 Partitioning: Double Stratified Split
To achieve the target distribution (e.g., $70/15/15$), the implementation performs two sequential stratified splits using `train_test_split`:

1.  **First Split:** Separates the **Test** set ($15\%$) from the **Train+Val** group.
2.  **Second Split:** Separates the **Validation** set ($15\%/\sim85\%$) from the **Train** set.
- `stratify=df["category"]`: Ensures the sentiment balance remains identical in all three artifacts.

---

## 4. Performance & Persistence

### 4.1 Parquet Pipeline
The implementation utilizes the **Apache Parquet** format via `to_parquet()`.
- **Schema Preservation:** Unlike CSV, Parquet stores metadata about column types, preventing the "sentiment-as-float" bug.
- **I/O Efficiency:** Compressed columnar storage significantly reduces the time for subsequent training stages to load the split into RAM.

### 4.2 Logging & Observability
The worker component logs the "Class Distribution" for the training split, providing immediate visual confirmation that the stratification worked as expected.

---

## 5. Error Handling & Robustness

- **Information Leakage Prevention:** The use of `random_state` from `params.yaml` across both splits ensures that the partition is immutable across independent experiment runs.
- **Empty Split Guard:** If any split results in zero rows (due to hyper-aggressive filtering), the component raises a `ValueError` before saving any files, preventing "downstream poisoning."
- **NLTK Resilience:** Downloads are wrapped in `quiet=True` and executed before configuration loading to ensure the engine is ready.

---

## 6. Execution Logic Summary
1.  **Bootstrap:** Download NLTK resources and create processed directories.
2.  **Ingest:** Load raw `reddit_comments.csv` via the central `RAW_PATH` constant.
3.  **Encode:** Map labels to standardized integers `{0, 1, 2}`.
4.  **Refine:** Execute Regex + Tokenization pipeline on the `clean_comment` feature.
5.  **Partition:** Execute double-layered stratified splitting.
6.  **Serialize:** Write **Train**, **Val**, and **Test** splits to Parquet artifacts.
7.  **Finalize:** Log distribution metrics and return success.
