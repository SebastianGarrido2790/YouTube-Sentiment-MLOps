# Technical Implementation Report: Feature Comparison Study

**Stage:** 03 — Feature Comparison  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Scikit-Learn + Transformers + MLflow

---

## 1. Overview
The Feature Comparison study is implemented as a specialized experiment runner that evaluates the trade-offs between sparse high-dimensional features (**TF-IDF**) and dense contextual embeddings (**DistilBERT**). It serves as a data-driven decision gate for the subsequent training pipeline.

---

## 2. Technical Stack
- **Experiment Tracking:** `mlflow`.
- **Transformers:** `torch` + Hugging Face `transformers` (DistilBERT).
- **ML Baseline:** `RandomForestClassifier`.
- **Parallel Computing:** GPU-aware torch inference.

---

## 3. Implementation Workflow

### 3.1 Lazy Dependency Management
Because DistilBERT requires heavy dependencies (`torch`, `transformers`), the implementation uses a **Lazy Import** strategy inside the `get_distilbert_embeddings` function. This ensures that the primary pipeline remains lightweight and can run TF-IDF experiments even if deep learning libraries are not installed.

### 3.2 Deep Learning Embedding Logic
When enabled, the DistilBERT path executes the following:
1.  **Tokenization:** Converts clean text into tensors with a 512-token limit and padding.
2.  **Batched Inference:** Processes text in configurable chunks (default: 32) to manage GPU memory.
3.  **Mean Pooling:** Instead of just using the `[CLS]` token, the implementation performs global average pooling over the sequence dimension:
    ```python
    outputs = model(**inputs)
    pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
    ```
4.  **CPU Guard:** Explicitly raises a `RuntimeError` if a user attempts DistilBERT on a CPU, preventing the pipeline from hanging on slow inference.

### 3.3 TF-IDF Grid Search
The implementation iterates over the `ngram_ranges` list from `params.yaml`. For each range, it:
1.  Instantiates a `TfidfVectorizer`.
2.  Fits on the **Train** set and transforms the **Val** set.
3.  Trains a baseline `RandomForestClassifier`.

---

## 4. MLflow Logging Strategy

The study utilizes **Nested Runs** to keep the experiment UI organized.

- **Parameters:** Logs `ngram_range`, `max_features`, and `vectorizer_type`.
- **Metrics:** Tracks standard classification metrics (`f1_score`, `precision_score`, `recall_score`, `accuracy`).
- **Artifacts:** Generates and logs a **Confusion Matrix** PNG for each configuration.
- **Run Discovery:** Uses `setup_experiment` to ensure all study logs are grouped under "Exp - Feature Comparison".

---

## 5. Security & Robustness

- **Information Leakage:** The study strictly uses the **Validation set** for evaluation, ensuring that the Test set remains "unseen" for final model reporting.
- **Reproducibility:** A fixed `random_state: 42` is shared between the vectorizer and the classifier.
- **Fast Exit:** If the raw data is malformed or columns are missing, the component raises a `ValueError` during the loading phase.

---

## 6. Execution Logic Summary
1.  **Bootstrap:** Load `FeatureComparisonConfig` via the Configuration Manager.
2.  **Initialize Tracking:** Configure MLflow Tracking URI and Experiment ID.
3.  **Ingest:** Load training and validation splits from Parquet artifacts.
4.  **Loop (TF-IDF):** 
    - Generate features for each N-gram configuration.
    - Train baseline RF.
    - Log results to MLflow.
5.  **Conditional (BERT):** 
    - If `use_distilbert=True`, generate 768-dim dense embeddings.
    - Train baseline RF.
    - Log results to MLflow.
6.  **Analyze:** Finalize the study and provide the MLflow Run ID for dashboarding.
