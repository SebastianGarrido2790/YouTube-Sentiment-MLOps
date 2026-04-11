# Technical Implementation Report: Feature Tuning Study

**Stage:** 04 — Feature Tuning  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Scikit-Learn + MLflow + Matplotlib

---

## 1. Overview
The Feature Tuning implementation performs an automated grid search to optimize the TF-IDF vocabulary size (`max_features`). It isolates this hyperparameter while keeping the N-gram configuration constant, allowing for a precise evaluation of the trade-off between dimensionality (feature count) and predictive performance.

---

## 2. Technical Stack
- **Experiment Tracking:** `mlflow`.
- **Feature Engineering:** `TfidfVectorizer`.
- **Benchmarking:** `RandomForestClassifier`.
- **Visualization:** `matplotlib` (via the `evaluate_and_log` utility).

---

## 3. Implementation Workflow

### 3.1 Loop-Driven Search Engine
The core logic resides in a standard Python `for` loop that iterates through the `max_features_values` provided in `params.yaml`.

```python
for max_features in config.max_features_values:
    run_max_features_experiment(
        max_features=max_features,
        ngram_range=tuple(config.best_ngram_range),
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
    )
```

### 3.2 Feature Density Analysis
For every candidate value (e.g., 1000, 5000, 10000), the implementation:
1.  **Re-instantiates the Vectorizer:** This ensures a clean vocabulary is built for each specific limit.
2.  **Applies `min_df=2`:** This "hands-on" technical choice filters out hapax legomena (words appearing once), significantly reducing the noise and dimensionality before the `max_features` limit is even applied.
3.  **Encapsulates fit/transform:** Uses the **Validation set** for evaluation to avoid over-optimizing for the training data.

---

## 4. Documentation & Artifact Strategy

### 4.1 MLflow Run Hierarchy
Each iteration is logged as an independent run within the "Exp - TFIDF Max Features" experiment. This allows the team to use the MLflow "Parallel Coordinates Plot" to visualize the relationship between `max_features` and `val_f1_score`.

### 4.2 Physical Figure Export
Unlike the comparison study, this stage is configured to save physical PNG artifacts to `reports/figures/tfidf_max_features/`.
- **Confusion Matrix:** Every run exports its matrix.
- **DVC Tracking:** DVC monitors the entire figures directory as an `out`. This ensures that the documentation for the tuning phase is version-controlled alongside the code and data.

---

## 5. Robustness & Design Patterns

- **Separation of Concerns:** The implementation uses the `evaluate_and_log` utility from `src/utils/feature_utils.py`. This ensures that the exact same metric calculation logic is used in both the comparison (Stage 03) and the tuning (Stage 04) phases, preventing "metric drift."
- **Immutable Logic:** By using a fixed `random_state: 42` for the RandomForest benchmark, the implementation ensures that variations in performance are due to the feature count, not the stochastic nature of the forest.
- **Fail-Fast Configuration:** If `best_ngram_range` is missing or incorrectly formatted in `params.yaml`, the `ConfigurationManager` raises an exception at the start of the `main()` function, preventing wasted compute time.

---

## 6. Execution Logic Summary
1.  **Initialize:** Load `FeatureTuningConfig` and setup MLflow.
2.  **Ingest:** Load Train/Val splits from Parquet artifacts into memory.
3.  **Iterate:** For each `max_features` candidate:
    - Build TF-IDF vocabulary.
    - Transform texts to sparse feature matrices.
    - Train the fixed-architecture RandomForest.
    - Generate validation metrics.
4.  **Evidence:** Log metrics to MLflow and export Confusion Matrix PNGs to disk.
5.  **Clean up:** Log the final Run ID and finish.
