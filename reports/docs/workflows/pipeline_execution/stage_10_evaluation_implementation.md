# Technical Implementation Report: Comparative Model Evaluation

**Stage:** 10 — Model Evaluation  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Scikit-Learn + Matplotlib + MLflow

---

## 1. Overview
The Model Evaluation implementation is a polymorphic scoring engine. It abstracts the differences between varied model architectures (Sklearn, XGBoost, Transformers) to perform a direct, unbiased comparison on the held-out Test Set. The implementation focuses on three pillars: **Fair Benchmarking**, **Visual Diagnostics**, and **Automated Champion Selection**.

---

## 2. Technical Stack
- **Evaluation Engine:** `scikit-learn` (Metrics & Preprocessing).
- **Visualization:** `matplotlib` + `seaborn`.
- **Inference Logic:** `torch` (for DistilBERT) + `pickle` (for GBTs/Logistic).
- **Experiment Tracking:** `mlflow` (Parent/Child Nested Runs).

---

## 3. Implementation Workflow

### 3.1 Standardized Inference Wrapper
The core implementation detail is the `get_model_predictions` method. It detects the incoming model type and adapts the data format:
- **Standard (LGBM/RF/LR):** Executes `.predict_proba(X_test)`.
- **XGBoost:** Converts sparse matrices to `xgb.DMatrix` before generating probabilities.
- **DistilBERT:** Uses a `torch` DataLoader to stream the Test set through the transformer's multi-head attention layers, returning raw logits mapped to probabilities via Softmax.

### 3.2 Evaluation & Metric Generation
The implementation executes a standardized scoring block for every candidate:
1.  **Macro-ROC AUC:** Uses `LabelBinarizer` to transform multiclass labels into One-vs-Rest (OvR) binary arrays. It then calculates the Area Under the Curve for each sentiment label.
2.  **Confusion Matrices:** Plots a heatmap of `Actual` vs `Predicted`. This is critical for identifying "Bias Zones" (e.g., if a model consistently mislabels Negative comments as Neutral).
3.  **JSON Persistence:** Writes a per-model `_test_metrics.json` file, allowing DVC to track performance metrics at the file level.

---

## 4. Champion Selection & Contract Handover

### 4.1 Automated Selection Logic
The implementation doesn't just rank models; it programmatically selects the "Production Winner" based on the **Macro AUC** score:
```python
winner = max(model_results, key=lambda x: x["test_macro_auc"])
```
*Why Macro AUC?* It is the most resilient metric against class imbalance, rewarding models that distinguish between all three sentiment labels with equal intensity.

### 4.2 The Best Model Run Info (JSON Contract)
The implementation culminates in the production of `best_model_run_info.json`. This file acts as the bridge to the deployment stage, containing:
- **run_id:** The unique MLflow internal ID of the champion model.
- **model_name:** The algorithm name.
- **timestamp:** Execution date for auditability.

---

## 5. Robustness & Design Patterns

- **Null-Model Safety:** The implementation includes defensive logic to handle "Placeholders." If a training stage was disabled (e.g., DistilBERT), the evaluator logs the absence and skips that specific evaluation column without crashing the entire study.
- **Parent-Child Telemetry:** To maintain a clean MLflow UI, the study begins by ending any existing runs and starting a new "Parent" run. Each model evaluation is then recorded as a "Child" run inside this context.
- **Visual Parity:** All plots use a consistent color-blind accessible palette and standardized axis labels, ensuring that the `comparative_roc_curve.png` is readable by all stakeholders.

---

## 6. Execution Logic Summary
1.  **Bootstrap:** Load the Test Set parity splits (`X_test`, `y_test`) and `LabelEncoder`.
2.  **Ingest:** Fetch model artifacts from the `artifacts/` sub-directories.
3.  **Benchmark:** For each available model:
    - Generate probabilities and discrete predictions.
    - Calculate F1-Macro and ROC-AUC metrics.
    - Export individual Confusion Matrix PNGs.
4.  **Visualize:** Overlay ROC curves for all models into a single comparative figure.
5.  **Promote:** Programmatically select the champion based on AUC performance.
6.  **Handover:** Write the winner's MLflow Run ID to the `best_model_run_info.json`.
