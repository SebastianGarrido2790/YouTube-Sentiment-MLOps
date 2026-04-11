# Technical Implementation Report: Hyperparameter Optimization Pipeline

**Stage:** 08 — Hyperparameter Tuning  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Optuna + LightGBM + XGBoost + MLflow

---

## 1. Overview
The Hyperparameter Optimization (HPO) implementation leverages the Optuna framework to find the "Champion" configuration for gradient-boosted trees. It implements an advanced research loop that automates parameter suggestion, trial evaluation, and final model artifact generation within a strictly governed MLOps context.

---

## 2. Technical Stack
- **Optimization Strategy:** `Optuna` (TPE Sampler).
- **Estimators:** `lightgbm.LGBMClassifier` and `xgboost.XGBClassifier`.
- **Telemetry:** `mlflow` (Nested Runs API).
- **Serialization:** `pickle` / `joblib` for model bundles and hyperparameter states.

---

## 3. Implementation Workflow

### 3.1 The Optuna Study Layer
The implementation initializes a stateful study using `optuna.create_study(direction="maximize")`.
- **Direction:** Target is the **Macro F1-Score** on the validation split.
- **Pruning:** Implements the `OptunaPruningCallback` from `optuna.integration.mlflow` to terminate non-promising trials after the first few boosting iterations, significantly reducing token/compute waste.

### 3.2 Dynamic Search Spaces
The core `objective(trial)` function implements differentiated search logics based on the target model type:

#### LightGBM Search Space:
- `num_leaves`: [31, 256]
- `learning_rate`: [0.01, 0.3] (log-uniform)
- `feature_fraction`: [0.4, 1.0]
- `bagging_fraction`: [0.4, 1.0]

#### XGBoost Search Space:
- `max_depth`: [3, 10]
- `eta` (learning_rate): [0.01, 0.3] (log-uniform)
- `gamma`: [0, 5]
- `colsample_bytree`: [0.5, 1.0]

### 3.3 Nested Telemetry Pattern
To manage the high volume of experimental data, the implementation uses the **MLflow Nested Run** pattern:
1.  **Parent Run:** The "Study" run (e.g., `HPO_LightGBM_Search`).
2.  **Child Runs:** Each individual trial is logged as a child. This keeps the primary MLflow dashboard clean, as trial-specific parameters (`trial_2_learning_rate`) are isolated from production-ready parameters.

---

## 4. Retraining & Preservation

### 4.1 Champion Extraction
Once the search loop (`n_trials`) is complete, the implementation extracts the best parameters via `study.best_params`.

### 4.2 Final Retraining Loop
Instead of saving a "trial" model (which may have been partially trained due to pruning), the implementation performs a **Clean-Room Retrain**:
- Instantiates a fresh model with `best_params`.
- Trains it on the full training set until convergence.
- Persists the result as an **Atomic Model Bundle** (Estimator + LabelEncoder).

---

## 5. Robustness & Resource Management

- **Memory Safety:** Large sparse feature matrices are loaded once and shared across all trials via the objective function closure, preventing memory leaks during high-iteration studies.
- **Fail-Safe Search:** The implementation includes a `try-except` block around the `study.optimize` call. If an individual trial crashes (e.g., due to an invalid param combination), Optuna logs the failure but continues with the next trial, ensuring the pipeline completes.
- **Categorical Integrity:** The implementation explicitly sets `objective="multiclass"` and `num_class=3` for both boosting libraries to ensure they are configured for the sentiment classification task correctly.

---

## 6. Execution Logic Summary
1.  **Initialize:** Load CLI arguments to determine model type (`--model`).
2.  **Bootstrap:** Load sparse features from `artifacts/feature_engineering/`.
3.  **Search:** Execute Optuna's `TPE` sampler for $N$ trials.
4.  **Audit:** Record trial-specific metrics and loss curves into MLflow child runs.
5.  **Promote:** Identify the `best_params` and execute the final retraining cycle.
6.  **Persist:** Export `best_model.pkl`, `best_hyperparams.pkl`, and the `metrics.json` report.
