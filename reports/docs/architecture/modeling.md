## Baseline Model Training Script: src/models/baseline_logistic.py

Based on the MLflow metrics from imbalance tuning experiments, the baseline uses **class_weight="balanced"** for simple, deterministic imbalance handling. This approach is preferred for baseline models as it's built-in, stable, and fully reproducible without synthetic data generation.

This script loads pre-engineered features from `models/features/`, trains a Logistic Regression baseline with balanced class weights, and logs to MLflow (accuracy, macro F1, per-class F1). It evaluates on train/val/test splits for comprehensive benchmarking.

#### Usage and Best Practices
- **Run**: `python -m src.models.baseline_logistic`
- **Reliability**: Uses deterministic class weights; robust logging of per-class F1.
- **Scalability**: Sparse matrices efficient; handles large feature spaces.
- **Maintainability**: Modular design with shared helpers; DVC tracks outputs.
- **Adaptability**: Simple baseline for comparison with advanced models.

The baseline establishes a reliable benchmark before applying complex methods or hyperparameter tuning.

### Advanced Model Training Script: src/models/hyperparameter_tuning.py

This centralized script handles hyperparameter optimization for multiple models (LightGBM, XGBoost) using Optuna. It tunes text features (from `models/features/X_train.npz` and `y_train.npy`) over 30 trials per model, using validation F1 (macro) as the objective. Results are logged to MLflow with nested runs for organized tracking.

Using `params.yaml`, you can configure:
- `hyperparameter_tuning.lightgbm.n_trials`
- `hyperparameter_tuning.xgboost.n_trials`

#### Usage and Best Practices
- **Execution**: `python -m src.models.hyperparameter_tuning --model lightgbm` or `--model xgboost`
- **Reliability**: Parent runs group trials; best parameters saved for final evaluation.
- **Scalability**: Efficient sparse matrix handling; parallelizable Optuna trials.
- **Maintainability**: Centralized tuning logic reduces code duplication.
- **Adaptability**: Easy to extend for new models by adding objective functions.

Best models are saved to `models/advanced/` for final evaluation and comparison.

### DistilBERT Training Script: src/models/distilbert_training.py

DistilBERT fine-tuning is handled separately. The inputs are tokenized text via `datasets` and `transformers` libraries, handling label mapping (`-1`, `0`, `1` -> `0`, `1`, `2`) automatically. The script uses Hugging Face transformers with Optuna hyperparameter tuning over learning rate, batch size, and weight decay.

#### Usage and Best Practices
- **Execution**: Controlled via `params.yaml` (`train.distilbert.enable: true` or `false`)
- **Reliability**: Requires CUDA typically; has internal checks to skip if GPU is unavailable or `enable=false`.
- **Scalability**: GPU recommended; uses `Trainer` API for optimization.
- **Maintainability**: Separate script keeps heavy dependencies isolated.

DistilBERT training is optional and can be toggled based on pipeline configuration.

---

### Model Evaluation Script: src/models/model_evaluation.py

The final evaluation script loads the best models (LightGBM, XGBoost, Logistic Baseline) and evaluates them on the held-out test set (`X_test`, `y_test`). It generates comprehensive metrics, confusion matrices, ROC curves, and determines a "champion" based on macro F1 and AUC.

#### Key Features
- **Comprehensive Evaluation**: Test set evaluation with classification reports, confusion matrices, and ROC curves
- **Artifact Generation**: Saves plots and metrics for DVC tracking and reporting.
- **Champion Selection**: Outputs `best_model_run_info.json` used by the model registry.
- **MLflow Integration**: Logs final test metrics for model comparisons.

#### Usage
- **Execution**: `python -m src.models.model_evaluation`
- **Outputs**: Evaluation report, confusion matrix, ROC curve, test metrics JSON, champion info JSON.
- **Integration**: Prepares model for registration stage with performance thresholds.

---
 
### Current Model Performance

Based on hyperparameter tuning experiments with optimal features (typically unigrams, max_features=5000):

| Model | Macro F1 Score (Val) | Method |
| :--- | :--- | :--- |
| **LightGBM** | **~0.80** | Optuna tuning with 30 trials |
| **Logistic Regression** | **~0.79** | Baseline with class_weight="balanced" |
| **XGBoost** | **~0.78** | Optuna tuning with 30 trials |

**LightGBM achieved the highest performance** during Optuna hyperparameter optimization, making it a strong candidate. However, the final `model_evaluation` stage determines the true winner on the unseen test set.

### Rationale for F1-Score Focus

Macro F1-score is prioritized due to class imbalance and the need for balanced performance across all sentiment classes (Negative, Neutral, Positive). This metric:
- **Handles imbalance** better than accuracy by balancing precision and recall per class.
- **Aligns with task requirements** for detecting minority classes accurately.
- **Supports MLOps** by providing a single metric for model comparison and selection.

---

## DVC Pipeline Integration

The modeling pipeline is fully integrated with DVC for reproducible execution. Below is a conceptual view of `dvc.yaml` dependencies:

```yaml
# Baseline model training
baseline_model:
  cmd: python -m src.models.baseline_logistic
  deps:
    - models/features/X_train.npz
    - models/features/y_train.npy
    - src/models/baseline_logistic.py
  outs:
    - models/baseline/logistic_baseline.pkl

# Hyperparameter tuning (LightGBM)
hyperparameter_tuning_lightgbm:
  cmd: python -m src.models.hyperparameter_tuning --model lightgbm
  params:
    - train.hyperparameter_tuning.lightgbm.n_trials
  outs:
    - models/advanced/lightgbm_model.pkl
    - models/advanced/lightgbm_best_hyperparams.pkl
  metrics:
    - models/advanced/lightgbm_metrics.json

# Model evaluation on test set
model_evaluation:
  cmd: python -m src.models.model_evaluation
  deps:
    - models/advanced/lightgbm_model.pkl
    - models/features/X_test.npz
    - models/features/y_test.npy
  outs:
    - models/advanced/evaluation/best_model_run_info.json
    - reports/figures/evaluation/comparative_roc_curve.png
  metrics:
    - models/advanced/evaluation/lightgbm_test_metrics.json
```

This structure ensures:
- **Reproducibility**: All dependencies and parameters are tracked.
- **Modularity**: Each stage can run independently.
- **Metrics Tracking**: DVC tracks key performance metrics across runs.
- **Pipeline Orchestration**: Clear dependency chain from data to evaluation.

## MLflow Logging Architecture

The hyperparameter tuning scripts use a nested MLflow logging structure that creates:

- **Parent Runs**: "LightGBM_Optuna_Study" and "XGBoost_Optuna_Study" encapsulate entire optimization studies.
- **Child Trials**: Individual trials logged as nested runs (e.g., "LightGBM_Trial_0", "LightGBM_Trial_1").

This design prevents UI clutter while maintaining organized experiment tracking:

| Level | Contents | Purpose |
|-------|----------|---------|
| **Parent** | Best parameters, aggregated metrics, study tags | High-level study comparison and audit trail |
| **Child** | Trial-specific parameters, per-trial metrics | Detailed hyperparameter exploration and analysis |

Benefits:
- **Organization**: Trials grouped under single parent run.
- **Comparability**: Easy cross-model study comparison at parent level.
- **Reproducibility**: Complete parameter and metric history preserved.
- **DVC Integration**: JSON metrics enable `dvc metrics diff` for pipeline version comparison.
