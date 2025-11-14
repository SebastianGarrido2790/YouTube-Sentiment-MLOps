## Baseline Model Training Script: src/models/baseline_logistic.py

Based on the MLflow metrics from imbalance tuning experiments, the baseline uses **class_weight="balanced"** for simple, deterministic imbalance handling. This approach is preferred for baseline models as it's built-in, stable, and fully reproducible without synthetic data generation.

This script loads pre-engineered TF-IDF features from `models/features/`, trains a Logistic Regression baseline with balanced class weights, and logs to MLflow (accuracy, macro F1, per-class F1). It evaluates on train/val/test splits for comprehensive benchmarking.

#### Usage and Best Practices
- **Run**: `uv run python -m src.models.baseline_logistic`
- **Reliability**: Uses deterministic class weights; robust logging of per-class F1.
- **Scalability**: Sparse matrices efficient; handles large feature spaces.
- **Maintainability**: Modular design with shared helpers; DVC tracks outputs.
- **Adaptability**: Simple baseline for comparison with advanced models.

The baseline establishes a reliable benchmark before applying complex methods like ADASYN or hyperparameter tuning.

### Advanced Model Training Script: src/models/hyperparameter_tuning.py

This centralized script handles hyperparameter optimization for multiple models (LightGBM, XGBoost) using Optuna. It tunes TF-IDF features with ADASYN oversampling (best method from imbalance experiments) over 30 trials per model, using validation F1 (macro) as objective. Results logged to MLflow with nested runs for organized tracking.

Dependencies are already in `pyproject.toml`:
```
optuna>=3.6
xgboost>=2.0
lightgbm>=4.3
```

#### Usage and Best Practices
- **Execution**: `uv run python -m src.models.hyperparameter_tuning --model lightgbm` or `--model xgboost`
- **Reliability**: Parent runs group trials; best parameters saved for final evaluation.
- **Scalability**: Efficient sparse matrix handling; parallelizable Optuna trials.
- **Maintainability**: Centralized tuning logic reduces code duplication.
- **Adaptability**: Easy to extend for new models by adding objective functions.

Best models are saved to `models/advanced/` for final evaluation and comparison.

### BERT Training Script: src/models/bert_training.py

BERT fine-tuning is handled separately with proper label encoding (shifting {-1,0,1} to {0,1,2}) for PyTorch compatibility. The script uses Hugging Face transformers with Optuna hyperparameter tuning over learning rate, batch size, and weight decay.

#### Usage and Best Practices
- **Execution**: Controlled via params.yaml (`train.bert.enable: true`)
- **Reliability**: Proper label encoding prevents PyTorch cross-entropy errors.
- **Scalability**: GPU recommended; uses `accelerate` for distributed training.
- **Maintainability**: Separate script keeps heavy dependencies isolated.

BERT training is optional and disabled by default based on current pipeline configuration.

---

### Model Evaluation Script: src/models/model_evaluation.py

The final evaluation script loads the best LightGBM model from hyperparameter tuning and evaluates it on the held-out test set. It generates comprehensive metrics, confusion matrices, ROC curves, and markdown reports for stakeholder review.

#### Key Features
- **Comprehensive Evaluation**: Test set evaluation with classification reports, confusion matrices, and ROC curves
- **Artifact Generation**: Saves plots and metrics for DVC tracking and reporting
- **MLflow Integration**: Logs final test metrics for model registry decisions
- **Report Generation**: Creates markdown evaluation reports with visualizations

#### Usage
- **Execution**: `uv run python -m src.models.model_evaluation`
- **Outputs**: Evaluation report, confusion matrix, ROC curve, test metrics JSON
- **Integration**: Prepares model for registration stage with performance thresholds

---
 
### Current Model Performance

Based on hyperparameter tuning experiments with ADASYN oversampling and optimal TF-IDF features (max_features=1000, ngram_range=(1,1)):

| Model | Best Macro F1 Score | Validation Method |
| :--- | :--- | :--- |
| **LightGBM** | **~0.80** | Optuna tuning with 30 trials |
| **Logistic Regression** | **~0.79** | Baseline with class_weight="balanced" |
| **XGBoost** | **~0.78** | Optuna tuning with 30 trials |

**LightGBM achieved the highest performance** during Optuna hyperparameter optimization, making it the selected model for final evaluation and potential deployment.

### Rationale for F1-Score Focus

Macro F1-score is prioritized due to class imbalance (Negative: ~22%, Neutral: ~35%, Positive: ~43%) and the need for balanced performance across all sentiment classes. This metric:
- **Handles imbalance** better than accuracy by balancing precision and recall
- **Aligns with task requirements** for detecting both positive and negative sentiment accurately
- **Supports MLOps** by providing a single metric for model comparison and selection

---

## DVC Pipeline Integration

The modeling pipeline is fully integrated with DVC for reproducible execution:

```yaml
# Baseline model training
baseline_model:
  cmd: uv run python -m src.models.baseline_logistic
  deps:
    - models/features/X_train.npz
    - models/features/y_train.npy
    - src/models/baseline_logistic.py
  outs:
    - models/baseline/logistic_baseline.pkl

# Hyperparameter tuning for advanced models
hyperparameter_tuning_lightgbm:
  cmd: uv run python -m src.models.hyperparameter_tuning --model lightgbm
  params:
    - hyperparameter_tuning.lightgbm.n_trials
  outs:
    - models/advanced/lightgbm_model.pkl
    - models/advanced/lightgbm_best_hyperparams.pkl
  metrics:
    - models/advanced/lightgbm_metrics.json

# Model evaluation on test set
model_evaluation:
  cmd: uv run python -m src.models.model_evaluation
  deps:
    - models/advanced/lightgbm_model.pkl
    - models/features/X_test.npz
    - models/features/y_test.npy
  outs:
    - models/advanced/evaluation/lightgbm_evaluation_run.json
    - reports/lightgbm_evaluation_report.md
  metrics:
    - models/advanced/evaluation/lightgbm_test_metrics.json
```

This structure ensures:
- **Reproducibility**: All dependencies and parameters are tracked
- **Modularity**: Each stage can run independently
- **Metrics Tracking**: DVC tracks key performance metrics across runs
- **Pipeline Orchestration**: Clear dependency chain from data to evaluation

## MLflow Logging Architecture

The hyperparameter tuning scripts use a nested MLflow logging structure that creates:

- **Parent Runs**: "LightGBM_Optuna_Study" and "XGBoost_Optuna_Study" encapsulate entire optimization studies
- **Child Trials**: Individual trials logged as nested runs (e.g., "LightGBM_Trial_0", "LightGBM_Trial_1")

This design prevents UI clutter while maintaining organized experiment tracking:

| Level | Contents | Purpose |
|-------|----------|---------|
| **Parent** | Best parameters, aggregated metrics, study tags | High-level study comparison and audit trail |
| **Child** | Trial-specific parameters, per-trial metrics | Detailed hyperparameter exploration and analysis |

Benefits:
- **Organization**: 30 trials grouped under single parent run
- **Comparability**: Easy cross-model study comparison at parent level
- **Reproducibility**: Complete parameter and metric history preserved
- **DVC Integration**: JSON metrics enable `dvc metrics diff` for pipeline version comparison

---

## Production Inference Service

### FastAPI Service: app/predict_model.py

The production inference service is implemented as a FastAPI application located in the `app/` directory with the following design objectives:

| Goal                        | Description                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Serve predictions**       | Expose a `/predict` endpoint that loads the best model and returns sentiment predictions. |
| **Use MLflow registry**     | Fetch the latest registered model from MLflow model registry. |
| **CPU-friendly**            | Lightweight, runs without GPU dependencies. |
| **Decoupled preprocessing** | Reuse saved `tfidf_vectorizer.pkl` and `label_encoder.pkl` artifacts from `models/features/`. |
| **Robust error handling**   | Handle malformed requests, missing artifacts, or registry issues gracefully. |

#### Usage
- **Local development**: `uv run python -m app.predict_model`
- **Docker deployment**: Multi-stage Dockerfile with `uv` for dependency management
- **API endpoint**: `POST /predict` with JSON payload containing text comments

#### Optional Enhancements
1. **Automatic model reloading** every N minutes (using background tasks)
2. **Token-based authentication** for production endpoints
3. **CORS middleware** for Chrome extension integration
4. **Request logging middleware** (structured JSON logs)
5. **Health check endpoint** for monitoring

### Chrome Extension Integration

The inference service is designed to integrate with the Chrome extension (`chrome-extension/`) for real-time YouTube comment sentiment analysis:

- **Content Script**: `youtube_api.js` extracts comments from YouTube pages
- **Popup Interface**: `popup.html` + `popup.js` provides user interface
- **API Communication**: Direct calls to FastAPI service for predictions
- **Deployment**: Can be deployed via AWS Lambda, ECS, or standalone containers

This creates a complete end-to-end pipeline from YouTube comment extraction to real-time sentiment prediction.

