# System Architecture & Data Flow — YouTube Sentiment MLOps

## Overview
This document presents a professional, production-oriented system architecture and data flow for the **YouTube Sentiment MLOps** project. It maps components (API, model service, storage, CI/CD, monitoring) and shows how data moves from collection to serving and retraining. The design prioritizes **reliability, scalability, maintainability, and adaptability**.

---

## 1. High-level architecture (text diagram)

```
[YouTube Data Source] ---> [Data Ingestion Script (DVC)] ---> [Data Lake (S3: Raw / Processed)]
                                   |                                
                                   v
[Training Pipeline (DVC + MLflow + Optuna)]
   |--> Feature Engineering (TF-IDF / BERT)
   |--> Imbalance Handling (SMOTE / ADASYN)
   |--> Model Training (Logistic Regression / XGBoost / LightGBM / DistilBERT)
   |--> Evaluation & Champion Selection
   |--> Model Registry (MLflow)
            |
            v
[Inference Service (FastAPI + Docker)] <--- [User / Client App]
            |
            v
[Prediction Logging (S3 / DB)] ---> [Monitoring Dashboard (Grafana / MLflow)]
```

---

## 2. Component responsibilities

- **Data Pipeline (DVC)**
  - **Ingestion**: `data/download_dataset.py` fetches raw comments (e.g., from Reddit/YouTube).
  - **Preparation**: `data/make_dataset.py` cleans text, encodes labels, and performs stratified splitting.
  - **Feature Engineering**: `features/feature_engineering.py` generates TF-IDF vectors or BERT embeddings.
  - **Tuning**: Dedicated stages (`feature_tuning`, `imbalance_tuning`) optimize n-grams, max features, and sampling strategies.

- **Training & Experimentation (MLflow + Optuna)**
  - **Baseline**: Logistic Regression with balanced class weights for rapid benchmarking.
  - **Advanced Models**: LightGBM and XGBoost with Optuna hyperparameter optimization.
  - **Deep Learning**: DistilBERT fine-tuning for complex semantic capture (optional stage).
  - **Evaluation**: `models/model_evaluation.py` compares all trained models on hold-out test set.
  - **Registry**: `models/register_model.py` promotes the best model if it exceeds the F1 threshold.

- **Model Storage & Artifacts**
  - **MLflow Model Registry**: Central repository for versioned models (Staging, Production).
  - **DVC Remote**: Tracks large files (datasets, `.pkl` models, `.npz` features) in S3.
  - **Experiments**: MLflow tracking server logs parameters, metrics, and artifacts for every run.

- **Configuration Management**
  - `params.yaml`: Single source of truth for all pipeline parameters (paths, hyperparameters, thresholds).
  - `src/config/manager.py`: Pydantic-based configuration validator ensuring type safety.

- **CI/CD & Automation**
  - **GitHub Actions**: Runs linting, unit tests (`pytest`), and dvc checks on push.
  - **Docker**: Encapsulates the training and inference environment for reproducibility.

---

## 3. Data flow (detailed sequence)

1.  **Ingestion**: Raw data is downloaded and saved to `data/raw/`. DVC tracks this file.
2.  **Preparation**: Raw text is cleaned, normalized, and split into Train/Val/Test parquet files in `data/processed/`.
3.  **Featurization**: Text is converted to numerical features (TF-IDF or Embeddings) and saved as compressed `npz` files in `models/features/`.
4.  **Tuning**: 
    - `feature_tuning` finds optimal TF-IDF parameters.
    - `imbalance_tuning` finds best resampling method (e.g., SMOTE, ADASYN).
5.  **Training**: multiple models (Logistic, XGBoost, LightGBM, DistilBERT) are trained on the prepared features.
6.  **Evaluation**: All models are evaluated on the Test set. Metrics (F1, Accuracy, AUC) are logged to MLflow and JSON files.
7.  **Registration**: The `register_model` stage identifies the "Champion" model based on F1 score and registers it in MLflow if usage criteria are met.

---

## 4. DVC & MLflow integration (practical snippets)

**`dvc.yaml` (stages example)**:
```yaml
stages:
  data_preparation:
    cmd: python -m src.data.make_dataset
    deps: [data/raw/reddit_comments.csv]
    outs: [data/processed/train.parquet]

  model_evaluation:
    cmd: python -m src.models.model_evaluation
    deps: 
      - models/advanced/lightgbm_model.pkl
      - models/baseline/logistic_baseline.pkl
    outs: [models/advanced/evaluation/best_model_run_info.json]
```

**`src/models/model_evaluation.py` (concept)**:
```python
# Select best model
best_model_name = max(metrics, key=lambda k: metrics[k]['test_macro_f1'])
# Save run info for registration
with open("best_model_run_info.json", "w") as f:
    json.dump({"run_id": run_id, "model_name": best_model_name}, f)
```

---

## 5. Monitoring & Observability
- **Model quality**: Per-class Precision/Recall/F1 on validation and test sets.
- **Drift detection**: Comparison of input feature distributions over time (planned).
- **System Health**: Logs captured via `src/utils/logger.py` and MLflow traces.

---

## 6. Next steps
- **Deployment**: Implement FastAPI wrapper around the registered model.
- **Containerization**: Create `Dockerfile` for serving the API.
- **Real-time**: Connect to YouTube API for live comment analysis.
