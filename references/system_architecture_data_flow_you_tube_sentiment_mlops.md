# System Architecture & Data Flow — YouTube Sentiment MLOps

## Overview
This document presents a professional, production-oriented system architecture and data flow for the **YouTube Sentiment MLOps** project. It maps components (Chrome extension, API, model service, storage, CI/CD, monitoring) and shows how data moves from collection to serving and retraining. The design prioritizes **reliability, scalability, maintainability, and adaptability**.

---

## 1. High-level architecture (text diagram)

```
[Chrome Extension] ---> [API Gateway / Load Balancer] ---> [FastAPI Inference Service (Docker/ECS)] ---> [ML Model (cached, TorchServe or in-process)]
                                  |                                |                                   |
                                  |                                |                                   +--> [MLflow Model Registry]
                                  |                                |                                   |
                                  |                                +--> [S3: raw / processed data] <--+-- [DVC Remote]
                                  |                                |                                   |
                                  |                                +--> [RDS / DynamoDB: metadata & usage logs]
                                  |
                                  +--> [Auth Service / API Keys]

Background & offline components:
[CI/CD - GitHub Actions] -> builds Docker image -> pushes to ECR -> deploys to ECS/Fargate
[Training Pipeline (DVC + MLflow + Optuna)] -> runs on Dev/Prod EC2 or Batch -> stores artifacts in S3 & MLflow

Monitoring & Observability:
[CloudWatch / Prometheus + Grafana] <- metrics/logs <- inference service
[Alerting (SNS / PagerDuty)] <- CloudWatch alarms

Optional: [Kinesis / Kafka] for high-throughput comment stream ingestion
```

---

## 2. Component responsibilities

- **Chrome Extension**
  - Capture comment text, video metadata, timestamp, language, video id
  - Batch and throttle sends to backend to respect user privacy and rate limits
  - Include unique client id (anonymized) for telemetry

- **API Gateway / Load Balancer**
  - Authenticate requests (API key / JWT)
  - Route to inference instances
  - Throttle, rate-limit, and apply WAF rules

- **FastAPI Inference Service** (containerized)
  - Endpoint `/predict` accepts JSON: `{comment, video_id, timestamp, lang}`
  - Preprocessing pipeline (tokenization / normalization)
  - Model inference (TF-IDF + classifier OR BERT)
  - Caching layer (Redis) for repeated inference
  - Log request/response metrics and sampled raw texts to S3 for labeling/drift analysis

- **Model Storage & Registry**
  - MLflow Model Registry manages model versions, stages (Staging, Production)
  - Model artifacts stored in S3 and referenced by MLflow
  - DVC tracks datasets and intermediate artifacts; remote storage is S3

- **Training & Experimentation**
  - DVC pipeline stages (data_ingest -> preprocess -> featurize -> train -> eval -> register)
  - Optuna for hyperparameter search, MLflow for experiment logging
  - Scheduled retraining (e.g., weekly) or triggered by drift detection

- **CI/CD**
  - GitHub Actions: lint/tests → build Docker → push to ECR → deploy to ECS/Fargate
  - Model deployment can be automated via MLflow registry webhook + GitHub Actions

- **Monitoring & Retraining**
  - Metrics: latency, throughput, error rate, per-class F1, prediction distribution
  - Drift detection: Compare live distribution vs. training distribution (JSD, KS)
  - Alerts for model performance drop; automated data-snapshot + retrain job

---

## 3. Data flow (detailed sequence)

1. User navigates YouTube; Chrome extension collects a batch of comments (with minimal metadata).
2. Extension sends HTTP POST to `/ingest` (API) with API key.
3. API Gateway auths request; forwards to FastAPI inference endpoint.
4. FastAPI preprocesses, applies model, returns `{label, score, model_version}`.
5. Sampled raw comments + predictions are written to S3 (`/raw_samples/`) and to a logging DB for analytics.
6. Periodic job (DVC `data_ingest`) aggregates S3 raw samples into DVC-tracked `data/raw` and triggers `data_preprocessing`.
7. DVC + MLflow training pipeline runs, logs experiments to MLflow, and stores artifacts in S3 and `models/`.
8. Once a model passes evaluation gates, MLflow registers a new model version and optionally triggers CI job to deploy.

---

## 4. API design (minimal)

**POST /predict**
- Input: `{ "comment": "text", "video_id": "id", "timestamp": "ISO8601", "lang": "en" }`
- Output: `{ "label": "positive|neutral|negative", "score": 0.92, "model_version": "v1.2.3" }`
- Constraints:  payload limit 2 KB; rate limit 10 req/s per key (example)

**POST /ingest** (for raw storage and telemetry)
- Input: `{ "comment":..., "metadata": {...} }`
- Behavior: write to S3 raw bucket (authenticated, encrypted)

**GET /health**
- Output: `{ "status":"ok", "model_version":"v1.2.3", "uptime": 12345 }`

---

## 5. DVC & MLflow integration (practical snippets)

`dvc.yaml` (stages example):
```yaml
stages:
  ingest:
    cmd: python src/data/download_dataset.py
    outs:
      - data/raw
  preprocess:
    cmd: python src/data/make_dataset.py
    deps: [data/raw]
    outs:
      - data/processed
  train:
    cmd: python src/models/train.py
    deps: [data/processed]
    outs:
      - models/production
```

`mlflow_config.py` (concept):
```python
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_S3_ARTIFACT = os.getenv('MLFLOW_S3_ARTIFACT', 's3://your-bucket/mlflow-artifacts')
```

---

## 6. CI/CD (GitHub Actions) — pipeline sketch

1. `on: push` to `main` or `on: workflow_dispatch`
2. Steps:
   - Checkout
   - Setup Python & install `uv` deps (use `uv install` / `uv sync`)
   - Run `pytest` and lint
   - Run DVC `dvc repro --no-run-cache` (lightweight validation)
   - Build Docker image and push to ECR
   - Deploy to ECS via `aws ecs update-service` or use Terraform

---

## 7. Security & Privacy
- Use encrypted S3 buckets and KMS keys
- Use AWS IAM roles for ECS tasks
- Keep PII minimal; anonymize client ids; add explicit opt-out in extension
- Secure API keys; store secrets in AWS Secrets Manager or GitHub Secrets

---

## 8. Monitoring & Observability (recommended KPIs)
- **Model quality**: Per-class Precision/Recall/F1 (sampled labeled set)
- **Prediction distribution**: percent Positive/Neutral/Negative (hourly)
- **Latency**: p50/p95/p99
- **Errors**: 4xx/5xx counts
- **Data drift**: JSD between live and train token distributions

---

## 9. Config file (params.yaml) — minimal example

```yaml
data:
  raw_path: data/raw
  processed_path: data/processed
training:
  seed: 42
  batch_size: 32
  epochs: 3
model:
  name: bert-base-uncased
  max_len: 128
inference:
  batch_predict_size: 16
  max_payload_kb: 2
```

---

## 10. Next steps / decisions to confirm
- Preferred serving platform: **ECS/Fargate** vs **Lambda** (cost vs latency trade-offs).
- Offline training infra: local runner, EC2, or AWS Batch/Spot instances.
- Real-time stream handling: direct HTTP vs message broker (Kinesis/Kafka) for scale.


---

*Document prepared for review. If you want, I can produce a visual diagram (PNG/SVG) or an infrastructure-as-code starter (Terraform) next.*

