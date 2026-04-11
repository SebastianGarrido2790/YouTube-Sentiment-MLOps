# DVC Pipeline Architecture Report

**Project:** YouTube Sentiment Analysis — Hybrid Agentic MLOps System  
**Version:** 2.0.0  
**Date:** 2026-04-11  
**Status:** Production-Ready

---

## Executive Summary

This document describes the complete pipeline architecture of the YouTube Sentiment Analysis project. The system operates on a **dual-layer orchestration model**: a **DVC-managed DAG** for automated, reproducible data science execution, and a **FastAPI Orchestrator Microservice** (`main.py`) as the centralized entry point for development and local execution.

These two layers are **complementary, not competing**. DVC handles dependency tracking, caching, and CI/CD triggering; `main.py` provides human-facing control, asynchronous background execution, and real-time AgentOps telemetry. Together, they fulfill the **FTI (Feature, Training, Inference)** pattern from raw data ingestion through to production model registration.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION DUAL-LAYER                           │
│                                                                         │
│  ┌─────────────────┐                    ┌───────────────────────────┐   │
│  │  dvc.yaml (DAG) │  ←── CI/CD / prod  │  main.py (FastAPI)        │   │
│  │  Reproducible   │                    │  Development / Local      │   │
│  │  Automated      │                    │  AgentOps Metrics         │   │
│  │  Cached stages  │                    │  Async Background Tasks   │   │
│  └────────┬────────┘                    └─────────────┬─────────────┘   │
│           │                                           │                 │
│           └──────────────────┬────────────────────────┘                 │
│                              ▼                                          │
│           ┌──────────────────────────────────────────┐                  │
│           │          FTI PIPELINE STAGES             │                  │
│           └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. FTI Pipeline — Full DAG

The pipeline follows the **Feature → Training → Inference (FTI)** pattern. Each stage is an independently deployable component.

```
data_ingestion
      │
      ├──► data_validation (GX data contracts)
      │
      └──► data_preparation
                │
                ├──► feature_comparison  (transient, MLflow experiments)
                │
                ├──► feature_tuning      (transient, MLflow experiments)
                │
                ├──► imbalance_tuning    (transient, MLflow experiments)
                │
                └──► feature_engineering
                              │
                              ├──► baseline_model   ──────────────────────┐
                              │                                           │
                              ├──► hyperparameter_tuning_lightgbm ────────┤
                              │                                           │
                              ├──► hyperparameter_tuning_xgboost  ────────┤
                              │                                           │
                              └──► train_distilbert ────────────────────► model_evaluation
                                                                               │
                                                                         register_model
```

### 2.1 Stage Dependency Mapping

| Stage | Inputs | Key Outputs | Config Params |
|:---|:---|:---|:---|
| `data_ingestion` | Remote URL | `data/raw/reddit_comments.csv` | `data_ingestion.*` |
| `data_validation` | Raw CSV | `artifacts/gx/` (GX suites) | `data_validation.*` |
| `data_preparation` | Raw CSV | `train/val/test .parquet` | `test_size`, `random_state` |
| `feature_comparison` | Processed parquets | *(transient — MLflow only)* | `feature_comparison.*` |
| `feature_tuning` | Processed parquets | `reports/figures/tfidf_max_features/` | `feature_tuning.*` |
| `imbalance_tuning` | Processed parquets | `reports/figures/imbalance_methods/` | `imbalance_tuning.*` |
| `feature_engineering` | Processed parquets | `artifacts/models/features/` (`.npz`, `.npy`, `.pkl`) | `feature_engineering.*` |
| `baseline_model` | Feature matrices | `logistic_baseline.pkl`, `baseline_metrics.json` | `train.logistic_baseline.*` |
| `hyperparameter_tuning_lightgbm` | Feature matrices | `lightgbm_model.pkl`, `lightgbm_metrics.json` | `train.hyperparameter_tuning.lightgbm` |
| `hyperparameter_tuning_xgboost` | Feature matrices | `xgboost_model.pkl`, `xgboost_metrics.json` | `train.hyperparameter_tuning.xgboost` |
| `train_distilbert` | Processed parquets | `distilbert_model.pkl`, `distilbert_metrics.json` | `train.distilbert.*` |
| `model_evaluation` | All model artifacts + features | `best_model_run_info.json`, confusion matrices, ROC curves | `model_evaluation.models` |
| `register_model` | Evaluation results | MLflow Production registration | `register.f1_threshold` |

---

## 3. `main.py` — Orchestrator Microservice

### 3.1 Design Rationale

`main.py` was refactored from a traditional sequential script into a **FastAPI microservice** to align with **Tools as Microservices** and **AgentOps — MLOps for Agentic Systems**.

While `dvc repro` handles automated DAG execution in CI/CD, `main.py` provides:

| Capability | DVC (`dvc repro`) | `main.py` (Orchestrator) |
|:---|:---:|:---:|
| Dependency caching | ✅ | — |
| DAG stage ordering | ✅ | ✅ (hardcoded DAG) |
| CI/CD integration | ✅ | — |
| Async background execution | — | ✅ |
| AgentOps metrics tracking | — | ✅ |
| Run lifecycle management | — | ✅ |
| HTTP API control surface | — | ✅ |
| Retry / Agentic Healing | — | ✅ |
| Development entry point | — | ✅ |

### 3.2 API Endpoints

```
GET  /health                → System heartbeat (service, project name)
POST /v1/train              → Triggers full FTI pipeline as background task
GET  /v1/status/{run_id}    → Retrieves lifecycle state of a specific run
GET  /v1/metrics            → Exposes AgentOps metrics for auditing
```

### 3.3 Pipeline Stage DAG (as defined in `run_training_pipeline`)

The orchestrator defines an ordered execution list mirroring the DVC DAG:

```python
stages = [
    ("Data Ingestion",      DataIngestionPipeline()),      # stage_01
    ("Data Validation",     DataValidationPipeline()),     # stage_01b
    ("Data Preparation",    DataPreparationPipeline()),    # stage_02
    ("Feature Engineering", FeatureEngineeringPipeline()), # stage_03
    ("Model Training",      DistilBERTTrainingPipeline()), # stage_04c
    ("Model Evaluation",    ModelEvaluationPipeline()),    # stage_05
    ("Model Registration",  ModelRegistrationPipeline()),  # stage_06
]
```

> [!NOTE]
> The orchestrator runs the main pipeline path (with DistilBERT). Experimental stages
> (`feature_comparison`, `feature_tuning`, `imbalance_tuning`, and parallel LightGBM/XGBoost
> tuning) are intentionally run exclusively via `dvc repro` since they are research tools,
> not production pipeline stages.

### 3.4 Agentic Healing — Retry Logic

Each stage is wrapped in a **retry loop with exponential backoff simulation** (Rule 2.7 — Agentic Healing):

```
For each stage:
    attempt 0  → execute pipeline.main()
    if fails:
        attempt 1  → wait 1s (backoff) → retry
    if fails:
        attempt 2  → wait 1s (backoff) → retry
    if fails after max_retries:
        → raise exception
        → record failed_stage in CustomExceptionError metadata
        → increment AGENT_METRICS.failed_tool_calls
```

This prevents transient I/O or network errors from crashing the entire pipeline.

---

## 4. AgentOps Metrics Layer

### 4.1 Schema (`src/entity/api_entity.py`)

```python
class AgentOpsMetrics(BaseModel):
    total_plans_executed: int   # Total pipeline runs triggered
    failed_tool_calls:    int   # Individual stage failures
    plan_success_rate:    float # Completed runs / total runs
    tool_call_accuracy:   float # Successful stages / total stages per run
    avg_retry_latency:    float # Avg seconds spent in retry loops

class PipelineStatus(BaseModel):
    run_id:        str                   # UUID
    status:        str                   # pending | running | completed | failed
    current_stage: str                   # Active stage name
    error:         str | None            # Failure message
    metrics:       AgentOpsMetrics | None
```

### 4.2 Metric Computation Flow

```
POST /v1/train called
        │
        ▼
  run_id = uuid4()
  PIPELINE_REGISTRY[run_id] = PipelineStatus(status="pending")
  background_tasks.add_task(run_training_pipeline, run_id)
        │
        ▼  (background)
  status.status = "running"
  AGENT_METRICS.total_plans_executed += 1
        │
        ▼  (for each stage)
  pipeline.main()
  successful_tools += 1
  [on retry: total_retries++, track latency]
        │
        ▼  (finally block — always executes)
  plan_success_rate  = completed_runs / total_plans_executed
  tool_call_accuracy = successful_tools / total_tools
  avg_retry_latency  = total_retry_latency / total_retries (if any)
  status.metrics     = AGENT_METRICS (snapshot)
```

### 4.3 Auditing via Exception Enrichment

Failures are captured with **rich domain metadata** via `CustomExceptionError`:

```python
CustomExceptionError(
    error_message=e,
    error_detail=sys,
    agent_metadata={
        "run_id":                   run_id,
        "failed_stage":             status.current_stage,
        "total_successful_stages":  successful_tools,
        "retries_attempted":        total_retries,
    }
)
```

This guarantees that automated workflows **fail loudly** in tracing logs while providing the context needed for self-correction.

---

## 5. Configuration Management

All pipeline behavior is controlled through versioned YAML files — no naked strings in code.

### 5.1 Configuration Files

| File | Role |
|:---|:---|
| `config/params.yaml` | Tunable hyperparameters only (model params, thresholds, split ratios) |
| `config/config.yaml` | Immutable system paths (artifact dirs, API URLs, infra settings) |
| `config/schema.yaml` | Data contract — enforced by GX in `data_validation` stage |

### 5.2 Configuration Flow

```
params.yaml + config.yaml + schema.yaml
            │
            ▼
  ConfigurationManager (singleton)
            │
            ▼
  AppConfig (Pydantic, frozen=True, extra="forbid")
  SystemConfig
  SchemaConfig
            │
            ├──► Per-stage config getters
            │    (e.g., get_feature_engineering_config())
            │
            └──► AgentConfig (merged from params + infra)
```

> [!IMPORTANT]
> `AppConfig` uses `extra="forbid"` on all nested schemas. Any undeclared key in
> `params.yaml` will raise a `ValidationError` at startup, preventing silent
> misconfiguration.

---

## 6. Data Lineage & Artifact Map

```
data/
└── raw/
    └── reddit_comments.csv                        ← data_ingestion output

artifacts/
├── gx/                                            ← data_validation output (GX suites)
├── data/
│   └── processed/
│       ├── train.parquet                          ← data_preparation output
│       ├── val.parquet
│       └── test.parquet
└── models/
    ├── features/
    │   ├── X_train.npz / X_val.npz / X_test.npz  ← feature_engineering output
    │   ├── y_train.npy / y_val.npy / y_test.npy
    │   ├── vectorizer.pkl
    │   └── label_encoder.pkl
    ├── baseline/
    │   ├── logistic_baseline.pkl                  ← baseline_model output
    │   └── baseline_metrics.json
    └── advanced/
        ├── lightgbm_model.pkl                     ← hyperparameter_tuning_lightgbm
        ├── lightgbm_metrics.json
        ├── xgboost_model.pkl                      ← hyperparameter_tuning_xgboost
        ├── xgboost_metrics.json
        ├── distilbert_model.pkl                   ← train_distilbert output
        ├── distilbert_metrics.json
        └── evaluation/
            ├── best_model_run_info.json           ← model_evaluation output (champion)
            ├── lightgbm_test_metrics.json
            ├── xgboost_test_metrics.json
            └── logistic_baseline_test_metrics.json

reports/
└── figures/
    ├── tfidf_max_features/                        ← feature_tuning output
    ├── imbalance_methods/                         ← imbalance_tuning output
    └── evaluation/
        ├── lightgbm_confusion_matrix.png          ← model_evaluation output
        ├── xgboost_confusion_matrix.png
        └── comparative_roc_curve.png
```

---

## 7. Development Workflow

### 7.1 Running the Full Pipeline

| Context | Command | Description |
|:---|:---|:---|
| **Development / Local** | `uv run python main.py` | Starts FastAPI orchestrator on `http://localhost:8080` |
| **Trigger pipeline run** | `POST http://localhost:8080/v1/train` | Async pipeline execution with AgentOps tracking |
| **Check run status** | `GET http://localhost:8080/v1/status/{run_id}` | Poll lifecycle state |
| **View metrics** | `GET http://localhost:8080/v1/metrics` | Inspect AgentOps telemetry |
| **CI/CD / Automated** | `dvc repro` | Full DAG reproduction with caching |
| **Partial rerun** | `dvc repro <stage_name>` | Rerun from a specific stage |
| **DAG visualization** | `dvc dag` | Print the dependency graph |
| **Metrics diff** | `dvc metrics diff` | Compare metrics across Git commits |

### 7.2 Validation

Before pushing to version control, always run the full system health check:

```bash
.\validate_system.bat
```

This checks all 4 pillars:
1. **Pillar 1** — Pyright (0 errors) + Ruff (linting + formatting)
2. **Pillar 2** — Pytest (76 tests, ≥50% coverage gate)
3. **Pillar 3** — DVC pipeline status (lineage integrity)
4. **Pillar 4** — FastAPI service health (Inference + Insights ports)

---

## 8. Orchestration Decision Rationale

### Why Maintain Both DVC and `main.py`?

This is a deliberate architectural decision, not redundancy.

**DVC (`dvc.yaml`) is the source of truth for:**
- Reproducibility — stage outputs are hashed and cached
- MLOps integrity — `dvc metrics diff` tracks experiment drift across commits
- CI/CD — GitHub Actions can trigger `dvc repro` to validate every PR
- Collaboration — any team member can reproduce the pipeline from `dvc.lock`

**`main.py` (Orchestrator Microservice) is the source of truth for:**
- Development UX — a single HTTP call triggers the pipeline instead of CLI chaining
- AgentOps telemetry — real-time metrics that DVC cannot natively expose
- Agentic integration — future AI agents call `/v1/train` to trigger retraining autonomously (Rule 1.3, Rule 1.14)
- Async execution — pipeline runs in the background; caller is not blocked
- Failure auditing — enriched `CustomExceptionError` metadata for self-healing diagnostics

> [!TIP]
> Think of DVC as the **contract** (what the pipeline does and how it is tracked) and
> `main.py` as the **interface** (how humans and agents interact with the pipeline at
> runtime). One validates the system; the other operates it.

---

## 9. Future Roadmap

| Priority | Enhancement | Rationale |
|:---|:---|:---|
| High | Persist `PIPELINE_REGISTRY` to Redis/PostgreSQL | Survive service restarts; support multi-replica deployment |
| High | Docker Compose integration | Containerize Orchestrator alongside MLflow and inference services |
| Medium | Grafana dashboard for `AgentOps` metrics | Visualize `plan_success_rate` and `avg_retry_latency` over time |
| Medium | GitHub Actions `dvc repro` workflow | Automate pipeline validation on every PR |
| Low | LangGraph integration for multi-agent orchestration | Enable autonomous retraining triggered by model drift detection |
| Low | `dvc studio` integration | Cloud-based experiment tracking and team collaboration |

---

*Last updated: 2026-04-11 — Hybrid Agentic MLOps System v2.0.0*
