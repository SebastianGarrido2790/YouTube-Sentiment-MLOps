# Product Requirements Document (PRD)
## YouTube Sentiment Analysis MLOps Pipeline

| **Field**          | **Value**                                             |
| :----------------- | :---------------------------------------------------- |
| **Version**        | v1.0                                                  |
| **Date**           | 2026-04-03                                            |
| **Author**         | Sebastian Garrido                                     |
| **Status**         | Active — Hardening Phase                              |
| **Linked Charter** | [`project_charter.md`](./project_charter.md)          |

---

## Project Analogy

> **This system is a "digital focus group" built into the YouTube interface.**
>
> Imagine a TV network that, after every broadcast, instantly receives a report from 10,000 audience members categorized by sentiment, topic, and trend over time, without a survey, without a call center. That's what a focus group gives traditional media. This system gives content creators the same intelligence, in real time, on every video, powered by a machine learning model trained and maintained through an automated MLOps pipeline.
>
> DVC is the production floor that ensures every model is built the same way every time. MLflow is the quality control lab that decides which model is good enough to ship. FastAPI is the dispatch system that gets the product to the consumer. And the Chrome Extension is the storefront where the creator reads the report.

---

## 1. Problem Statement & Opportunity

### 1.1 Problem

YouTube content creators generating substantial viewership (>10K views/video) are unable to synthesize audience sentiment at scale. The platform provides:
- A binary like/dislike ratio (no textual sentiment)
- A raw comment feed (no aggregation or trend detection)
- No topic-level sentiment breakdown (no aspect analysis)

This forces creators to either **ignore comment sentiment** or **manually sample** comments, which is a non-scalable, biased process that leaves actionable feedback on the table.

### 1.2 Engineering Opportunity

Building this system end-to-end requires solving several non-trivial engineering problems:
1. Training a robust, multi-class sentiment classifier on **surrogate domain data** (Reddit) and generalizing it to YouTube comments
2. Designing a **reproducible MLOps pipeline** that ensures any team member can retrain from scratch and produce the same model
3. **Eliminating training-serving skew** — ensuring the text preprocessing and feature engineering at training time are identical to what runs in production
4. Delivering insights through a **browser-native interface** that requires no user installation beyond a Chrome Extension

### 1.3 Strategic Value (Portfolio)

This project is designed not only to solve the technical problem above, but to serve as a **portfolio differentiator** demonstrating:
- FTI (Feature-Training-Inference) pipeline architecture
- Production-grade Python development (strict typing, Pydantic schemas, linting)
- MLOps tooling mastery (DVC + MLflow + Docker + GitHub Actions)
- Full end-to-end product delivery (ML backend → Chrome Extension frontend)

---

## 2. Goals & Non-Goals

### Goals ✅

| Goal | Success Criterion |
| :--- | :--- |
| **Accurate Sentiment Classification** | Macro F1 ≥ 0.75 on the held-out test set across all three classes |
| **Reproducible Pipeline** | `dvc repro` from a clean state produces identical artifacts and metrics |
| **Production-Gated Registration** | No model registers to the MLflow Production alias unless F1 threshold is met |
| **Low-Latency Inference** | `/predict` endpoint responds in < 500ms p95 for ≤ 100 comment batches |
| **Real-Time User Experience** | Chrome Extension displays results within 5 seconds of user clicking "Analyze" |
| **Granular ABSA** | DeBERTa-based ABSA correctly assigns aspect-level sentiment for user-defined topics |
| **CI/CD Integrity** | All commits pass lint, type check, test coverage, and security scanning gates |

### Non-Goals ❌

| Non-Goal | Rationale |
| :--- | :--- |
| **Multilingual support** | English-only scope; extension to other languages requires a separate modeling effort |
| **Real-time comment streaming** | System analyzes on-demand batches; live comment streams are out of scope |
| **User authentication / accounts** | Stateless API design; no user identity management required |
| **YouTube comment write operations** | System is read-only; no automated comment posting or moderation |
| **Training on labeled YouTube data** | Reddit is the training proxy; domain adaptation is a future enhancement |
| **Mobile support** | Chrome Desktop only; mobile browser extensions are not a current target |

---

## 3. Key Features & Requirements

### 3.1 Feature 1 — Automated MLOps Pipeline

**Description:** A 12-stage DVC DAG that ingests raw Reddit data, engineers features, trains models, evaluates performance, and registers the champion model — all reproducible from a single `dvc repro` command.

| Requirement ID | Requirement | Priority |
| :--- | :--- | :---: |
| PIPE-01 | Pipeline must be fully reproducible from a clean state | MUST |
| PIPE-02 | Every stage must have explicit `deps`, `params`, `outs`, and `metrics` in `dvc.yaml` | MUST |
| PIPE-03 | Hyperparameter optimization must use Optuna with ≥ 30 trials per model | MUST |
| PIPE-04 | All experiments must be logged to MLflow with parent/child run structure | MUST |
| PIPE-05 | Model registration must be gated by a configurable F1 threshold (`params.yaml`) | MUST |
| PIPE-06 | Pipeline must support TF-IDF and DistilBERT feature strategies, configurable via `params.yaml` | SHOULD |
| PIPE-07 | Data artifacts must be versioned and remotely stored via DVC (AWS S3) | SHOULD |

### 3.2 Feature 2 — Inference Microservices

**Description:** Two FastAPI services that serve predictions and insights, loaded from the MLflow Model Registry with a local fallback.

| Requirement ID | Requirement | Priority |
| :--- | :--- | :---: |
| API-01 | Inference API must expose `/predict` and `/predict_absa` endpoints | MUST |
| API-02 | Insights API must expose `/wordcloud`, `/pie_chart`, and `/trend_chart` endpoints | MUST |
| API-03 | ABSA model must be lazy-loaded at first request, not at startup | MUST |
| API-04 | Both APIs must include `/health` endpoints with model loading status | MUST |
| API-05 | APIs must load models from MLflow Registry first, local path second | MUST |
| API-06 | Both APIs must have CORS enabled with `chrome-extension://*` origin restriction | MUST |
| API-07 | API startup time must not exceed 10 seconds (excluding ABSA lazy load) | SHOULD |
| API-08 | All endpoints must use Pydantic request/response models for validation | MUST |

### 3.3 Feature 3 — Chrome Extensions

**Description:** Two browser extensions that integrate directly into the YouTube UI, offering standard sentiment and aspect-based analysis.

| Requirement ID | Requirement | Priority |
| :--- | :--- | :---: |
| EXT-01 | Standard Extension must scrape comments from YouTube DOM and call `/predict` | MUST |
| EXT-02 | ABSA Extension must accept user-defined aspects and call `/predict_absa` | MUST |
| EXT-03 | Extensions must display results within the popup UI without page reload | MUST |
| EXT-04 | All API calls must use `AbortController` with a 30-second timeout | MUST |
| EXT-05 | Extensions must display a loading state during API calls | SHOULD |
| EXT-06 | API key must be stored in `chrome.storage.local`, not hardcoded in source | SHOULD |

### 3.4 Feature 4 — Visualizations

**Description:** Server-side generated charts and visualizations for the Insights API.

| Requirement ID | Requirement | Priority |
| :--- | :--- | :---: |
| VIZ-01 | Sentiment distribution pie chart (Positive/Neutral/Negative percentages) | MUST |
| VIZ-02 | Top token wordcloud per sentiment class | MUST |
| VIZ-03 | Monthly sentiment trend line chart | SHOULD |
| VIZ-04 | All charts must be returned as base64-encoded images via JSON | MUST |

### 3.5 Feature 5 — Operational Quality

**Description:** Engineering standards enforced throughout the codebase for maintainability and production readiness.

| Requirement ID | Requirement | Priority |
| :--- | :--- | :---: |
| OPS-01 | All Python code must pass `ruff` lint and format checks | MUST |
| OPS-02 | All Python code must pass `pyright` type checking in CI | MUST |
| OPS-03 | Test coverage must be ≥ 60% (enforced by `pytest-cov` gate in CI) | MUST |
| OPS-04 | Docker image must pass Trivy CVE scan (CRITICAL severity = blocking) | MUST |
| OPS-05 | No secrets committed to repository; `.env.example` must document all required variables | MUST |
| OPS-06 | `src/constants/__init__.py` is the single source of truth for all project paths | MUST |
| OPS-07 | A shared `text_preprocessing.py` module must be used by both training and inference | MUST |
| OPS-08 | All Pydantic schema models must use `ConfigDict(extra="forbid")` | SHOULD |

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  CHROME EXTENSIONS (Presentation Layer)                              │
│  ┌─────────────────────┐     ┌──────────────────────────────────┐    │
│  │  Standard Extension  │     │  ABSA Extension                  │    │
│  │  popup.js → /predict │     │  popup.js → /predict_absa        │    │
│  └──────────┬──────────┘     └───────────────┬──────────────────┘    │
└─────────────┼──────────────────────────────────┼─────────────────────┘
              │ HTTP POST                         │ HTTP POST
              ▼                                   ▼
┌──────────────────────────┐   ┌────────────────────────────────────┐
│  Inference API (:8000)    │   │  Insights API (:8001)              │
│  FastAPI + Uvicorn        │   │  FastAPI + Uvicorn                 │
│  ├── /predict             │   │  ├── /pie_chart                    │
│  ├── /predict_absa        │   │  ├── /wordcloud                    │
│  └── /health              │   │  ├── /trend_chart                  │
│                            │   │  └── /health                      │
│  Models:                   │   │                                    │
│  - LightGBM (Champion)    │   │  Models: re-uses inference utils   │
│  - DeBERTa (ABSA)         │   └────────────────────────────────────┘
└───────────┬──────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  MLflow Model Registry              │
│  Champion: LightGBM / XGBoost       │
│  Fallback: Local artifact path      │
└───────────┬─────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────┐
│  DVC Pipeline (12-stage DAG)                                      │
│  data_ingestion → data_preparation → feature_comparison           │
│  → feature_tuning → imbalance_tuning → feature_engineering        │
│  → baseline_model → hpo_lightgbm → hpo_xgboost                    │
│  → train_distilbert → model_evaluation → register_model           │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Requirements

| Dataset | Source | Characteristics |
| :--- | :--- | :--- |
| **Reddit Comments** | [Himanshu-1703/reddit-sentiment-analysis](https://github.com/Himanshu-1703/reddit-sentiment-analysis) | ~26K labeled rows; classes: {-1, 0, 1} → normalized to {0, 1, 2} |
| **Train / Val / Test Split** | 70% / 15% / 15% (stratified) | Parquet format; stratified by sentiment class |
| **Feature Artifacts** | DVC-tracked `models/features/` | Sparse `.npz` matrices, `.npy` label arrays, `.pkl` label encoder |

---

## 6. Constraints & Dependencies

- **Python ≥ 3.11** — required for modern union type syntax (`str | None`) and performance improvements
- **CUDA GPU** — required for DistilBERT fine-tuning stage (skippable via `params.yaml`)
- **MLflow Server** — must be running locally or remotely for model registry operations
- **YouTube Data API Key** — required for YouTube API comment fetching (DOM scraping is the primary path)
- **AWS S3 + ECR** — required for DVC remote storage and production container deployment
- **Chrome ≥ Manifest V3** — the extensions use Manifest V3 APIs

---

## 7. Success Metrics

| Metric | Target | Measurement Point |
| :--- | :---: | :--- |
| Model Macro F1 (test set) | ≥ 0.75 | `models/advanced/evaluation/*.json` via MLflow |
| Model Registration Rate | 100% of runs ≥ threshold | `register_model` stage log |
| API p95 Response Time | < 500ms | Manual load testing / future OpenTelemetry tracing |
| CI Pipeline Success Rate | > 95% | GitHub Actions dashboard |
| Test Coverage | ≥ 60% | `pytest-cov` report in CI |
| Pyright Error Count | 0 | Pyright CI step |

---

## 8. Release Milestones

| Milestone | Deliverables | Status |
| :--- | :--- | :---: |
| **M1 — Core Pipeline** | 12-stage DVC DAG, MLflow tracking, model registration | ✅ Complete |
| **M2 — Inference APIs** | Dual FastAPI services, MLflow model loading, ABSA endpoint | ✅ Complete |
| **M3 — Chrome Extensions** | Standard + ABSA extensions, visualizations | ✅ Complete |
| **M4 — Containerization** | Dockerfile, CI/CD pipeline, GitHub Actions | ✅ Complete |
| **M5 — Hardening** | Security, type safety, training-serving integrity, test coverage | 🔄 In Progress |
| **M6 — Portfolio Polish** | Makefile, pre-commit, CONTRIBUTING.md, Model Card, GX validation | ⬜ Planned |
