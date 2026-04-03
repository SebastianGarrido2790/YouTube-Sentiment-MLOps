# Technical Roadmap
## YouTube Sentiment Analysis MLOps Pipeline

| **Field**     | **Value**                                                              |
| :------------ | :--------------------------------------------------------------------- |
| **Version**   | v1.0                                                                   |
| **Date**      | 2026-04-03                                                             |
| **Author**    | Sebastian Garrido                                                      |
| **Status**    | Active — M5 Hardening Phase                                            |
| **Linked To** | [`prd.md`](./prd.md), [`user_story.md`](./user_story.md), [`project_charter.md`](./project_charter.md) |

---

## Roadmap Overview

```
M1 ─────── M2 ─────── M3 ─────── M4 ─────── M5 ─────── M6
 Core       Inference  Chrome     Containers  Hardening  Portfolio
 Pipeline   APIs       Extensions  & CI/CD    & Security  Polish
    ✅         ✅         ✅          ✅        🔄 ACTIVE  ⬜ Planned
```

> [!IMPORTANT]
> Milestones M1–M4 are **complete**. This roadmap serves as both a retrospective record and a forward specification for the active M5 hardening phase and planned M6 portfolio polish.

---

## Milestone 1 — Core MLOps Pipeline ✅

**Goal:** A fully reproducible 12-stage DVC pipeline that ingests raw data, engineers features, trains models under controlled experiment conditions, evaluates champions, and registers to a model registry.

**Linked PRD Requirements:** PIPE-01 through PIPE-07

---

### Phase 1.1 — Data Foundation

**Objective:** Build a reliable, versioned data ingestion and preparation system that produces consistent train/val/test splits.

| Task | Specification | File(s) |
| :--- | :--- | :--- |
| **Data Ingestion** | Download Reddit CSV from public URL; store at `data/raw/reddit_comments.csv`; DVC-tracked output | `src/data/download_dataset.py` |
| **Text Cleaning** | Regex: `[^a-zA-Z\s]` → alphabetic-only; NLTK tokenization; stopword removal; drop tokens ≤ 2 chars | `src/data/make_dataset.py` |
| **Label Normalization** | Map `{-1, 0, 1}` → `{0, 1, 2}` for SMOTE and XGBoost compatibility; LabelEncoder persisted as `.pkl` | `src/data/make_dataset.py` |
| **Stratified Split** | 70% / 15% / 15% (train/val/test); double stratification with `random_state=42`; output as Parquet | `src/data/make_dataset.py` |

**DVC Stage:** `data_ingestion`, `data_preparation`  
**Outputs:** `data/raw/reddit_comments.csv`, `data/processed/{train,val,test}.parquet`

---

### Phase 1.2 — Feature Engineering

**Objective:** Determine the optimal feature representation strategy and engineer a composite feature matrix combining TF-IDF and lexicon-derived features.

| Task | Specification | File(s) |
| :--- | :--- | :--- |
| **Feature Strategy Comparison** | Benchmark TF-IDF (unigrams, bigrams) vs. DistilBERT embeddings on validation Macro F1 via MLflow | `src/features/tfidf_vs_distilbert.py` |
| **TF-IDF Hyperparameter Tuning** | Grid search over `max_features` values (1K–10K); identify optimal vocabulary size | `src/features/tfidf_max_features.py` |
| **Imbalance Strategy Tuning** | Compare ADASYN, SMOTE-ENN, class_weights, oversampling, undersampling; select ADASYN | `src/features/imbalance_tuning.py` |
| **Composite Feature Engineering** | `hstack([tfidf_matrix, derived_features])` where derived = `pos_ratio`, `neg_ratio`, `char_len`, `word_len` | `src/features/feature_engineering.py` |

**DVC Stages:** `feature_comparison`, `feature_tuning`, `imbalance_tuning`, `feature_engineering`  
**Outputs:** Sparse `.npz` feature matrices, `label_encoder.pkl` at `models/features/`

---

### Phase 1.3 — Model Training & Evaluation

**Objective:** Train competing models via automated HPO, evaluate on held-out test data, and select the champion.

| Task | Specification | File(s) |
| :--- | :--- | :--- |
| **Logistic Baseline** | `class_weight="balanced"`, `solver="liblinear"`, `max_iter=2000`; deterministic benchmark | `src/models/baseline_logistic.py` |
| **LightGBM HPO** | Optuna 30 trials; TPE sampler; nested MLflow runs; save best `.pkl` + hyperparams | `src/models/hyperparameter_tuning.py` |
| **XGBoost HPO** | Optuna 30 trials; native XGBoost API; parallel to LightGBM; same nested logging pattern | `src/models/hyperparameter_tuning.py` |
| **DistilBERT Fine-tuning** | Optional (GPU-dependent); HuggingFace Trainer + Optuna; toggled via `params.yaml` | `src/models/distilbert_training.py` |
| **Champion Evaluation** | Test set evaluation for all models; confusion matrix + ROC curves; write `best_model_run_info.json` | `src/models/model_evaluation.py` |
| **Quality-Gated Registration** | Read champion info; gate on `test_macro_f1 >= 0.75`; register to MLflow with "Production" alias | `src/models/register_model.py` |

**DVC Stages:** `baseline_model`, `hyperparameter_tuning_lightgbm`, `hyperparameter_tuning_xgboost`, `train_distilbert`, `model_evaluation`, `register_model`  
**Outputs:** Champion model in MLflow Registry; evaluation figures at `reports/figures/evaluation/`

---

## Milestone 2 — Inference Microservices ✅

**Goal:** Two production FastAPI services that load the champion model from the MLflow Registry and serve predictions and visualizations.

**Linked PRD Requirements:** API-01 through API-08

---

### Phase 2.1 — Inference API

**Objective:** A low-latency API for sentiment classification and ABSA, with resilient model loading.

| Task | Specification | File(s) |
| :--- | :--- | :--- |
| **Model Loading Strategy** | Priority: MLflow Registry → `PREFER_LOCAL_MODEL` override → local artifact path | `app/inference_utils.py` |
| **`/predict` Endpoint** | Accepts `{"texts": [...]}`, preprocesses, vectorizes, predicts; returns class labels + probabilities | `app/main.py` |
| **`/predict_absa` Endpoint** | Accepts `{"text": "...", "aspects": [...]}`, routes to DeBERTa NLP pipeline | `app/main.py` |
| **Lazy ABSA Loading** | DeBERTa model initialized only on first request via `global absa_model` guard | `app/main.py` |
| **`/health` Endpoint** | Returns API status, model name, load strategy, and version metadata | `app/main.py` |
| **Pydantic I/O Models** | All request/response bodies typed with `BaseModel`; automatic validation + OpenAPI docs | `app/main.py` |

---

### Phase 2.2 — Insights API

**Objective:** A visualization service generating server-side charts returned as base64-encoded images.

| Task | Specification | File(s) |
| :--- | :--- | :--- |
| **`/pie_chart`** | Accepts sentiment counts; returns Matplotlib pie chart as base64-encoded PNG | `app/insights_api.py` |
| **`/wordcloud`** | Accepts classified comments; generates per-class wordcloud via `wordcloud` library | `app/insights_api.py` |
| **`/trend_chart`** | Accepts timestamped data; generates monthly sentiment trend line chart | `app/insights_api.py` |
| **CORS Configuration** | `allow_origins=["*"]` for development; to be restricted to `chrome-extension://*` | `app/insights_api.py` |
| **Preprocessing Reuse** | Uses same `insights_api.py` preprocessing logic as training (skew — to be fixed in M5) | `app/insights_api.py` |

---

## Milestone 3 — Chrome Extensions ✅

**Goal:** Two working browser extensions providing visible, interactive sentiment analysis directly on YouTube video pages.

**Linked PRD Requirements:** EXT-01 through EXT-06, VIZ-01 through VIZ-04

| Task | Specification | Location |
| :--- | :--- | :--- |
| **Standard Extension** | Manifest V3; DOM comment scraping; calls `/predict`; renders pie chart in popup | `chrome-extension/` |
| **ABSA Extension** | Same base structure; accepts user-defined aspects; calls `/predict_absa` | `chrome-extension-absa/` |
| **Timeout Handling** | `AbortController` with 30-second timeout on all fetch calls | `popup.js` |
| **YouTube API Support** | `youtube_api.js` scaffolding for official YouTube Data API v3 integration | `youtube_api.js` |

---

## Milestone 4 — Containerization & CI/CD ✅

**Goal:** The system is containerized, deployable via Docker, and protected by a GitHub Actions CI/CD pipeline that enforces quality gates.

**Linked PRD Requirements:** OPS-01, OPS-04

| Task | Specification | File(s) |
| :--- | :--- | :--- |
| **Dockerfile** | Multi-stage build; `uv sync --frozen --no-dev` for production; `HEALTHCHECK` on port 8000 | `Dockerfile` |
| **CI Pipeline — Lint** | `ruff check src/ app/ tests/` + `ruff format --check` | `.github/workflows/ci_cd.yaml` |
| **CI Pipeline — Test** | `pytest` on all test files | `.github/workflows/ci_cd.yaml` |
| **CI Pipeline — Security** | Trivy container scan (currently non-blocking — to be hardened in M5) | `.github/workflows/ci_cd.yaml` |
| **CI Pipeline — Build** | Docker build + push to AWS ECR | `.github/workflows/ci_cd.yaml` |
| **LocalStack Simulation** | CI LocalStack-based simulation for AWS ECR/ECS deployment validation without real credentials | `.github/workflows/ci_cd.yaml` |
| **DVC Remote** | AWS S3 bucket as DVC remote storage for data and model artifacts | `.dvc/config` |

---

## Milestone 5 — Hardening & Security 🔄 (Active)

**Goal:** Elevate the codebase from MVP to production-grade engineering standards — resolving all critical security, type safety, and training-serving integrity gaps identified in the codebase review.

**Linked PRD Requirements:** OPS-01 through OPS-08  
**Linked Review:** [`codebase_review.md`](../evaluations/codebase_review.md)

---

### Phase 5.1 — Security & Quick Wins

**Estimated effort:** 30 minutes

| Task | Priority | Status |
| :--- | :---: | :---: |
| Revoke compromised YouTube API key + clean Git history | 🔴 CRITICAL | ⬜ Pending |
| Create `.env.example` with all required variable placeholders | 🔴 CRITICAL | ⬜ Pending |
| Create empty `app/__init__.py` to formalize package boundary | 🟠 HIGH | ⬜ Pending |
| Create empty `src/py.typed` for PEP 561 compliance | 🟡 MEDIUM | ⬜ Pending |
| Move `import pickle` from function body to module level in `data_loader.py` | 🟢 LOW | ⬜ Pending |
| Complete path migration: deprecate `src/utils/paths.py`, use `src.constants` everywhere | 🟢 LOW | ⬜ Pending |

---

### Phase 5.2 — Type Safety & CI Hardening

**Estimated effort:** 1–2 hours

| Task | Priority | Status |
| :--- | :---: | :---: |
| Add `[tool.pyright]` section to `pyproject.toml` (`pythonVersion="3.11"`, `typeCheckingMode="standard"`) | 🔴 CRITICAL | ⬜ Pending |
| Add `pyright>=1.1.350` to `[project.optional-dependencies] dev` | 🔴 CRITICAL | ⬜ Pending |
| Add `pyright src/ app/` step to `ci_cd.yaml` (blocking on error) | 🔴 CRITICAL | ⬜ Pending |
| Expand `[tool.ruff]` — add `target-version`, `select`, `isort`, `line-length=100` | 🟠 HIGH | ⬜ Pending |
| Replace all legacy `typing.Optional/List/Dict/Tuple/Union` imports project-wide | 🔴 CRITICAL | ⬜ Pending |
| Separate `pytest`, `ruff`, `pyright` into `[project.optional-dependencies] dev` | 🟠 HIGH | ⬜ Pending |
| Add `pytest-cov>=4.1.0` to dev deps; add `--cov-fail-under=60` gate to CI | 🟠 HIGH | ⬜ Pending |
| Add `model_config = ConfigDict(extra="forbid")` to all Pydantic schema models | 🟡 MEDIUM | ⬜ Pending |
| Fix `use_distilbert: str` → `use_distilbert: bool` in `schemas.py` + update `params.yaml` | 🟡 MEDIUM | ⬜ Pending |

---

### Phase 5.3 — Training-Serving Integrity

**Estimated effort:** 2–3 hours

> [!CAUTION]
> This phase addresses the most impactful quality risk in the system. Training-serving skew silently degrades model performance in production without triggering any error.

| Task | Priority | Status |
| :--- | :---: | :---: |
| Create `src/utils/text_preprocessing.py` with authoritative `clean_text()` function | 🟠 HIGH | ⬜ Pending |
| Refactor `src/data/make_dataset.py` to import from `text_preprocessing.py` | 🟠 HIGH | ⬜ Pending |
| Refactor `app/main.py` to import and call `clean_text()` before vectorization | 🟠 HIGH | ⬜ Pending |
| Refactor `app/insights_api.py` to use shared `clean_text()` | 🟠 HIGH | ⬜ Pending |
| Create `src/utils/feature_utils.py` with authoritative `build_derived_features()` | 🟠 HIGH | ⬜ Pending |
| Refactor `src/features/feature_engineering.py` to import `build_derived_features()` | 🟠 HIGH | ⬜ Pending |
| Refactor `app/inference_utils.py` to import `build_derived_features()` | 🟠 HIGH | ⬜ Pending |
| Add CORS middleware to `app/main.py` (`allow_origins=["chrome-extension://*"]`) | 🟡 MEDIUM | ⬜ Pending |
| Add API versioning prefix (`/v1/`) to both FastAPI routers | 🟢 LOW | ⬜ Pending |
| Rewrite `app/test_inference.py` as proper `pytest` tests using `TestClient` | 🟡 MEDIUM | ⬜ Pending |
| Create `docker-compose.yml` (MLflow + Inference API + Insights API) OR update README | 🟢 LOW | ⬜ Pending |

---

## Milestone 6 — Portfolio Polish ⬜ (Planned)

**Goal:** Elevate the project from production-ready to portfolio-differentiating — adding observability, data quality contracts, and developer experience tooling that demonstrates MLOps depth.

**Linked PRD Requirements:** OPS-02 through OPS-06

---

### Phase 6.1 — Developer Experience

**Estimated effort:** 1–2 hours

| Task | Priority | Specification |
| :--- | :---: | :--- |
| Create `Makefile` with standard targets | 🟠 HIGH | `make install`, `make lint`, `make typecheck`, `make test`, `make pipeline`, `make serve`, `make docker`, `make clean` |
| Create `.pre-commit-config.yaml` | 🟠 HIGH | Hooks: `ruff`, `ruff-format`, `pyright` (on staged Python files) |
| Remove `sys.path` hack from `tests/conftest.py` | 🟡 MEDIUM | Replace with editable install via `uv pip install -e .` in CI setup step |
| Create `CONTRIBUTING.md` | 🟡 MEDIUM | Development workflow, branching strategy, code standards, testing guide |

---

### Phase 6.2 — Observability

**Estimated effort:** 2–3 hours

| Task | Priority | Specification |
| :--- | :---: | :--- |
| Add structured JSON logging for production | 🟡 MEDIUM | `json_log_formatter` or `python-json-logger`; conditionally activate in production ENV |
| Add OpenTelemetry instrumentation to FastAPI | 🟢 LOW | Span-level visibility for prediction latency, model load time, feature engineering time |
| Harden Trivy scan to blocking (`exit-code: "1"` for CRITICAL) | 🟡 MEDIUM | Convert security scan from notification to gate |
| Add `bandit` Python security scan to CI | 🟡 MEDIUM | `bandit -r src/ app/ -ll` as CI step |
| Add Chrome Extension API key input UI | 🟡 MEDIUM | `chrome.storage.local` persistence; remove hardcoded field from `popup.js` |

---

### Phase 6.3 — Data Quality Contracts

**Estimated effort:** 3–4 hours

| Task | Priority | Specification |
| :--- | :---: | :--- |
| Integrate Great Expectations (GX) into `data_preparation` stage | 🟢 LOW | Expectations: null% threshold, text length range, label balance check |
| Store GX suites as versioned artifacts in `data/contracts/` | 🟢 LOW | DVC-tracked; fails pipeline if data drift exceeds contract boundaries |

---

### Phase 6.4 — Documentation & Portfolio

**Estimated effort:** 2–3 hours

| Task | Priority | Specification |
| :--- | :---: | :--- |
| Write Model Card | 🟡 MEDIUM | Model description, training data domain gap, per-class metrics, bias considerations, version log |
| Add `ModelEvaluationConfig` field validation | 🟢 LOW | `models: list[str] = Field(..., min_length=1)` to prevent empty model list silent no-op |

---

## Dependency Map & Completion Gates

```
Phase 5.1 (Security)
    └─→ Phase 5.2 (Type Safety)    # Pyright can only be configured after deps are clean
            └─→ Phase 5.3 (Skew)   # Shared modules must be typed before CI can pass
                    └─→ Phase 6.1 (DevEx)     # pre-commit needs a clean codebase to enforce
                            └─→ Phase 6.2 (Observability)
                                    └─→ Phase 6.3 (GX)
                                            └─→ Phase 6.4 (Documentation)
```

---

## Architecture Evolution Log

| Version | Change | Rationale |
| :--- | :--- | :--- |
| v0.1 | Single Jupyter notebook | Initial exploration |
| v0.2 | Migrated to `src/` module structure | Production gap — separation of concerns |
| v0.3 | Added DVC pipeline (`dvc.yaml`) | Reproducibility and artifact versioning |
| v0.4 | Added MLflow experiment tracking + registry | Experiment auditability and model governance |
| v0.5 | Dual FastAPI services added | Clean API/visualization separation |
| v0.6 | Chrome Extensions built | End-to-end product demonstration |
| v0.7 | Docker + CI/CD (GitHub Actions) | Portability and quality enforcement |
| v0.8 | `src/constants/__init__.py` + `src/config/schemas.py` | Centralized path and config management |
| **v0.9** | **M5 Hardening (current)** | **Production-grade engineering standards** |
| v1.0 (planned) | M6 Portfolio Polish | GX, OpenTelemetry, Makefile, Model Card |

---

## Definition of Done

A milestone is considered **complete** when:

1. ✅ All tasks in the phase checklist are completed and verified
2. ✅ CI/CD pipeline passes all gates (lint, type check, test coverage, security scan)
3. ✅ New code is covered by tests (coverage gate enforced)
4. ✅ No `pyright` errors on modified files
5. ✅ Documentation is updated to reflect the change (architecture docs, CHANGELOG if applicable)
6. ✅ Changes are commited to Git with a descriptive message and pushed to `main`
