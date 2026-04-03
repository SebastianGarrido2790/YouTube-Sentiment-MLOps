# Project Charter: YouTube Sentiment Analysis MLOps Pipeline

| **Field**          | **Value**                                                                   |
| :----------------- | :-------------------------------------------------------------------------- |
| **Project Name**   | YouTube Sentiment Analysis MLOps Pipeline                                   |
| **Author**         | Sebastian Garrido                                                           |
| **Date**           | 2026-04-03                                                                  |
| **Version**        | v1.0                                                                        |
| **Status**         | In Hardening — Post-MVP Production Readiness Phase                          |
| **Repository**     | [SebastianGarrido2790/Youtube-Sentiment-MLOPS](https://github.com/SebastianGarrido2790/Youtube-Sentiment-MLOPS) |

---

## 1. What exactly am I going to build?

A **production-grade MLOps sentiment analysis pipeline** that classifies YouTube video comments as Positive, Neutral, or Negative, delivered through two bespoke Chrome Extensions that inject real-time sentiment insights directly into the YouTube interface.

The system extends beyond simple sentiment classification to include **Aspect-Based Sentiment Analysis (ABSA)**, enabling content creators to understand not just *how* their audience feels, but *what specific topics* they feel strongly about, all powered by an automated, reproducible DVC pipeline managed through an MLflow registry.

---

## 2. Who is it intended for?

**Primary User:** YouTube content creators — vloggers, educators, journalists, and channel operators — who want to understand their audience's emotional response without manually reading hundreds of comments.

**Secondary Audience (Portfolio):** Hiring managers, technical recruiters, and senior ML engineers evaluating the system as a demonstration of **MLOps engineering maturity, production-grade Python development, and end-to-end system design**.

---

## 3. What problem does the product solve?

### Surface Problem
YouTube creators receive hundreds, sometimes thousands of comments per video. Reading them manually is impossible at scale. The platform's own "Like/Dislike" count provides no granularity around *sentiment quality*, *topic distribution*, or *negative signal detection*.

### Real Engineering Problem
This project solves a **training-serving integration challenge**: how do you build a sentiment model that is:
1. **Trained on surrogate data** (Reddit comments, a domain proxy for social media sentiment), yet reliably serves real-world YouTube comments?
2. **Reproducible and auditable** via versioned pipelines, so experiment outcomes can be trusted and reproduced?
3. **Served in real time** through a browser extension without a native ML runtime, requiring a clean separation between the model inference backend and the frontend presentation layer?
4. **Maintainable** as the production environment evolves, through a Model Registry that gatekeeps promotion and an automated CI/CD pipeline that prevents regressions?

---

## 4. How is it going to work?

The system operates as three synchronized layers:

**Layer 1 — MLOps Pipeline (Backend):**
A 12-stage DVC DAG processes raw Reddit sentiment data through ingestion, preparation, feature engineering (TF-IDF or DistilBERT embeddings), imbalance resolution (ADASYN), dual-framework hyperparameter optimization (LightGBM + XGBoost via Optuna), DistilBERT fine-tuning, test evaluation, and automated champion registration into MLflow.

**Layer 2 — Inference Core (Microservices):**
Two FastAPI services serve the trained models:
- **Inference API (`:8000`)** — handles `/predict` (sentiment classification) and `/predict_absa` (aspect-based sentiment via DeBERTa).
- **Insights API (`:8001`)** — generates visualizations: pie charts, wordclouds, and monthly sentiment trend graphs.

Both services load models from the MLflow Registry at startup, with a local artifact fallback for resilience. The ABSA model is lazy-loaded to ensure fast API startup.

**Layer 3 — Presentation (Chrome Extensions):**
Two Vanilla JS Chrome Extensions (Standard Sentiment + ABSA) scrape YouTube comments from the DOM, send them to the appropriate API endpoint, and render the results inline in the popup UI.

---

## 5. What is the expected result (technically)?

| Outcome | Specification |
| :--- | :--- |
| **Model Performance** | Macro F1 ≥ 0.75 on held-out test set (enforced by `register.f1_threshold` in `params.yaml`) |
| **Pipeline Reproducibility** | `dvc repro` executes the full 12-stage DAG from a clean state, producing identical artifacts |
| **API Availability** | Both FastAPI services respond in < 500ms p95 for a batch of ≤ 100 comments |
| **CI/CD Gate** | All commits pass Ruff linting, Pytest suite, and Trivy security scan before merging |
| **Model Registry** | Champion model promoted to MLflow "Production" alias with lineage to its training run |
| **Extension Usability** | Chrome Extensions successfully analyze comments on any public YouTube video page |

---

## 6. What steps do I need to take to achieve this result?

### Phase 1 — Data Foundation
1. Ingest Reddit comment dataset (CSV) via `download_dataset.py`
2. Clean, label-normalize, and stratify-split into train/val/test Parquet files
3. Engineer features: TF-IDF vectorization + derived lexicon features (`pos_ratio`, `neg_ratio`, `char_len`, `word_len`)

### Phase 2 — Model Development
4. Run feature strategy comparison: TF-IDF vs. DistilBERT embeddings via MLflow experiment
5. Tune TF-IDF hyperparameters (max_features, ngram_range) against validation Macro F1
6. Evaluate imbalance handling strategies (ADASYN, SMOTE-ENN, class weights)
7. Train Logistic Regression baseline
8. Optuna hyperparameter optimization for LightGBM and XGBoost (30 trials each)
9. Optional: DistilBERT Optuna fine-tuning (GPU-dependent)

### Phase 3 — Evaluation & Registration
10. Evaluate all models on unseen test set; select champion by Macro AUC
11. Gate registration: only register if `test_macro_f1 ≥ 0.75`
12. Promote champion to MLflow Production alias

### Phase 4 — Serving
13. Build FastAPI Inference + Insights APIs with MLflow Registry loading
14. Containerize with Docker; define healthchecks
15. Build and load both Chrome Extensions

### Phase 5 — Operations
16. Configure GitHub Actions CI/CD: lint → test → Trivy scan → Docker build → deploy
17. Connect DVC remote to AWS S3 for artifact persistence
18. Document architecture, decisions, runbooks, and workflows

### Phase 6 — Hardening (Current Phase)
19. Security: revoke exposed API key, add `.env.example`
20. Type safety: enforce Pyright in CI, migrate legacy `typing` imports
21. Training-serving integrity: extract shared preprocessing + feature modules
22. Developer experience: Makefile, pre-commit, CONTRIBUTING.md

---

## 7. What could go wrong along the way?

| Risk | Likelihood | Mitigation |
| :--- | :---: | :--- |
| **Domain Shift** — Reddit ≠ YouTube sentiment distribution | High | Monitor prediction confidence distributions at inference time; plan for fine-tuning on YouTube-native data |
| **Training-Serving Skew** — preprocessing logic diverges between pipeline and API | Medium | Extract shared `text_preprocessing.py` and `feature_utils.py` modules (Phase 6 goal) |
| **API Key Exposure** — environment secrets committed to Git | **OCCURRED** | Revoke key immediately; clean Git history; enforce `.env.example` |
| **Class Imbalance** — Neutral class underrepresented after cleaning | Medium | ADASYN oversampling; monitor per-class F1 in MLflow |
| **ABSA DeBERTa Latency** — heavyweight model slows inference | Medium | Lazy-load strategy already implemented; add async warmup endpoint |
| **MLflow Registry Downtime** — model loading fails at API startup | Low | Local artifact fallback via `PREFER_LOCAL_MODEL` env var |
| **Chrome Extension Breakage** — YouTube DOM structure changes break DOM scraping | Medium | Monitor for YouTube UI updates; add official YouTube Data API fallback path |
| **CI/CD Drift** — type errors silently accumulate without Pyright enforcement | Medium | Add Pyright step to CI as Phase 6 priority |

---

## 8. What tools should I use to develop this project?

### Core Stack

| Layer | Tool | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.11 + JavaScript (Vanilla) | Modern typing support; lightweight browser frontend |
| **Package Manager** | `uv` | Rust-based, deterministic resolution, 10-100x faster than pip |
| **API Framework** | FastAPI + Uvicorn | High-performance async Python ASGI; auto-generated OpenAPI docs |
| **ML Frameworks** | Scikit-learn, LightGBM, XGBoost, PyTorch + HuggingFace Transformers | Breadth of coverage: classical gradient boosters + transformer ABSA |
| **Pipeline Orchestration** | DVC | Reproducible DAG with dependency tracking + remote storage integration |
| **Experiment Tracking** | MLflow | Nested runs, Model Registry, UI dashboard |
| **Hyperparameter Optimization** | Optuna | TPE sampler, trial pruning, multi-objective support |
| **Containerization** | Docker + Docker Compose | Consistent deployment environments |
| **CI/CD** | GitHub Actions | Native integration with repository; multi-job matrix support |

### Quality & Safety Toolchain

| Tool | Purpose |
| :--- | :--- |
| `ruff` | Fast lint + format + import sort (replaces flake8, isort, black) |
| `pyright` | Static type checking; strict enforcement of 100% type hint coverage |
| `pytest` + `pytest-cov` | Unit + integration tests with coverage gating |
| `trivy` | Container + dependency CVE scanning |
| `bandit` | Python security static analysis |
| `pre-commit` | Local quality gates before every commit |

---

## 9. What are the main concepts involved and how are they related?

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                  FEATURE PIPELINE (Data Engineering)            │
 │  Reddit CSV  →  Text Cleaning  →  Stratified Split              │
 │            →  TF-IDF + Lexicon Features  →  Feature Store       │
 └──────────────────────────────┬──────────────────────────────────┘
                                │ versioned Parquet + .npz artifacts
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                  TRAINING PIPELINE (Model Development)          │
 │  Feature Store → Imbalance Handling (ADASYN)                    │
 │               → Optuna HPO (LightGBM, XGBoost)                  │
 │               → DistilBERT Fine-Tuning (optional)               │
 │               → Champion Selection by AUC                       │
 │               → MLflow Model Registry (gated by F1 ≥ 0.75)      │
 └──────────────────────────────┬──────────────────────────────────┘
                                │ registered model artifact
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                  INFERENCE PIPELINE (Model Serving)             │
 │                                                                 │
 │  Inference API (:8000)          Insights API (:8001)            │
 │  ├── /predict  (LightGBM)       ├── /wordcloud                  │
 │  └── /predict_absa (DeBERTa)    ├── /trend_chart                │
 │                                 └── /pie_chart                  │
 └──────────────────────────────┬──────────────────────────────────┘
                                │ JSON responses
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                  PRESENTATION LAYER (Chrome Extensions)         │
 │  Standard Extension → DOM scrape → /predict → Pie Chart         │
 │  ABSA Extension     → DOM scrape → /predict_absa → Per-Aspect   │
 └─────────────────────────────────────────────────────────────────┘
```

**Key Conceptual Relationships:**

- **DVC ↔ MLflow:** DVC tracks *what was run and with what data*; MLflow tracks *what was learned and which model is best*. Together they provide full lineage.
- **FTI Decoupling:** Feature, Training, and Inference pipelines are independent. Changing the feature strategy does not require redeploying the API.
- **Domain Proxy Pattern:** Reddit data is used as a training proxy for YouTube sentiment, acknowledging the domain gap as a known, monitored risk.
- **Dual Model Strategy:** LightGBM (speed, efficiency, general sentiment) + DeBERTa (contextual understanding, ABSA) serve complementary use cases within the same system.
- **Singleton ConfigurationManager:** Bridges `params.yaml` (static config) and all pipeline stages through Pydantic-validated, typed accessor methods — the single source of truth for all hyperparameters and paths.
- **F1-Score as Production Gateway:** Macro F1 is used rather than accuracy because class imbalance makes accuracy misleading; it ensures balanced detection across all three sentiment classes.
