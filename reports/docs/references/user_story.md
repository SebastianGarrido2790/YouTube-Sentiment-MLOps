# User Story & Problem Framing
## YouTube Sentiment Analysis MLOps Pipeline

| **Field**     | **Value**                                    |
| :------------ | :------------------------------------------- |
| **Version**   | v1.0                                         |
| **Date**      | 2026-04-03                                   |
| **Author**    | Sebastian Garrido                            |
| **Linked To** | [`prd.md`](./prd.md), [`project_charter.md`](./project_charter.md) |

---

## Part 1: User Personas

### Persona A — The Independent Creator

**Name:** Marco, 28  
**Channel:** Tech reviews & tutorials, 45K subscribers  
**Technical Level:** Non-technical (uses YouTube Studio analytics but doesn't code)

**Context:**
Marco uploads three videos per week. His last video on "Budget Laptops Under $500" received 1,200 comments in 48 hours. He spent two hours manually skimming through them and still felt uncertain about the overall reaction. He noticed complaints about the "speaker quality" section but couldn't tell if it was a minority view or a widespread concern.

**Quote:** *"I know the comments are where the real feedback is, but there's just too many. I feel like I'm reading 5% of them and guessing the rest."*

**Pain Points:**
- No scalable way to aggregate sentiment
- Cannot identify which *aspects* of the video drove negative reaction
- Lacks time-series trend data to know if sentiment is improving or declining across videos

---

### Persona B — The MLOps Engineer (Hiring Target / Technical Evaluator)

**Name:** Rocio, 38  
**Role:** ML Platform Lead at a Series B analytics company  
**Technical Level:** Expert — reviews portfolios of candidates for senior ML Engineer roles

**Context:**
Rocio is evaluating Sebastian's portfolio. She's seen dozens of "I trained a model on Kaggle" projects. She's looking for evidence of **systemic thinking** — reproducibility, production concerns, failure mode awareness, and clean engineering.

**What Rocio looks for:**
- Does the pipeline have actual MLOps infrastructure (not just a notebook)?
- Is there a data contract, or is it "garbage in, garbage out"?
- Is training-serving skew addressed, or just silently accepted?
- Are there tests? Is CI/CD real or performative?

**Quote:** *"Anyone can fine-tune a model. Show me you understand what happens after the model is trained."*

---

### Persona C — The Data Scientist Building This (Sebastian)

**Name:** Sebastian, 35  
**Role:** Aspiring MLOps/Data Science Engineer  
**Technical Level:** Intermediate-to-Advanced — strong Python, growing expertise in MLOps tooling

**Context:**
Sebastian is building this project both to solve a real problem and to demonstrate engineering maturity. He's moving away from Jupyter notebooks and into production-grade system design. He wants this project to reflect the standards he'd apply at work: typed, tested, reproducible, and documented.

**Personal Goals:**
- Build confidence in MLOps tooling (DVC, MLflow, FastAPI, Docker)
- Create a portfolio project that provably goes beyond "tutorial-grade" work
- Develop the habit of treating data science as a disciplined engineering practice

---

## Part 2: User Stories

### Epic 1: Content Creator Experience

```
AS A YouTube content creator with a high-volume comment section
I WANT to click a button and instantly see how my audience feels about my video
SO THAT I can improve my content based on aggregated, data-driven feedback
WITHOUT reading thousands of comments manually.
```

---

**Story 1.1 — Standard Sentiment Snapshot**

> **As a** YouTube creator browsing my own video page,  
> **I want to** click the Chrome Extension and see a breakdown of Positive / Neutral / Negative comment sentiment as a pie chart,  
> **So that** I can gauge the overall audience reaction in under 10 seconds.

**Acceptance Criteria:**
- [ ] Extension icon appears in Chrome toolbar when on a YouTube video page
- [ ] Clicking "Analyze" triggers comment scraping from the page DOM
- [ ] Comments are sent to the `/predict` endpoint and a pie chart renders in the popup
- [ ] If fewer than 10 comments are found, a warning message is displayed
- [ ] Results render within 5 seconds of clicking "Analyze"

---

**Story 1.2 — Aspect-Based Deep Dive**

> **As a** content creator who received mixed feedback on a specific topic,  
> **I want to** enter keywords like "audio quality, graphics, pacing" into the ABSA extension  
> **So that** I can see whether the sentiment for each topic is positive, neutral, or negative independently.

**Acceptance Criteria:**
- [ ] ABSA extension accepts a comma-separated list of aspect keywords
- [ ] Comments are concatenated and sent to `/predict_absa` with the aspects list
- [ ] Results display per-aspect sentiment labels in the popup UI
- [ ] An error message is shown if the API is unreachable (within 30-second timeout)

---

**Story 1.3 — Historical Trend Awareness**

> **As a** creator with a long publishing history,  
> **I want to** see how comment sentiment has trended month-over-month across recent videos  
> **So that** I can understand if my content strategy is resonating better or worse over time.

**Acceptance Criteria:**
- [ ] The Insights API `/trend_chart` endpoint accepts a YouTube video or channel identifier
- [ ] A monthly line chart is returned and rendered in the extension
- [ ] The chart handles gaps in data gracefully (no errors on sparse months)

---

### Epic 2: MLOps Pipeline Reliability

```
AS A data scientist maintaining this pipeline
I WANT the entire training workflow to be reproducible and automated
SO THAT I can retrain, experiment, and register new models without manual intervention
AND without producing inconsistent or stale artifacts.
```

---

**Story 2.1 — One-Command Reproducibility**

> **As a** developer cloning this repository,  
> **I want to** run `dvc repro` and produce the exact same final model as the original author,  
> **So that** I can trust the pipeline's integrity and begin experimenting from a verified baseline.

**Acceptance Criteria:**
- [ ] `dvc repro` executes all stages in dependency order without errors
- [ ] `dvc params diff` shows no differences between a fresh run and the locked state
- [ ] `random_state=42` is set at all stochastic stages (e.g. splitting, ADASYN, Optuna)
- [ ] Final model performance is within ±0.005 Macro F1 of the documented baseline

---

**Story 2.2 — Experiment Comparison**

> **As a** data scientist running hyperparameter experiments,  
> **I want to** view all Optuna trial results organized under a single parent MLflow run,  
> **So that** I can compare strategies and identify the best configuration without scrolling through hundreds of flat runs.

**Acceptance Criteria:**
- [ ] Each model's Optuna study creates exactly one parent MLflow run
- [ ] Each trial is logged as a nested child run with `val_macro_f1`, `val_accuracy`, and all hyperparameters
- [ ] The parent run records the best hyperparameters and a custom `"study_complete"` tag on completion

---

**Story 2.3 — Quality-Gated Registration**

> **As a** pipeline maintainer protecting production from degraded models,  
> **I want to** have a configurable F1 threshold that automatically blocks registration of underperforming models,  
> **So that** only champion models that meet minimum quality standards are promoted to production.

**Acceptance Criteria:**
- [ ] `register.f1_threshold` in `params.yaml` controls the registration gate (default: `0.75`)
- [ ] If `test_macro_f1 < threshold`, the registration stage logs a warning and exits without promoting
- [ ] If `test_macro_f1 >= threshold`, the model is registered with the "Production" alias in MLflow
- [ ] The stage handles both legacy (pre-2.9.0) and modern MLflow version transitions gracefully

---

### Epic 3: Engineering Quality & Security

```
AS A hiring manager or senior engineer reviewing this project
I WANT to see evidence of mature engineering practices in the codebase
SO THAT I can be confident this work reflects production-ready standards
NOT just a prototype wrapped in documentation.
```

---

**Story 3.1 — Type-Safe Codebase**

> **As a** code reviewer evaluating this project,  
> **I want to** run `pyright src/ app/` and see zero type errors,  
> **So that** I can trust that the type annotations are meaningful and enforced by tooling, not decorative.

**Acceptance Criteria:**
- [ ] `pyright` passes with `typeCheckingMode = "standard"` and zero reported errors
- [ ] No `Optional[str]` or `List[str]` legacy typing — all modern PEP 604 (`str | None`, `list[str]`)
- [ ] A `pyright` step exists in the GitHub Actions CI workflow and blocks the pipeline on failure

---

**Story 3.2 — Secret-Safe Repository**

> **As a** developer onboarding to this project,  
> **I want to** find a `.env.example` file that documents all required environment variables,  
> **So that** I can configure my local environment without guessing or reading source code.

**Acceptance Criteria:**
- [ ] `.env.example` exists with documented placeholder values for `ENV`, `MLFLOW_TRACKING_URI`, `YOUTUBE_API_KEY`, and `PREFER_LOCAL_MODEL`
- [ ] No real API keys or credentials appear in any tracked file in the repository
- [ ] The CI pipeline fails if `gitleaks` or equivalent detects secrets in committed files

---

**Story 3.3 — Zero Training-Serving Skew**

> **As a** data scientist deploying this model to production,  
> **I want to** guarantee that the text preprocessing applied at training time is identical to what runs in the API,  
> **So that** the model receives the same feature distribution it was trained on — preventing silent accuracy degradation.

**Acceptance Criteria:**
- [ ] A single `src/utils/text_preprocessing.py` module contains the authoritative `clean_text()` function
- [ ] `src/data/make_dataset.py`, `app/main.py`, and `app/insights_api.py` all import from this shared module
- [ ] A single `src/utils/feature_utils.py` module contains the authoritative `build_derived_features()` function
- [ ] No duplicate preprocessing logic exists anywhere in `src/` or `app/`

---

## Part 3: Problem Framing Summary

### The Jobs-to-Be-Done (JTBD) Framework

| User | Job | Current Solution | Pain |
| :--- | :--- | :--- | :--- |
| Creator (Marco) | Understand audience reaction quickly | Manually reading comments | Time-consuming, biased, non-scalable |
| Creator (Marco) | Identify which content elements resonated | Guessing from "top comments" | No topic-level granularity |
| Creator (Marco) | Track sentiment trends over time | Ad-hoc export + manual review | No automation, no visualization |
| Engineer (Rocio) | Evaluate MLOps maturity | Reading README and code | Hard to assess depth without running the system |
| Builder (Sebastian) | Build production-grade ML systems | Notebooks / scripts | No reproducibility, no type safety, no monitoring |

### The Core Tension

> **What users want:** instant, visual, topic-specific sentiment insights.  
> **What is hard:** building a reliable, reproducible, skew-free ML system that serves those insights consistently at scale.  

The Chrome Extension is the 5% of the system that users see. This project documents and implements the 95% that makes it trustworthy: the DVC pipeline, the MLflow registry, the typed API layer, and the operational harness that keeps it honest.
