# YouTube Sentiment Analysis — Codebase Review & Production Readiness Assessment

| **Date** | 2026-04-02 |
| **Version** | v1.0 |
| **Overall Score** | **6.7 / 10** |
| **Current Status** | **NOT YET PRODUCTION-READY** |

**Scope:** Full codebase — ~25 Python source files across `src/` and `app/`, 4 test files, 1 CI/CD workflow, 1 YAML config (`params.yaml`), 1 Dockerfile, 2 Chrome Extensions (JS), `pyproject.toml`, and 11 documentation files in `reports/docs/`.

---

## Overall Verdict

The **YouTube Sentiment Analysis MLOps Pipeline** is an **ambitious and well-scoped portfolio project** that demonstrates a strong understanding of the MLOps lifecycle. The 12-stage DVC pipeline is impressively comprehensive — covering data ingestion, preparation, feature comparison (TF-IDF vs. DistilBERT), TF-IDF tuning, imbalance handling, feature engineering, baseline modeling, dual-framework hyperparameter optimization (LightGBM + XGBoost), DistilBERT fine-tuning, model evaluation, and automated model registration. The dual-API architecture (Inference on port 8000 + Insights/Visualization on port 8001) feeding two bespoke Chrome Extensions is a creative and impressive end-to-end product demonstration.

**However**, the project has several critical gaps that prevent it from meeting the **Python-Development Standard**: no `pyright` enforcement, no coverage gates, legacy `typing` imports, missing developer onboarding files (`.env.example`, `Makefile`, `.pre-commit-config.yaml`), and a `pyproject.toml` that lacks essential tooling configuration. The `app/` inference layer also has a significant **training-serving skew** in derived feature engineering. These are all solvable gaps, and the architectural foundation is strong enough to support a rapid elevation to production-grade status.

---

## 1. Strengths ✅

### 1.1 Architecture & Design

| Strength | Evidence |
|:---|:---|
| **Extended FTI Pattern** | 12-stage DVC pipeline (`dvc.yaml`) covering the full data lifecycle from raw ingestion through model registration — far beyond a typical tutorial project |
| **Pydantic Config Schemas** | [schemas.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/schemas.py) defines a full `AppConfig` root model with nested Pydantic schemas for every pipeline stage — strict validation at construction time |
| **Singleton ConfigurationManager** | [manager.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/manager.py) implements a Singleton pattern with typed accessor methods, ensuring config is loaded once and accessed consistently across all pipeline stages |
| **Environment-Aware MLflow** | [mlflow_config.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/utils/mlflow_config.py) implements a 3-level priority chain (env var → env-based default → YAML fallback) with production runtime guard that `raise RuntimeError` when `MLFLOW_TRACKING_URI` is missing in production |
| **Centralized Paths** | [src/constants/](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/constants/__init__.py) establishes the single source of truth for project-wide paths — integrated into `logger` and `mlflow_config` to eliminate hardcoded strings |
| **Dual-API Architecture** | Inference API (`app/main.py`:8000) handles predictions & ABSA; Insights API (`app/insights_api.py`:8001) generates charts, wordclouds, and trend graphs — clean separation of prediction vs. visualization concerns |
| **Lazy-Loaded ABSA** | [main.py:150-155](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/main.py#L150-L155) uses `global absa_model` with lazy initialization, preventing the heavyweight DeBERTa model from blocking API startup |

### 1.2 MLOps Pipeline

| Strength | Evidence |
|:---|:---|
| **12-Stage DVC DAG** | Full `dvc.yaml` with explicit `deps`, `params`, `outs`, and `metrics` — reproducible and cacheable across all stages |
| **MLflow Integration** | Experiment tracking across comparison, tuning, training, and evaluation stages using nested Parent/Child runs with proper tagging |
| **Automated Champion Selection** | [model_evaluation.py:401-417](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/models/model_evaluation.py#L401-L417) selects champion model based on highest Test Macro AUC and persists run info to JSON for downstream registration |
| **F1 Threshold Gating** | [register_model.py:100-105](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/models/register_model.py#L100-L105) applies a configurable F1 threshold from `params.yaml` before model registration — prevents poor models from reaching production |
| **MLflow Version Handling** | [register_model.py:128-162](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/models/register_model.py#L128-L162) gracefully handles both legacy stage transitions (pre-2.9.0) and modern tag-based deployment |
| **Dual-Model Loading Strategy** | [inference_utils.py:76-155](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/inference_utils.py#L76-L155) implements MLflow Registry → Local fallback with `PREFER_LOCAL_MODEL` env override — resilient to registry downtime |
| **ADASYN Imbalance Handling** | [data_loader.py:96-115](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/models/helpers/data_loader.py#L96-L115) applies ADASYN with `random_state=42` for reproducible oversampling |
| **Dual-Framework Optuna** | [hyperparameter_tuning.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/models/hyperparameter_tuning.py) supports both LightGBM (sklearn API) and XGBoost (native API) in a single script with strategy pattern |
| **Artifacts Persistence Record** | Adherence to **Rule 2.12** — the pipeline is architected to save and version all reusable artifacts (scalers, encoders, stratified splits) to ensure full lifecycle persistence and reproducibility |


### 1.3 Data Processing

| Strength | Evidence |
|:---|:---|
| **NLP Text Pipeline** | [make_dataset.py:46-67](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/data/make_dataset.py#L46-L67) implements regex cleaning, NLTK tokenization, stopword removal, and minimum token length filtering |
| **Label Normalization** | [make_dataset.py:93-106](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/data/make_dataset.py#L93-L106) maps `{-1, 0, 1} → {0, 1, 2}` for compatibility with ML tools (SMOTE, XGBoost) while preserving original labels |
| **Derived Feature Engineering** | [feature_engineering.py:87-107](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/features/feature_engineering.py#L87-L107) creates lexicon-based `pos_ratio`/`neg_ratio` and length features, combined with TF-IDF via sparse `hstack` |
| **Feature Strategy Pattern** | TF-IDF vs. DistilBERT selection is configurable via `params.yaml` — supports switching feature representation without code changes |
| **Stratified Splitting** | [make_dataset.py:119-134](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/data/make_dataset.py#L119-L134) uses double stratified split with proportional validation sizing to preserve class distribution |

### 1.4 Chrome Extensions

| Strength | Evidence |
|:---|:---|
| **Full Product Demo** | Two working Chrome Extensions (Standard Sentiment + ABSA) provide a tangible, user-facing product that distinguishes this from typical notebook-only projects |
| **Rich Visualizations** | Extensions request pie charts, wordclouds, and monthly trend graphs server-side — demonstrating end-to-end API integration |
| **Timeout Handling** | [popup.js:109-131](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/chrome-extension/popup.js#L109-L131) uses `AbortController` with 30-second timeout for all API calls |

### 1.5 Documentation

| Strength | Evidence |
|:---|:---|
| **Six-Pillar Structure** | `reports/docs/` follows `architecture/`, `decisions/`, `evaluations/`, `references/`, `runbooks/`, `workflows/` taxonomy — 11 documentation files |
| **Module Docstrings** | Every Python file has a module-level docstring explaining purpose, usage, and dependencies |
| **Google-style Docstrings** | Functions document Args, Returns, and Raises consistently |
| **README Quality** | Rich badges (CI/CD, Python, Docker, FastAPI, MLflow, DVC), architecture overview, usage guide, and tech stack table |

---

## 2. Weaknesses & Gaps 🔴

### 2.1 CRITICAL: No `pyright` Configuration or CI Enforcement

> [!CAUTION]
> `pyproject.toml` has **no** `[tool.pyright]` section, no `pyright` in dependencies, and the CI workflow does **not** run any type checking. The "100% type hint coverage" standard from the Python-Development Standard is completely unenforced.

**Gaps found:**
- No `[tool.pyright]` section with `pythonVersion` or `typeCheckingMode`
- No `pyright` dependency in `pyproject.toml`
- CI workflow `ci_cd.yaml` only runs `ruff check` and `ruff format` — no type checking step
- Several modules use `Optional[str]` instead of `str | None` (e.g., [logger.py:11](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/utils/logger.py#L11), [manager.py:17](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/manager.py#L17))

**Impact:** Type errors silently pass through the pipeline. Typos in function signatures, incorrect return types, and Pydantic model misuses are not caught until runtime.

**Recommendation:**
1. Add to `pyproject.toml`:
```toml
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "standard"
```
2. Add `pyright>=1.1.350` to dev dependencies.
3. Add a CI step: `uv run pyright src/ app/`

---

### 2.2 CRITICAL: Missing `.env.example` File

> [!CAUTION]
> No `.env.example` file exists in the repository. The `.env` file is correctly gitignored, **but the actual `.env` file is also committed** (it appears in the directory listing and is readable, containing a real YouTube API key: `AIzaSyAW9eHOlS_GZgHlW6tSOYHWAtUkRJY0Hio`). This is a **security vulnerability**.

**Impact:**
1. **API Key Exposure:** The YouTube API key in `.env` is exposed in the Git history. Even if removed now, the key is compromised. It must be revoked and regenerated.
2. **Onboarding Failure:** New contributors have no template to know which variables are required (`ENV`, `MLFLOW_TRACKING_URI`, `YOUTUBE_API_KEY`, `PREFER_LOCAL_MODEL`).

**Recommendation:**
1. **Immediately** revoke the compromised YouTube API key and generate a new one.
2. Remove `.env` from Git history using `git filter-repo` or BFG Repo Cleaner.
3. Create a `.env.example` file:
```env
# Environment: local | staging | production
ENV=local
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
YOUTUBE_API_KEY=your_key_here
PREFER_LOCAL_MODEL=false
```

---

### 2.3 CRITICAL: Legacy `typing` Imports Across Codebase

> [!WARNING]
> Multiple files use legacy `typing.List`, `typing.Dict`, `typing.Optional` imports instead of modern PEP 604 builtins. Since the project requires Python ≥3.10, these are unnecessary.

| File | Import |
|:---|:---|
| [schemas.py:10](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/schemas.py#L10) | `from typing import List, Optional` |
| [manager.py:17](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/manager.py#L17) | `from typing import Optional` |
| [tfidf_vs_distilbert.py:23](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/features/tfidf_vs_distilbert.py#L23) | `from typing import Any, List, Optional, Tuple, Union` |
| [feature_utils.py:10](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/features/helpers/feature_utils.py#L10) | `from typing import Any, Dict, Tuple, Union, Optional` |
| [main.py:38](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/main.py#L38) | `from typing import List` |
| [insights_api.py:36](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/insights_api.py#L36) | `from typing import List, Dict, Any` |
| [feature_engineering.py:30](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/features/feature_engineering.py#L30) | `from typing import Optional, Tuple, Union` |
| [logger.py:11](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/utils/logger.py#L11) | `from typing import Optional` |

**Recommendation:** Replace all legacy imports project-wide:
- `List[str]` → `list[str]`
- `Dict[str, Any]` → `dict[str, Any]`
- `Optional[str]` → `str | None`
- `Tuple[int, int]` → `tuple[int, int]`
- `Union[A, B]` → `A | B`

---

### 2.4 HIGH: Training-Serving Skew in Derived Feature Engineering

> [!WARNING]
> The derived feature engineering logic (`char_len`, `word_len`, `pos_ratio`, `neg_ratio`) is **duplicated** across training and inference with **subtle differences**.

| Location | Purpose | Lexicon Implementation |
|:---|:---|:---|
| [feature_engineering.py:87-107](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/features/feature_engineering.py#L87-L107) | Training | Uses a set literal within a local function; `count_lexicon_ratio` is an inner function with no type hints |
| [inference_utils.py:158-193](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/inference_utils.py#L158-L193) | Inference | Uses `Set[str]` typed variables; `count_lexicon_ratio` has type hints |

While the lexicon word sets are currently **identical**, this is only by coincidence, not by design. Any future edit to one function without updating the other introduces silent degradation.

**Impact:** **Training-serving skew risk.** If the lexicon sets or the ratio calculation diverge, the model receives different feature values at inference than it was trained on.

**Recommendation:** Create `src/utils/feature_utils.py` containing a single shared `build_derived_features()` function. Both `feature_engineering.py` and `inference_utils.py` must import and call this same function — eliminating the DRY violation.

---

### 2.5 HIGH: Minimal `[tool.ruff]` Configuration in `pyproject.toml`

> [!IMPORTANT]
> `pyproject.toml` declares `[tool.ruff]` but only has `exclude = ["notebooks"]` and a per-file ignore for `conftest.py`. There is no `target-version`, no `line-length`, no `select` rules, and no import sorting configuration.

```toml
[tool.ruff]
exclude = ["notebooks"]

[tool.ruff.lint.per-file-ignores]
"tests/conftest.py" = ["E402"]
```

**Impact:** Ruff runs with default rules — no import sorting enforced, no f-string enforcement, no bugbear checks, no simplification rules. The Python-Development Standard mandates these explicitly.

**Recommendation:**
```toml
[tool.ruff]
target-version = "py311"
line-length = 100
exclude = ["notebooks"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "N", "W", "B", "SIM", "C4", "RUF"]

[tool.ruff.lint.isort]
known-first-party = ["src", "app"]

[tool.ruff.lint.per-file-ignores]
"tests/conftest.py" = ["E402"]
```

---

### 2.6 HIGH: No `pytest-cov` and No Coverage Gate in CI

> [!WARNING]
> `pytest-cov` is not listed in `pyproject.toml`. The CI workflow runs `uv run pytest` without any coverage reporting or threshold enforcement. Test coverage can silently regress.

**Current CI test step:**
```yaml
- name: Run Tests
  run: uv run pytest
```

**Impact:** There is no visibility into what percentage of the codebase is covered by tests. Coverage regressions are invisible.

**Recommendation:**
1. Add `pytest-cov>=4.1.0` to dependencies.
2. Update CI:
```yaml
- name: Run Tests with Coverage
  run: uv run pytest --cov=src --cov=app --cov-fail-under=60 --cov-report=term-missing tests/
```

---

### 2.7 HIGH: Preprocessing Mismatch Between Training and Inference

> [!IMPORTANT]
> The text preprocessing logic in `insights_api.py` is **materially different** from the training pipeline's preprocessing in `make_dataset.py`.

| Step | Training ([make_dataset.py:46-67](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/data/make_dataset.py#L46-L67)) | Inference ([insights_api.py:161-178](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/insights_api.py#L161-L178)) |
|:---|:---|:---|
| Regex pattern | `[^a-zA-Z\s]` → removes ALL non-alpha | `[^A-Za-z0-9\s!?.,]` → keeps digits and punctuation |
| Stopwords | Full NLTK stopwords set | NLTK stopwords **minus** `{not, but, however, no, yet}` |
| Lemmatization | None | WordNet lemmatization applied |
| Short token filter | Tokens ≤ 2 chars removed | Not applied |

**Impact:** **Training-serving skew.** The model was trained on text cleaned to alphabetic-only with full stopword removal and no lemmatization. At inference time, it receives text with digits, punctuation, retained negation words, and lemmatized tokens. This fundamentally degrades prediction quality.

Meanwhile, `app/main.py:102` does **no preprocessing at all** — it passes raw `data.texts` directly to `vectorizer.transform()`.

**Recommendation:** Create a single `src/utils/text_preprocessing.py` module with the authoritative `clean_text()` function. All three consumers (training, `main.py`, `insights_api.py`) must import from this shared module.

---

### 2.8 HIGH: `app/` Directory Is Not a Python Package

> [!IMPORTANT]
> The `app/` directory has **no `__init__.py`** file. While `uvicorn app.main:app` works due to Python's namespace package behavior, this can cause import resolution issues with `pyright` and packaging tools.

**Impact:** Type checkers and IDE tools may not properly resolve cross-module imports within `app/`. The lack of a package marker signals that this was not designed as a proper Python package.

**Recommendation:** Create an empty `app/__init__.py` file.

---

### 2.9 HIGH: Mixed Dependency Boundaries — `pytest` in Production Dependencies

> [!IMPORTANT]
> `pyproject.toml` lists `pytest>=7.0` and `ruff>=0.1.0` in the main `dependencies` array instead of separating them into `[project.optional-dependencies] dev`.

```toml
dependencies = [
    ...
    "pytest>=7.0",
    "ruff>=0.1.0",
    ...
]
```

**Impact:** The Docker production container installs testing and linting tools unnecessarily, bloating the image by ~50MB and including development-only packages in the attack surface.

**Recommendation:** Restructure `pyproject.toml`:
```toml
[project]
dependencies = [
    # Only production dependencies
    "pandas>=2.0",
    "scikit-learn>=1.3",
    ...
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "pyright>=1.1.350",
]
```

In the Dockerfile, use `uv sync --frozen --no-dev` (which you already do — but it currently installs dev tools because they're in `dependencies`).

---

### 2.10 MEDIUM: Hardcoded API Key in Chrome Extension Source

> [!WARNING]
> [popup.js:10](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/chrome-extension/popup.js#L10) has a hardcoded API key field:
> ```javascript
> const API_KEY = "";  // Use your YouTube API key
> ```
> While currently empty, the `.env` file DOES contain a real API key (§2.2). The extension provides no input mechanism for the user to enter their own key dynamically at runtime. The `README.md` instructs users to "paste your Google API Key when prompted" but the extension never prompts.

**Impact:** Users must edit source code to use the extension, or the extension is non-functional out of the box.

**Recommendation:** Add an input field in the extension popup for the API key, stored in `chrome.storage.local`.

---

### 2.11 MEDIUM: `FeatureEngineeringConfig.use_distilbert` Is Typed as `str`, Not `bool`

> [!IMPORTANT]
> [schemas.py:91](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/schemas.py#L91) declares:
> ```python
> use_distilbert: str = Field(description="String boolean ('True'/'False')...")
> ```
> And [params.yaml:50](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/params.yaml#L50) stores `"False"` as a string.

This forces the downstream consumer ([feature_engineering.py:234-238](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/features/feature_engineering.py#L234-L238)) to perform manual string parsing:
```python
if isinstance(use_distilbert_val, str):
    use_distilbert = use_distilbert_val.lower() == "true"
```

**Impact:** TypeScript-style boolean confusion. Pydantic can natively coerce `"False"` to `False` if the field is typed as `bool`.

**Recommendation:** Change the schema to `use_distilbert: bool` and the `params.yaml` value to `false` (bare YAML boolean). Pydantic v2 handles this natively.

---

### 2.12 MEDIUM: `app/test_inference.py` Is a Manual Script, Not an Automated Test

> [!NOTE]
> [test_inference.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/test_inference.py) is a `requests`-based integration test that requires a running server. It uses `if __name__ == "__main__":` with `print()` statements and `sys.exit(1)` — not `pytest` assertions. It is not discovered by `pytest` and does not contribute to CI coverage.

**Impact:** The API endpoints have **zero automated test coverage** in CI. Any regression in `/predict` or `/predict_absa` goes undetected.

**Recommendation:** Rewrite using `httpx.AsyncClient` and `pytest` with FastAPI's `TestClient`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_positive():
    response = client.post("/predict", json={"texts": ["I love this!"]})
    assert response.status_code == 200
    assert "predictions" in response.json()
```

---

### 2.13 MEDIUM: No `ConfigDict(extra="forbid")` on Config Schemas

> [!IMPORTANT]
> The Pydantic schemas in [schemas.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/schemas.py) do not use `model_config = ConfigDict(extra="forbid")`. Any typo in `params.yaml` keys will be silently accepted by Pydantic's default permissive behavior.

**Example:** A typo like `use_dstilbert` instead of `use_distilbert` in `params.yaml` would pass validation silently, and the schema's `use_distilbert` field would fall back to its default or be missing — producing silent failures.

**Recommendation:** Add `model_config = ConfigDict(extra="forbid")` to the `AppConfig` root model and propagate to all nested models.

---

### 2.14 MEDIUM: Insights API Has CORS `allow_origins=["*"]` — No CORS on Main API

> [!NOTE]
> [insights_api.py:88-93](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/insights_api.py#L88-L93) enables CORS with `allow_origins=["*"]` (required for Chrome Extensions), but [main.py](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/app/main.py) has **no CORS middleware**. This means the Chrome Extension cannot call `/predict` or `/predict_absa` from the main API.

**Impact:** The Chrome Extensions can only reach the Insights API endpoints. The main API's ABSA endpoint is unreachable from browser context.

**Recommendation:** Add CORS middleware to `main.py` with appropriate origin restrictions:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)
```

---

### 2.15 MEDIUM: No `py.typed` Marker File

> [!NOTE]
> No `py.typed` marker file exists in `src/`. This file signals PEP 561 compliance to downstream consumers and type checkers. Its absence means `pyright` may not fully analyze the package.

**Recommendation:** Create an empty `src/py.typed` file.

---

### 2.16 LOW: `data_loader.py` Has a Lazy `import pickle` Inside Function Body

> [!NOTE]
> [data_loader.py:58](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/models/helpers/data_loader.py#L58) imports `pickle` inside `load_feature_data()`:
> ```python
> import pickle
> with open(feature_files["label_encoder"], "rb") as f:
>     le = pickle.load(f)
> ```
> All other standard library imports are at module level.

**Impact:** Minor readability inconsistency. `pickle` is a standard library module and has no performance cost at module level.

---

### 2.17 LOW: Root `Dockerfile` Healthcheck Uses `curl` But Extension Needs Two APIs

The Dockerfile only exposes port 8000 and only includes the main API app. The Insights API (port 8001) is not containerized separately. There is no `docker-compose.yml` to orchestrate both services together.

**Impact:** The Docker deployment documented in the README (`docker-compose up --build`) references a `docker/` directory that does not exist in the repository structure.

**Recommendation:** Either:
1. Create a `docker-compose.yml` at the project root with services for both APIs + MLflow, OR
2. Update the README to remove the Docker Compose reference and document single-container deployment.

---

### 2.18 LOW: No Security Scanning in CI

| Gap | Impact |
|:---|:---|
| Trivy scan has `exit-code: "0"` | Vulnerabilities are reported but **never block** the pipeline |
| No `bandit` or `safety` step | Insecure code patterns and CVE-vulnerable dependencies ship undetected |

The CI workflow already includes Trivy, but with `exit-code: "0"` it functions as a notification only, not as a gate.

**Recommendation:** Set `exit-code: "1"` on Trivy for CRITICAL severity, and add:
```yaml
- name: Python Security Scan
  run: uv run pip install bandit && uv run bandit -r src/ app/ -ll
```

---

### 2.19 LOW: `conftest.py` Uses `sys.path.append` Instead of Proper Package Installation

> [!NOTE]
> [conftest.py:5-6](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/tests/conftest.py#L5-L6) manually appends the project root to `sys.path`. This is fragile and unnecessary if the project is installed in editable mode (`uv pip install -e .`).

**Impact:** Tests may resolve imports differently than the production application. The `sys.path` hack can mask broken import configurations.

---

### 2.20 LOW: No API Versioning

All FastAPI endpoints (`/predict`, `/predict_absa`, `/health`) are registered at the root level. There is no API versioning prefix.

**Recommendation:** Use `APIRouter(prefix="/v1")`:
```python
router = APIRouter(prefix="/v1")

@router.post("/predict")
def predict(...): ...

app.include_router(router)
```

---

### 2.21 LOW: `ModelEvaluationConfig` Has Only One Field

> [!NOTE]
> [schemas.py:158-163](file:///c:/Users/sebas/Desktop/youtube-sentiment-analysis/src/config/schemas.py#L158-L163):
> ```python
> class ModelEvaluationConfig(BaseModel):
>     models: List[str]
> ```
> This Pydantic model adds no validation constraints (e.g., no `min_length=1`, no enum restriction on model names). An empty list `[]` would pass validation and produce a silent no-op in the evaluation stage.

**Recommendation:** Add `models: list[str] = Field(..., min_length=1)`.

---

### 2.22 LOW: Incomplete Migration to `src.constants`

> [!NOTE]
> While `src/constants/__init__.py` has been established as the single source of truth for paths, some components like `ConfigurationManager` still default to hardcoded `"params.yaml"` strings, and `src/utils/paths.py` still contains redundant, unaligned path definitions.

**Impact:** Maintaining two path utility modules increases technical debt and risks path divergence if only one is updated.

**Recommendation:** Fully migrate all path logic from `src/utils/paths.py` to `src/constants/__init__.py`, update `ConfigurationManager` to use `PARAMS_FILE_PATH`, and delete the legacy `paths.py` module.

---

## 3. Recommendations for Portfolio Differentiation 🚀

These are enhancements that go beyond "fixing gaps" and would make this project **stand out to elite employers**:

### 3.1 Add a `Makefile` for Developer Experience

A `Makefile` with standardized targets provides instant developer productivity:
```makefile
help:       ## Show available targets
install:    ## Install all dependencies
lint:       ## Run ruff check + format
typecheck:  ## Run pyright
test:       ## Run pytest with coverage
pipeline:   ## Run full DVC pipeline
serve:      ## Start both APIs + MLflow
docker:     ## Build and run Docker containers
clean:      ## Remove artifacts
```

### 3.2 Add `.pre-commit-config.yaml`

Enforce quality gates locally before commits reach CI:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
```

### 3.3 Add Great Expectations (GX) Data Validation (Rule 2.1)

The current pipeline has basic validation (label check, empty check), but no statistical data quality contracts. Add:
- Value distribution expectations (null % thresholds)
- Text length range checks
- Label balance monitoring
- Store GX suites as versioned artifacts in `data/contracts/`

### 3.4 Add Structured JSON Logging for Production (Rule 2.2)

The current logger uses human-readable format. For observability platforms (Datadog, ELK, CloudWatch), add a JSON formatter option:
```python
import json_log_formatter
handler.setFormatter(json_log_formatter.JSONFormatter())
```

### 3.5 Add OpenTelemetry Tracing (Rule 4.2)

Instrument FastAPI with span-level visibility into prediction latency, model loading time, and feature engineering overhead:
```toml
"opentelemetry-api>=1.20.0"
"opentelemetry-sdk>=1.20.0"
"opentelemetry-instrumentation-fastapi>=0.41b0"
```

### 3.6 Add Model Card Documentation

Following [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993):
- Model description and intended use
- Training data characteristics (Reddit → YouTube domain gap)
- Per-class evaluation metrics
- Limitations and ethical considerations (bias in sentiment lexicons)
- Versioning and change log

### 3.7 Add `CONTRIBUTING.md`

Document the development workflow, testing strategy, and code standards. This demonstrates team-readiness and engineering maturity.

### 3.8 Extract Shared Text Preprocessing & Feature Engineering

Create `src/utils/text_preprocessing.py` and `src/utils/feature_utils.py` as single-source-of-truth modules shared by training and inference — eliminating the skew issues identified in §2.4 and §2.7.

### 3.9 Add End-to-End Integration Tests

Write `pytest`-based tests using FastAPI's `TestClient` that exercise the full `/predict` and `/predict_absa` paths, including preprocessing, vectorization, and model inference — all without a running server.

### 3.10 Add `docker-compose.yml` for Full-Stack Development

Define a compose file that starts MLflow, the Inference API, and the Insights API together with health-checked dependencies:
```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports: ["5000:5000"]
  inference-api:
    build: .
    ports: ["8000:8000"]
    depends_on:
      mlflow: { condition: service_healthy }
  insights-api:
    build:
      context: .
      target: insights
    ports: ["8001:8001"]
```

---

## 4. Summary Scorecard

| Category | Score | Notes |
|:---|:---:|:---|
| **Architecture** | 8.5/10 | Extended FTI pattern, Pydantic config schemas, dual-API design, Chrome Extensions — impressive scope |
| **Code Quality** | 5.5/10 | Legacy typing imports, DRY violations in feature engineering and preprocessing, no `extra="forbid"`, string-typed bool |
| **Type Safety** | 3.5/10 | No `pyright` enforcement anywhere — CI, pre-commit, or dev dependencies. Legacy `typing` imports throughout |
| **Testing** | 4.5/10 | 4 test files (config, data validation, model pipeline), but no API endpoint tests, no coverage gate, manual `test_inference.py` |
| **CI/CD** | 6/10 | Good structure (test → build → deploy), Trivy scan, AWS integration — but no type checking, no coverage gate, no blocking security scan |
| **Security** | 3/10 | **API key committed to repository** (`.env` with YouTube key), Trivy non-blocking, CORS misconfiguration, no `bandit` |
| **Documentation** | 8/10 | Six-pillar report structure, module docstrings, README excellence — minor gaps in developer onboarding (no `.env.example`, no `CONTRIBUTING.md`) |
| **MLOps Maturity** | 8/10 | 12-stage DVC DAG, MLflow with Parent/Child runs, F1 threshold gating, model registry version handling — no Makefile, no pre-commit |
| **Training-Serving Integrity** | 4/10 | **Two** independent skew vectors: derived features (§2.4) and text preprocessing (§2.7). Both are critical for prediction quality |
| **Developer Experience** | 5/10 | No Makefile, no `.env.example`, no `docker-compose.yml`, `sys.path` hack in tests, `pytest` not in dev group |
| **TOTAL** | **6.7 / 10** | **NOT YET PROD-READY** |

**Overall: 6.7/10** — A creative and architecturally ambitious project with a strong MLOps foundation and an impressive end-to-end product demonstration (Chrome Extensions). The critical gaps are concentrated in **type safety enforcement**, **training-serving integrity**, and **security hygiene**. These are all systematically solvable — the project's modular architecture actually makes hardening straightforward.

---

## 5. Prioritized Action Plan

> [!TIP]
> Phases are ordered by impact and effort. Phase 1 should take ~1 hour; Phase 2 ~2 hours; Phase 3 ~3-4 hours; Phases 4-5 are longer-term portfolio investments.

### Phase 1: Security & Quick Wins (30 min)

- [ ] **Revoke compromised YouTube API key** and regenerate ([§2.2](#22-critical-missing-envexample-file))
- [ ] **Remove `.env` from Git history** using `git filter-repo` ([§2.2](#22-critical-missing-envexample-file))
- [ ] **Create `.env.example`** with placeholder values ([§2.2](#22-critical-missing-envexample-file))
- [ ] **Create `app/__init__.py`** ([§2.8](#28-high-app-directory-is-not-a-python-package))
- [ ] **Create `src/py.typed`** ([§2.15](#215-medium-no-pytyped-marker-file))
- [ ] **Move lazy `import pickle` to module level** in `data_loader.py` ([§2.16](#216-low-data_loaderpy-has-a-lazy-import-pickle-inside-function-body))
- [ ] **Complete path migration to `src.constants`** ([§2.22](#222-low-incomplete-migration-to-srcconstants))

### Phase 2: Type Safety & CI Hardening (1-2 hours)

- [ ] **Add `[tool.pyright]` to `pyproject.toml`** and `pyright>=1.1.350` to dev deps ([§2.1](#21-critical-no-pyright-configuration-or-ci-enforcement))
- [ ] **Expand `[tool.ruff]` configuration** with `target-version`, `select`, and `isort` ([§2.5](#25-high-minimal-toolruff-configuration-in-pyprojecttoml))
- [ ] **Replace all legacy `typing` imports** with modern PEP 604 builtins project-wide ([§2.3](#23-critical-legacy-typing-imports-across-codebase))
- [ ] **Separate dev dependencies** from production in `pyproject.toml` ([§2.9](#29-high-mixed-dependency-boundaries--pytest-in-production-dependencies))
- [ ] **Add `pytest-cov` with 60% coverage gate** in CI ([§2.6](#26-high-no-pytest-cov-and-no-coverage-gate-in-ci))
- [ ] **Add `pyright` CI step** in `ci_cd.yaml` ([§2.1](#21-critical-no-pyright-configuration-or-ci-enforcement))
- [ ] **Add `model_config = ConfigDict(extra="forbid")`** to all Pydantic schemas ([§2.13](#213-medium-no-configdictextraforbid-on-config-schemas))
- [ ] **Fix `use_distilbert: str` → `bool`** schema type ([§2.11](#211-medium-featureengineeringconfiguse_distilbert-is-typed-as-str-not-bool))

### Phase 3: Training-Serving Integrity (2-3 hours)

- [ ] **Create shared `src/utils/text_preprocessing.py`** — fix preprocessing skew ([§2.7](#27-high-preprocessing-mismatch-between-training-and-inference))
- [ ] **Create shared `src/utils/feature_utils.py`** — fix derived feature skew ([§2.4](#24-high-training-serving-skew-in-derived-feature-engineering))
- [ ] **Add CORS middleware to `main.py`** ([§2.14](#214-medium-insights-api-has-cors-allow_origins--no-cors-on-main-api))
- [ ] **Add API versioning (`/v1/` router)** to both APIs ([§2.20](#220-low-no-api-versioning))
- [ ] **Rewrite `test_inference.py` as `pytest` tests** using `TestClient` ([§2.12](#212-medium-apptest_inferencepy-is-a-manual-script-not-an-automated-test))
- [ ] **Fix Docker/README mismatch** — create `docker-compose.yml` or update docs ([§2.17](#217-low-root-dockerfile-healthcheck-uses-curl-but-extension-needs-two-apis))
- [ ] **Implement Artifacts Persistence Mandate (Rule 2.12)** — Centralize all non-code outputs into a versioned `artifacts/` root directory for absolute lifecycle persistence.


### Phase 4: Developer Experience (1-2 hours)

- [ ] **Add `Makefile`** ([§3.1](#31-add-a-makefile-for-developer-experience))
- [ ] **Add `.pre-commit-config.yaml`** ([§3.2](#32-add-pre-commit-configyaml))
- [ ] **Remove `sys.path` hack** from `conftest.py`, use editable install ([§2.19](#219-low-conftestpy-uses-syspathappend-instead-of-proper-package-installation))
- [ ] **Add `CONTRIBUTING.md`** ([§3.7](#37-add-contributingmd))
- [ ] **Add `docker-compose.yml`** ([§3.10](#310-add-docker-composeyml-for-full-stack-development))

### Phase 5: Portfolio Differentiation

- [ ] **Add Great Expectations data validation** ([§3.3](#33-add-great-expectations-gx-data-validation-rule-21))
- [ ] **Add structured JSON logging** ([§3.4](#34-add-structured-json-logging-for-production-rule-22))
- [ ] **Add OpenTelemetry tracing** ([§3.5](#35-add-opentelemetry-tracing-rule-42))
- [ ] **Add Model Card** ([§3.6](#36-add-model-card-documentation))
- [ ] **Set Trivy `exit-code: "1"`** and add `bandit` to CI ([§2.18](#218-low-no-security-scanning-in-ci))
- [ ] **Add Chrome Extension API key input UI** ([§2.10](#210-medium-hardcoded-api-key-in-chrome-extension-source))
