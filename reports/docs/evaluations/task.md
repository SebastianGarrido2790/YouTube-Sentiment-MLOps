# YouTube Sentiment Analysis — Codebase Hardening Task Tracker

## Phase 1: Security & Quick Wins
- [ ] Create `.env.example` with placeholder values
- [ ] Create `app/__init__.py`
- [ ] Create `src/py.typed`
- [ ] Move lazy `import pickle` to module level in `data_loader.py`
- [ ] Complete path migration to `src.constants` (remove `src.utils.paths`)
- [ ] ⚠️ MANUAL: Revoke compromised YouTube API key + clean git history

## Phase 2: Type Safety & CI Hardening
- [ ] Add `[tool.pyright]` to `pyproject.toml` + pyright dep
- [ ] Expand `[tool.ruff]` configuration
- [ ] Replace all legacy `typing` imports with PEP 604 builtins
- [ ] Separate dev dependencies from production in `pyproject.toml`
- [ ] Add `pytest-cov` with coverage gate in CI
- [ ] Add `pyright` CI step in `ci_cd.yaml`
- [ ] Add `ConfigDict(extra="forbid")` to Pydantic schemas
- [ ] Fix `use_distilbert: str` → `bool` in schema + params.yaml

## Phase 3: Training-Serving Integrity
- [ ] Create shared `src/utils/text_preprocessing.py`
- [ ] Create shared `src/utils/feature_utils.py` for derived features
- [ ] Refactor `feature_engineering.py` to use shared modules
- [ ] Refactor `app/main.py` + `app/insights_api.py` to use shared modules
- [ ] Add CORS middleware to `main.py`
- [ ] Rewrite `test_inference.py` as `pytest` tests using `TestClient`

## Phase 4: Developer Experience
- [ ] Add `Makefile`
- [ ] Add `.pre-commit-config.yaml`
- [ ] Remove `sys.path` hack from `conftest.py`
- [ ] Add `CONTRIBUTING.md`
