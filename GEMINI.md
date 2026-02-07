# 🤖 GEMINI CLI Customization for MLOps Project

This file provides context to the Gemini model for its role as an AI assistant in the development of an **End-to-End MLOps Pipeline for Real-Time YouTube Sentiment Analysis**.

## 🚀 1. Project Goal

The primary objective is to master modern ML engineering and MLOps best practices by building a production-grade system that automatically processes YouTube comments, predicts sentiment in real-time (via a Chrome extension/FastAPI service), and ensures **reproducibility, automation, and continuous monitoring**.

## 🛠️ 2. Core Technology Stack

Please assume all code examples, advice, and configuration relate to this specific stack:

| Layer | Tool | Notes |
| :--- | :--- | :--- |
| **Project Methodology** | CRISP-DM + MLOps | For lifecycle and structure. |
| **Python** | Python 3.11 | Preferred environment. |
| **Dependencies** | **`uv`** + `pyproject.toml` | For fast, reproducible dependency management. |
| **Data Versioning** | **DVC** | To track data and pipeline stages (`dvc.yaml`). |
| **Experiment Tracking** | **MLflow** | To log parameters, metrics, models, and manage the model registry. |
| **Model Serving** | **FastAPI** + **Docker** | For the real-time inference API. |
| **CI/CD** | **GitHub Actions** | For automated testing, Docker builds, and deployment triggers. |
| **Cloud** | **AWS** (S3, ECR, ECS/Lambda, CloudWatch) | Primary cloud infrastructure. |
| **ML Models** | Logistic Regression (Baseline) & **Transformer/BERT** and boosted models (XGBoost, LightGBM) | Focus on text embeddings (TF-IDF, BERT). |
| **Configuration** | **Pydantic** (Strict Schemas) | Centralized, typed config (`src/config/`). |
| **Testing** | **pytest** | Unit and integration testing framework. |

## ✨ 3. Design Pillars (Non-Negotiable Requirements)

All proposed solutions, code, and architectural advice **must prioritize** these four requirements, as they define the project's production-readiness:

1.  **Reliability:** Use version control (Git/DVC/MLflow) and implement a comprehensive unit/integration testing suite (`pytest`).
2.  **Scalability:** Design modular components, favor containerization (Docker), and utilize AWS auto-scaling features.
3.  **Maintainability:** Insist on **clean, modular code** with clear module boundaries. Use **strict typing** (Pydantic) for configuration via `src/config/manager.py` instead of raw dictionaries.
4.  **Adaptability:** Use config schemas (`src/config/schemas.py`) to enforce parameter validity while allowing easy swapping of models or hyperparameters.

## 📝 4. Preferred Response Style

When responding to development queries, please:

* **Be brief and direct.**
* Provide **fully complete, self-contained code snippets** when asked for configuration or script examples.
* **Enforce Strict Config**: Always prefer `ConfigurationManager` over `dvc.api.params_show()` for parameter loading.
* Focus on **MLOps best practices**: **Automation, Reproducibility, and Monitoring**.
* Reference the specific project files/structure (e.g., `dvc.yaml`, `src/models/advanced_training.py`, `pyproject.toml`).

## ❓ 5. Example Interaction Topics

You can expect prompts related to the following specific development tasks:

* Writing **DVC stage definitions** for data preprocessing and training.
* Configuring **MLflow logging** within the `src/models/advanced_training.py` script.
* Creating a **`Dockerfile`** for the FastAPI inference service.
* Designing **GitHub Actions workflows** for CI (testing) and CD (Docker build/push).
* Setting up **`uv`** for a `pyproject.toml` dependency group.
* Implementing **model drift detection** logic using AWS/MLflow features.
