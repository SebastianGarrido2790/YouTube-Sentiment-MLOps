# Project Overview: End-to-End MLOps Pipeline for Real-Time YouTube Sentiment Analysis

Let’s structure the project overview around **CRISP-DM** while embedding **MLOps best practices** and your design pillars: **reliability, scalability, maintainability, and adaptability.**

Below is the complete **strategic plan** for how we’ll build our end-to-end YouTube Sentiment Analysis MLOps system.

---

## 🧭 1. CRISP-DM Framework (Professional Data Science Lifecycle)

We’ll use **CRISP-DM** as the methodological backbone, but integrate it with **modern MLOps components** for full reproducibility, automation, and monitoring.

| CRISP-DM Stage                  | Objective                                                    | Key Deliverables                                                                                       | MLOps Integrations                            |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| **1. Business Understanding**   | Define problem scope, success metrics, and system goals.     | - Problem statement<br>- KPI definition (accuracy, latency, uptime)<br>- System architecture blueprint | GitHub README, system design diagram          |
| **2. Data Understanding**       | Collect, explore, and validate text data.                    | - EDA notebook<br>- Data validation checks (missing, imbalance)<br>- DVC data versioning               | DVC tracking, `data/raw`, `data/interim`      |
| **3. Data Preparation**         | Clean, preprocess, and feature-engineer text data.           | - Tokenization, stopword removal<br>- TF-IDF and BERT embeddings<br>- Balanced training set            | Modular `src/features/` scripts, DVC pipeline |
| **4. Modeling**                 | Train baseline and advanced models, log experiments.         | - Baseline Logistic Regression<br>- Transformer-based fine-tuning<br>- Hyperparameter optimization     | MLflow experiment tracking, Optuna            |
| **5. Evaluation**               | Validate models using consistent metrics.                    | - Model report (F1, Precision, Recall)<br>- Model card (metadata)<br>- Champion model selection        | MLflow model registry, automated evaluation   |
| **6. Deployment**               | Serve model via API and integrate with Chrome extension.     | - REST API (FastAPI or AWS Lambda)<br>- Docker container                                               | Docker + CI/CD (GitHub Actions + AWS ECR/ECS) |
| **7. Monitoring & Maintenance** | Track performance drift and retrain as needed.               | - Drift detection<br>- Model version updates<br>- Scheduled retraining                                 | MLflow + DVC + AWS CloudWatch/Lambda triggers |

---

## 🏗️ 2. System Architecture Overview

### **High-Level Flow**

1. **Chrome Extension** (Planned) → Captures YouTube comments (via DOM scraping or API).
2. **Backend API (FastAPI)** (Planned) → Sends comment data to the inference endpoint.
3. **Model Service (Containerized)** → Predicts sentiment using the deployed model.
4. **Data Lake (S3/DVC)** → Stores raw & processed comment data.
5. **Pipeline Orchestration** → **DVC + MLflow** automate preprocessing, training, and deployment.
6. **CI/CD** → GitHub Actions for testing, training triggers, Docker builds, and AWS deployment.

### **Technology Stack**

| Layer                      | Tool                              | Purpose                             |
| -------------------------- | --------------------------------- | ----------------------------------- |
| **Data Versioning**        | DVC                               | Track raw → processed data lineage  |
| **Experiment Tracking**    | MLflow                            | Log parameters, metrics, models     |
| **Environment Management** | `uv` + `pyproject.toml`           | Reproducible and lightweight        |
| **Model Serving**          | FastAPI + Docker                  | Production-ready sentiment API      |
| **Orchestration**          | GitHub Actions                    | Continuous Integration and Delivery |
| **Cloud Infrastructure**   | AWS (S3, Lambda, ECS, CloudWatch) | Deployment & monitoring             |
| **Storage**                | PostgreSQL / S3                   | Store model metadata and datasets   |
| **Testing**                | pytest                            | Unit & integration tests            |

---

## ⚙️ 3. MLOps Pipeline Design

Each pipeline step is modular and tracked via **DVC + MLflow**. The current implementation includes:

```yaml
stages:
  data_ingestion:      # Download raw comments
  data_preparation:    # Clean, split, encode labels
  feature_comparison:  # Compare TF-IDF settings
  feature_tuning:      # Optimize n-grams/max_features
  imbalance_tuning:    # Select best resampling strategy (SMOTE, ADASYN)
  feature_engineering: # Generate final feature matrices
  baseline_model:      # Train Logistic Regression benchmark
  hyperparameter_tuning_lightgbm: # Tune LightGBM via Optuna
  hyperparameter_tuning_xgboost:  # Tune XGBoost via Optuna
  train_distilbert:    # Fine-tune DistilBERT (optional)
  model_evaluation:    # Compare all models on Test set
  register_model:      # Promote Champion based on F1 threshold
```

This structure ensures that changes in early stages (e.g., data prep) automatically trigger re-runs of dependent downstream stages (modeling/evaluation).

---

## 📦 4. Infrastructure and Automation

### **CI/CD Workflow (GitHub Actions)**

* **CI Stage**:
  * Run linting (flake8, black)
  * Execute unit tests (`pytest`) covering:
    * Configuration loading
    * Data validation logic
    * Model pipeline orchestration
  * Validate DVC stages (`dvc repro --dry-run`)
* **CD Stage** (Planned):
  * Build Docker image → push to AWS ECR
  * Deploy container to ECS or Lambda
  * Register model version in MLflow registry

### **Dockerization**

* Base image: `python:3.12-slim`
* Layers:
  * Install system deps
  * Install `uv` + dependencies (`uv sync`)
  * Copy source code
  * Expose FastAPI port
* Use multi-stage builds for lightweight production images.

---

## 🧩 5. Design Requirements Integration

| Design Requirement  | How It’s Achieved                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **Reliability**     | Version control (Git/DVC), comprehensive test suite (`tests/`), strict config validation      |
| **Scalability**     | Modular pipeline, efficient data formats (Parquet/NPZ), distributed training ready            |
| **Maintainability** | Clean code, clear module boundaries, centralized logging (`src/utils/logger.py`)              |
| **Adaptability**    | Config-driven pipeline (`params.yaml`), easily swap BERT/TF-IDF, flexible retraining triggers |

---

## 🧠 6. Implementation Status (Phases)

| Phase       | Component                       | Status          | Focus                                                             |
| ----------- | ------------------------------- | --------------- | ----------------------------------------------------------------- |
| **Phase 1** | Data ingestion & preprocessing  | ✅ Complete      | Automate dataset download, cleaning, and validation (DVC tracked) |
| **Phase 2** | Feature engineering             | ✅ Complete      | TF-IDF tuning, imbalance handling comparison                      |
| **Phase 3** | Modeling                        | ✅ Complete      | Baseline + XGBoost/LightGBM/DistilBERT + Optuna optimization      |
| **Phase 4** | Experiment tracking             | ✅ Complete      | Integrate MLflow logging and registry                             |
| **Phase 5** | Evaluation & Registration       | ✅ Complete      | Automated champion selection and MLflow Model Registry promotion  |
| **Phase 6** | Deployment                      | ✅ Complete      | Dockerize FastAPI inference service                               |
| **Phase 7** | CI/CD pipeline                  | ✅ Complete      | GitHub Actions + AWS deployment automation                        |
| **Phase 8** | Real-time inference integration | ✅ Complete      | Connect Chrome extension → inference API                          |

---

## 📋 7. Documentation and Governance

* **README.md** → Project overview, setup, and contribution guidelines.
* **Reports/** → Contain experiment results and data visualizations.
* **References/** → Include ML design document, model card, and system diagram.
* **Logging & Config Management**:
  * `.env` → API keys and secrets
  * `params.yaml` → Centralized hyperparameters/configuration
  * `src/utils/logger.py` → Unified logging for all modules
  * `src/config/manager.py` → Type-safe configuration loading

---

### ✅ Next Steps

1. **Cloud Deployment**: Set up AWS infrastructure (S3, ECS/Lambda).
