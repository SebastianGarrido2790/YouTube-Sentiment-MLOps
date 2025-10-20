# Project Overview: End-to-End MLOps Pipeline for Real-Time YouTube Sentiment Analysis

Let’s structure the project overview around **CRISP-DM** while embedding **MLOps best practices** and your design pillars: **reliability, scalability, maintainability, and adaptability.**
Below is the complete **strategic plan** for how we’ll build our end-to-end YouTube Sentiment Analysis MLOps system.

---

## 🧭 1. CRISP-DM Framework (Professional Data Science Lifecycle)

We’ll use **CRISP-DM** as the methodological backbone, but integrate it with **modern MLOps components** for full reproducibility, automation, and monitoring.

| CRISP-DM Stage                  | Objective                                                    | Key Deliverables                                                                                       | MLOps Integrations                            |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| **1. Business Understanding**   | Define problem scope, success metrics, and system goals.     | - Problem statement<br>- KPI definition (accuracy, latency, uptime)<br>- System architecture blueprint | GitHub README, system design diagram          |
| **2. Data Understanding**       | Collect, explore, and validate text data (YouTube comments). | - EDA notebook<br>- Data validation checks (missing, imbalance)<br>- DVC data versioning               | DVC tracking, `data/raw`, `data/interim`      |
| **3. Data Preparation**         | Clean, preprocess, and feature-engineer text data.           | - Tokenization, stopword removal<br>- TF-IDF and BERT embeddings<br>- Balanced training set            | Modular `src/features/` scripts, DVC pipeline |
| **4. Modeling**                 | Train baseline and advanced models, log experiments.         | - Baseline Logistic Regression<br>- Transformer-based fine-tuning<br>- Hyperparameter optimization     | MLflow experiment tracking, Optuna            |
| **5. Evaluation**               | Validate models using consistent metrics.                    | - Model report (F1, Precision, Recall)<br>- Model card (metadata)                                      | MLflow model registry, automated evaluation   |
| **6. Deployment**               | Serve model via API and integrate with Chrome extension.     | - REST API (FastAPI or AWS Lambda)<br>- Docker container                                               | Docker + CI/CD (GitHub Actions + AWS ECR/ECS) |
| **7. Monitoring & Maintenance** | Track performance drift and retrain as needed.               | - Drift detection<br>- Model version updates<br>- Scheduled retraining                                 | MLflow + DVC + AWS CloudWatch/Lambda triggers |

---

## 🏗️ 2. System Architecture Overview

### **High-Level Flow**

1. **Chrome Extension** → Captures YouTube comments (via DOM scraping or API).
2. **Backend API (FastAPI)** → Sends comment data to the inference endpoint.
3. **Model Service (Containerized)** → Predicts sentiment using the deployed model.
4. **Data Lake (S3)** → Stores raw & processed comment data.
5. **Pipeline Orchestration** → DVC + MLflow automate preprocessing, training, and deployment.
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

---

## ⚙️ 3. MLOps Pipeline Design

Each pipeline step will be modular and tracked via **DVC + MLflow**:

```bash
dvc.yaml
├── stages:
│   ├── data_ingestion:
│   │     cmd: python src/data/download_dataset.py
│   │     outs: data/raw/
│   ├── data_preprocessing:
│   │     cmd: python src/data/make_dataset.py
│   │     deps: [data/raw/]
│   │     outs: data/processed/
│   ├── feature_engineering:
│   │     cmd: python src/features/feature_engineering.py
│   │     outs: models/features/
│   ├── train_model:
│   │     cmd: python src/models/advanced_training.py
│   │     deps: [data/processed/]
│   │     outs: models/advanced/
│   ├── evaluate_model:
│   │     cmd: python src/models/model_evaluation.py
│   │     deps: [models/advanced/]
│   └── deploy_model:
│         cmd: python src/models/register_model.py
│         deps: [models/advanced/]
```

---

## 📦 4. Infrastructure and Automation

### **CI/CD Workflow (GitHub Actions)**

* **CI Stage**:

  * Run linting (flake8, black)
  * Execute unit tests (pytest)
  * Validate DVC stages
* **CD Stage**:

  * Build Docker image → push to AWS ECR
  * Deploy container to ECS or Lambda
  * Register model version in MLflow registry

### **Dockerization**

* Base image: `python:3.11-slim`
* Layers:

  * Install system deps
  * Install uv + dependencies
  * Copy source code
  * Expose FastAPI port
* Use multi-stage builds for lightweight production images.

---

## 🧩 5. Design Requirements Integration

| Design Requirement  | How It’s Achieved                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **Reliability**     | Version control (Git/DVC), testing suite, monitoring via CloudWatch                           |
| **Scalability**     | Modular pipeline, AWS ECS auto-scaling, distributed training (Accelerate)                     |
| **Maintainability** | Clean code, clear module boundaries, logging system (`src/utils/logger.py`)                   |
| **Adaptability**    | Config-driven pipeline (`params.yaml`), easily swap BERT/TF-IDF, flexible retraining triggers |

---

## 🧠 6. Planned Modules (Implementation Phases)

| Phase       | Component                       | Focus                                                             |
| ----------- | ------------------------------- | ----------------------------------------------------------------- |
| **Phase 1** | Data ingestion & preprocessing  | Automate dataset download, cleaning, and validation (DVC tracked) |
| **Phase 2** | Feature engineering             | TF-IDF + BERT comparison (feature performance analysis)           |
| **Phase 3** | Modeling                        | Baseline → Transformer fine-tuning + Optuna optimization          |
| **Phase 4** | Experiment tracking             | Integrate MLflow logging and registry                             |
| **Phase 5** | Deployment                      | Dockerize FastAPI inference service                               |
| **Phase 6** | CI/CD pipeline                  | GitHub Actions + AWS deployment automation                        |
| **Phase 7** | Real-time inference integration | Connect Chrome extension → inference API                          |
| **Phase 8** | Monitoring & retraining loop    | Detect drift, retrain via DVC + MLflow automation                 |

---

## 📋 7. Documentation and Governance

* **README.md** → Project overview, setup, and contribution guidelines.
* **Reports/** → Contain experiment results and data visualizations.
* **References/** → Include ML design document, model card, and system diagram.
* **Logging & Config Management**:

  * `.env` → API keys and secrets
  * `params.yaml` → Centralized hyperparameters/configuration
  * `src/utils/logger.py` → Unified logging for all modules

---

### ✅ Next Steps (Before Implementation)

1. Define **project KPIs** (accuracy target, latency, API response time).
2. Design **system architecture diagram** (logical + infrastructure).
3. Finalize **data versioning policy** (e.g., keep only 3 latest processed versions).
4. Prepare **DVC pipeline skeleton** (no execution yet).
5. Create **MLflow tracking server** setup plan (local first, then AWS S3 backend).
