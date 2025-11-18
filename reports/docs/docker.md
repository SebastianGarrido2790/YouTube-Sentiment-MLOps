# Docker Setup

Given our goal (a **simple, reliable, single-stage Docker setup** optimized for local development and CI/CD), here’s how to design a **cache-efficient container** that avoids reinstalling dependencies on every code change while remaining clean and reproducible under `uv`.

---

## 🧱 1. Dockerfile (Single-Stage, Cached Dependencies)

```dockerfile
# ============================================================
# Production-grade FastAPI container for YouTube Sentiment API
# ============================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS-level dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install uv and project dependencies separately for caching
# ------------------------------------------------------------
RUN pip install --no-cache-dir uv

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (will only rebuild if deps change)
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------
# Copy project code
# ------------------------------------------------------------
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set environment variable for MLflow tracking (optional)
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
    ENV=production

# Default command (production mode)
CMD ["uv", "run", "uvicorn", "app.predict_model:app", "--host", "0.0.0.0", "--port", "8000"]
```

### ✅ Key Features

* **Layer caching**: Dependencies are only reinstalled when `pyproject.toml` or `uv.lock` changes.
* **Minimal base image** (`python:3.11-slim`) → lightweight.
* **Single-stage** → simpler than a multistage build.
* **`uv sync --frozen`** ensures reproducibility and speed.
* **Non-root best practice** can be added if required for security (`RUN useradd appuser && USER appuser`).

---

## 🧩 2. `.dockerignore`

Prevent unnecessary files from bloating your build context:

```
# Python / virtual env
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
.python-version

# Logs and artifacts
logs/
mlruns/
mlflow.db
mlartifacts/
*.log

# Data and reports
data/
reports/
notebooks/
references/
models/
.DS_Store

# DVC / Git
.dvc/
.dvcignore
.git/
.gitignore

# IDE / misc
.vscode/
.idea/
```

---

## ⚙️ 3. `docker-compose.yml` (Local Development)

This setup lets you mount your local code for fast iteration **without reinstalling dependencies**:

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: youtube_sentiment_api
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./app:/app/app
      - ./src:/app/src
      - ./data:/app/data
      - ./models:/app/models
    command: uv run uvicorn app.predict_model:app --host 0.0.0.0 --port 8000 --reload
```

### ✅ Notes

* **Mounted volumes** (`app/`, `src/`) allow instant code updates without rebuild.
* **Dependencies remain cached** unless you modify dependency files.

---

## 🚀 4. GitHub Actions CI/CD (`.github/workflows/ci.yml`)

A minimal but complete workflow for testing + building the container:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        run: pip install uv

      - name: Sync dependencies
        run: uv sync --frozen --no-dev

      - name: Run unit tests
        run: uv run pytest -v app/test_inference.py

      - name: Build Docker image
        run: docker build -t youtube-sentiment-api .

      - name: Run health check
        run: docker run --rm -p 8000:8000 -d youtube-sentiment-api && sleep 10 && curl -f http://127.0.0.1:8000/health
```

Later, you can extend this to:

* Push to **AWS ECR** (`aws-actions/amazon-ecr-login`).
* Deploy to **AWS ECS** or **Lambda** via separate `deploy.yml`.

---

## 🧠 5. Developer Workflow Summary

| Task                       | Command                                                                           |
| -------------------------- | --------------------------------------------------------------------------------- |
| Build container            | `docker compose build`                                                            |
| Run locally (hot reload)   | `docker compose up`                                                               |
| Rebuild dependencies       | `docker compose build --no-cache` (only if `pyproject.toml` or `uv.lock` changed) |
| Run tests inside container | `docker compose exec api uv run pytest`                                           |
| Check logs                 | `docker compose logs -f api`                                                      |

---

## Dockerized FastAPI inference service is running *exactly* as intended.

Let’s unpack what’s happening and confirm next steps for production-readiness.

---

### ✅ What Worked

1. **Docker build cache works correctly**

   * You only rebuilt dependencies once (`uv sync` took ~220s) — subsequent builds will be instant unless `pyproject.toml` or `uv.lock` changes.
   * The `COPY . .` layer and `--reload` mode ensure hot reloading of your Python code during development.

2. **API successfully containerized**

   * The service exposed port `8000` → accessible at `http://127.0.0.1:8000`.
   * `curl` test returned a valid response:

     ```json
     {"predictions":["Positive"],"encoded_labels":[2],"feature_shape":[1,1004]}
     ```

     meaning the model, vectorizer, and encoder loaded flawlessly inside Docker.

3. **MLflow fallback logic worked**

   * The log shows:

     ```
     ⚠️ MLflow registry unavailable or alias not found. Loaded local LightGBM model...
     ```

     This confirms that the failover mechanism correctly defaulted to a local `.pkl` model when the registry wasn’t reachable — excellent reliability behavior.

---

### ⚙️ Minor Warnings and Fixes

| Warning                                            | Meaning                                        | Recommendation                                                                                                                       |
| -------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `the attribute version is obsolete`                | Docker Compose v2 ignores the `version` field  | Remove the first line (`version: "3.9"`) from `docker-compose.yml`                                                                   |
| `UserWarning: X does not have valid feature names` | Sklearn warning when reusing TF-IDF vectorizer | Safe to ignore in inference; you can suppress it via `warnings.filterwarnings("ignore", category=UserWarning)` in `predict_model.py` |
| `Converting data to scipy sparse matrix`           | LightGBM internal optimization notice          | Benign; LightGBM handles this automatically                                                                                          |

---

### 🧩 Optional: Improve Developer Workflow

If you want to make **code edits without full rebuilds**, ensure your `docker-compose.yml` uses **mounted volumes** (which already is there):

```yaml
volumes:
  - ./app:/app/app
  - ./src:/app/src
```

That’s why the logs show:

```
INFO:     Will watch for changes in these directories: ['/app']
```

Uvicorn is detecting local file changes and reloading automatically — perfect for development.

---

### 🚀 Recommended Next Steps

Now that the local container runs perfectly, you can move forward with production deployment.

#### 1. Configure MLflow Tracking on AWS EC2 or S3

* Launch an MLflow tracking server (EC2 + S3 + SQLite/Postgres).
* Update your `.env`:

  ```bash
  MLFLOW_TRACKING_URI=http://<your-ec2-ip>:5000
  ENV=production
  ```
* Then rebuild the container (`docker compose build`) to make it point to the remote URI.

#### 2. Add a CI/CD Deployment Step

* Extend `.github/workflows/ci.yml` to:

  * Log in to **AWS ECR**
  * Push the image
  * Trigger **ECS** or **Lambda** deployment
* You’ll then have a fully automated, versioned pipeline:

  1. Test →
  2. Build Docker image →
  3. Push to ECR →
  4. Deploy to ECS →
  5. Load latest model from MLflow.

#### 3. Optional Enhancements

* Add health endpoint monitoring (`/health`) in GitHub Actions or CloudWatch.
* Create a lightweight `Makefile` for one-liner commands:

  ```makefile
  build:
  	docker compose build
  up:
  	docker compose up
  test:
  	curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"texts":["Great work!"]}'
  ```

---

### ✅ Next Step: 

**AWS ECR + ECS-ready GitHub Actions workflow (`deploy.yml`)**

Designed to:

* build the same container,
* push to your ECR repository,
* and automatically update an ECS service.

---

## Docker System Cleanup

| Command | Effect |
|-------|--------|
| `docker builder prune -af` | **Clears BuildKit cache** → fixes `parent snapshot does not exist` |
| `docker system prune -af --volumes` | Removes: <br> • Stopped containers <br> • Unused networks <br> • Dangling images <br> • **All volumes** (including old `.venv`) |

> **Safe**: Only removes *unused* resources. Your code, MLflow DB, and DVC data are preserved.

---

### Run Order

```bash
# 1. Stop everything
docker compose down -v

# 2. Full Docker system cleanup (safe, removes everything unused)
docker builder prune -af && docker system prune -af --volumes

# 3. Rebuild & start
docker compose build --no-cache && docker compose up

# Go to the project root (if you are in the docker/ folder) and run:
cd ..
# Take down any existing containers first
docker compose -f docker/docker-compose.yml down
# Run the build command, specifying the compose file's location
docker compose -f docker/docker-compose.yml build && docker compose -f docker/docker-compose.yml up
```

Your API will start and connect to MLflow at `host.docker.internal:5000`.

