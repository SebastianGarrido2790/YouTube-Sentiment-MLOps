# Docker Configuration for MLOps

This document provides a comprehensive overview of the Docker setup for the YouTube Sentiment Analysis project. The configuration is designed to create a reproducible, isolated, and secure environment for production deployment, adhering to MLOps best practices.

## 1. Overview

Docker is a cornerstone of this project, used to containerize the FastAPI inference service and its dependencies. This approach provides several key benefits:

-   **Reproducibility**: The application runs in the exact same environment every time.
-   **Security**: Runs as a non-privileged user with minimal attack surface.
-   **Portability**: The container can be run anywhere Docker is installed.
-   **Scalability**: Containerized applications are easy to scale horizontally.

The Docker setup consists of two main files located in the project root:
-   [`Dockerfile`](../../Dockerfile): A blueprint for building the secure application image.
-   [`.dockerignore`](../../.dockerignore): A list of files to exclude from the build context.

> **Note**: We favor a simplified, single-file Docker approach. Complex orchestration (like `docker-compose`) has been removed to keep the deployment logic straightforward and focused on the single inference service artifact.

## 2. `Dockerfile` Explained

The `Dockerfile` defines the steps to create a secure, production-ready container image. It is optimized for security (non-root user), reliability (healthchecks), and size (multi-stage caching).

### Key Security & MLOps Features

1.  **Non-Root Execution**: The container creates and switches to `appuser` (UID 1000). This prevents a compromised container from having root access to the host system.
2.  **Health Checks**: The `HEALTHCHECK` instruction allows orchestrators (like AWS ECS or Docker Swarm) to know if the service is actually ready to receive traffic, not just if the process is running.
3.  **Strict Dependency Management**: Uses `uv sync --frozen` to ensure the installed packages exactly match `uv.lock`, guaranteeing reproducibility.
4.  **Minimal Base Image**: Uses `python:3.11-slim` to reduce the image size and potential vulnerability surface.

## 3. `.dockerignore` Explained

The [`.dockerignore`](../../.dockerignore) file is essential for keeping the build context small and secure. It excludes:

-   **Data (`data/`)**: prevents gigabytes of training data from bloating the image.
-   **Models (`models/`, `mlruns/`)**: Models should be loaded from a model registry (like MLflow) or mounted at runtime, not baked into the code image.
-   **Secrets (`.env`)**: prevents accidental leakage of API keys.
-   **Git/DVC metafiles**: unnecessary for the production runtime.

## 4. How to Build and Run

### 4.1 Build the Image

Run this command from the project root. The first build might take a few minutes, but subsequent builds will be extremely fast thanks to `uv` cache mounting.

```bash
docker build -t youtube-sentiment-api:latest .
```

> **Optimization Note**: We use `--mount=type=cache,target=/root/.cache/uv` in the Dockerfile. This persists the `uv` package cache between builds, meaning you won't redownload dependencies unless `uv.lock` changes.

### 4.2 Run the Container

Running locally requires mapping the port.

```bash
docker run -d \
  --name youtube-api \
  -p 8000:8000 \
  youtube-sentiment-api:latest
```

### 4.3 Verify Health

You should see the container become healthy in ~10 seconds.

```bash
docker ps
# Look for "(healthy)" in the STATUS column
```

Or manually:

```bash
curl http://localhost:8000/health
# Output: {"status":"ok","message":"YouTube Sentiment Analysis API is running."}
```

## 5. Advanced Configuration & Optimizations

### 5.1 Offline / Isolated Docker Mode (`PREFER_LOCAL_MODEL`)

By default, the application attempts to connect to the MLflow Model Registry to fetch the champion model. In isolated Docker environments (or without internet), this connection may fail or time out.

To force the application to use the **local model artifact**, set `PREFER_LOCAL_MODEL=true` and mount your models directory.

```bash
docker run -d \
  --name youtube-api \
  -e PREFER_LOCAL_MODEL=true \
  -v ${PWD}/models:/app/models \
  -p 8000:8000 \
  youtube-sentiment-api:latest
```

-   **`PREFER_LOCAL_MODEL=true`**: Checks for `models/advanced/lightgbm_model.pkl` FIRST. This avoids the 30s MLflow connection timeout.
-   **Default (`false`)**: Checks MLflow Registry first, falls back to local only on failure.

### 5.2 Lazy Loading (ABSA Model)

The Aspect-Based Sentiment Analysis (ABSA) model is large (~800MB). To prevent the API from hanging during startup, we implemented **Lazy Loading**.

-   **Startup**: The API starts instantly (focusing on the main lightweight model).
-   **First Request**: The ABSA model is loaded into memory only when `/predict_absa` is first called.
-   **Effect**: 
    -   `docker run` -> API ready in ~5 seconds.
    -   First `curl .../predict_absa` -> Takes ~5-10 seconds (one-time cost).
    -   Subsequent requests -> Instant.
