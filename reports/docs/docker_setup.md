# Docker Configuration for MLOps

This document provides a comprehensive overview of the Docker setup for the YouTube Sentiment Analysis project. The configuration is designed to create a reproducible, isolated, and efficient environment for both local development and production deployment, adhering to MLOps best practices.

## 1. Overview

Docker is a cornerstone of this project, used to containerize the FastAPI inference service and its dependencies. This approach provides several key benefits:

-   **Reproducibility**: The application runs in the exact same environment every time, regardless of the host machine.
-   **Isolation**: Project dependencies are isolated from the host system, preventing conflicts.
-   **Portability**: The container can be run anywhere Docker is installed, from a developer's laptop to a cloud server.
-   **Scalability**: Containerized applications are easy to scale horizontally.

The Docker setup consists of three main files located in the `docker/` directory:
-   `Dockerfile`: A blueprint for building the application image.
-   `.dockerignore`: A list of files to exclude from the build context.
-   `docker-compose.yml`: A tool for defining and running the multi-container local development environment.

## 2. `Dockerfile` Explained

The `Dockerfile` defines the steps to create a container image for the FastAPI application. It is optimized for small image size and efficient builds using layer caching.

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

# ------------------------------------------------------------
# Copy project code
# ------------------------------------------------------------
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command (production mode)
CMD ["uv", "run", "uvicorn", "app.predict_model:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key Stages:**

1.  **Base Image**: It starts from `python:3.11-slim`, a lightweight official Python image.
2.  **Dependency Installation**:
    *   It first installs `uv`, our fast package manager.
    *   It then copies only the dependency files (`pyproject.toml` and `uv.lock`) and installs the packages. This is a crucial optimization. Docker caches this layer, and it will only be rebuilt if these specific files change, not on every code change.
3.  **Code Copying**: The application source code (`app/` and `src/`) is copied into the image in a separate layer.
4.  **Port Exposure**: It exposes port `8000` for the FastAPI application.
5.  **Default Command**: The `CMD` instruction specifies the command to run when the container starts: `uvicorn` to serve the FastAPI application.

## 3. `.dockerignore` Explained

The `.dockerignore` file is essential for keeping the Docker image small and the build process fast. It lists files and directories that should be excluded from the build context sent to the Docker daemon.

**Key Exclusions:**

-   **Python artifacts and virtual environments**: `__pycache__/`, `.venv/`, etc.
-   **MLOps artifacts**: `data/`, `models/`, `mlruns/`, `logs/`. These are often very large and are not needed inside the application image. The model is loaded from the MLflow Model Registry at runtime, not from local files.
-   **Git and DVC history**: `.git/`, `.dvc/`.
-   **Development files**: `notebooks/`, `reports/`, etc.

By excluding these files, we ensure a minimal build context, which leads to faster builds and a smaller, more secure final image.

## 4. `docker-compose.yml` for Local Development

The `docker-compose.yml` file orchestrates a multi-container environment for local development, making it easy to run the entire application stack with a single command. It defines two services: `api` and `mlflow`.

```yaml
services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: youtube_sentiment_api
    ports:
      - "8000:8000"
    env_file:
      - ../.env
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ../app:/app/app
      - ../src:/app/src
      - ../pyproject.toml:/app/pyproject.toml
      - ../uv.lock:/app/uv.lock
    command: >
      uv run uvicorn app.predict_model:app
      --host 0.0.0.0 --port 8000 --reload
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow_server
    ports:
      - "5000:5000"
    command: >
      mlflow server ...
    volumes:
      - ./mlruns:/app/mlruns
      - ./mlflow.db:/app/mlflow.db
```

**`api` Service:**

-   **`build`**: It builds the Docker image using the `Dockerfile` in the `docker/` directory with the project root as the context.
-   **`ports`**: It maps port `8000` of the container to port `8000` on the host machine.
-   **`env_file`**: It loads environment variables from the `.env` file in the project root.
-   **`environment`**: It sets the `MLFLOW_TRACKING_URI` to `http://mlflow:5000`, allowing the `api` service to communicate with the `mlflow` service using its service name.
-   **`volumes`**: This is key for local development. It mounts the local `app/` and `src/` directories into the container. This allows for **hot-reloading**: any changes made to the code on the host machine are immediately reflected in the container without needing to rebuild the image.
-   **`command`**: It overrides the `CMD` from the `Dockerfile` to run `uvicorn` with the `--reload` flag, enabling hot-reloading.
-   **`depends_on`**: It ensures that the `mlflow` service is started before the `api` service.

**`mlflow` Service:**

-   **`image`**: It uses a standard MLflow image from the GitHub Container Registry.
-   **`ports`**: It maps port `5000` for the MLflow UI.
-   **`volumes`**: It mounts the local `mlruns/` directory and `mlflow.db` file into the container, ensuring that all MLflow experiments and artifacts are persisted on the host machine.

## 5. How to Use the Docker Setup

Here is a simple workflow for using the Docker setup for local development. All commands should be run from the project root directory.

1.  **Build the Docker Images**:
    This command builds the `api` image as defined in the `docker-compose.yml` file.

    ```bash
    docker compose -f docker/docker-compose.yml build
    ```

2.  **Start the Services**:
    This command starts the `api` and `mlflow` services in the background.

    ```bash
    docker compose -f docker/docker-compose.yml up -d
    ```

3.  **Access the Services**:
    -   The **FastAPI application** will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.
    -   The **MLflow UI** will be available at `http://localhost:5000`.

4.  **View Logs**:
    To view the logs for the running services, use the following command:

    ```bash
    docker compose -f docker/docker-compose.yml logs -f
    ```

5.  **Stop the Services**:
    To stop the services and remove the containers, use the following command:

    ```bash
    docker compose -f docker/docker-compose.yml down
    ```

6.  **Run Commands Inside the Container**:
    You can execute commands directly inside the running `api` container. For example, to run the test suite:

    ```bash
    docker compose -f docker/docker-compose.yml exec api uv run pytest -v
    ```

