# Docker Configuration for MLOps

This document provides a comprehensive overview of the Docker setup for the YouTube Sentiment Analysis project. The configuration is designed to create a reproducible, isolated, and efficient environment for both local development and production deployment, adhering to MLOps best practices.

## 1. Overview

Docker is a cornerstone of this project, used to containerize the FastAPI inference service and its dependencies. This approach provides several key benefits:

-   **Reproducibility**: The application runs in the exact same environment every time, regardless of the host machine.
-   **Isolation**: Project dependencies are isolated from the host system, preventing conflicts.
-   **Portability**: The container can be run anywhere Docker is installed, from a developer's laptop to a cloud server.
-   **Scalability**: Containerized applications are easy to scale horizontally.

The Docker setup consists of four main files located in the `docker/` directory:
-   [`Dockerfile`](docker/Dockerfile): A blueprint for building the application image.
-   [`.dockerignore`](docker/.dockerignore): A list of files to exclude from the build context.
-   [`docker-compose.yml`](docker/docker-compose.yml): Development environment with hot-reloading and MLflow integration.
-   [`docker-compose.dev.yml`](docker/docker-compose.dev.yml): Lightweight development environment connecting to host MLflow.

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
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key Stages:**

1.  **Base Image**: It starts from [`python:3.11-slim`](docker/Dockerfile:4), a lightweight official Python image that provides a minimal Python environment.
2.  **Environment Configuration**: Sets [`PYTHONDONTWRITEBYTECODE=1`](docker/Dockerfile:10) to prevent `.pyc` file generation and [`PYTHONUNBUFFERED=1`](docker/Dockerfile:11) for real-time logging.
3.  **System Dependencies**: Installs essential build tools ([`build-essential`](docker/Dockerfile:15), [`curl`](docker/Dockerfile:15), [`git`](docker/Dockerfile:15)) needed for Python package compilation.
4.  **Package Manager Installation**: Installs [`uv`](docker/Dockerfile:21), a fast Python package manager that significantly speeds up dependency installation.
5.  **Dependency Layer Optimization**:
    - Copies only [`pyproject.toml`](docker/Dockerfile:24) and [`uv.lock`](docker/Dockerfile:24) first
    - Runs [`uv sync --frozen --no-dev`](docker/Dockerfile:27) to install production dependencies only
    - This layer is cached and only rebuilds when dependency files change, dramatically speeding up rebuilds during development
6.  **Application Code**: Copies the entire project with [`COPY . .`](docker/Dockerfile:32) after dependencies are installed.
7.  **Port Configuration**: Exposes port [`8000`](docker/Dockerfile:35) for the FastAPI application.
8.  **Environment Variables**: Sets default [`MLFLOW_TRACKING_URI`](docker/Dockerfile:38) and [`ENV=production`](docker/Dockerfile:39) for runtime configuration.
9.  **Production Command**: Runs [`uvicorn`](docker/Dockerfile:42) through [`uv run`](docker/Dockerfile:42) to serve the FastAPI application at [`app.main:app`](docker/Dockerfile:42).

## 3. `.dockerignore` Explained

The [`.dockerignore`](docker/.dockerignore) file is essential for keeping the Docker image small and the build process fast. It lists files and directories that should be excluded from the build context sent to the Docker daemon.

### Current Contents

```dockerignore
# ------------------------------------------------------------
# 1. Python Environment & Build Artifacts (Necessary Exclusions)
# ------------------------------------------------------------
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
.python-version
.vscode/

# ------------------------------------------------------------
# 2. DVC / MLflow / MLOps Artifacts (Primary Cause of Large Context)
# ------------------------------------------------------------
# Exclude ALL directories containing large, generated files, data, and models.
# The model will be loaded from the MLflow Registry, not from local files.
data/         # Contains raw, interim, and processed datasets (Gigabytes)
models/       # Contains trained models, evaluation artifacts (large files)
mlruns/       # MLflow run metadata and artifacts
mlflow_data/  # Includes mlruns, mlartifacts, and mlflow.db
mlflow.db
logs/         # Application logs
reports/      # Generated analysis, figures, docs

# DVC/Git-related files
.dvc/
dvc.lock
.dvcignore
.git/
.gitignore

# ------------------------------------------------------------
# 3. Development/Documentation/Tooling Files
# ------------------------------------------------------------
notebooks/    # Jupyter notebooks
references/   # Data dictionaries, manuals, design docs
reset-venv-full.ps1
GEMINI.md
.DS_Store

# We keep:
# - pyproject.toml / uv.lock (for dependency installation)
# - src/ and app/ (for source code)
# - params.yaml (for runtime configuration)
# - docker/ (for the Dockerfile/docker-compose, though they don't get copied into the image, they are needed for context)
```

### Key Exclusions and Rationale

**1. Python Artifacts & Virtual Environments**
- [`__pycache__/`](docker/.dockerignore:4), [`*.pyc`](docker/.dockerignore:5), [`.venv/`](docker/.dockerignore:8): Compiled Python files and virtual environments are host-specific and shouldn't be in the container.

**2. MLOps Artifacts (Largest Impact)**
- [`data/`](docker/.dockerignore:18): Raw and processed datasets can be gigabytes in size
- [`models/`](docker/.dockerignore:19): Trained model files and evaluation artifacts
- [`mlruns/`](docker/.dockerignore:20), [`mlflow.db`](docker/.dockerignore:22): MLflow tracking data and metadata
- [`logs/`](docker/.dockerignore:23): Application logs that should be written to stdout/stderr in containers

**3. Version Control & Development Files**
- [`.git/`](docker/.dockerignore:30), [`.dvc/`](docker/.dockerignore:27): Version control history not needed in production
- [`notebooks/`](docker/.dockerignore:36), [`references/`](docker/.dockerignore:37): Development and documentation files

**4. Sensitive Files**
- [`.env`](docker/.dockerignore:9): Environment variables should be passed at runtime, not baked into images

### Files Explicitly Kept
- [`pyproject.toml`](docker/.dockerignore:43), [`uv.lock`](docker/.dockerignore:43): Dependency definitions
- [`src/`](docker/.dockerignore:44), [`app/`](docker/.dockerignore:44): Application source code
- [`params.yaml`](docker/.dockerignore:44): Runtime configuration parameters

By excluding these files, we ensure a minimal build context (typically <50MB vs potentially >1GB), which leads to faster builds, smaller images, and improved security.

## 4. Docker Compose Environments

The project includes two Docker Compose configurations for different use cases: a **full development environment** with integrated MLflow, and a **lightweight development environment** that connects to a host-running MLflow.

### 4.1 `docker-compose.yml` - Full Development Environment

The [`docker-compose.yml`](docker/docker-compose.yml) file orchestrates a complete multi-container development environment with both the FastAPI application and MLflow tracking server.

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
      uv run uvicorn app.main:app
      --host 0.0.0.0 --port 8000 --reload
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow_server
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri sqlite:////app/mlflow.db
      --default-artifact-root /app/mlruns
      --host 0.0.0.0
      --port 5000
      --allowed-hosts *
    volumes:
      - ./mlruns:/app/mlruns
      - ./mlflow.db:/app/mlflow.db

volumes:
  mlflow_data:
    driver: local
```

#### `api` Service Configuration

-   **[`build`](docker/docker-compose.yml:3)**: Builds from the project root context using the [`Dockerfile`](docker/Dockerfile)
-   **[`container_name`](docker/docker-compose.yml:6)**: Named `youtube_sentiment_api` for easy reference
-   **[`ports`](docker/docker-compose.yml:7)**: Maps host port [`8000`](docker/docker-compose.yml:8) to container port [`8000`](docker/docker-compose.yml:8)
-   **[`env_file`](docker/docker-compose.yml:10)**: Loads environment variables from [`.env`](docker/docker-compose.yml:11) in project root
-   **[`environment`](docker/docker-compose.yml:12)**: Overrides [`MLFLOW_TRACKING_URI`](docker/docker-compose.yml:15) to use internal Docker network
-   **[`volumes`](docker/docker-compose.yml:16)**: **Hot-reloading mounts** for rapid development
    - [`../app:/app/app`](docker/docker-compose.yml:18): Source code changes reflect immediately
    - [`../src:/app/src`](docker/docker-compose.yml:19): Library code changes reflect immediately
    - [`../pyproject.toml:/app/pyproject.toml`](docker/docker-compose.yml:21): Dependency updates trigger reload
    - [`../uv.lock:/app/uv.lock`](docker/docker-compose.yml:22): Lock file updates trigger reload
-   **[`command`](docker/docker-compose.yml:25)**: Runs with [`--reload`](docker/docker-compose.yml:27) flag for automatic restart on code changes
-   **[`extra_hosts`](docker/docker-compose.yml:28)**: Enables [`host.docker.internal`](docker/docker-compose.yml:29) for host network access
-   **[`depends_on`](docker/docker-compose.yml:30)**: Ensures MLflow starts before API

#### `mlflow` Service Configuration

-   **[`image`](docker/docker-compose.yml:34)**: Official MLflow image from GitHub Container Registry
-   **[`container_name`](docker/docker-compose.yml:35)**: Named `mlflow_server`
-   **[`ports`](docker/docker-compose.yml:36)**: Maps host port [`5000`](docker/docker-compose.yml:37) to container port [`5000`](docker/docker-compose.yml:37)
-   **[`command`](docker/docker-compose.yml:38)**: Configures SQLite backend and artifact storage
-   **[`volumes`](docker/docker-compose.yml:46)**: Persists MLflow data to host directories
    - [`./mlruns:/app/mlruns`](docker/docker-compose.yml:47): Model artifacts and experiment data
    - [`./mlflow.db:/app/mlflow.db`](docker/docker-compose.yml:48): SQLite database for metadata

### 4.2 `docker-compose.dev.yml` - Lightweight Development Environment

The [`docker-compose.dev.yml`](docker/docker-compose.dev.yml) provides a minimal setup for developers who already have MLflow running on their host machine.

```yaml
services:
  api-dev:
    build:
      context: .. # Moves context up to project root
      dockerfile: docker/Dockerfile
    container_name: youtube-sentiment-dev
    ports:
      - "8000:8000"
    volumes:
      # Sync local code to container for hot-reloading
      - ../app:/app/app
      - ../src:/app/src
      # Mount models so you can drop in new .pkl files and test instantly
      - ../models:/app/models
    environment:
      - ENV=development
      - MLFLOW_TRACKING_URI=http://host.docker.internal:5000
    extra_hosts:
      - "host.docker.internal:host-gateway"
    # 'uv run uvicorn' to ensure the uvicorn executable is found within the uv environment.
    command: uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Key Differences from `docker-compose.yml`

| Feature | `docker-compose.yml` | `docker-compose.dev.yml` |
|---------|---------------------|-------------------------|
| **Services** | `api` + `mlflow` | `api-dev` only |
| **MLflow Connection** | Internal Docker network (`http://mlflow:5000`) | Host machine (`http://host.docker.internal:5000`) |
| **Volume Mounts** | Code + dependencies | Code + models directory |
| **Use Case** | Complete isolated environment | Connect to existing host MLflow |
| **Resource Usage** | Higher (2 containers) | Lower (1 container) |

#### When to Use Each Environment

**Use `docker-compose.yml` when:**
- Starting fresh with no MLflow server running
- Need complete isolation from host machine
- Want to test the full production-like setup
- Working on MLflow integration features

**Use `docker-compose.dev.yml` when:**
- Already have MLflow running locally on port 5000
- Want faster startup time (single container)
- Need to access models directory from host
- Developing API features independently

## 5. How to Build and Run Containers

### 5.1 Quick Start Commands

All commands should be run from the **project root directory**.

#### Full Development Environment (with MLflow)

```bash
# Build the Docker images
docker compose -f docker/docker-compose.yml build

# Start all services (API + MLflow) in detached mode
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f

# Stop services
docker compose -f docker/docker-compose.yml down
```

#### Lightweight Development Environment (connect to host MLflow)

```bash
# Build the Docker image
docker compose -f docker/docker-compose.dev.yml build

# Start only the API service
docker compose -f docker/docker-compose.dev.yml up -d

# View logs
docker compose -f docker/docker-compose.dev.yml logs -f

# Stop service
docker compose -f docker/docker-compose.dev.yml down
```

### 5.2 Accessing Services

Once running, access your services at:

- **FastAPI Application**: http://localhost:8000
- **API Documentation (Swagger UI)**: http://localhost:8000/docs
- **API Documentation (ReDoc)**: http://localhost:8000/redoc
- **MLflow UI** (full environment only): http://localhost:5000

### 5.3 Running Commands Inside Containers

Execute commands in running containers:

```bash
# Run tests in the API container
docker compose -f docker/docker-compose.yml exec api uv run pytest -v

# Check Python environment
docker compose -f docker/docker-compose.yml exec api uv run python --version

# Install additional packages (development only)
docker compose -f docker/docker-compose.yml exec api uv add package-name

# Access container shell
docker compose -f docker/docker-compose.yml exec api bash
```

### 5.4 Building for Production

For production deployment without development dependencies:

```bash
# Build production image
docker build -f docker/Dockerfile -t youtube-sentiment-api:prod .

# Run production container
docker run -d \
  --name youtube-sentiment-prod \
  -p 8000:8000 \
  --env-file .env \
  -e MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 \
  youtube-sentiment-api:prod
```
## 6. Environment Variables and Configuration

### Required Environment Variables

The application requires the following environment variables to function properly:

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | URI for MLflow tracking server | `http://mlflow:5000` or `http://host.docker.internal:5000` |
| `ENV` | Environment mode | `development` or `production` |
| `GEMINI_API_KEY` | Google Gemini API key for inference | `your-api-key-here` |

### Environment File Setup

Create a `.env` file in the project root:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000  # For docker-compose.yml
# MLFLOW_TRACKING_URI=http://host.docker.internal:5000  # For docker-compose.dev.yml

# Application Environment
ENV=development

# API Keys (required for inference)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Custom model names
# MODEL_NAME=your_registered_model_name
# MODEL_VERSION=1
```

### Configuration Priority

1. **Docker Compose `environment` section**: Highest priority, overrides everything
2. **`.env` file**: Loaded by `env_file` directive in Docker Compose
3. **Dockerfile `ENV` instructions**: Default values in the image
4. **Application defaults**: Fallback values in the code

## 7. Volume Mounts and Data Persistence

### Development Volume Mounts

#### In `docker-compose.yml`:

```yaml
volumes:
  # Code directories for hot-reloading
  - ../app:/app/app           # FastAPI application code
  - ../src:/app/src           # Shared library code
  - ../pyproject.toml:/app/pyproject.toml  # Dependencies
  - ../uv.lock:/app/uv.lock   # Dependency lock file
  
  # MLflow data persistence (on host)
  - ./mlruns:/app/mlruns      # Model artifacts
  - ./mlflow.db:/app/mlflow.db # Metadata database
```

**Purpose**: Enables hot-reloading and persists MLflow data across container restarts.

#### In `docker-compose.dev.yml`:

```yaml
volumes:
  # Code directories for hot-reloading
  - ../app:/app/app           # FastAPI application code
  - ../src:/app/src           # Shared library code
  # Models directory access
  - ../models:/app/models     # Local model files for testing
```

**Purpose**: Connects to host MLflow and enables local model testing.

### Production Volume Strategy

For production, **avoid bind mounts** for code. Use:

1. **Named volumes** for persistent data (MLflow artifacts, databases)
2. **No volumes** for application code (code is baked into the image)
3. **External storage** for large models (S3, GCS, etc.)

Example production volume setup:
```yaml
# In production docker-compose.yml
volumes:
  mlflow_artifacts:    # Named volume for MLflow
    driver: local
  postgres_data:       # Named volume for database
    driver: local
```

## 8. Service Dependencies and Networking

### Service Dependency Chain

```
docker-compose.yml:
mlflow (tracking server) ← api (FastAPI application)

docker-compose.dev.yml:
api-dev (FastAPI only, connects to host MLflow)
```

### Internal Docker Networking

#### In `docker-compose.yml`:

- **Network**: Docker creates a default bridge network
- **Service Discovery**: Containers can reach each other by service name
- **API → MLflow Communication**: `http://mlflow:5000`
- **Host → Containers**: `http://localhost:8000` and `http://localhost:5000`

#### Network Flow:

1. **API Container** sends requests to `http://mlflow:5000`
2. **Docker DNS** resolves `mlflow` to the MLflow container's IP
3. **MLflow Container** receives requests on port 5000
4. **Response** travels back through the internal network

### Host Network Access

#### In `docker-compose.dev.yml`:

- **Purpose**: Connect to MLflow running on host machine
- **Configuration**: `extra_hosts: - "host.docker.internal:host-gateway"`
- **Connection**: `http://host.docker.internal:5000`
- **Use Case**: Developer has MLflow running locally outside Docker

#### Host → Container Communication:

```bash
# From host machine
curl http://localhost:8000/health          # → API container
curl http://localhost:5000                 # → MLflow container (full env)

# From API container to host
curl http://host.docker.internal:5000      # → Host MLflow (dev env)
```

### Port Mapping Reference

| Service | Container Port | Host Port | Access URL |
|---------|---------------|-----------|------------|
| FastAPI API | 8000 | 8000 | http://localhost:8000 |
| MLflow UI | 5000 | 5000 | http://localhost:5000 |
| MLflow (internal) | 5000 | - | http://mlflow:5000 (container network) |

## 9. Troubleshooting and Best Practices

### Common Issues and Solutions

#### 1. **Build Context Too Large**
**Symptom**: Docker build is very slow or runs out of memory
**Solution**: Verify [`.dockerignore`](docker/.dockerignore) is excluding large directories:
```bash
# Check what's being sent to Docker
docker build -f docker/Dockerfile . --progress=plain
```

#### 2. **MLflow Connection Failed**
**Symptom**: API cannot connect to MLflow
**Solutions by environment**:

**For `docker-compose.yml`:**
```bash
# Check if MLflow container is running
docker compose -f docker/docker-compose.yml ps

# Check MLflow logs
docker compose -f docker/docker-compose.yml logs mlflow

# Verify network connectivity from API container
docker compose -f docker/docker-compose.yml exec api curl http://mlflow:5000
```

**For `docker-compose.dev.yml`:**
```bash
# Verify MLflow is running on host
curl http://localhost:5000

# Check host.docker.internal resolution
docker compose -f docker/docker-compose.yml exec api getent hosts host.docker.internal
```

#### 3. **Hot Reloading Not Working**
**Symptom**: Code changes don't trigger automatic restart
**Solution**:
- Verify volume mounts in Docker Compose file
- Check file permissions: `ls -la app/ src/`
- Ensure `--reload` flag is in the command
- Check uvicorn logs: `docker compose logs api`

#### 4. **Port Already in Use**
**Symptom**: `bind: address already in use`
**Solution**:
```bash
# Find what's using the port
lsof -i :8000  # or :5000

# Stop conflicting service
# OR use different ports in docker-compose.yml
```

#### 5. **Model Loading Failures**
**Symptom**: API starts but cannot load models from MLflow
**Solution**:
```bash
# Verify model exists in MLflow
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=your_model

# Check API logs for specific errors
docker compose -f docker/docker-compose.yml logs api

# Verify GEMINI_API_KEY is set
docker compose -f docker/docker-compose.yml exec api env | grep GEMINI
```

### Best Practices

#### Development Workflow
1. **Use `docker-compose.dev.yml`** for daily development (faster startup)
2. **Use `docker-compose.yml`** for integration testing and MLflow features
3. **Always use `--build`** when dependencies change: `docker compose up --build`
4. **Clean up regularly**: `docker system prune` to free disk space

#### Production Deployment
1. **Never use `--reload`** in production
2. **Don't mount source code** volumes in production
3. **Use specific image tags**, not `latest`
4. **Set resource limits** in Docker Compose:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

#### Performance Optimization
1. **Leverage build cache**: Order Dockerfile instructions correctly
2. **Use multi-stage builds** for smaller production images
3. **Minimize layer count**: Combine related RUN commands
4. **Use `.dockerignore`** aggressively to reduce build context

#### Debugging Commands
```bash
# View all containers
docker ps -a

# Inspect container details
docker inspect container_name

# Execute shell in container
docker exec -it container_name bash

# View real-time logs
docker logs -f container_name

# Check network connectivity
docker network ls
docker network inspect network_name
```

### Health Checks

The API includes a health endpoint for monitoring:

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
{"status":"ok","message":"YouTube Sentiment Analysis API is running."}
```

For production deployments, add Docker health checks:

```yaml
# In docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```


