# ============================================================
# Production-grade FastAPI container for YouTube Sentiment API
# ============================================================
FROM python:3.11-slim

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=1000
RUN useradd -m -u "${UID}" appuser

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS-level dependencies (minimal)
# curl is added for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install uv and project dependencies separately for caching
# ------------------------------------------------------------
RUN pip install --no-cache-dir uv

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (will only rebuild if deps change)
# We use the system python environment or create a venv.
# uv sync defaults to creating a .venv in the current directory (/app/.venv).
# --mount=type=cache persists the uv cache across builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ------------------------------------------------------------
# Copy project code
# ------------------------------------------------------------
COPY . .

# Change ownership of the application directory to the non-privileged user
RUN chown -R appuser:appuser /app

# Switch to non-privileged user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Healthcheck configuration (Phase 6 Requirement)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variable for MLflow tracking (optional)
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
    ENV=production

# Default command (production mode)
# Using `uv run` which will detect the .venv in /app/.venv
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]