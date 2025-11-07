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
