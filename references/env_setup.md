### Project Environment Setup Guide

To establish a reproducible development environment using uv (a fast, Rust-based Python package manager), follow these steps. This setup assumes Python 3.10+ is installed globally. uv will manage virtual environments, dependencies, and locks for consistency across local, CI/CD, and AWS deployments.

#### Step 1: Install uv
Run the following in your terminal (macOS/Linux/Windows with pip):
```bash
pip install uv
```
Verify installation:
```bash
uv --version
```
Expected output: `uv x.y.z` (e.g., 0.4+).

#### Step 2: Initialize the Project
Navigate to your project root directory. If you haven't already initialized the project:

```bash
uv init --python 3.12
```

This generates:
- `pyproject.toml`: For dependencies and project metadata.
- `.venv`: Virtual environment (auto-activated in supported shells).

Activate the environment (if not auto-activated):
- macOS/Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

#### Step 3: Configure Project Metadata & Dependencies
Ensure your `pyproject.toml` includes the comprehensive list of dependencies used in this project:

```toml
[project]
name = "youtube-sentiment-mlops"
version = "0.1.0"
description = "End-to-end MLOps pipeline for real-time YouTube sentiment analysis"
authors = [{name = "Your Name", email = "your.email@example.com"}]
dependencies = [
    "pandas>=2.0",
    "python-dotenv>=1.0",
    "scikit-learn>=1.3",
    "nltk>=3.8",
    "mlflow>=2.8",
    "dvc>=3.0",
    "docker>=7.0",  # For local containerization
    "boto3>=1.28",  # AWS integration
    "requests>=2.31",  # API calls (e.g., YouTube)
    "jupyter==1.0.0",
    "seaborn>=0.13.2",
    "wordcloud>=1.9.4",
    "torch>=2.0",  # For efficient inference
    "imbalanced-learn>=0.12.0",
    "optuna>=3.6",
    "xgboost>=2.0",
    "lightgbm>=4.3",
    "transformers>=4.40",
    "datasets>=2.20",  # For BERT data loading
    "accelerate>=0.30", # For distributed training
    "ipykernel>=6.30.1",
    "sentencepiece>=0.2.0",
    "pytest>=7.0",
    "pydantic>=2.0",
]
requires-python = ">=3.10, <3.13"
```

Install dependencies and generate the lockfile:
```bash
uv sync
```
Verify installation: `uv pip list` should show installed packages.

#### Step 4: Initialize Git and DVC
For version control:
```bash
git init
git add .
git commit -m "Initial project structure"
```
For data versioning (DVC):
```bash
uv run dvc init
# Add raw data tracking (after running download_dataset.py)
uv run dvc add data/raw/reddit_comments.csv
git add data/raw/reddit_comments.csv.dvc .gitignore
git commit -m "Initialize DVC and track raw data"
```

#### Step 5: Test the Setup
1. **Run the Pipeline**: Validate the end-to-end pipeline using DVC.
   ```bash
   uv run dvc repro
   ```
   This will execute all stages defined in `dvc.yaml`, from data ingestion to model registration.

2. **Run Tests**: Ensure the codebase is stable.
   ```bash
   uv run pytest
   ```

#### Best Practices for Maintenance
- **Execution**: Always use `uv run <command>` for scripts to ensure environment isolation.
- **Dependency Management**: Add new packages with `uv add <package>` or update with `uv sync --upgrade`.
- **Secrets**: Add secrets (e.g., `YOUTUBE_API_KEY`) to `.env` and load via `python-dotenv`. **Never commit `.env`**.
- **Containerization**: The environment is designed to be easily containerized using `uv`'s lockfile for reproducible builds.

This setup ensures reliability (locked deps), scalability (uv's speed), maintainability (TOML config), and adaptability (easy dep swaps).