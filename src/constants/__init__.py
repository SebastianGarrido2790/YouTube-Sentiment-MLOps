"""
Project-wide path constants.

Single source of truth for all configuration file paths, data directories,
and artifact locations. Every pipeline module should import paths from here
instead of hardcoding strings.
"""

from pathlib import Path

# --- Project Root ---
# Automatically finds the top-level directory (the one containing 'src/')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Configuration & Schema ---
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"
PARAMS_FILE_PATH = CONFIG_DIR / "params.yaml"
SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"

# --- Model & Reports ---
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = REPORTS_DIR / "docs"

# --- Logs & MLflow ---
LOGS_DIR = PROJECT_ROOT / "logs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# --- Data Directories ---
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

# --- Artifact Directories ---
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "data" / "processed"

# Specific file splits (DVC Artifacts)
TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"
VAL_PATH = PROCESSED_DATA_DIR / "val.parquet"

MODELS_DIR = ARTIFACTS_DIR / "models"
GX_DIR = ARTIFACTS_DIR / "gx"

# Specific file paths
RAW_PATH = RAW_DATA_DIR / "reddit_comments.csv"

# Advanced and Baseline model directories
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
ADVANCED_DIR = MODELS_DIR / "advanced"
FEATURES_DIR = MODELS_DIR / "features"
EVAL_DIR = ADVANCED_DIR / "evaluation"

# Additional figure directories
EVAL_FIG_DIR = FIGURES_DIR / "evaluation"
TFIDF_FIGURES_DIR = FIGURES_DIR / "tfidf_max_features"
IMBALANCE_FIGURES_DIR = FIGURES_DIR / "imbalance_methods"

# --- Ensure directories exist ---
directories_to_create = [
    REPORTS_DIR,
    FIGURES_DIR,
    DOCS_DIR,
    LOGS_DIR,
    MLRUNS_DIR,
    EXTERNAL_DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    GX_DIR,
    BASELINE_MODEL_DIR,
    ADVANCED_DIR,
    FEATURES_DIR,
    EVAL_DIR,
    EVAL_FIG_DIR,
    TFIDF_FIGURES_DIR,
    IMBALANCE_FIGURES_DIR,
]

for path in directories_to_create:
    path.mkdir(parents=True, exist_ok=True)
