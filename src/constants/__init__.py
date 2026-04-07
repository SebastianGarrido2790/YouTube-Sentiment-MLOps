"""
Project-wide path constants.

Single source of truth for all configuration file paths, data directories,
and artifact locations.
"""

from pathlib import Path

import yaml

# Provide standard hardcoded constants for the config system itself
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"
PARAMS_FILE_PATH = CONFIG_DIR / "params.yaml"
SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"


# Parse the system configuration dict directly to prevent circular imports
with open(CONFIG_FILE_PATH, encoding="utf-8") as _f:
    _sys = yaml.safe_load(_f)

# Dynamic bindings
REPORTS_DIR = PROJECT_ROOT / _sys["reports"]["root_dir"]
FIGURES_DIR = PROJECT_ROOT / _sys["reports"]["figures_dir"]
DOCS_DIR = PROJECT_ROOT / _sys["reports"]["docs_dir"]
MLRUNS_DIR = PROJECT_ROOT / _sys["ops"]["mlruns_dir"]

RAW_DATA_DIR = PROJECT_ROOT / _sys["data"]["raw_dir"]
EXTERNAL_DATA_DIR = PROJECT_ROOT / _sys["data"]["external_dir"]
ARTIFACTS_DIR = PROJECT_ROOT / _sys["artifacts_root"]
PROCESSED_DATA_DIR = PROJECT_ROOT / _sys["data"]["processed_dir"]

TRAIN_PATH = PROJECT_ROOT / _sys["data"]["train_path"]
TEST_PATH = PROJECT_ROOT / _sys["data"]["test_path"]
VAL_PATH = PROJECT_ROOT / _sys["data"]["val_path"]

MODELS_DIR = PROJECT_ROOT / _sys["models"]["root_dir"]
GX_DIR = PROJECT_ROOT / _sys["ops"]["gx_dir"]

RAW_PATH = PROJECT_ROOT / _sys["data"]["raw_path"]
BASELINE_MODEL_DIR = PROJECT_ROOT / _sys["models"]["baseline_dir"]
ADVANCED_DIR = PROJECT_ROOT / _sys["models"]["advanced_dir"]
FEATURES_DIR = PROJECT_ROOT / _sys["models"]["features_dir"]
EVAL_DIR = PROJECT_ROOT / _sys["models"]["evaluation_dir"]

EVAL_FIG_DIR = PROJECT_ROOT / _sys["reports"]["eval_fig_dir"]
TFIDF_FIGURES_DIR = PROJECT_ROOT / _sys["reports"]["tfidf_fig_dir"]
IMBALANCE_FIGURES_DIR = PROJECT_ROOT / _sys["reports"]["imbalance_fig_dir"]
