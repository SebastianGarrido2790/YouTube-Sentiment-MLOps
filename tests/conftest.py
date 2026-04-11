"""
Test Configuration and Fixtures

This module provides shared fixtures for the YouTube Sentiment Analysis test suite,
including mock configuration files and common components.
"""

from pathlib import Path

import pytest
import yaml

from src.config.configuration import ConfigurationManager


@pytest.fixture
def mock_params_yaml(tmp_path: Path) -> str:
    """
    Creates a temporary mock params.yaml file for testing.
    Includes all necessary sections to satisfy the AppConfig strict schema.
    """
    params = {
        "data_ingestion": {
            "url": "http://example.com/data.csv",
            "output_path": "data/raw/data.csv",
        },
        "data_validation": {
            "null_threshold_percent": 0.05,
            "min_text_length": 5,
            "max_text_length": 500,
        },
        "data_preparation": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "feature_comparison": {
            "mlflow_uri": "http://localhost:5000",
            "ngram_ranges": [[1, 1]],
            "max_features": 1000,
            "use_distilbert": False,
            "batch_size": 32,
            "n_estimators": 10,
            "max_depth": 5,
        },
        "feature_tuning": {
            "max_features_values": [100, 200],
            "best_ngram_range": [1, 1],
            "n_estimators": 10,
            "max_depth": 5,
        },
        "imbalance_tuning": {
            "imbalance_methods": ["smote"],
            "best_max_features": 100,
            "best_ngram_range": [1, 1],
            "rf_n_estimators": 10,
            "rf_max_depth": 5,
        },
        "feature_engineering": {
            "use_distilbert": False,
            "distilbert_batch_size": 32,
        },
        "train": {
            "logistic_baseline": {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 100,
                "C": 1.0,
            },
            "hyperparameter_tuning": {
                "lightgbm": {"n_trials": 1},
                "xgboost": {"n_trials": 1},
            },
            "distilbert": {
                "enable": False,
                "n_trials": 1,
                "batch_size": [8],
                "lr": [1e-5],
                "weight_decay": [0.01],
            },
        },
        "model_evaluation": {"models": ["logistic_baseline"]},
        "register": {"f1_threshold": 0.5},
        "agent": {
            "model_name": "google-gla:gemini-2.0-flash-lite",
            "max_comments": 100,
            "fallback_enabled": True,
            "fallback_model_name": "groq:llama-3.1-8b-instant",
        },
    }

    p = tmp_path / "params.yaml"
    with open(p, "w") as f:
        yaml.dump(params, f)

    return str(p)


@pytest.fixture
def mock_config_yaml(tmp_path: Path) -> str:
    """Creates a temporary mock config.yaml for system paths."""
    config = {
        "artifacts_root": "artifacts",
        "data": {
            "raw_dir": "data/raw",
            "external_dir": "data/external",
            "processed_dir": "data/processed",
            "raw_path": "data/raw/dataset.csv",
            "train_path": "data/processed/train.csv",
            "test_path": "data/processed/test.csv",
            "val_path": "data/processed/val.csv",
        },
        "models": {
            "root_dir": "models",
            "baseline_dir": "models/baseline",
            "advanced_dir": "models/advanced",
            "features_dir": "models/features",
            "evaluation_dir": "models/evaluation",
        },
        "reports": {
            "root_dir": "reports",
            "figures_dir": "reports/figures",
            "docs_dir": "reports/docs",
            "eval_fig_dir": "reports/figures/eval",
            "tfidf_fig_dir": "reports/figures/tfidf",
            "imbalance_fig_dir": "reports/figures/imbalance",
        },
        "ops": {
            "logs_dir": "logs",
            "mlruns_dir": "mlruns",
            "gx_dir": "gx",
        },
        "agent": {
            "inference_api_url": "http://localhost:8000",
            "insights_api_url": "http://localhost:8080",
            "tool_timeout_seconds": 10,
        },
    }

    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)

    return str(p)


@pytest.fixture
def mock_schema_yaml(tmp_path: Path) -> str:
    """Creates a temporary mock schema.yaml for data contracts."""
    schema = {
        "columns": {"text": "string", "label": "integer"},
        "target": "label",
    }
    p = tmp_path / "schema.yaml"
    with open(p, "w") as f:
        yaml.dump(schema, f)
    return str(p)


@pytest.fixture
def config_manager(mock_params_yaml: str, mock_config_yaml: str, mock_schema_yaml: str) -> ConfigurationManager:
    """
    Initializes a ConfigurationManager with mock files.
    Ensures singletons are reset for test isolation.
    """
    # Reset singleton effectively by clearing its internal state or creating new
    ConfigurationManager._instance = None
    return ConfigurationManager(
        config_path=mock_config_yaml,
        params_path=mock_params_yaml,
        schema_path=mock_schema_yaml,
    )
