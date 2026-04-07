"""
Shared Test Configuration and Fixtures

This module contains shared pytest fixtures and setup logic for the
YouTube Sentiment Analysis test suite. It handles project path resolution,
NLTK resource management, and configuration mocking.
"""

import sys
from pathlib import Path

# Add project root to sys.path so we can import src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Ensure NLTK resources are available for tests
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

import pytest
import yaml

from src.config.configuration import ConfigurationManager


@pytest.fixture
def mock_params_yaml(tmp_path: Path) -> str:
    """
    Creates a temporary params.yaml for testing.

    This fixture generates a comprehensive mock parameters file containing
    settings for all pipeline stages (data ingestion, preparation, tuning, etc.)
    to ensure the ConfigurationManager can be tested without production files.

    Args:
        tmp_path: Pytest built-in fixture for temporary directory paths.

    Returns:
        str: Absolute path to the created mock params.yaml file.
    """
    params = {
        "data_ingestion": {
            "url": "http://example.com/data.csv",
            "output_path": "data/raw/test.csv",
        },
        "data_preparation": {"test_size": 0.2, "random_state": 123},
        "feature_comparison": {
            "mlflow_uri": "http://localhost:5000",
            "ngram_ranges": [[1, 1]],
            "max_features": 1000,
            "use_distilbert": False,
            "batch_size": 16,
            "n_estimators": 100,
            "max_depth": 10,
        },
        "feature_tuning": {
            "max_features_values": [100, 200],
            "best_ngram_range": [1, 1],
            "n_estimators": 100,
            "max_depth": 10,
        },
        "imbalance_tuning": {
            "imbalance_methods": ["smote"],
            "best_max_features": 1000,
            "best_ngram_range": [1, 1],
            "rf_n_estimators": 100,
            "rf_max_depth": 10,
        },
        "feature_engineering": {
            "use_distilbert": "False",
            "distilbert_batch_size": 16,
            "best_max_features": 1000,
            "best_ngram_range": "[1, 1]",  # Kept as string/optional based on schema but let's check
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
    }

    p = tmp_path / "params.yaml"
    with open(p, "w") as f:
        yaml.dump(params, f)

    return str(p)


@pytest.fixture
def config_manager(mock_params_yaml: str) -> ConfigurationManager:
    """
    Initializes a ConfigurationManager with a mock params file.

    This fixture resets the ConfigurationManager singleton instance to ensure
    test isolation and hydrates it with the temporary mock parameters.

    Args:
        mock_params_yaml: Path to the mock parameters file from the `mock_params_yaml` fixture.

    Returns:
        ConfigurationManager: A ready-to-use configuration manager instance.
    """
    # Reset singleton
    ConfigurationManager._instance = None
    return ConfigurationManager(params_path=mock_params_yaml)
