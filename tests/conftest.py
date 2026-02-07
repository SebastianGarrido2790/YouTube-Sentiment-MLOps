import pytest
import yaml
from src.config.manager import ConfigurationManager


@pytest.fixture
def mock_params_yaml(tmp_path):
    """Creates a temporary params.yaml for testing."""
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
            "max_features_values": "[100, 200]",
            "best_ngram_range": "(1,1)",
            "n_estimators": 100,
            "max_depth": 10,
        },
        "imbalance_tuning": {
            "imbalance_methods": "['smote']",
            "best_max_features": 1000,
            "best_ngram_range": "(1,1)",
            "rf_n_estimators": 100,
            "rf_max_depth": 10,
        },
        "feature_engineering": {
            "use_distilbert": "False",
            "distilbert_batch_size": 16,
            "best_max_features": 1000,
            "best_ngram_range": "(1,1)",
        },
        "train": {
            "logistic_baseline": {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 100,
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
def config_manager(mock_params_yaml):
    # Reset singleton
    ConfigurationManager._instance = None
    return ConfigurationManager(params_path=mock_params_yaml)
