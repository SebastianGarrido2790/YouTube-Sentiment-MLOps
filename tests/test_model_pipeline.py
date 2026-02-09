import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.models.baseline_logistic import train_baseline
from src.config.schemas import LogisticBaselineConfig


@pytest.fixture
def mock_data():
    """Create dummy training data."""
    X_train = np.random.rand(10, 5)
    X_val = np.random.rand(5, 5)
    X_test = np.random.rand(5, 5)
    y_train = np.random.randint(0, 2, 10)
    y_val = np.random.randint(0, 2, 5)
    y_test = np.random.randint(0, 2, 5)

    mock_le = MagicMock()
    mock_le.inverse_transform.side_effect = lambda x: x  # Identity for simplification

    return X_train, X_val, X_test, y_train, y_val, y_test, mock_le


@pytest.fixture
def mock_config():
    """Create a dummy logistic baseline configuration."""
    return LogisticBaselineConfig(
        model_type="LogisticRegression",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=100,
        C=1.0,
    )


@patch("src.models.baseline_logistic.load_feature_data")
@patch("src.models.baseline_logistic.mlflow")
@patch("src.models.baseline_logistic.save_baseline_metrics_json")
@patch("src.models.baseline_logistic.save_model_bundle")
def test_train_baseline(
    mock_save_bundle,
    mock_save_metrics,
    mock_mlflow,
    mock_load_data,
    mock_data,
    mock_config,
):
    """Test the train_baseline function end-to-end with mocks."""

    # Setup mocks
    mock_load_data.return_value = mock_data

    # Mock active run info
    mock_mlflow.active_run.return_value.info.run_id = "test_run_id"

    # Run training
    train_baseline(config=mock_config)

    # Assertions
    # 1. Data loaded?
    mock_load_data.assert_called_once()

    # 2. MLflow run started?
    mock_mlflow.start_run.assert_called()

    # 3. Model trained and logged?
    # We can check if sklearn.log_model was called
    mock_mlflow.sklearn.log_model.assert_called()

    # 4. Metrics logged?
    # We check if log_metrics_to_mlflow (which calls mlflow.log_metric) was effectively used.
    # Since we didn't patch log_metrics_to_mlflow explicitly but it uses mlflow,
    # we can check mlflow.log_metric calls.
    assert mock_mlflow.log_metric.called

    # 5. Files saved locally?
    mock_save_metrics.assert_called_once()
    mock_save_bundle.assert_called_once()
