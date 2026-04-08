"""
Model Pipeline Orchestration Suite

Automated integration tests for the model training and evaluation pipelines.
This suite uses comprehensive mocking to verify that the pipeline conductor
correctly orchestrates data loading, experiment tracking with MLflow,
and artifact persistence.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.components.baseline_model import BaselineModel
from src.entity.config_entity import LogisticBaselineConfig


@pytest.fixture
def mock_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MagicMock]:
    """
    Creates dummy training, validation, and test data for pipeline testing.

    Returns:
        Tuple: A tuple containing (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder).
    """
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
def mock_config() -> LogisticBaselineConfig:
    """
    Creates a dummy logistic baseline configuration.

    Returns:
        LogisticBaselineConfig: A mock configuration with balanced weights.
    """
    return LogisticBaselineConfig(
        model_type="LogisticRegression",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=100,
        C=1.0,
    )


@patch("src.components.baseline_model.load_feature_data")
@patch("src.components.baseline_model.mlflow")
@patch("src.components.baseline_model.log_metrics_to_mlflow")
@patch("src.components.baseline_model.save_baseline_metrics_json")
@patch("src.components.baseline_model.save_model_bundle")
def test_train_baseline(
    mock_save_bundle: MagicMock,
    mock_save_metrics: MagicMock,
    mock_log_metrics: MagicMock,
    mock_mlflow: MagicMock,
    mock_load_data: MagicMock,
    mock_data: tuple,
    mock_config: LogisticBaselineConfig,
):
    """
    Tests the `train_baseline` function end-to-end with mocks.

    This test verifies that the pipeline conductor correctly coordinates:
    1. Loading feature data from the artifact store.
    2. Starting and managing an MLflow experiment run.
    3. Logging model artifacts and performance metrics.
    4. Persisting final metrics and model bundles locally.

    Args:
        mock_save_bundle: Mock for model bundle persistence.
        mock_save_metrics: Mock for JSON metrics persistence.
        mock_mlflow: Mock for MLflow tracking API.
        mock_load_data: Mock for data loading utility.
        mock_data: Data fixture providing dummy arrays.
        mock_config: Configuration fixture.
    """

    # Setup mocks
    mock_load_data.return_value = mock_data

    # Mock active run info
    mock_mlflow.active_run.return_value.info.run_id = "test_run_id"

    # Run training
    baseline = BaselineModel(config=mock_config)
    baseline.train_baseline()

    # Assertions
    # 1. Data loaded?
    mock_load_data.assert_called_once()

    # 2. MLflow run started?
    mock_mlflow.start_run.assert_called()

    # 3. Model trained and logged?
    mock_mlflow.sklearn.log_model.assert_called()

    # 4. Metrics logged?
    assert mock_mlflow.log_metric.called

    # 5. Files saved locally?
    mock_save_metrics.assert_called_once()
    mock_save_bundle.assert_called_once()
