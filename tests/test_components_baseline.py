"""
Unit Tests for Baseline Model Training

Tests the training pipeline for Logistic Regression.
Mocks MLflow, data loading, and model persistence to isolate training logic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.components.baseline_model import BaselineModel
from src.entity.config_entity import LogisticBaselineConfig


@pytest.fixture
def mock_data():
    """Provides dummy feature matrices and labels."""
    X_train = np.random.rand(10, 5)
    X_val = np.random.rand(5, 5)
    X_test = np.random.rand(5, 5)
    y_train = np.array([0, 1] * 5)
    y_val = np.array([0, 1, 0, 1, 0])
    y_test = np.array([1, 0, 1, 0, 1])

    le = MagicMock()
    le.inverse_transform.side_effect = lambda x: x  # Identity for tests

    return X_train, X_val, X_test, y_train, y_val, y_test, le


@patch("src.components.baseline_model.load_feature_data")
@patch("src.components.baseline_model.mlflow")
@patch("src.components.baseline_model.log_metrics_to_mlflow")
@patch("src.components.baseline_model.save_baseline_metrics_json")
@patch("src.components.baseline_model.save_model_bundle")
def test_train_baseline_success(
    mock_save_bundle, mock_save_metrics, mock_log_metrics, mock_mlflow, mock_load_data, mock_data
):
    """Tests a successful baseline training run."""
    mock_load_data.return_value = mock_data
    mock_mlflow.active_run.return_value.info.run_id = "test_run"

    config = LogisticBaselineConfig(
        model_type="LogisticRegression", class_weight="balanced", solver="lbfgs", max_iter=100, C=1.0
    )

    baseline = BaselineModel(config)
    baseline.train_baseline()

    assert mock_load_data.called
    assert mock_mlflow.start_run.called
    # Check that sklearn.log_model was called via mock_mlflow
    assert mock_mlflow.sklearn.log_model.called
    assert mock_save_bundle.called
    assert mock_save_metrics.called
