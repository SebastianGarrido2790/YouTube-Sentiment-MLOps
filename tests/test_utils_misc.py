"""
Unit Tests for Miscellaneous Utilities

Tests the training utilities, logger, and MLflow helpers.
"""

import json
from unittest.mock import patch

from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.train_utils import save_baseline_metrics_json, save_model_bundle, save_test_metrics_json


def test_save_baseline_metrics_json(tmp_path, monkeypatch):
    """Tests saving baseline metrics to JSON."""
    monkeypatch.setattr("src.utils.train_utils.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.utils.train_utils.BASELINE_MODEL_DIR", tmp_path)

    save_baseline_metrics_json(0.85)
    metrics_file = tmp_path / "baseline_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        data = json.load(f)
        assert data["val_macro_f1"] == 0.85


def test_save_test_metrics_json(tmp_path, monkeypatch):
    """Tests saving final test metrics to JSON."""
    monkeypatch.setattr("src.utils.train_utils.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.utils.train_utils.EVAL_DIR", tmp_path)

    report = {"macro avg": {"f1-score": 0.8}, "weighted avg": {"f1-score": 0.9}}
    save_test_metrics_json("dummy_model", report)
    metrics_file = tmp_path / "dummy_model_test_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        data = json.load(f)
        assert data["test_macro_f1"] == 0.8


@patch("src.utils.train_utils.pickle.dump")
def test_save_model_bundle(mock_pickle, tmp_path, monkeypatch):
    """Tests pickling a model bundle."""
    monkeypatch.setattr("src.utils.train_utils.PROJECT_ROOT", tmp_path)
    save_path = tmp_path / "model.pkl"
    save_model_bundle({"model": "test"}, save_path)
    assert mock_pickle.called


def test_get_logger():
    """Tests logger initialization."""
    log = get_logger("test_logger", headline="TEST")
    assert log is not None
    assert log.name == "test_logger"


def test_get_mlflow_uri():
    """Tests MLflow configuration retrieval."""
    uri = get_mlflow_uri()
    assert uri is not None
    assert "mlruns" in uri or "http" in uri
