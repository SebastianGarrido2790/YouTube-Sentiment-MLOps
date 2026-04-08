"""
Unit Tests for Model Evaluation Component

Tests the model loading, prediction, and ROC plotting logic.
Mocks MLflow and Matplotlib to ensure no side effects.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig


@pytest.fixture
def eval_config():
    return ModelEvaluationConfig(models=["logistic_baseline", "lightgbm"])


@pytest.fixture
def model_eval(eval_config):
    return ModelEvaluation(eval_config)


def test_load_model_artifact_baseline(model_eval, tmp_path, monkeypatch):
    """Tests loading the baseline model bundle."""
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    monkeypatch.setattr("src.components.model_evaluation.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.components.model_evaluation.BASELINE_MODEL_DIR", baseline_dir)

    import pickle

    model_file = baseline_dir / "logistic_baseline.pkl"
    mock_model = "dummy_model"
    with open(model_file, "wb") as f:
        pickle.dump({"model": mock_model}, f)

    loaded = model_eval.load_model_artifact("logistic_baseline")
    assert loaded is not None


def test_load_model_artifact_advanced(model_eval, tmp_path, monkeypatch):
    """Tests loading advanced model artifacts."""
    advanced_dir = tmp_path / "advanced"
    advanced_dir.mkdir()
    monkeypatch.setattr("src.components.model_evaluation.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.components.model_evaluation.ADVANCED_DIR", advanced_dir)

    import pickle

    model_file = advanced_dir / "lightgbm_model.pkl"
    mock_model = "dummy_model"
    with open(model_file, "wb") as f:
        pickle.dump(mock_model, f)

    loaded = model_eval.load_model_artifact("lightgbm")
    assert loaded is not None


def test_evaluate_model_lgbm(model_eval):
    """Tests evaluation of an LGBM model."""
    mock_model = MagicMock()
    # Simulate LGBMClassifier by including it in the type string
    mock_model.__class__.__name__ = "LGBMClassifier"

    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    mock_model.predict.return_value = y
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

    report, cm, y_pred_proba = model_eval.evaluate_model(mock_model, X, y)

    assert "accuracy" in report
    assert cm.shape == (2, 2)
    assert y_pred_proba.shape == (2, 2)


@patch("src.components.model_evaluation.plt")
@patch("src.components.model_evaluation.mlflow")
def test_plot_comparative_roc_curve(mock_mlflow, mock_plt, model_eval, tmp_path, monkeypatch):
    """Tests the ROC curve plotting logic."""
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    monkeypatch.setattr("src.components.model_evaluation.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.components.model_evaluation.EVAL_FIG_DIR", fig_dir)

    y_test_bin = np.array([[1, 0], [0, 1]])
    roc_results = [{"name": "model1", "proba": np.array([[0.8, 0.2], [0.1, 0.9]])}]
    labels = ["0", "1"]

    model_eval.plot_comparative_roc_curve(y_test_bin, roc_results, labels)

    assert mock_plt.savefig.called
    assert mock_mlflow.log_artifact.called


@patch("src.components.model_evaluation.load_feature_data")
@patch("src.components.model_evaluation.load_text_data")
@patch("src.components.model_evaluation.mlflow")
@patch("src.components.model_evaluation.log_metrics_to_mlflow")
@patch("src.components.model_evaluation.log_confusion_matrix_as_artifact")
@patch("src.components.model_evaluation.save_test_metrics_json")
@patch("src.components.model_evaluation.save_best_model_run_info")
def test_run_evaluation(
    mock_save_best,
    mock_save_metrics,
    mock_log_cm,
    mock_log_metrics,
    mock_mlflow,
    mock_load_text,
    mock_load_feat,
    model_eval,
):
    """Tests the main orchestration method of ModelEvaluation."""
    # Mock data loading
    mock_le = MagicMock()
    mock_le.classes_ = np.array([-1, 0, 1])
    mock_load_feat.return_value = (None, None, np.array([[1], [2], [3]]), None, None, np.array([-1, 0, 1]), mock_le)
    mock_load_text.return_value = (None, None, MagicMock())

    # Mock model loading within the class
    model_eval.load_model_artifact = MagicMock(return_value=MagicMock())
    model_eval.evaluate_model = MagicMock(
        return_value=(
            {"macro avg": {"f1-score": 0.8}},
            np.zeros((3, 3)),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        )
    )
    model_eval.plot_comparative_roc_curve = MagicMock()

    model_eval.run_evaluation()

    assert mock_mlflow.start_run.called
    assert mock_save_metrics.called
    assert mock_save_best.called
