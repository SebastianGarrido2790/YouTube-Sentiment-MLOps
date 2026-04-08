"""
Pipeline Integration Test Suite

This suite verifies the orchestration of individual pipeline stages.
It uses comprehensive mocking to isolate the pipeline flow from the heavy
lifting of components (training, cleaning, etc.) while ensuring that
all stages are invoked correctly and their metadata is properly managed.
"""

from unittest.mock import patch

from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_01b_data_validation import DataValidationPipeline
from src.pipeline.stage_02_data_preparation import DataPreparationPipeline
from src.pipeline.stage_03_feature_engineering import FeatureEngineeringPipeline
from src.pipeline.stage_04a_baseline_model import BaselineModelPipeline
from src.pipeline.stage_04b_hyperparameter_tuning import HyperparameterTuningPipeline
from src.pipeline.stage_04c_distilbert_training import DistilBERTTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_06_register_model import ModelRegistrationPipeline


# ============================================================
# Stage 1: Data Ingestion
# ============================================================
@patch("src.pipeline.stage_01_data_ingestion.DataIngestion")
def test_stage_01_ingestion(mock_ingestion):
    """Verifies that the Ingestion stage initializes and executes correctly."""
    pipeline = DataIngestionPipeline()
    pipeline.main()
    mock_ingestion.return_value.download_file.assert_called_once()


# ============================================================
# Stage 1b: Data Validation
# ============================================================
@patch("src.pipeline.stage_01b_data_validation.DataValidation")
def test_stage_01b_validation(mock_validation):
    """Verifies that the Validation stage initializes and executes correctly."""
    pipeline = DataValidationPipeline()
    pipeline.main()
    mock_validation.return_value.validate_raw_data.assert_called_once()


# ============================================================
# Stage 2: Data Preparation
# ============================================================
@patch("src.pipeline.stage_02_data_preparation.DataPreparation")
def test_stage_02_preparation(mock_prep):
    """Verifies that the Preparation stage initializes and executes correctly."""
    pipeline = DataPreparationPipeline()
    pipeline.main()
    mock_prep.return_value.prepare_reddit_dataset.assert_called_once()


# ============================================================
# Stage 3: Feature Engineering
# ============================================================
@patch("src.pipeline.stage_03_feature_engineering.FeatureEngineering")
def test_stage_03_features(mock_feat):
    """Verifies that the Feature Engineering stage initializes and executes correctly."""
    pipeline = FeatureEngineeringPipeline()
    pipeline.main()
    mock_feat.return_value.build_features.assert_called_once()


# ============================================================
# Stage 4a: Baseline Training
# ============================================================
@patch("src.pipeline.stage_04a_baseline_model.setup_experiment")
@patch("src.pipeline.stage_04a_baseline_model.BaselineModel")
def test_stage_04a_baseline(mock_baseline, mock_setup):
    """Verifies that the Baseline Training stage initializes and executes correctly."""
    pipeline = BaselineModelPipeline()
    pipeline.main()
    mock_baseline.return_value.train_baseline.assert_called_once()


# ============================================================
# Stage 4b: Hyperparameter Tuning
# ============================================================
@patch("src.pipeline.stage_04b_hyperparameter_tuning.setup_experiment")
@patch("src.pipeline.stage_04b_hyperparameter_tuning.HyperparameterTuning")
def test_stage_04b_tuning(mock_tuner, mock_setup):
    """Verifies that the Tuning stage initializes and executes data-driven tuning."""
    pipeline = HyperparameterTuningPipeline()
    pipeline.main("lightgbm")
    mock_tuner.return_value.tune_model.assert_called_once_with("lightgbm")


# ============================================================
# Stage 4c: DistilBERT Training
# ============================================================
@patch("src.pipeline.stage_04c_distilbert_training.setup_experiment")
@patch("src.pipeline.stage_04c_distilbert_training.DistilBERTTraining")
def test_stage_04c_bert(mock_bert, mock_setup):
    """Verifies that the DistilBERT stage initializes and executes training."""
    pipeline = DistilBERTTrainingPipeline()
    pipeline.main()
    mock_bert.return_value.fine_tune.assert_called_once()


# ============================================================
# Stage 5: Model Evaluation
# ============================================================
@patch("src.pipeline.stage_05_model_evaluation.setup_experiment")
@patch("src.pipeline.stage_05_model_evaluation.ModelEvaluation")
def test_stage_05_evaluation(mock_eval, mock_setup):
    """Verifies that the Evaluation stage initializes and executes across metrics."""
    pipeline = ModelEvaluationPipeline()
    pipeline.main()
    mock_eval.return_value.run_evaluation.assert_called_once()


# ============================================================
# Stage 6: Model Registration
# ============================================================
@patch("src.pipeline.stage_06_register_model.mlflow")
@patch("src.pipeline.stage_06_register_model.ModelRegistration")
def test_stage_06_registration(mock_reg, mock_mlflow):
    """Verifies that the Registration stage initializes and executes for champion models."""
    pipeline = ModelRegistrationPipeline()
    pipeline.main()
    mock_reg.return_value.run_registration.assert_called_once()
