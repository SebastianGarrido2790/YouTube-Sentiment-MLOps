"""
Execution Conductor: DistilBERT Training.

This pipeline stage orchestrates the fine-tuning of the DistilBERT language model.
It acts as a thin execution wrapper that delegates data processing, tokenization,
and Optuna tuning logic to the `DistilBERTTraining` component.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.pipeline.stage_04c_distilbert_training
"""

from src.components.distilbert_training import DistilBERTTraining
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.mlflow_tracking_utils import setup_experiment

logger = get_logger(__name__, headline="DistilBERT_Training.py")


class DistilBERTTrainingPipeline:
    """
    Pipeline stage execution class orchestrating the DistilBERT fine-tuning phase.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self):
        """Execute the DistilBERT training step by triggering the worker component."""
        logger.info("🚀 Starting DistilBERT training pipeline... 🚀")

        config_manager = ConfigurationManager()
        distilbert_config = config_manager.get_distilbert_config()

        mlflow_uri = get_mlflow_uri()
        setup_experiment("DistilBERT - Advanced Tuning", mlflow_uri)

        distilbert_training = DistilBERTTraining(config=distilbert_config)
        distilbert_training.fine_tune()


if __name__ == "__main__":
    try:
        pipeline = DistilBERTTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
