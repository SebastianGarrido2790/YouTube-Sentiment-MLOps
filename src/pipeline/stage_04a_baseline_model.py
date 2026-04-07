"""
Execution Conductor: Logistic Regression Baseline.

This pipeline stage orchestrates the training of the baseline model.
It delegates the actual model training and artifact logging to the
`BaselineModel` component, acting purely as a thin execution wrapper.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.pipeline.stage_04a_baseline_model
"""

from src.components.baseline_model import BaselineModel
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.mlflow_tracking_utils import setup_experiment

logger = get_logger(__name__, headline="baseline_logistic_training.py")


class BaselineModelPipeline:
    """
    Pipeline stage execution class orchestrating the baseline training phase.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self):
        """Execute the baseline training step by triggering the worker component."""
        logger.info("🚀 Starting baseline Logistic Regression training... 🚀")

        config_manager = ConfigurationManager()
        baseline_config = config_manager.get_logistic_baseline_config()

        mlflow_uri = get_mlflow_uri()
        setup_experiment("Model Training - Baseline Logistic Regression", mlflow_uri)

        baseline_model = BaselineModel(config=baseline_config)
        baseline_model.train_baseline()


if __name__ == "__main__":
    try:
        pipeline = BaselineModelPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
