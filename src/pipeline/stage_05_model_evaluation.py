"""
Execution Conductor: Model Evaluation.

This pipeline stage orchestrates the comparative evaluation of all trained models.
It acts as a thin wrapper that initiates the `ModelEvaluation` component,
delegating the evaluation, validation, ROC plotting, and Champion selection logic to it.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.pipeline.stage_05_model_evaluation
"""

from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.mlflow_tracking_utils import setup_experiment

logger = get_logger(__name__, headline="model_evaluation.py")

EXPERIMENT_NAME = "Final Model Evaluation - Test Set"


class ModelEvaluationPipeline:
    """
    Pipeline stage execution class orchestrating the evaluation and comparison phase.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self):
        """Execute the validation phase by triggering the model evaluation worker."""
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_model_evaluation_config()

        mlflow_uri = get_mlflow_uri()
        setup_experiment(EXPERIMENT_NAME, mlflow_uri)

        model_evaluation = ModelEvaluation(config=eval_config)
        model_evaluation.run_evaluation()


if __name__ == "__main__":
    try:
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
