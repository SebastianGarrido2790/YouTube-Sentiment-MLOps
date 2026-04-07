"""
Execution Conductor: Automated Model Registration.

This pipeline stage orchestrates the champion model validation and registration process.
It serves as an execution wrapper that triggers the `ModelRegistration` component
to parse the evaluation results and deploy the artifact to the MLflow registry
if the required F1 threshold is met.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.pipeline.stage_06_register_model
"""

import mlflow

from src.components.register_model import ModelRegistration
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri

logger = get_logger(__name__, headline="register_model.py")


class ModelRegistrationPipeline:
    """
    Pipeline stage execution class orchestrating champion registry tagging.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self):
        """Execute the registration phase by triggering the registration worker."""
        logger.info("🚀 Starting automated model registration workflow... 🚀")

        config_manager = ConfigurationManager()
        register_config = config_manager.get_register_config()

        logger.info(f"Using F1 threshold for registration: {register_config.f1_threshold}")

        mlflow_uri = get_mlflow_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Using MLflow Tracking URI: {mlflow_uri}")

        model_registration = ModelRegistration(config=register_config)
        model_registration.run_registration()


if __name__ == "__main__":
    try:
        pipeline = ModelRegistrationPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
