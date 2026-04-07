"""
Execution Conductor: Hyperparameter Tuning (Optuna).

This pipeline stage orchestrates the optimization of hyperparameters for
gradient boosting models (LightGBM and XGBoost). It acts as a thin execution
wrapper that delegates all objective evaluation and training logic to the
`HyperparameterTuning` component.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.pipeline.stage_04b_hyperparameter_tuning --model lightgbm
    uv run python -m src.pipeline.stage_04b_hyperparameter_tuning --model xgboost
"""

import argparse

from src.components.hyperparameter_tuning import HyperparameterTuning
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.mlflow_tracking_utils import setup_experiment

logger = get_logger(__name__, headline="hyperparameter_tuning.py")


class HyperparameterTuningPipeline:
    """
    Pipeline stage execution class orchestrating the Optuna hyperparameter tuning phase.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self, model_name: str):
        """
        Execute the hyperparameter optimization step by triggering the worker component.

        Args:
            model_name (str): The name of the model ('lightgbm' or 'xgboost').
        """
        logger.info(f"🚀 Starting hyperparameter tuning for {model_name.upper()}... 🚀")

        config_manager = ConfigurationManager()
        train_config = config_manager.get_train_config()

        mlflow_uri = get_mlflow_uri()
        setup_experiment(f"Hyperparameter Tuning - {model_name.upper()}", mlflow_uri)

        hyperparameter_tuning = HyperparameterTuning(config=train_config)
        hyperparameter_tuning.tune_model(model_name)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            choices=["lightgbm", "xgboost"],
            help="The model to tune.",
        )
        args = parser.parse_args()

        pipeline = HyperparameterTuningPipeline()
        pipeline.main(args.model)
    except Exception as e:
        logger.exception(e)
        raise e
