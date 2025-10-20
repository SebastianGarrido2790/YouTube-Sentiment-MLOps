"""
Utility functions for feature engineering and model evaluation.
Includes MLflow setup, data loading, DVC parameter parsing,
and standardized model evaluation with logging."""

import ast
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import ClassifierMixin
from scipy.sparse import spmatrix

# --- Project Utilities ---
from src.utils.paths import TRAIN_PATH, VAL_PATH
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri

# Setup logger for the utility module
logger = get_logger(__name__)


def setup_mlflow_run(experiment_name: str, params_path: str = "params.yaml") -> None:
    """
    Standardizes MLflow configuration (URI and Experiment).
    Takes the URI string from get_mlflow_uri and executes the global MLflow setup commands
    (mlflow.set_tracking_uri, mlflow.set_experiment).
    It manages the global state of the MLflow client.
    This should be called once at the start of a script.

    Args:
        experiment_name (str): Name of the MLflow experiment to use or create.
        params_path (str): Path to params.yaml for fallback URI (default: project root).

    Maintainability: It enforces the correct sequence of setting the URI, setting the experiment, and logging the action.
    """
    mlflow_uri = get_mlflow_uri(params_path)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow URI set to: {mlflow_uri}")
    logger.info(f"MLflow Experiment set to: {experiment_name}")


def load_train_val_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and returns the processed train and validation datasets from Parquet files.

    Reliability: Includes robust error handling for missing files.
    """
    logger.info(f"Loading data from {TRAIN_PATH} and {VAL_PATH}...")
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        logger.info(
            f"Data loaded: Train shape {train_df.shape}, Val shape {val_df.shape}"
        )
        return train_df, val_df
    except FileNotFoundError as e:
        logger.error(
            f"Data file not found. Ensure 'make_dataset.py' has been executed. Error: {e}"
        )
        raise


def parse_dvc_param(param_value: str, name: str, expected_type: type = str) -> Any:
    """
    Safely converts string parameters passed from DVC (params.yaml) into their
    native Python objects (e.g., list, tuple, int, float).

    Maintainability: Centralizes the complex and error-prone logic of
                     `ast.literal_eval` used for complex parameters.
    """
    param_value = str(param_value).strip()
    try:
        # Use ast.literal_eval for safe parsing of Python literals (lists, tuples, dicts)
        if expected_type in (list, tuple, dict):
            parsed_value = ast.literal_eval(param_value)
            if not isinstance(parsed_value, expected_type):
                raise ValueError(
                    f"Type mismatch: Expected {expected_type}, got {type(parsed_value)}"
                )
            return parsed_value

        # For simple types (str, int, float)
        return expected_type(param_value)

    except (ValueError, SyntaxError) as e:
        logger.error(
            f"Error parsing DVC parameter '{name}' with value '{param_value}'. Expected type: {expected_type.__name__}. Error: {e}"
        )
        # Re-raise to ensure the DVC stage fails with a clear message
        raise ValueError(
            f"Invalid DVC parameter format for '{name}'. Check params.yaml."
        ) from e


def evaluate_and_log(
    model: ClassifierMixin,
    X_val: Union[np.ndarray, spmatrix],
    y_val: np.ndarray,
    run_name: str,
    params: Dict[str, Any],
    tags: Dict[str, str],
    output_dir: Optional[Path] = None,
    log_model: bool = False,  # Set to True only for final, best model
) -> Dict[str, float]:
    """
    Standardizes model evaluation, metric calculation, and (optionally) artifact logging
    to MLflow for a given experiment run.

    If output_dir is provided, a confusion matrix is saved; otherwise, only metrics are logged.

    Reproducibility: Ensures all experiments log the same set of parameters,
                     tags, metrics, and the crucial confusion matrix plot.
    """
    logger.info(f"Evaluating model: {run_name}")
    y_pred = model.predict(X_val)

    # --- Metrics ---
    accuracy = accuracy_score(y_val, y_pred)
    # Use zero_division=0 to prevent warnings when a class is never predicted
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    # Extract key aggregate metrics for comparison
    metrics = {
        "val_accuracy": accuracy,
        "val_f1_score_macro": report["macro avg"]["f1-score"],
        "val_precision_macro": report["macro avg"]["precision"],
        "val_recall_macro": report["macro avg"]["recall"],
    }

    # --- MLflow Logging ---
    mlflow.log_params(params)
    mlflow.set_tags(tags)
    mlflow.log_metrics(metrics)
    mlflow.log_text(json.dumps(report, indent=4), "classification_report.json")
    logger.info(f"Key metrics logged: {metrics}")

    # --- Confusion Matrix Plot ---
    if output_dir is not None:
        fig = plt.figure(figsize=(10, 8))
        # NOTE: Ensure label order is consistent [-1, 0, 1]
        cm = confusion_matrix(y_val, y_pred, labels=[-1, 0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative (-1)", "Neutral (0)", "Positive (1)"],
            yticklabels=["Negative (-1)", "Neutral (0)", "Positive (1)"],
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title(f"Confusion Matrix\n{run_name}")

        # Save the figure locally and to MLflow
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"cm_{run_name}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(str(plot_path))
        plt.close(fig)

    # --- Model Logging ---
    if log_model:
        mlflow.sklearn.log_model(model, "model")
        logger.info("Trained model logged to MLflow Model Registry.")

    return metrics
