"""
Utility functions for MLflow experiment tracking actions.
"""

import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.paths import REPORTS_DIR, EVAL_FIG_DIR

logger = get_logger(__name__, headline="mlflow_tracking_utils.py")


# ---------------------------------------------------------------------
# 1. MLflow Setup
# ---------------------------------------------------------------------
def setup_experiment(experiment_name: str, mlflow_uri: str):
    """Initialize MLflow tracking with URI and experiment name."""
    mlflow.set_tracking_uri(mlflow_uri)

    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d")
    full_name = f"{experiment_name} - {timestamp}"

    try:
        # If the experiment is already set, this is a no-op
        mlflow.set_experiment(full_name)
    except mlflow.exceptions.MlflowException as e:
        # Handles case where a deleted experiment is attempted to be set
        if "deleted" in str(e).lower() or "not found" in str(e).lower():
            logger.info(
                f"Experiment '{full_name}' not found/deleted. Creating a new one."
            )
            mlflow.set_experiment(full_name)

    logger.info(
        f"MLflow experiment initialized → {full_name} | URI: {mlflow.get_tracking_uri()}"
    )


# ---------------------------------------------------------------------
# 2. Metric Logging
# ---------------------------------------------------------------------
def log_metrics_to_mlflow(metrics: dict):
    """Log a dictionary of metrics to the active MLflow run."""
    mlflow.log_metrics(metrics)
    logger.info(f"Logged {len(metrics)} metrics to MLflow.")


# ---------------------------------------------------------------------
# 3. Artifact Logging (e.g., Confusion Matrix)
# ---------------------------------------------------------------------
def log_confusion_matrix_as_artifact(cm, model_name: str, labels: list):
    """
    Generate and log a Confusion Matrix plot to the active MLflow run.
    This plot is saved locally first, then uploaded as an artifact.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Confusion Matrix for {model_name.upper()} (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save to DVC-tracked path first
    filepath = EVAL_FIG_DIR / f"{model_name}_confusion_matrix.png"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    plt.close()

    # Log to MLflow
    mlflow.log_artifact(str(filepath), artifact_path="evaluation_plots")
    logger.info(
        f"Confusion Matrix saved to {filepath.relative_to(REPORTS_DIR.parent)} and logged to MLflow."
    )
