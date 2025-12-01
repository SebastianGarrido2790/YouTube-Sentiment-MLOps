"""
Trains Logistic Regression baseline on engineered features using DVC parameters.

Logs experiment to MLflow; saves the model bundle (model + LabelEncoder) locally for DVC tracking.
Parameters are loaded from params.yaml by default.

Usage (DVC - preferred):
    uv run dvc repro
    Run specific pipeline stage:
    uv run dvc repro baseline_model

Usage (local cli override only)
    uv run python -m src.models.baseline_logistic --C 0.5 --max_iter 1000

Requirements:
    - Processed features in models/features/.
    - Parameters defined in params.yaml under `train.logistic_baseline`.
    - MLflow server running.

Desing:
    - Parameters are read from params.yaml via dvc.api (single source of truth).
    - CLI arguments are optional and only for quick local testing overrides.
    - Reproducibility is prioritized by warning users about CLI overrides.
"""

import argparse
from typing import Any, Dict

import dvc.api
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- Project Utilities ---
from src.models.helpers.data_loader import load_feature_data
from src.models.helpers.mlflow_tracking_utils import (
    log_metrics_to_mlflow,
    setup_experiment,
)
from src.models.helpers.train_utils import (
    save_baseline_metrics_json,
    save_model_bundle,
)
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import BASELINE_MODEL_DIR

# --- Logging Setup ---
logger = get_logger(__name__, headline="baseline_logistic_training.py")


def load_params() -> Dict[str, Any]:
    """
    Load logistic regression parameters from params.yaml using DVC.
    Falls back gracefully if running outside a DVC pipeline.
    """
    try:
        logger.info("Loading params via dvc.api")
        params = dvc.api.params_show()
        return params["train"]["logistic_baseline"]
    except Exception as e:
        logger.warning(f"Could not load params via dvc.api: {e}")
        logger.warning("Falling back to script defaults (only for local debugging).")
        return {
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 2000,
            "C": 1.0,
            "random_state": 42,
        }


def train_baseline(
    C: float,
    max_iter: int,
    solver: str,
    class_weight: str,
    random_state: int = 42,
) -> None:
    """Train Logistic Regression baseline and log to MLflow."""

    # --- Load engineered features using helper ---
    logger.info("Loading pre-engineered TF-IDF features and labels...")
    X_train, X_val, X_test, y_train, y_val, y_test, le = load_feature_data()

    # --- Model Configuration ---
    params = {
        "C": C,
        "max_iter": max_iter,
        "solver": solver,
        "class_weight": class_weight,
        "random_state": random_state,
    }

    # Ensure clean MLflow run state
    mlflow.end_run()

    with mlflow.start_run(run_name="Baseline_LogReg_TFIDF_Balanced"):
        # --- Tags ---
        mlflow.set_tags(
            {
                "stage": "model_training",
                "model_type": "LogisticRegression",
                "imbalance_method": f"class_weight_{class_weight}",
                "experiment_type": "baseline_modeling",
                "description": "Baseline Logistic Regression with balanced class weights on TF-IDF features",
            }
        )

        # --- Log Parameters ---
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("feature_dim", X_train.shape[1])

        # --- Train Model ---
        logger.info(f"Training Logistic Regression baseline (params={params})...")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # --- Predict ---
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # --- Inverse transform labels ---
        y_val_orig = le.inverse_transform(y_val)
        y_test_orig = le.inverse_transform(y_test)
        y_pred_val_orig = le.inverse_transform(y_pred_val)
        y_pred_test_orig = le.inverse_transform(y_pred_test)

        # --- Compute Metrics ---
        val_acc = accuracy_score(y_val_orig, y_pred_val_orig)
        val_f1 = f1_score(y_val_orig, y_pred_val_orig, average="macro")
        test_acc = accuracy_score(y_test_orig, y_pred_test_orig)
        test_f1 = f1_score(y_test_orig, y_pred_test_orig, average="macro")

        # --- Log metrics via helper ---
        log_metrics_to_mlflow(
            {
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
            }
        )

        # --- Save baseline metric locally for DVC tracking ---
        save_baseline_metrics_json(score=val_f1)

        # --- Per-class F1 breakdown ---
        report = classification_report(y_test_orig, y_pred_test_orig, output_dict=True)
        for label, metrics in report.items():
            if label not in ("accuracy", "macro avg", "weighted avg"):
                mlflow.log_metric(f"test_f1_{label}", metrics["f1-score"])

        logger.info(
            f"✅ Logistic Regression baseline complete | "
            f"Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}"
        )

        # --- Log Model Bundle ---
        model_bundle = {"model": model, "encoder": le}
        mlflow.sklearn.log_model(sk_model=model_bundle, artifact_path="model")

        # --- Save Locally for DVC Tracking ---
        save_model_bundle(
            model_bundle=model_bundle,
            save_path=BASELINE_MODEL_DIR / "logistic_baseline.pkl",
        )

        logger.info(
            f"🎯 MLflow Run completed | Run ID: {mlflow.active_run().info.run_id}"
        )


def main() -> None:
    """Parse args and run baseline training using DVC params as source of truth."""
    params = load_params()
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression baseline. Params from params.yaml by default."
    )

    parser.add_argument(
        "--C", type=float, required=False, help="Inverse regularization strength."
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        required=False,
        help="Maximum number of iterations for solver.",
    )
    parser.add_argument(
        "--solver", type=str, required=False, help="Algorithm to use in optimization."
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        required=False,
        help="Weights associated with classes.",
    )
    args = parser.parse_args()

    # Consolidate parameters (CLI overrides DVC-loaded params)
    final_params = params.copy()
    overridden_keys = []
    for key in final_params:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            final_params[key] = cli_val
            overridden_keys.append(key)

    if overridden_keys:
        logger.warning(
            "CLI overrides detected for: %s. This run may not be reproducible with 'dvc repro'.",
            ", ".join(overridden_keys),
        )

    logger.info("🚀 Starting baseline Logistic Regression training...")

    # --- MLflow Setup ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment("Model Training - Baseline Logistic Regression", mlflow_uri)

    train_baseline(
        C=final_params.get("C", 1.0),
        max_iter=final_params.get("max_iter", 2000),
        solver=final_params.get("solver", "liblinear"),
        class_weight=final_params.get("class_weight", "balanced"),
    )


if __name__ == "__main__":
    main()
