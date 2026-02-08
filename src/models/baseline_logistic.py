"""
Trains Logistic Regression baseline on engineered features using MLOps standards.

Logs experiment to MLflow; saves the model bundle (model + LabelEncoder) locally for DVC tracking.
Parameters are loaded strictly from params.yaml via ConfigurationManager.

Usage:
Run the entire pipeline:
    uv run dvc repro               # Uses params.yaml → fully reproducible
Run specific pipeline stage:
    uv run python -m src.models.baseline_logistic

Requirements:
    - Processed features in models/features/.
    - Parameters defined in params.yaml under `train.logistic_baseline`.
    - MLflow server must be running (e.g., uv run python -m mlflow server --host 127.0.0.1 --port 5000).
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- Project Utilities ---
from src.config.manager import ConfigurationManager
from src.config.schemas import LogisticBaselineConfig
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


def train_baseline(config: LogisticBaselineConfig, random_state: int = 42) -> None:
    """Train Logistic Regression baseline and log to MLflow."""

    # --- Load engineered features using helper ---
    logger.info("Loading pre-engineered TF-IDF features and labels...")
    X_train, X_val, X_test, y_train, y_val, y_test, le = load_feature_data()

    # --- Model Configuration ---
    # Using strict types from Pydantic config
    # Using strict types from Pydantic config
    params = {
        "C": config.C,
        "max_iter": config.max_iter,
        "solver": config.solver,
        "class_weight": config.class_weight,
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
                "imbalance_method": f"class_weight_{config.class_weight}",
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
            f"✅ Baseline Logistic Regression Complete | "
            f"Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f} ✅"
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
            f"🎯 MLflow Run completed | Run ID: {mlflow.active_run().info.run_id} 🎯"
        )


def main() -> None:
    """Run baseline training using ConfigurationManager as source of truth."""
    logger.info("🚀 Starting baseline Logistic Regression training... 🚀")

    # --- Parameter Loading via ConfigurationManager ---
    config_manager = ConfigurationManager()
    baseline_config = config_manager.get_logistic_baseline_config()

    # --- MLflow Setup ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment("Model Training - Baseline Logistic Regression", mlflow_uri)

    train_baseline(config=baseline_config)


if __name__ == "__main__":
    main()
