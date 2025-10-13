"""
Trains Logistic Regression baseline on engineered features with class_weight='balanced'
for intrinsic imbalance handling.

Logs experiment to MLflow; saves the model bundle (model + LabelEncoder) locally for DVC tracking.

Usage:
    uv run python -m src.models.baseline_logistic

Design Considerations:
- Reliability: Uses class weights for simple, effective imbalance handling; robust logging of per-class F1.
- Maintainability: Simple model, centralized path and logging utilities.
- Decoupling: Loads features from .npz/.npy files, independent of feature generation script.
"""

import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

# --- Project Utilities ---
from src.utils.paths import MODELS_DIR, PROJECT_ROOT
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.logger import get_logger

# --- Logging Setup ---
logger = get_logger(__name__, headline="baseline_logistic.py")

# --- MLflow Setup ---
mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Model Training - Baseline Logistic Regression")

# --- Path Setup ---
FEATURES_DIR = MODELS_DIR / "features" / "engineered_features"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_baseline() -> None:
    """
    Train Logistic Regression with class_weight='balanced' for imbalance handling.
    Logs to MLflow with consistent stage, metrics, and tagging conventions.
    """

    logger.info("Loading engineered features...")
    X_train = load_npz(FEATURES_DIR / "X_train.npz").tocsr()
    X_val = load_npz(FEATURES_DIR / "X_val.npz").tocsr()
    X_test = load_npz(FEATURES_DIR / "X_test.npz").tocsr()
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_val = np.load(FEATURES_DIR / "y_val.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")

    with open(FEATURES_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    original_labels = le.classes_  # [-1, 0, 1]

    # --- Model Parameters ---
    params = {
        "C": 1.0,
        "max_iter": 2000,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42,
    }

    mlflow.end_run()  # ensure no previous run is active

    with mlflow.start_run(run_name="LogReg_Baseline_TFIDF_Balanced"):
        # --- Tags ---
        mlflow.set_tag("stage", "model_training")
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("imbalance_method", "class_weight_balanced")
        mlflow.set_tag("feature_type", "TF-IDF (max_features=1000)")
        mlflow.set_tag("experiment_type", "baseline_modeling")
        mlflow.set_tag(
            "description", "Logistic Regression baseline with class_weight='balanced'"
        )

        # --- Log Parameters ---
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("feature_dim", X_train.shape[1])

        # --- Train Model ---
        logger.info(
            "Training Logistic Regression baseline (class_weight='balanced')..."
        )
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # --- Predict on Validation & Test ---
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # Decode labels back to original {-1, 0, 1}
        y_val_orig = le.inverse_transform(y_val)
        y_test_orig = le.inverse_transform(y_test)
        y_pred_val_orig = le.inverse_transform(y_pred_val)
        y_pred_test_orig = le.inverse_transform(y_pred_test)

        # --- Compute Metrics ---
        val_acc = accuracy_score(y_val_orig, y_pred_val_orig)
        val_f1 = f1_score(y_val_orig, y_pred_val_orig, average="macro")
        test_acc = accuracy_score(y_test_orig, y_pred_test_orig)
        test_f1 = f1_score(y_test_orig, y_pred_test_orig, average="macro")

        # Log aggregate metrics
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_macro_f1", val_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_macro_f1", test_f1)

        # Log detailed per-class F1 on test set
        report = classification_report(y_test_orig, y_pred_test_orig, output_dict=True)
        for label in original_labels:
            mlflow.log_metric(f"test_f1_{label}", report[str(label)]["f1-score"])

        logger.info(
            f"✅ Baseline Logistic Regression completed | "
            f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
            f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}"
        )

        # --- Log Model Bundle to MLflow ---
        model_bundle = {"model": model, "encoder": le}
        mlflow.sklearn.log_model(
            sk_model=model_bundle,
            artifact_path="model",
        )

        # --- Save Locally for DVC Tracking ---
        local_path = BASELINE_MODEL_DIR / "logistic_baseline.pkl"
        with open(local_path, "wb") as f:
            pickle.dump(model_bundle, f)
        logger.info(
            f"Model bundle saved locally to: {local_path.relative_to(PROJECT_ROOT)}"
        )

        logger.info(
            f"🎯 MLflow Run completed | Run ID: {mlflow.active_run().info.run_id}"
        )


if __name__ == "__main__":
    logger.info("🚀 Starting baseline Logistic Regression training...")
    train_baseline()
