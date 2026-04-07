"""
Component for training a Logistic Regression baseline model.

This module provides the worker component that utilizes pre-engineered TF-IDF features
to train a baseline Logistic Regression model while logging metrics to MLflow.
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.constants import BASELINE_MODEL_DIR
from src.entity.config_entity import LogisticBaselineConfig
from src.utils.data_loader import load_feature_data
from src.utils.logger import get_logger
from src.utils.mlflow_tracking_utils import log_metrics_to_mlflow
from src.utils.train_utils import save_baseline_metrics_json, save_model_bundle

logger = get_logger(__name__, headline="baseline_model_component")


class BaselineModel:
    """
    Component responsible for training the baseline Logistic Regression model.

    Attributes:
        config (LogisticBaselineConfig): Configuration parameters for the model.
        random_state (int): Seed for reproducibility.
    """

    def __init__(self, config: LogisticBaselineConfig, random_state: int = 42):
        """
        Initialize the BaselineModel component.

        Args:
            config (LogisticBaselineConfig): Configuration for Logistic Regression.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.config = config
        self.random_state = random_state

    def train_baseline(self):
        """
        Train the Logistic Regression baseline model using pre-engineered TF-IDF features.

        Procedure:
            1. Load the TF-IDF features and labels.
            2. Train a LogisticRegression classifier.
            3. Evaluate validation and test F1 / Accuracy.
            4. Log all parameters, tags, and metrics into MLflow.
            5. Save the local `.pkl` model bundle for DVC tracking.
        """
        logger.info("Loading pre-engineered TF-IDF features and labels...")
        X_train, X_val, X_test, y_train, y_val, y_test, le = load_feature_data()

        params = {
            "C": self.config.C,
            "max_iter": self.config.max_iter,
            "solver": self.config.solver,
            "class_weight": self.config.class_weight,
            "random_state": self.random_state,
        }

        mlflow.end_run()

        with mlflow.start_run(run_name="Baseline_LogReg_TFIDF_Balanced"):
            mlflow.set_tags(
                {
                    "stage": "model_training",
                    "model_type": "LogisticRegression",
                    "imbalance_method": f"class_weight_{self.config.class_weight}",
                    "experiment_type": "baseline_modeling",
                    "description": "Baseline Logistic Regression with balanced class weights on TF-IDF features",
                }
            )

            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("feature_dim", X_train.shape[1])

            logger.info(f"Training Logistic Regression baseline (params={params})...")
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)

            y_val_orig = le.inverse_transform(y_val)
            y_test_orig = le.inverse_transform(y_test)
            y_pred_val_orig = le.inverse_transform(y_pred_val)
            y_pred_test_orig = le.inverse_transform(y_pred_test)

            val_acc = accuracy_score(y_val_orig, y_pred_val_orig)
            val_f1 = f1_score(y_val_orig, y_pred_val_orig, average="macro")
            test_acc = accuracy_score(y_test_orig, y_pred_test_orig)
            test_f1 = f1_score(y_test_orig, y_pred_test_orig, average="macro")

            log_metrics_to_mlflow(
                {
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "test_accuracy": test_acc,
                    "test_macro_f1": test_f1,
                }
            )

            save_baseline_metrics_json(score=val_f1)

            report = classification_report(y_test_orig, y_pred_test_orig, output_dict=True)
            for label, metrics in report.items():
                if label not in ("accuracy", "macro avg", "weighted avg"):
                    mlflow.log_metric(f"test_f1_{label}", metrics["f1-score"])

            logger.info(f"✅ Baseline Logistic Regression Complete | Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f} ✅")

            model_bundle = {"model": model, "encoder": le}
            mlflow.sklearn.log_model(sk_model=model_bundle, artifact_path="model")

            save_model_bundle(
                model_bundle=model_bundle,
                save_path=BASELINE_MODEL_DIR / "logistic_baseline.pkl",
            )

            logger.info(f"🎯 MLflow Run completed | Run ID: {mlflow.active_run().info.run_id} 🎯")
