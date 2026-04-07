"""
Component for evaluating and comparing machine learning models.

This module provides the worker component responsible for dynamically loading
trained model artifacts (Logistic Regression, LightGBM, XGBoost, DistilBERT),
calculating test set metrics, plotting a comparative Macro-Average ROC curve,
and selecting the champion model based on the highest Test Macro AUC.
"""

import pickle
from itertools import cycle

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from src.constants import (
    ADVANCED_DIR,
    BASELINE_MODEL_DIR,
    EVAL_FIG_DIR,
    PROJECT_ROOT,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.data_loader import load_feature_data, load_text_data
from src.utils.logger import get_logger
from src.utils.mlflow_tracking_utils import (
    log_confusion_matrix_as_artifact,
    log_metrics_to_mlflow,
)
from src.utils.train_utils import (
    save_best_model_run_info,
    save_test_metrics_json,
)

logger = get_logger(__name__, headline="model_evaluation_component")


class ModelEvaluation:
    """
    Component handling model evaluation, comparison, and MLflow logging.

    Attributes:
        config (ModelEvaluationConfig): Model evaluation configuration from params.yaml.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation component.

        Args:
            config (ModelEvaluationConfig): Evaluation configuration.
        """
        self.config = config

    def load_model_artifact(self, model_name: str):
        """
        Load a serialized `.pkl` model artifact from the respective directories.

        Args:
            model_name (str): The name of the model ('logistic_baseline', 'lightgbm', etc.)

        Returns:
            object: The loaded model object, or None if skipped.

        Raises:
            FileNotFoundError: If the expected model file cannot be found.
            ValueError: If an unknown `model_name` is provided.
        """
        if model_name == "logistic_baseline":
            filepath = BASELINE_MODEL_DIR / "logistic_baseline.pkl"
            logger.info(f"Loading baseline model bundle from: {filepath.relative_to(PROJECT_ROOT)}")

            try:
                with open(filepath, "rb") as f:
                    model_bundle = pickle.load(f)
                model = model_bundle.get("model")
                if model is None:
                    raise ValueError("Model object not found in the baseline bundle.")
                return model
            except FileNotFoundError:
                logger.error(f"Baseline model file not found at: {filepath}")
                raise
        elif model_name in ["lightgbm", "xgboost", "distilbert"]:
            filepath = ADVANCED_DIR / f"{model_name}_model.pkl"
            if model_name == "distilbert" and filepath.exists() and filepath.stat().st_size == 0:
                logger.warning(f"DistilBERT artifact at {filepath} is empty (skipped training). Skipping evaluation.")
                return None

            logger.info(f"Loading advanced model artifact from: {filepath.relative_to(PROJECT_ROOT)}")

            try:
                with open(filepath, "rb") as f:
                    model = pickle.load(f)
                return model
            except FileNotFoundError:
                logger.error(f"Advanced model file not found at: {filepath}")
                raise
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                "Must be 'lightgbm', 'xgboost', 'distilbert', or 'logistic_baseline'."
            )

    def evaluate_model(self, model, x_test, y_test, model_name: str = ""):
        """
        Compute classification report, confusion matrix, and prediction probabilities.

        Args:
            model (object): The loaded model object.
            x_test (np.ndarray): The test features.
            y_test (np.ndarray): The test labels.
            model_name (str): The name of the model.

        Returns:
            tuple: A tuple containing (report_dict, confusion_matrix, y_pred_proba).
        """
        try:
            if "LGBMClassifier" in str(type(model)):
                y_pred = model.predict(x_test)
                y_pred_proba = model.predict_proba(x_test)
            elif "xgboost.core.Booster" in str(type(model)):
                dtest = xgb.DMatrix(x_test, label=y_test)
                y_pred_proba = model.predict(dtest)
                y_pred = np.argmax(y_pred_proba, axis=1)
            elif "transformers" in str(type(model)) or model_name == "distilbert":
                if pipeline is None:
                    logger.warning("transformers library not found. Skipping DistilBERT evaluation.")
                    return None, None, None

                # Check if x_test is raw text (needed for HF pipeline)
                if not isinstance(x_test, list) or not isinstance(x_test[0], str):
                    logger.warning("DistilBERT evaluation requires raw text list. Skipping.")
                    return None, None, None

                pipe = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer="distilbert-base-uncased",
                    device=-1,
                    return_all_scores=True,
                )
                preds = pipe(x_test)

                # preds is a list of lists of dicts [{'label': 'LABEL_0', 'score': 0.1}, ...]
                y_pred_proba = np.zeros((len(x_test), 3))
                for i, pred_list in enumerate(preds):
                    for pred_dict in pred_list:
                        # Extract class index from LABEL_0 format
                        class_idx = int(pred_dict["label"].split("_")[-1])
                        y_pred_proba[i, class_idx] = pred_dict["score"]

                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = model.predict(x_test)
                y_pred_proba = model.predict_proba(x_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            logger.info("Model evaluation completed on test set.")
            return report, cm, y_pred_proba
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def plot_comparative_roc_curve(self, y_test_bin, roc_results: list, labels: list):
        """
        Generate and save a comparative plot of Macro-Average ROC curves for all models.

        Args:
            y_test_bin (np.ndarray): LabelBinarizer transformed test labels (One-vs-Rest format).
            roc_results (list): List of dictionaries containing `name` and `proba`.
            labels (list): List of string class labels.
        """
        n_classes = len(labels)
        file_path = EVAL_FIG_DIR / "comparative_roc_curve.png"

        plt.figure(figsize=(12, 10))
        colors = cycle(["blue", "green", "red", "cyan", "magenta", "yellow"])

        for result, color in zip(roc_results, colors, strict=False):
            model_name = result["name"]
            y_pred_proba = result["proba"]

            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            plt.plot(
                fpr["macro"],
                tpr["macro"],
                color=color,
                lw=2,
                label=f"{model_name} (Macro-Avg AUC = {roc_auc['macro']:.3f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Chance (AUC = 0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Comparative Macro-Average OvR ROC Curves (Test Set)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        mlflow.log_artifact(str(file_path))
        logger.info(f"Comparative ROC Curve saved to {file_path.relative_to(PROJECT_ROOT)} and logged to MLflow.")

    def log_model_to_mlflow(self, model, model_name: str):
        """
        Dynamically log the model artifact back to the MLflow child run.

        Args:
            model (object): The loaded model.
            model_name (str): The name of the model to resolve its flavor (sklearn, lgb, xgb).
        """
        if "LGBMClassifier" in str(type(model)):
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=None,
            )
        elif "xgboost.core.Booster" in str(type(model)):
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name=None,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None,
            )
        logger.info(f"Model artifact for {model_name} logged to MLflow run.")

    def run_evaluation(self):
        """
        Main orchestration method for model comparison.

        Iterates over the models defined in the config, loads their test data,
        evaluates metrics, logs to MLflow (Nested Runs), plots comparative curves,
        and finally identifies and saves the details of the champion model.
        """
        model_list = self.config.models

        try:
            _, _, X_test, _, _, y_test, le = load_feature_data(validate_files=True)
            labels = le.classes_.tolist()
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)

            try:
                _, _, test_df = load_text_data()
                X_test_text = test_df["clean_comment"].astype(str).tolist()
            except FileNotFoundError:
                X_test_text = None

        except Exception as e:
            logger.error(f"Data loading error: {e}")
            raise

        with mlflow.start_run(run_name="Model_Comparison_Test_Set") as parent_run:
            logger.info(f"🚀 Starting model comparison for: {model_list} 🚀")
            mlflow.set_tag("task", "Comparative Evaluation")
            mlflow.log_param("models_evaluated", ", ".join(model_list))

            roc_results = []
            champion_metrics = []

            for model_name in model_list:
                logger.info(f"--- Evaluating model: {model_name} ---")

                if model_name == "distilbert" and X_test_text is None:
                    logger.info("DistilBERT requires raw text for evaluation, but it was not found. Skipping.")
                    continue

                with mlflow.start_run(run_name=f"Evaluation_{model_name}", nested=True) as child_run:
                    try:
                        mlflow.set_tag("model_name", model_name)
                        mlflow.set_tag("parent_run_id", parent_run.info.run_id)

                        if model_name == "distilbert":
                            # Use X_test_text for DistilBERT
                            import os

                            from transformers import AutoModelForSequenceClassification

                            # Auto-resolve checkpoint dir if trained
                            res_dir = ADVANCED_DIR / "distilbert_results"
                            checkpoints = (
                                [d for d in os.listdir(res_dir) if d.startswith("checkpoint")]
                                if res_dir.exists()
                                else []
                            )
                            if not checkpoints:
                                logger.warning(f"No checkpoint found for DistilBERT in {res_dir}. Evaluation skipped.")
                                continue

                            best_ckpt = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                            model_path = res_dir / best_ckpt

                            # Load HuggingFace PyTorch model directly for pipeline
                            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                            report, cm, y_pred_proba = self.evaluate_model(model, X_test_text, y_test, model_name)
                        else:
                            model = self.load_model_artifact(model_name)
                            if model is None:
                                logger.warning(f"Feature processing skipped for {model_name}. Skipping evaluation.")
                                continue

                            report, cm, y_pred_proba = self.evaluate_model(model, X_test, y_test, model_name)

                        if report is None:
                            continue

                        test_macro_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")

                        flat_metrics = {
                            f"test_{label}_{metric_name.replace('-score', '_f1')}": value
                            for label, metrics in report.items()
                            if isinstance(metrics, dict)
                            for metric_name, value in metrics.items()
                            if metric_name not in ("support")
                        }
                        flat_metrics["test_macro_auc"] = test_macro_auc
                        log_metrics_to_mlflow(flat_metrics)

                        save_test_metrics_json(model_name, report)

                        log_confusion_matrix_as_artifact(cm, model_name, labels)

                        self.log_model_to_mlflow(model, model_name)

                        roc_results.append({"name": model_name, "proba": y_pred_proba})

                        champion_metrics.append(
                            {
                                "model_name": model_name,
                                "run_id": child_run.info.run_id,
                                "test_macro_auc": test_macro_auc,
                                "test_macro_f1": report["macro avg"]["f1-score"],
                            }
                        )

                        logger.info(
                            f"✅ Evaluation complete for {model_name}. "
                            f"Test Macro F1: {report['macro avg']['f1-score']:.4f} | "
                            f"Test Macro AUC: {test_macro_auc:.4f} ✅"
                        )

                    except Exception as e:
                        logger.error(f"Failed to evaluate model {model_name}: {e}")
                        mlflow.set_tag("status", "FAILED")
                        mlflow.log_param("error", str(e))
                        continue

            if roc_results:
                logger.info("Generating comparative ROC curve...")
                self.plot_comparative_roc_curve(y_test_bin, roc_results, labels)

                if champion_metrics:
                    logger.info("Selecting champion model...")
                    champion = max(champion_metrics, key=lambda item: item["test_macro_auc"])

                    logger.info(
                        f"🏆 Champion selected: {champion['model_name']} "
                        f"(AUC: {champion['test_macro_auc']:.4f}). "
                        f"Saving run info for registration. 🏆"
                    )

                    save_best_model_run_info(run_id=champion["run_id"], model_name=champion["model_name"])
                else:
                    logger.error("No models were successfully evaluated. Cannot select champion.")
            else:
                logger.warning("No models were successfully evaluated. Skipping ROC curve and champion selection.")

            logger.info(f"🏁 Model comparison complete. View Parent Run: {parent_run.info.run_id} 🏁")
