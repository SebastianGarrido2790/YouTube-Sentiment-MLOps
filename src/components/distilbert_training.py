"""
Component for fine-tuning the DistilBERT language model.

This module provides the worker component that handles tokenization, Optuna
hyperparameter tuning, and Hugging Face Trainer orchestration for DistilBERT.
"""

import mlflow
import mlflow.transformers
import numpy as np
import optuna

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None

from datasets import Dataset
from sklearn.metrics import f1_score

from src.constants import ADVANCED_DIR
from src.entity.config_entity import DistilBERTConfig
from src.utils.data_loader import load_text_data
from src.utils.logger import get_logger
from src.utils.train_utils import save_hyperparams_bundle, save_metrics_json

logger = get_logger(__name__, headline="DistilBERT_Training_Component")


class DistilBERTTraining:
    """
    Component that orchestrates Optuna trials and Hugging Face training.

    Attributes:
        config (DistilBERTConfig): Configuration specifying hyperparameter bounds and settings.
    """

    def __init__(self, config: DistilBERTConfig):
        """
        Initialize the DistilBERTTraining component.

        Args:
            config (DistilBERTConfig): DistilBERT configuration from params.yaml.
        """
        self.config = config

    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function for trial evaluation.

        Tokenizes the raw text datasets, initializes Hugging Face AutoModelForSequenceClassification,
        and trains it on a set of hyperparameters drawn from the current trial.

        Args:
            trial (optuna.trial.Trial): The current Optuna trial.

        Returns:
            float: The macro F1 score evaluated on the validation dataset.
        """
        train_df, val_df, _ = load_text_data()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        def tokenize(batch):
            tokenized = tokenizer(
                batch["clean_comment"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            tokenized["labels"] = batch["category"] + 1
            return tokenized

        train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
        val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

        num_train_epochs = trial.suggest_int("num_epochs", 2, 5)
        per_device_train_batch_size = trial.suggest_categorical("batch_size", self.config.batch_size)
        learning_rate = trial.suggest_float("lr", min(self.config.lr), max(self.config.lr), log=True)
        weight_decay = trial.suggest_float("weight_decay", min(self.config.weight_decay), max(self.config.weight_decay))

        output_dir = ADVANCED_DIR / "distilbert_results"
        logging_dir = ADVANCED_DIR / "distilbert_logs"

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",  # type: ignore
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            logging_dir=str(logging_dir),
            disable_tqdm=True,
            report_to="none",
            fp16=torch.cuda.is_available() if torch else False,
        )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=1)
            return {"macro_f1": f1_score(labels, preds, average="macro")}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,  # type: ignore
            compute_metrics=compute_metrics,
        )

        with mlflow.start_run(run_name=f"DistilBERT_Trial_{trial.number}", nested=True):
            logger.info(f"🧪 Trial {trial.number} started... 🧪")
            trainer.train()
            results = trainer.evaluate()
            f1 = results["eval_macro_f1"]

            mlflow.log_params(trial.params)
            mlflow.log_metric("val_macro_f1", f1)

            mlflow.transformers.log_model(
                transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                artifact_path="distilbert_model_trial",
                task="text-classification",
                metadata={"optuna_trial_id": trial.number, "macro_f1": f1},
            )

            logger.info(f"Trial {trial.number} finished with F1: {f1:.4f}")

        return f1

    def create_placeholder_artifacts(self):
        """
        Create dummy artifact files when DistilBERT training is bypassed.

        This guarantees that DVC stages relying on these outputs will continue
        running properly even if deep learning is disabled.
        """
        logger.warning("Skipping DistilBERT training. Creating placeholder artifacts for DVC continuity.")

        distilbert_model_path = ADVANCED_DIR / "distilbert_model.pkl"
        metrics_path = ADVANCED_DIR / "distilbert_metrics.json"
        hyperparams_path = ADVANCED_DIR / "distilbert_hyperparams.pkl"
        distilbert_results_dir = ADVANCED_DIR / "distilbert_results"

        distilbert_model_path.parent.mkdir(parents=True, exist_ok=True)
        distilbert_results_dir.mkdir(parents=True, exist_ok=True)

        with open(distilbert_model_path, "wb") as f:
            f.write(b"")
        with open(metrics_path, "w") as f:
            f.write('{"val_macro_f1": null}')
        with open(hyperparams_path, "wb") as f:
            f.write(b"")

        logger.info("Created placeholder artifacts → DVC continuity ensured.")

    def fine_tune(self):
        """
        Orchestrate the hyperparameter tuning and model saving process.

        Controls CUDA availability checks, Optuna study execution, and best trial
        logging into MLflow. Falls back to placeholders if DL is ignored.
        """
        if not self.config.enable:
            logger.info("i DistilBERT training is disabled in params.yaml.")
            self.create_placeholder_artifacts()
            return

        if not torch or not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available. DistilBERT training skipped to avoid long CPU times.")
            self.create_placeholder_artifacts()
            return

        logger.info("✅ CUDA available. Proceeding with DistilBERT fine-tuning.")

        study = optuna.create_study(direction="maximize")

        logger.info(f"🏃 Running {self.config.n_trials} trials...")
        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True,
        )

        best_f1 = study.best_value
        best_params = study.best_params

        logger.info(f"✅ Tuning complete | Best F1: {best_f1:.4f} | Best parameters: {best_params} ✅")

        with mlflow.start_run(run_name="DistilBERT_Best_Run"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_macro_f1", best_f1)
            mlflow.set_tag("best_trial_number", study.best_trial.number)

        save_hyperparams_bundle("distilbert", best_params, best_f1)
        save_metrics_json("distilbert", best_f1)

        logger.info(f"💾 Best DistilBERT trial ({study.best_trial.number}) logged. Artifacts saved.")
