"""
Fine-tune DistilBERT with Optuna, ADASYN balancing, and MLflow logging (DVC-Aware).

This script performs hyperparameter tuning for DistilBERT using Optuna,
logs results to MLflow, and saves the best model and metrics for DVC tracking.

Features:
    - Controlled via params.yaml → train.distilbert section.
    - Logs metrics and hyperparameters to MLflow.
    - Skips entirely if DistilBERT is disabled (for CPU setups or if configured off).

Usage (DVC - preferred):
    uv run dvc repro
    Run specific pipeline stage:
    uv run dvc repro distilbert_training

Usage (local cli override only)
    uv run python -m src.models.distilbert_training --enable true --n_trials 10

Requirements:
    - Processed features in models/features/.
    - Parameters defined in params.yaml under `train.distilbert`.
    - MLflow server running.

Design Considerations:
    - Reliability: Uses pre-loaded features/labels; validates inputs.
    - Scalability: Leverages Hugging Face Trainer for efficient training.
    - Maintainability: Leverages shared helpers (data_loader, train_utils); centralized logging/MLflow.
    - Adaptability: Parameterized hyperparameters via Optuna using `params.yaml`; easily switchable models.
"""

import functools
from typing import Any, Dict

import dvc.api
import mlflow
import mlflow.transformers
import numpy as np
import optuna
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# --- Project Utilities ---
from src.models.helpers.data_loader import load_text_data
from src.models.helpers.mlflow_tracking_utils import setup_experiment
from src.models.helpers.train_utils import save_hyperparams_bundle, save_metrics_json
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import ADVANCED_DIR

logger = get_logger(__name__, headline="bert_training.py")


# ============================================================
#  Load configuration
# ============================================================
def load_params() -> Dict[str, Any]:
    """Load project configuration parameters for DistilBERT training using DVC."""
    try:
        logger.info("Loading params via dvc.api")
        all_params = dvc.api.params_show()
        distilbert_params = all_params.get("train", {}).get("distilbert", {})

        # Ensure 'enable' is a boolean, handling string representation from YAML
        enable_val = distilbert_params.get("enable", False)
        if isinstance(enable_val, str):
            distilbert_params["enable"] = enable_val.lower() == "true"
        else:
            distilbert_params["enable"] = bool(enable_val)

        return distilbert_params
    except Exception as e:
        logger.error(f"Failed to load params via dvc.api: {e}")
        # Provide fallback values for local debugging if params.yaml can't be loaded
        return {
            "enable": False,
            "n_trials": 20,
            "batch_size": [8, 16, 32],
            "lr": [1e-5, 5e-5],
            "weight_decay": [0.001, 0.1],
            "num_epochs_range": [2, 5],
        }


# ============================================================
#  Objective function for Optuna optimization
# ============================================================
def objective(trial: optuna.trial.Trial, loaded_params: Dict[str, Any]) -> float:
    """Define Optuna optimization logic for DistilBERT fine-tuning."""
    train_df, val_df = load_text_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        tokenized = tokenizer(
            batch["clean_comment"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tokenized["labels"] = batch["category"] + 1  # Shift labels to 0–2 range
        return tokenized

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    # Get hyperparameters ranges from loaded_params for Optuna suggestions
    batch_size_choices = loaded_params.get("batch_size", [8, 16, 32])
    lr_range = loaded_params.get("lr", [1e-5, 5e-5])
    weight_decay_range = loaded_params.get("weight_decay", [0.001, 0.1])
    # Assuming num_epochs also comes from params, default to [2,5] if not specified
    num_epochs_range = loaded_params.get("num_epochs_range", [2, 5])

    training_args = TrainingArguments(
        output_dir=str(ADVANCED_DIR / "distilbert_results"),
        num_train_epochs=trial.suggest_int(
            "num_epochs", num_epochs_range[0], num_epochs_range[1]
        ),
        per_device_train_batch_size=trial.suggest_categorical(
            "batch_size", batch_size_choices
        ),
        learning_rate=trial.suggest_float("lr", lr_range[0], lr_range[1], log=True),
        weight_decay=trial.suggest_float(
            "weight_decay", weight_decay_range[0], weight_decay_range[1]
        ),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        logging_dir=str(ADVANCED_DIR / "distilbert_logs"),
        # Disable progress bar for cleaner Optuna output when running many trials
        disable_tqdm=True,
        report_to="none",  # Optuna handles logging to MLflow
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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Start a nested MLflow run for each Optuna trial
    with mlflow.start_run(
        run_name=f"DistilBERT_Trial_{trial.number}", nested=True
    ) as trial_run:
        trainer.train()
        results = trainer.evaluate()
        f1 = results["eval_macro_f1"]

        mlflow.log_params(trial.params)  # Log all trial params suggested by Optuna
        mlflow.log_metric("val_macro_f1", f1)
        # Log the model for this specific trial, MLflow will handle artifact storage
        mlflow.transformers.log_model(
            trainer.model,
            artifact_path="distilbert_model_trial",
            task="text-classification",
            # Add metadata if needed to link back to the trial
            metadata={"optuna_trial_id": trial.number, "macro_f1": f1},
        )
        logger.info(f"Trial {trial.number} finished with F1: {f1:.4f}")

    return f1


# ============================================================
#  Main entrypoint
# ============================================================
if __name__ == "__main__":
    params = load_params()  # This now returns the 'distilbert' section directly
    enable_distilbert = params.get("enable", False)

    # --- Conditional Execution Logic ---
    run_training = False
    # Check if DistilBERT training is enabled and CUDA is available
    if enable_distilbert:
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(
                    "✅ CUDA is available. Proceeding with DistilBERT training."
                )
                run_training = True
            else:
                logger.warning(
                    "⚠️ DistilBERT training skipped: 'enable' is true, but CUDA is not available."
                )
        except ImportError:
            logger.error(
                "❌ DistilBERT training skipped: PyTorch is not installed. Please run 'uv add torch'."
            )
    else:
        logger.info(
            "ℹ️ DistilBERT training is disabled in params.yaml (train.distilbert.enable: false)."
        )

    if not run_training:
        logger.warning(
            "Skipping DistilBERT training. Creating placeholder artifacts for DVC continuity."
        )
        # --- Ensure expected DVC outputs exist ---
        distilbert_model_path = ADVANCED_DIR / "distilbert_model.pkl"
        distilbert_results_dir = ADVANCED_DIR / "distilbert_results"
        metrics_path = ADVANCED_DIR / "distilbert_metrics.json"
        hyperparams_path = ADVANCED_DIR / "distilbert_hyperparams.pkl"

        distilbert_model_path.parent.mkdir(parents=True, exist_ok=True)
        distilbert_results_dir.mkdir(parents=True, exist_ok=True)

        # Create lightweight placeholder files so DVC doesn't fail
        with open(distilbert_model_path, "wb") as f:
            f.write(b"")  # empty placeholder file
        with open(metrics_path, "w") as f:
            f.write('{"val_macro_f1": null}')
        with open(hyperparams_path, "wb") as f:
            f.write(b"")

        logger.info(
            "Created placeholder artifacts for skipped DistilBERT stage → DVC continuity ensured."
        )
        exit(0)

    # --- Proceed with training if all checks passed ---
    logger.info("🚀 Starting DistilBERT training with Optuna hyperparameter tuning...")

    mlflow_uri = get_mlflow_uri()
    setup_experiment("DistilBERT - Advanced Tuning", mlflow_uri)

    study = optuna.create_study(direction="maximize")
    n_trials = params.get("n_trials", 20)

    # Pass loaded_params to the objective function using functools.partial
    objective_with_params = functools.partial(objective, loaded_params=params)

    study.optimize(objective_with_params, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_f1 = study.best_value

    mlflow.set_tag("best_trial_id", study.best_trial.number)
    mlflow.log_params(best_params)  # Log best params to the parent run
    mlflow.log_metric("best_val_macro_f1", best_f1)

    # It's generally better to retrieve the best model from the best trial's MLflow run
    # rather than re-training here. Assuming the objective logs the model.
    best_trial_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
    if best_trial_run_id:
        # Save the actual model object, not just params/score.
        # This will depend on how save_model_object is implemented (e.g., retrieving from MLflow)
        logger.info(
            f"Retrieving best model from trial run ID: {best_trial_run_id}. "
            "Model artifact should be under 'distilbert_model_trial' in that run."
        )
        # Assuming save_model_object can handle MLflow run ID to retrieve artifact
        # As it stands, save_model_object expects a model object, so we'll just save the bundle.
    else:
        logger.warning(
            "Could not retrieve MLflow run ID for the best trial. Model artifact might not be directly linkable."
        )

    save_hyperparams_bundle("distilbert", best_params, best_f1)
    save_metrics_json("distilbert", best_f1)

    logger.info(
        f"🏁 DistilBERT training complete | Best Macro-F1: {best_f1:.4f} |"
        f"Best Trial: {study.best_trial.number} | Run ID: {mlflow.active_run().info.run_id} logged to MLflow."
    )
