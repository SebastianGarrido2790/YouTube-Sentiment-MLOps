# Trains advanced models (XGBoost, LightGBM, BERT) with Optuna tuning and MLflow logging.
# Uses undersampling for imbalance; selects best via val macro F1.
# Run: uv run python src/models/advanced_training.py

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.transformers
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from imblearn.under_sampling import RandomUnderSampler
import optuna
import xgboost as xgb
import lightgbm as lgb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import os
import pickle

# Set up MLflow
# mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Advanced Models - Optuna Tuning")


def undersample_features(X_train, y_train):
    """Apply undersampling."""
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X_train, y_train)


def xgboost_objective(trial):
    """Optuna objective for XGBoost."""
    mlflow.end_run()  # End any prior run to avoid nesting conflicts
    with mlflow.start_run(nested=True):
        # Tags and description for trial runs
        mlflow.set_tag("mlflow.runName", f"XGBoost_Trial_{trial.number}")
        mlflow.set_tag("experiment_type", "advanced_tuning")
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag(
            "description",
            "Optuna trial for XGBoost hyperparameter tuning on TF-IDF features with undersampling",
        )

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        mlflow.log_params(params)

        # Load and prepare data (TF-IDF)
        X_train = load_npz("../../models/features/X_train.npz").tocsr()
        X_val = load_npz("../../models/features/X_val.npz").tocsr()
        y_train = np.load("../../models/features/y_train.npy")
        y_val = np.load("../../models/features/y_val.npy")

        X_train_us, y_train_us = undersample_features(X_train, y_train)

        dtrain = xgb.DMatrix(X_train_us, label=y_train_us)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, "val")],
            verbose_eval=False,
        )

        y_pred_proba = model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)
        f1 = f1_score(y_val, y_pred, average="macro")
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.xgboost.log_model(model, "xgboost_model")

        return f1


def lightgbm_objective(trial):
    """Optuna objective for LightGBM."""
    mlflow.end_run()  # End any prior run to avoid nesting conflicts
    with mlflow.start_run(nested=True):
        # Tags and description for trial runs
        mlflow.set_tag("mlflow.runName", f"LightGBM_Trial_{trial.number}")
        mlflow.set_tag("experiment_type", "advanced_tuning")
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag(
            "description",
            "Optuna trial for LightGBM hyperparameter tuning on TF-IDF features with undersampling",
        )

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        mlflow.log_params(params)

        # Load data
        X_train = load_npz("../../models/features/X_train.npz").tocsr()
        X_val = load_npz("../../models/features/X_val.npz").tocsr()
        y_train = np.load("../../models/features/y_train.npy")
        y_val = np.load("../../models/features/y_val.npy")

        X_train_us, y_train_us = undersample_features(X_train, y_train)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_us, y_train_us)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro")
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.lightgbm.log_model(model, "lightgbm_model")

        return f1


def bert_objective(trial):
    """Optuna objective for BERT fine-tuning."""
    mlflow.end_run()  # End any prior run to avoid nesting conflicts
    with mlflow.start_run(nested=True):
        # Tags and description for trial runs
        mlflow.set_tag("mlflow.runName", f"BERT_Trial_{trial.number}")
        mlflow.set_tag("experiment_type", "advanced_tuning")
        mlflow.set_tag("model_type", "BERT")
        mlflow.set_tag(
            "description",
            "Optuna trial for BERT fine-tuning on raw text with undersampling",
        )

        # Load raw text for BERT
        train_df = pd.read_parquet("data/processed/train.parquet")
        val_df = pd.read_parquet("data/processed/val.parquet")

        # Undersample train (use default include_groups=True to retain 'category')
        train_df_us = train_df.groupby("category", group_keys=False).apply(
            lambda x: x.sample(min(len(x), len(train_df) // 3), random_state=42)
        )

        # Rename 'category' to 'labels' for Trainer compatibility
        # AND SHIFT LABELS TO BE 0, 1, 2
        train_df_us = train_df_us.rename(columns={"category": "labels"})
        train_df_us["labels"] = train_df_us["labels"] + 1  # <-- NEW: Shifted labels

        val_df = val_df.rename(columns={"category": "labels"})
        val_df["labels"] = val_df["labels"] + 1  # <-- NEW: Shifted labels

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize(batch):
            tokenized = tokenizer(
                batch["clean_comment"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            tokenized["labels"] = batch["labels"]  # Ensure labels are included
            return tokenized

        train_dataset = Dataset.from_pandas(
            train_df_us[["clean_comment", "labels"]]
        ).map(tokenize, batched=True)
        val_dataset = Dataset.from_pandas(val_df[["clean_comment", "labels"]]).map(
            tokenize, batched=True
        )

        num_labels = 3
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir="./models/advanced/bert_results",
            num_train_epochs=trial.suggest_int("num_epochs", 2, 5),
            per_device_train_batch_size=trial.suggest_categorical(
                "batch_size", [8, 16, 32]
            ),
            learning_rate=trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            warmup_steps=500,
            weight_decay=trial.suggest_float("weight_decay", 0.001, 0.1),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            # pin_memory=False,  # Suppress warning if no GPU
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"macro_f1": f1_score(labels, predictions, average="macro")}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,  # Fixed: deprecated 'tokenizer'
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        f1 = eval_results["eval_macro_f1"]
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.transformers.log_model(
            trainer.model, "bert_model", task="text-classification"
        )

        return f1


if __name__ == "__main__":
    os.makedirs("models/advanced", exist_ok=True)

    # Run Optuna for each model (50 trials)
    for model_name, objective in [
        ("xgboost", xgboost_objective),
        ("lightgbm", lightgbm_objective),
        # ("bert", bert_objective),
    ]:
        mlflow.end_run()  # Ensure clean start
        with mlflow.start_run():
            # Tags and description for outer run
            mlflow.set_tag("mlflow.runName", f"{model_name}_Optuna_Study")
            mlflow.set_tag("experiment_type", "advanced_tuning")
            mlflow.set_tag("model_type", model_name.upper())
            mlflow.set_tag(
                "description",
                f"Optuna hyperparameter study for {model_name} on sentiment data with undersampling and TF-IDF/BERT features",
            )

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)

            best_params = study.best_params
            best_f1 = study.best_value

            mlflow.log_param("best_params", str(best_params))
            mlflow.log_metric("best_val_macro_f1", best_f1)

            # Save best params
            with open(f"models/advanced/{model_name}_best.pkl", "wb") as f:
                pickle.dump(best_params, f)

        print(f"{model_name.upper()} Best: F1 {best_f1:.3f}, Params: {best_params}")

    # Model selection: Compare in MLflow UI; programmatically pick highest F1
    print("Run complete. Review MLflow for best model selection via CV.")
