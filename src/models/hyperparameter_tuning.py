"""
Centralized Hyperparameter Optimization Script using Optuna (ConfigurationManager-Aware).

This script optimizes hyperparameters for LightGBM and XGBoost using Optuna.
It strictly adheres to MLOps best practices:
- **Reproducibility:** Configuration is loaded solely from `params.yaml` via `ConfigurationManager`.
- **Tracking:** All trials, parameters, and metrics are logged to MLflow.
- **Artifacts:** The best model, parameters, and metrics are saved for DVC tracking.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.models.hyperparameter_tuning --model lightgbm
    uv run python -m src.models.hyperparameter_tuning --model xgboost

Requirements:
    - Processed features in models/features/.
    - Parameters defined in params.yaml under `train.hyperparameter_tuning`.
    - MLflow server must be running (e.g., uv run python -m mlflow server --host 127.0.0.1 --port 5000).
"""

import argparse
from typing import Callable
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score

# --- Project Utilities ---
from src.config.manager import ConfigurationManager
from src.models.helpers.data_loader import apply_adasyn, load_feature_data
from src.models.helpers.mlflow_tracking_utils import setup_experiment
from src.models.helpers.train_utils import (
    save_hyperparams_bundle,
    save_metrics_json,
    save_model_object,
)
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri

logger = get_logger(__name__, headline="hyperparameter_tuning.py")


def get_objective(model_name: str) -> Callable[[optuna.Trial], float]:
    """Returns the appropriate objective function for the specified model."""
    if model_name == "lightgbm":
        return lightgbm_objective
    elif model_name == "xgboost":
        return xgboost_objective
    else:
        raise ValueError(f"Model '{model_name}' is not supported for tuning.")


def lightgbm_objective(trial: optuna.Trial) -> float:
    """Optuna objective function for LightGBM."""
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

    # Load engineered features and apply ADASYN
    # Placeholders to unpack 7 values returned by load_feature_data()
    logger.info(f"🧪 Trial {trial.number} started... 🧪")
    X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
    X_res, y_res = apply_adasyn(X_train, y_train)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")

    with mlflow.start_run(run_name=f"LightGBM_Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_macro_f1", f1)

    return f1


def xgboost_objective(trial: optuna.Trial) -> float:
    """Optuna objective function for XGBoost."""
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        # XGBoost requires verbosity control differently than LightGBM if needed
        "verbosity": 0,
    }

    # Load engineered features and apply ADASYN
    # Placeholders to unpack 7 values returned by load_feature_data()
    logger.info(f"🧪 Trial {trial.number} started... 🧪")
    X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
    X_res, y_res = apply_adasyn(X_train, y_train)

    # Use basic xgb.train for more control if mostly using DMatrix
    dtrain = xgb.DMatrix(X_res, label=y_res)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Note: xgb.train uses num_boost_round instead of n_estimators
    bst_params = params.copy()
    n_estimators = bst_params.pop("n_estimators")

    model = xgb.train(bst_params, dtrain, num_boost_round=n_estimators)

    y_pred_proba = model.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")

    with mlflow.start_run(run_name=f"XGBoost_Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_macro_f1", f1)

    return f1


def hyperparameter_optimization(model_name: str, n_trials: int):
    """Run Optuna study for the specified model."""
    objective_func = get_objective(model_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value

    logger.info(
        f"✅ Optuna tuning complete for {model_name} | Best F1: {best_score:.4f} | Best trial: {study.best_trial.number} ✅"
    )

    # Log best parameters and metrics to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_val_macro_f1", best_score)

    return best_params, best_score


def retrain_and_save_model(model_name: str, best_params: dict):
    """Retrain appropriate model with best params and save artifacts."""
    logger.info(f"🏆 Retraining best {model_name} model for consistency... 🏆")
    X_train, _, _, y_train, _, _, _ = load_feature_data()
    X_res, y_res = apply_adasyn(X_train, y_train)

    # Initialize best_model to None
    best_model = None

    if model_name == "lightgbm":
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_res, y_res)
        mlflow.lightgbm.log_model(best_model, artifact_path=f"best_{model_name}_model")
    elif model_name == "xgboost":
        # Merge static params with best_params
        static_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "verbosity": 0.0,
        }
        final_params = {**static_params, **best_params}

        # Handle n_estimators -> num_boost_round
        num_boost_round = final_params.pop("n_estimators", 100)

        dtrain = xgb.DMatrix(X_res, label=y_res)
        best_model = xgb.train(final_params, dtrain, num_boost_round=num_boost_round)
        mlflow.xgboost.log_model(best_model, artifact_path=f"best_{model_name}_model")

    if best_model:
        save_model_object(best_model, model_name)


def main() -> None:
    """Run hyperparameter tuning using ConfigurationManager."""

    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lightgbm", "xgboost"],
        help="The model to tune.",
    )
    args = parser.parse_args()
    model_name = args.model

    logger.info(f"🚀 Starting hyperparameter tuning for {model_name.upper()}... 🚀")

    # --- 2. Configuration Loading ---
    config_manager = ConfigurationManager()
    train_config = config_manager.get_train_config()

    # Retrieve model-specific tuning config
    n_trials = 20  # Default
    if model_name == "lightgbm":
        n_trials = train_config.hyperparameter_tuning.lightgbm.n_trials
    elif model_name == "xgboost":
        n_trials = train_config.hyperparameter_tuning.xgboost.n_trials
    else:
        # Should be caught by argparse, but for safety
        raise ValueError(f"Unknown model: {model_name}")

    logger.info(f"Running {n_trials} trials for {model_name}...")

    # --- 3. MLflow Setup ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment(f"Hyperparameter Tuning - {model_name.upper()}", mlflow_uri)

    # --- 4. Optimization Loop ---
    with mlflow.start_run(run_name=f"{model_name.upper()}_Optuna_Study"):
        best_params, best_score = hyperparameter_optimization(model_name, n_trials)

        # --- 5. Retrain & Save Artifacts ---
        retrain_and_save_model(model_name, best_params)

        # Save params and metrics for DVC
        save_hyperparams_bundle(model_name, best_params, best_score)
        save_metrics_json(model_name, best_score)

        logger.info(
            f"🎯 Best {model_name.upper()} run logged to MLflow | "
            f"Run ID: {mlflow.active_run().info.run_id} 🎯"
        )


if __name__ == "__main__":
    main()
