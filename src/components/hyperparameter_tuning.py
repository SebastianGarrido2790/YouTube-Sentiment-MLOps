"""
Component for hyperparameter tuning using Optuna.

This module provides the worker component responsible for tuning gradient boosting
models (LightGBM and XGBoost) using Optuna, logging parameter trials to MLflow,
and saving the optimal model schemas to disk.
"""

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score

from src.entity.config_entity import TrainConfig
from src.utils.data_loader import apply_adasyn, load_feature_data
from src.utils.logger import get_logger
from src.utils.train_utils import (
    save_hyperparams_bundle,
    save_metrics_json,
    save_model_object,
)

logger = get_logger(__name__, headline="hyperparameter_tuning_component")


class HyperparameterTuning:
    """
    Component handling the Optuna search and retraining for gradient-boosted models.

    Attributes:
        config (TrainConfig): Global training configuration parameters.
    """

    def __init__(self, config: TrainConfig):
        """
        Initialize the HyperparameterTuning component.

        Args:
            config (TrainConfig): Global training config specifying trial limits.
        """
        self.config = config

    def get_objective(self, model_name: str):
        """
        Retrieve the appropriate Optuna objective function based on the model name.

        Args:
            model_name (str): The name of the model ('lightgbm' or 'xgboost').

        Returns:
            Callable: The Optuna objective function.
        """
        if model_name == "lightgbm":
            return self.lightgbm_objective
        elif model_name == "xgboost":
            return self.xgboost_objective
        else:
            raise ValueError(f"Model '{model_name}' is not supported for tuning.")

    def lightgbm_objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for LightGBM.

        Args:
            trial (optuna.Trial): The Optuna trial instance.

        Returns:
            float: The validation macro F1 score.
        """
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

    def xgboost_objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for XGBoost.

        Args:
            trial (optuna.Trial): The Optuna trial instance.

        Returns:
            float: The validation macro F1 score.
        """
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }

        logger.info(f"🧪 Trial {trial.number} started... 🧪")
        X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
        X_res, y_res = apply_adasyn(X_train, y_train)

        dtrain = xgb.DMatrix(X_res, label=y_res)
        dval = xgb.DMatrix(X_val, label=y_val)

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

    def hyperparameter_optimization(self, model_name: str, n_trials: int):
        """
        Execute the Optuna hyperparameter optimization study.

        Args:
            model_name (str): Model name to tune ('lightgbm' or 'xgboost').
            n_trials (int): Number of Optuna trials to run.

        Returns:
            tuple: A tuple containing (best_params, best_score).
        """
        objective_func = self.get_objective(model_name)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(
            f"✅ Optuna tuning complete for {model_name} | Best F1: {best_score:.4f} | "
            f"Best trial: {study.best_trial.number} ✅"
        )

        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_macro_f1", best_score)

        return best_params, best_score

    def retrain_and_save_model(self, model_name: str, best_params: dict):
        """
        Retrain the best model with optimal parameters and save it locally & to MLflow.

        Args:
            model_name (str): Model name ('lightgbm' or 'xgboost').
            best_params (dict): The optimal parameters discovered by Optuna.
        """
        logger.info(f"🏆 Retraining best {model_name} model for consistency... 🏆")
        X_train, _, _, y_train, _, _, _ = load_feature_data()
        X_res, y_res = apply_adasyn(X_train, y_train)

        best_model = None

        if model_name == "lightgbm":
            best_model = lgb.LGBMClassifier(**best_params)
            best_model.fit(X_res, y_res)
            mlflow.lightgbm.log_model(best_model, artifact_path=f"best_{model_name}_model")
        elif model_name == "xgboost":
            static_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": 42,
                "verbosity": 0,
            }
            final_params = {**static_params, **best_params}

            num_boost_round = final_params.pop("n_estimators", 100)

            dtrain = xgb.DMatrix(X_res, label=y_res)
            best_model = xgb.train(final_params, dtrain, num_boost_round=num_boost_round)
            mlflow.xgboost.log_model(best_model, artifact_path=f"best_{model_name}_model")

        if best_model:
            save_model_object(best_model, model_name)

    def tune_model(self, model_name: str):
        """
        Main orchestration method for tuning, retraining, and saving.

        Args:
            model_name (str): The name of the model to tune ('lightgbm' or 'xgboost').
        """
        n_trials = 20
        if model_name == "lightgbm":
            n_trials = self.config.hyperparameter_tuning.lightgbm.n_trials
        elif model_name == "xgboost":
            n_trials = self.config.hyperparameter_tuning.xgboost.n_trials
        else:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Running {n_trials} trials for {model_name}...")

        with mlflow.start_run(run_name=f"{model_name.upper()}_Optuna_Study"):
            best_params, best_score = self.hyperparameter_optimization(model_name, n_trials)
            self.retrain_and_save_model(model_name, best_params)

            save_hyperparams_bundle(model_name, best_params, best_score)
            save_metrics_json(model_name, best_score)

            logger.info(
                f"🎯 Best {model_name.upper()} run logged to MLflow | Run ID: {mlflow.active_run().info.run_id} 🎯"
            )
