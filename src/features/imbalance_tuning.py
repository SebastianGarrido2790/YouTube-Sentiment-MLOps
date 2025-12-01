"""
Tune imbalance handling techniques using DVC parameters.

Applies techniques (SMOTE, ADASYN, etc.) to TF-IDF features on the train set,
trains a RandomForest classifier, and logs results to MLflow. Parameters are
managed by DVC.

Usage (DVC - preferred):
    uv run dvc repro               # Uses params.yaml → fully reproducible
    Run specific pipeline stage:
    uv run dvc repro imbalance_tuning

Usage (local cli override only):
    uv run python -m src.features.imbalance_tuning --rf_n_estimators 100

Requirements:
    - Parameters defined in params.yaml under `imbalance_tuning`.
    - Processed data available in data/processed/.
    - `uv sync` must be run for all dependencies.
    - MLflow server must be running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design:
    - Parameters are read from params.yaml via dvc.api (single source of truth).
    - CLI arguments are optional and only for quick local testing overrides.
    - Reproducibility is prioritized by warning users about CLI overrides.
"""

import argparse
from typing import Any, Dict, Tuple, Union

import dvc.api
import mlflow
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import spmatrix  # For sparse matrix type hint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Project Utilities ---
from src.features.helpers.feature_utils import (
    evaluate_and_log,
    load_train_val_data,
    parse_dvc_param,
)
from src.models.helpers.mlflow_tracking_utils import setup_experiment
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import IMBALANCE_FIGURES_DIR

# --- Logging Setup ---
logger = get_logger(__name__, headline="imbalance_tuning.py")


def load_params() -> Dict[str, Any]:
    """
    Load imbalance tuning parameters from params.yaml using DVC.
    Falls back gracefully if running outside a DVC pipeline.
    """
    try:
        logger.info("Loading params via dvc.api")
        params = dvc.api.params_show()
        return params["imbalance_tuning"]
    except Exception as e:
        logger.warning(f"Could not load params via dvc.api: {e}")
        logger.warning("Falling back to script defaults (only for local debugging).")
        return {
            "imbalance_methods": "['class_weights','oversampling','adasyn','undersampling','smote_enn']",
            "best_ngram_range": "(1,1)",
            "best_max_features": 1000,
            "rf_n_estimators": 200,
            "rf_max_depth": 15,
        }


def run_imbalanced_experiment(
    imbalance_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    n_estimators: int,
    max_depth: int,
) -> None:
    """
    Run experiment for specified imbalance handling method.

    Args:
        imbalance_method: Technique name ('class_weights', 'oversampling', 'adasyn', 'undersampling', 'smote_enn').
        ngram_range: N-gram range tuple.
        max_features: Maximum TF-IDF features.
        n_estimators: RF trees.
        max_depth: RF depth.
    """

    # --- Load data ---
    # NOTE: load_train_val_data loads train.parquet and val.parquet for hyperparameter tuning.
    train_df, val_df = load_train_val_data()

    X_train_text = train_df["clean_comment"].tolist()
    # Assuming 'category_encoded' is the numerical target variable (0, 1, 2)
    y_train = train_df["category_encoded"].values
    X_val_text = val_df["clean_comment"].tolist()
    y_val = val_df["category_encoded"].values

    logger.info(
        f"Data split: Train {train_df.shape[0]} ({np.bincount(y_train)}), Val {val_df.shape[0]}"
    )

    # --- Vectorization using TF-IDF (Fit on training, transform both) ---
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        lowercase=False,
        min_df=2,
    )
    X_train_vec: spmatrix = vectorizer.fit_transform(X_train_text)
    X_val_vec: spmatrix = vectorizer.transform(X_val_text)
    feature_dim = X_train_vec.shape[1]

    # --- Handle class imbalance (train set only) ---
    class_weight: Union[str, None] = None
    resampling_applied: bool = False

    if imbalance_method == "class_weights":
        class_weight = "balanced"
        logger.info("Using 'balanced' class weights. No resampling applied.")
    else:
        sampler = None
        if imbalance_method == "oversampling":
            sampler = SMOTE(random_state=42)
        elif imbalance_method == "adasyn":
            sampler = ADASYN(random_state=42)
        elif imbalance_method == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        elif imbalance_method == "smote_enn":
            sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42))
        else:
            raise ValueError(f"Unsupported imbalance method: {imbalance_method}")

        if sampler:
            resampling_applied = True
            logger.info(f"Applying {imbalance_method.upper()} to training data...")
            X_train_vec, y_train = sampler.fit_resample(X_train_vec, y_train)
            logger.info(
                f"New training sample size: {X_train_vec.shape[0]} (Classes: {np.bincount(y_train)})"
            )  # np.bincount for sanity check: count of samples per class after resampling

    # --- MLflow Tracking, Training, and Evaluation ---
    run_name = f"Imb_{imbalance_method}_Feat_{feature_dim}"
    logger.info(f"🚀 Running experiment: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # 1. Model Training
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        )
        model.fit(X_train_vec, y_train)

        # 2. Define Params and Tags for Logging
        params: Dict[str, Any] = {
            "vectorizer_type": "TF-IDF",
            "ngram_range": str(ngram_range),  # Log as string for consistency
            "max_features": max_features,
            "feature_dim": feature_dim,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "imbalance_method": imbalance_method,
            "class_weight_param": class_weight,
            "resampling_applied": resampling_applied,
        }
        tags: Dict[str, str] = {
            "experiment_type": "imbalance_tuning",
            "model_type": "RandomForestClassifier",
            "imbalance_method": imbalance_method,
        }

        # 3. Evaluation and Logging
        metrics = evaluate_and_log(
            model=model,
            X_val=X_val_vec,
            y_val=y_val,
            run_name=run_name,
            params=params,
            tags=tags,
            output_dir=IMBALANCE_FIGURES_DIR,
            log_model=True,  # Log the model to the MLflow registry
        )

        # 4. Log key metric to console
        # Use .get() to safely retrieve the metric, defaulting to 0.0 if the key is missing.
        val_accuracy = metrics.get("val_accuracy", 0.0)
        f1_score_1 = metrics.get("1_f1-score", 0.0)

        logger.info(f"Model Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Class 1 F1-Score: {f1_score_1:.4f}")

        logger.info(
            f"✅ Experiment finished: {run_name} | MLflow Run ID: {mlflow.last_active_run().info.run_id}"
        )


def main() -> None:
    """Parse args and run experiments, using DVC params as source of truth."""
    # --- DVC/CLI Parameter Loading ---
    params = load_params()
    parser = argparse.ArgumentParser(
        description="Tune imbalance handling methods. Params from params.yaml by default."
    )
    # Define arguments for optional CLI overrides, using names from params.yaml
    parser.add_argument(
        "--imbalance_methods",
        type=str,
        required=False,
        help="Override imbalance_methods from params.yaml.",
    )
    parser.add_argument(
        "--best_ngram_range",
        type=str,
        required=False,
        help="Override best_ngram_range from params.yaml.",
    )
    parser.add_argument(
        "--best_max_features",
        type=int,
        required=False,
        help="Override best_max_features from params.yaml.",
    )
    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        required=False,
        help="Override rf_n_estimators from params.yaml.",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        required=False,
        help="Override rf_max_depth from params.yaml.",
    )
    args = parser.parse_args()

    # --- Consolidate Parameters (CLI overrides DVC) ---
    final_params = {}
    overridden_keys = []
    for key, default_val in params.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            final_params[key] = cli_val
            overridden_keys.append(key)
        else:
            final_params[key] = default_val

    if overridden_keys:
        logger.warning(
            "CLI overrides detected for: %s. This run may not be reproducible with 'dvc repro'.",
            ", ".join(overridden_keys),
        )

    # --- MLflow Setup ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment("Exp - Imbalance Handling", mlflow_uri)

    # --- Parameter Parsing ---
    imbalance_methods = parse_dvc_param(
        final_params["imbalance_methods"], name="imbalance_methods", expected_type=list
    )
    ngram_range = parse_dvc_param(
        final_params["best_ngram_range"], name="best_ngram_range", expected_type=tuple
    )

    logger.info(
        f"--- Running imbalance experiments for methods: {imbalance_methods} ---"
    )
    for method in imbalance_methods:
        run_imbalanced_experiment(
            imbalance_method=method,
            ngram_range=ngram_range,
            max_features=final_params["best_max_features"],
            n_estimators=final_params["rf_n_estimators"],
            max_depth=final_params["rf_max_depth"],
        )
    logger.info(
        "--- Imbalance handling tuning complete. Analyze results in MLflow UI ---"
    )


if __name__ == "__main__":
    main()
