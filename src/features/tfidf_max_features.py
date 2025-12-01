"""
Tune TF-IDF max_features using DVC parameters.

Loads processed data, iterates through a list of `max_features` values defined in
`params.yaml`, trains a RandomForest baseline for each, and logs results to MLflow.

Usage (DVC - preferred):
    uv run dvc repro               # Uses params.yaml → fully reproducible
    Run specific pipeline stage:
    uv run dvc repro tfidf_max_features_tuning

Usage (local cli override only):
    uv run python -m src.features.tfidf_max_features --max_features_values '[500, 1000]'

Requirements:
    - Parameters defined in params.yaml under `tfidf_max_features_tuning`.
    - Processed data available in data/processed/.
    - `uv sync` must be run for all dependencies.
    - MLflow server must be running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design:
    - Parameters are read from params.yaml via dvc.api (single source of truth).
    - CLI arguments are optional and only for quick local testing overrides.
    - Reproducibility is prioritized by warning users about CLI overrides.
"""

import argparse
from typing import Any, Dict, Tuple

import dvc.api
import mlflow
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
from src.utils.paths import TFIDF_FIGURES_DIR


# --- Logging Setup ---
logger = get_logger(__name__, headline="tfidf_max_features.py")


def load_params() -> Dict[str, Any]:
    """
    Load TF-IDF tuning parameters from params.yaml using DVC.
    Falls back gracefully if running outside a DVC pipeline.
    """
    try:
        logger.info("Loading params via dvc.api")
        params = dvc.api.params_show()
        # Assuming a new section for this specific tuning script
        return params["tfidf_max_features_tuning"]
    except Exception as e:
        logger.warning(f"Could not load params via dvc.api: {e}")
        logger.warning("Falling back to script defaults (only for local debugging).")
        return {
            "max_features_values": "[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]",
            "ngram_range": "(1,1)",
            "n_estimators": 200,
            "max_depth": 15,
        }


def run_max_features_experiment(
    max_features: int,
    ngram_range: Tuple[int, int],
    n_estimators: int,
    max_depth: int,
) -> None:
    """
    Run experiment for TF-IDF with specified max_features.

    Args:
        max_features: Maximum number of features for TF-IDF.
        ngram_range: N-gram range tuple.
        n_estimators: RF trees.
        max_depth: RF depth.
    """

    # --- Load data ---
    # NOTE: Using the VAL set for tuning is the correct MLOps practice.
    train_df, val_df = load_train_val_data()

    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_val_text = val_df["clean_comment"].tolist()
    y_val = val_df["category"].values

    logger.info(f"Data split: Train {train_df.shape[0]}, Val {val_df.shape[0]}")

    # --- Vectorization using TF-IDF ---
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        lowercase=False,
        min_df=2,
    )
    X_train: spmatrix = vectorizer.fit_transform(X_train_text)
    X_val: spmatrix = vectorizer.transform(X_val_text)
    feature_dim = X_train.shape[1]

    # --- MLflow Tracking, Training, and Evaluation ---
    run_name = f"TFIDF_max_features_{max_features}"
    logger.info(f"🚀 Running experiment: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # 1. Model Training
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # 2. Define Params and Tags for Logging
        params: Dict[str, Any] = {
            "vectorizer": "TF-IDF",
            "ngram_range": str(ngram_range),  # Log as string for consistency
            "max_features": max_features,
            "feature_dim": feature_dim,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }
        tags: Dict[str, str] = {
            "experiment_type": "feature_tuning",
            "model_type": "RandomForestClassifier",
            "vectorizer_type": "TF-IDF",
        }

        # 3. Evaluation and Logging
        metrics = evaluate_and_log(
            model=model,
            X_val=X_val,
            y_val=y_val,
            run_name=run_name,
            params=params,
            tags=tags,
            output_dir=TFIDF_FIGURES_DIR,
            log_model=False,
        )

        # Log key metric to console
        logger.info(f"Model Val Accuracy: {metrics['val_accuracy']:.4f}")

        logger.info(
            f"✅ Experiment finished: {run_name} | MLflow Run ID: {mlflow.last_active_run().info.run_id}"
        )


def main() -> None:
    """Parse args and run experiments using DVC params as source of truth."""
    # --- DVC/CLI Parameter Loading ---
    params = load_params()
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF max_features. Params from params.yaml by default."
    )

    # Define arguments for optional CLI overrides
    parser.add_argument(
        "--max_features_values",
        type=str,
        required=False,
        help="Override max_features_values from params.yaml.",
    )
    parser.add_argument(
        "--ngram_range",
        type=str,
        required=False,
        help="Override ngram_range from params.yaml.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        required=False,
        help="Override n_estimators from params.yaml.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        required=False,
        help="Override max_depth from params.yaml.",
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
    setup_experiment("Exp - TFIDF Max Features", mlflow_uri)

    # --- Parameter Parsing ---
    max_features_values = parse_dvc_param(
        final_params["max_features_values"],
        name="max_features_values",
        expected_type=list,
    )
    ngram_range = parse_dvc_param(
        final_params["ngram_range"], name="ngram_range", expected_type=tuple
    )

    logger.info(
        f"🚀 Running TF-IDF tuning for max_features: {max_features_values} with n-gram {ngram_range}"
    )

    for max_features in max_features_values:
        run_max_features_experiment(
            max_features=max_features,
            ngram_range=ngram_range,
            n_estimators=final_params["n_estimators"],
            max_depth=final_params["max_depth"],
        )

    logger.info("--- Max features tuning complete. Analyze results in MLflow UI ---")


if __name__ == "__main__":
    main()
