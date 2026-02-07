"""
Tune TF-IDF max_features using MLOps best practices.

Loads processed data, iterates through a list of `max_features` values defined in
`params.yaml` (via ConfigurationManager), trains a RandomForest baseline for each,
and logs results to MLflow.

Usage:
Run the entire pipeline:
    uv run dvc repro               # Uses params.yaml → fully reproducible
Run specific pipeline stage:
    uv run python -m src.features.tfidf_max_features

Requirements:
    - Parameters defined in params.yaml under `feature_tuning`.
    - Processed data available in data/processed/.
    - MLflow server must be running (e.g., uv run python -m mlflow server --host 127.0.0.1 --port 5000).
"""

from typing import Any, Dict, Tuple
import mlflow
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Project Utilities ---
from src.config.manager import ConfigurationManager
from src.config.schemas import FeatureTuningConfig
from src.features.helpers.feature_utils import (
    evaluate_and_log,
    load_train_val_data,
)
from src.models.helpers.mlflow_tracking_utils import setup_experiment
from src.utils.logger import get_logger
from src.utils.paths import TFIDF_FIGURES_DIR


# --- Logging Setup ---
logger = get_logger(__name__, headline="tfidf_max_features.py")


def load_params() -> FeatureTuningConfig:
    """
    Load TF-IDF tuning parameters from params.yaml using ConfigurationManager.
    """
    try:
        logger.info("Loading params via ConfigurationManager")
        return ConfigurationManager().get_feature_tuning_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise e


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
    logger.info(f"🏁 Running experiment: {run_name}")

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
            f"Experiment finished: {run_name} | MLflow Run ID: {mlflow.last_active_run().info.run_id}"
        )


def main() -> None:
    """Run experiments using ConfigurationManager users as source of truth."""
    logger.info("🚀 Starting TF-IDF Max Features Tuning 🚀")

    # --- Parameter Loading ---
    config = load_params()

    # --- MLflow Setup ---
    from src.utils.mlflow_config import get_mlflow_uri

    mlflow_uri = get_mlflow_uri()
    setup_experiment("Exp - TFIDF Max Features", mlflow_uri)

    logger.info(
        f"Running TF-IDF tuning for max_features: {config.max_features_values} with n-gram {config.best_ngram_range}"
    )

    for max_features in config.max_features_values:
        run_max_features_experiment(
            max_features=max_features,
            ngram_range=tuple(config.best_ngram_range),  # Convert list to tuple
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
        )

    logger.info("✅ Max features tuning complete. Analyze results in MLflow UI ✅")


if __name__ == "__main__":
    main()
