"""
Tune TF-IDF max_features for sentiment feature engineering (unigrams).

Loads processed data, varies max_features, trains RandomForest baselines, logs to MLflow,
and saves artifacts/models using reusable helper functions.

Usage:
    uv run python -m src.features.tfidf_max_features --max_features_values '[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]'

Requirements:
    - Processed data in data/processed/.
    - uv sync (for scikit-learn, mlflow).
    - MLflow server running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design Considerations:
- Reliability: Input validation, consistent splits.
- Scalability: Sparse TF-IDF matrices.
- Maintainability: Logging, type hints, relative paths.
- Adaptability: Parameterized via args/params.yaml; extensible to other n-grams.
"""

import argparse
from typing import Tuple, Dict, Any
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix  # For sparse matrix type hint

# --- Project Utilities ---
from src.utils.paths import TFIDF_FIGURES_DIR
from src.utils.logger import get_logger
from src.features.helpers.feature_utils import (
    setup_mlflow_run,
    load_train_val_data,
    parse_dvc_param,
    evaluate_and_log,
)

# --- Logging Setup ---
logger = get_logger(__name__, headline="tfidf_max_features.py")


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
            "ngram_range": ngram_range,
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
    """Parse args and run experiments."""
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF max_features using RandomForest baseline with MLflow tracking."
    )
    parser.add_argument(
        "--max_features_values",
        type=str,
        default="[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]",
        help="Max features values as string list.",
    )
    parser.add_argument(
        "--ngram_range", type=str, default="(1,1)", help="N-gram range as string tuple."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="RF n_estimators."
    )
    parser.add_argument("--max_depth", type=int, default=15, help="RF max_depth.")
    args = parser.parse_args()

    # --- MLflow Setup ---
    setup_mlflow_run(experiment_name="Exp - TFIDF Max Features")

    # --- Parameter Parsing ---
    max_features_values = parse_dvc_param(
        args.max_features_values, name="max_features_values", expected_type=list
    )
    ngram_range = parse_dvc_param(
        args.ngram_range, name="ngram_range", expected_type=tuple
    )

    logger.info(
        f"--- Running TF-IDF tuning for max_features: {max_features_values} with n-gram {ngram_range} ---"
    )

    for max_features in max_features_values:
        run_max_features_experiment(
            max_features=max_features,
            ngram_range=ngram_range,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    logger.info("--- Max features tuning complete. Analyze results in MLflow UI ---")


if __name__ == "__main__":
    main()
