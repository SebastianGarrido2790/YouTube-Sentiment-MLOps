"""
Tune TF-IDF max_features for sentiment feature engineering (unigrams).

Loads processed data, varies max_features, trains RandomForest baselines, logs to MLflow,
and saves visualizations/models.

Usage:
    uv run python -m src.features.tfidf_max_features --max_features_values '[1000,2000]' --ngram_range '(1,1)'

Requirements:
    - Processed data in data/processed/.
    - uv sync (for scikit-learn, mlflow, seaborn).
    - MLflow server running.

Design Considerations:
- Reliability: Input validation, consistent splits.
- Scalability: Sparse TF-IDF matrices.
- Maintainability: Logging, type hints, relative paths.
- Adaptability: Parameterized via args/params.yaml; extensible to other n-grams.
"""

import argparse
import ast  # For safe list/tuple parsing from params
import logging
from typing import Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Import the modular MLflow URI loader
from src.utils.mlflow_utils import get_mlflow_uri

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths relative to project root
ROOT = Path(__file__).parent.parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"
FIGURES_DIR = ROOT / "reports" / "figures" / "tfidf_max_features"
MODELS_DIR = ROOT / "models" / "features" / "tfidf_max_features"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MLflow URI via utility function
mlflow_uri = get_mlflow_uri(params_path=str(ROOT / "params.yaml"))
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Exp - TFIDF max_features")
logger.info(f"Using MLflow Tracking URI: {mlflow_uri}")


def run_experiment_tfidf_max_features(
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
    # Load data
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Processed data missing. Run data_preparation first.")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category"].values
    logger.info(
        f"Loaded train: {len(X_train_text)} samples, test: {len(X_test_text)} samples."
    )

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        # tokenizer=lambda x: x.split(),  # Use pre-cleaned tokens
        lowercase=False,  # Already lowercased
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    feature_dim = X_train.shape[1]

    # Save vectorizer locally
    vectorizer_path = MODELS_DIR / f"tfidf_vectorizer_max_features_{max_features}.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved TF-IDF vectorizer: {vectorizer_path}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # MLflow Tracking
    with mlflow.start_run() as run:
        # Tags
        mlflow.set_tag("mlflow.runName", f"TFIDF_max_features_{max_features}")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RF with TF-IDF, max_features={max_features}, ngram={ngram_range}",
        )
        # Params
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("feature_dim", feature_dim)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        logger.info(f"Model Accuracy: {accuracy:.4f}")

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: TF-IDF, max_features={max_features}")
        plot_path = FIGURES_DIR / f"confusion_matrix_tfidf_max_{max_features}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(str(plot_path))

        # Log model
        model_path = f"random_forest_model_tfidf_max_{max_features}"
        mlflow.sklearn.log_model(model, model_path)

        # Save model locally
        local_model_path = MODELS_DIR / f"{model_path}.pkl"
        # with open(local_model_path, "wb") as f:
        #     pickle.dump(model, f)

        # Log vectorizer artifact
        logger.info(
            f"Experiment finished. MLflow Run ID: {run.info.run_id}; Model saved: {local_model_path}"
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

    # Parse lists/tuples
    max_features_values = ast.literal_eval(args.max_features_values.strip())
    ngram_range = ast.literal_eval(args.ngram_range.strip())

    if not isinstance(max_features_values, list) or not isinstance(ngram_range, tuple):
        raise ValueError(
            "max_features_list must be a list of ints and ngram_range_str must be a tuple of ints."
        )

    logger.info(f"Running TF-IDF tuning for max_features: {max_features_values}")

    for max_features in max_features_values:
        run_experiment_tfidf_max_features(
            max_features=max_features,
            ngram_range=ngram_range,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    logger.info("Tuning complete. View in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
