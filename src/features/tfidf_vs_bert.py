"""
Compare TF-IDF vs. BERT embeddings for sentiment feature engineering.

Loads processed data, generates embeddings, trains RandomForest baselines, logs to MLflow,
and saves visualizations/models. Tracks multiple TF-IDF n-gram variants.

Usage:
    uv run python -m src.features.tfidf_vs_bert --ngram_ranges '[(1,1),(1,2),(1,3)]' --max_features 5000

Requirements:
    - Processed data in data/processed/.
    - uv sync (for scikit-learn, transformers, torch, mlflow, seaborn).
    - MLflow server running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).


Design Considerations:
- Reliability: Input validation, error handling for device/embedding dims.
- Scalability: Batched BERT inference; sparse TF-IDF.
- Maintainability: Logging, type hints, relative paths.
- Adaptability: Parameterized via args/params.yaml; extensible to other embedders.
"""

import argparse
import ast  # For safe list parsing from params
import logging
import os
from pathlib import Path
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

# Conditional import for BERT (minimize overhead for TF-IDF runs)
import torch
from transformers import AutoTokenizer, AutoModel

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
FIGURES_DIR = ROOT / "reports" / "figures" / "tfidf_vs_bert"
MODELS_DIR = ROOT / "models" / "features"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MLflow URI via utility function
mlflow_uri = get_mlflow_uri(params_path=str(ROOT / "params.yaml"))
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Exp - TFIDF vs BERT")
logger.info(f"Using MLflow Tracking URI: {mlflow_uri}")


def get_bert_embeddings(
    texts: list, device: str = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled BERT embeddings for texts.

    Args:
        texts: List of input texts.
        device: Torch device ('cuda' or 'cpu').
        batch_size: Inference batch size.

    Returns:
        np.ndarray: Pooled embeddings (n_samples, 768).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            # Mean pool over non-special tokens
            # This is a common and simple way to get sentence embeddings from the hidden states
            pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def run_experiment(
    vectorizer_type: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    n_estimators: int,
    max_depth: int,
    vectorizer_name: str = "TF-IDF",
    batch_size: int = 32,
) -> None:
    """
    Run experiment for the given vectorizer type using a RandomForest baseline.

    Args:
        vectorizer_type: "TF-IDF" or "BERT".
        ngram_range: N-gram tuple.
        max_features (Optional[int]): Max features for TF-IDF.
        n_estimators: RF trees.
        max_depth: RF depth.
        vectorizer_name: Display name.
        batch_size: BERT batch size.
        mlflow_uri (str): URI for the MLflow tracking server.
    """
    # Load data
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Processed data missing. Run data_preparation first.")

    # Data loading
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category"].values
    logger.info(
        f"Loaded train: {len(X_train_text)} samples, test: {len(X_test_text)} samples."
    )

    # --- Vectorization ---
    if vectorizer_type == "TF-IDF":
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
            # We already cleaned and tokenized text in make_dataset, but TfidfVectorizer
            # often performs better with default tokenization if stopwords/cleaning is minimal.
            # However, since make_dataset.py is thorough, we rely on its output.
            # tokenizer=lambda x: x.split(),  # Use pre-cleaned, space-separated tokens
            lowercase=False,  # Text is already lowercased
            min_df=2,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        feature_dim = X_train.shape[1]
        # Save vectorizer locally
        vectorizer_path = (
            MODELS_DIR / f"tfidf_vectorizer_{vectorizer_name}_{ngram_range}.pkl"
        )
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Saved TF-IDF vectorizer: {vectorizer_path}")

    elif vectorizer_type == "BERT":
        logger.info("Generating BERT embeddings...")
        X_train = get_bert_embeddings(X_train_text, batch_size=batch_size)
        X_test = get_bert_embeddings(X_test_text, batch_size=batch_size)
        feature_dim = X_train.shape[1]  # 768
        vectorizer_path = None  # No serializable tokenizer needed for inference

    else:
        raise ValueError("Unsupported vectorizer_type")

    logger.info(f"Vectorization complete. Feature dimensions: {feature_dim}")

    # --- Train & Evaluate ---
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # --- MLflow Tracking ---
    with mlflow.start_run() as run:
        # Tags
        mlflow.set_tag(
            "mlflow.runName", f"{vectorizer_name}_{ngram_range}_{n_estimators}est"
        )
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RF with {vectorizer_name}, ngram={ngram_range}, features={feature_dim}",
        )
        # Params
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param(
            "max_features", max_features if vectorizer_type == "TF-IDF" else None
        )
        mlflow.log_param("feature_dim", feature_dim)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param(
            "batch_size", batch_size if vectorizer_type == "BERT" else None
        )
        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        logger.info(f"Model Accuracy: {accuracy:.4f}")

        # Classification report (log key metrics)
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Confusion matrix (artifact)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: {vectorizer_name}, {ngram_range}")
        plot_path = (
            FIGURES_DIR / f"confusion_matrix_{vectorizer_name}_{ngram_range}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(str(plot_path))

        # Log model (for later deployment/registration)
        model_path = f"random_forest_model_{vectorizer_name}_{ngram_range}"
        mlflow.sklearn.log_model(model, model_path)

        logger.info(f"Experiment finished. MLflow Run ID: {run.info.run_id}")

        # Save model locally
        local_model_path = MODELS_DIR / f"{model_path}.pkl"
        with open(local_model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(
            f"Logged metrics (accuracy: {accuracy:.4f}) and saved model: {local_model_path}"
        )


def main() -> None:
    """Parse args and run experiments."""
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF and BERT feature engineering with MLflow tracking."
    )
    parser.add_argument(
        "--ngram_ranges",
        type=str,
        default="[(1,1),(1,2),(1,3)]",
        help="N-gram ranges as string list.",
    )
    parser.add_argument(
        "--max_features", type=int, default=5000, help="Max TF-IDF features."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="BERT batch size.")
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="RF n_estimators."
    )
    parser.add_argument("--max_depth", type=int, default=15, help="RF max_depth.")
    args = parser.parse_args()

    # Parse ngram_ranges
    ngram_ranges = ast.literal_eval(args.ngram_ranges.strip())
    if not isinstance(ngram_ranges, list):
        raise ValueError("ngram_ranges must be a list of tuples.")

    # --- TF-IDF experiments with different n-grams (parameterized) ---
    logger.info("--- Starting TF-IDF Experiments ---")
    for ngram_range in ngram_ranges:
        run_experiment(
            "TF-IDF",
            ngram_range,
            args.max_features,
            args.n_estimators,
            args.max_depth,
            "TF-IDF",
            args.batch_size,
        )

    # # --- BERT experiment (single run) ---
    # logger.info("--- Starting BERT Experiment ---")
    # # BERT uses a fixed 768 features for bert-base-uncased
    # run_experiment(
    #     "BERT",
    #     (1, 1),
    #     args.max_features,
    #     args.n_estimators,
    #     args.max_depth,
    #     "BERT",
    #     args.batch_size,
    # )

    logger.info("Experiments complete. View in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
