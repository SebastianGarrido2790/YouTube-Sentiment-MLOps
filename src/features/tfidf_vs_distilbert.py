"""
Compare TF-IDF vs. DistilBERT embeddings using DVC parameters.

Loads processed data, generates embeddings using TF-IDF and (optionally) DistilBERT,
trains RandomForest baselines, and logs all results to MLflow for comparison.
This script is designed to be run as part of a DVC pipeline.

The feature_comparison stage should not have an `outs` section in dvc.yaml,
as its primary purpose is to log experiment metrics to MLflow, not to
produce versioned artifacts.

Usage (DVC - preferred):
    uv run dvc repro               # Uses params.yaml → fully reproducible
    Run specific pipeline stage:
    uv run dvc repro feature_comparison

Usage (local cli override only):
    uv run python -m src.features.tfidf_vs_distilbert --max_features 1000

Requirements:
    - Parameters defined in params.yaml under `feature_comparison`.
    - Processed data available in data/processed/.
    - `uv sync` must be run for all dependencies.
    - MLflow server must be running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design:
    - Parameters are read from params.yaml via dvc.api (single source of truth).
    - CLI arguments are optional and only for quick local testing overrides.
    - Reproducibility is prioritized by warning users about CLI overrides.
"""

import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import dvc.api
import mlflow
import numpy as np
from scipy.sparse import spmatrix  # For sparse matrix type hint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Project Utilities ---
from src.features.helpers.feature_utils import (
    evaluate_and_log,
    load_train_val_data,
    parse_dvc_param,
    str2bool,
)
from src.models.helpers.mlflow_tracking_utils import setup_experiment
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri

# --- Logging Setup ---
logger = get_logger(__name__, headline="tfidf_vs_distilbert.py")


def load_params() -> Dict[str, Any]:
    """
    Load feature comparison parameters from params.yaml using DVC.
    Falls back gracefully if running outside a DVC pipeline.
    """
    try:
        logger.info("Loading params via dvc.api")
        params = dvc.api.params_show()
        return params["feature_comparison"]
    except Exception as e:
        logger.warning(f"Could not load params via dvc.api: {e}")
        logger.warning("Falling back to script defaults (only for local debugging).")
        return {
            "ngram_ranges": "[(1,1), (1,2), (1,3)]",
            "max_features": 5000,
            "batch_size": 32,
            "n_estimators": 200,
            "max_depth": 15,
            "use_distilbert": False,
        }


def get_distilbert_embeddings(
    texts: List[str], device: Optional[str] = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled DistilBERT embeddings for texts.

    Args:
        texts (list): Cleaned text samples.
        device (str): "cuda" or "cpu". Auto-detected if None.
        batch_size (int): Batch size for batched inference.

    Returns:
        np.ndarray: Dense embeddings (n_samples, 768).
    """
    # --- Lazy imports (only executed if DistilBERT is actually used) ---
    try:
        import torch
        from tqdm import tqdm
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "DistilBERT embeddings require `torch`, `transformers`, and `tqdm`. "
            "Please install them via: uv add torch transformers tqdm"
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Skip if no GPU and user prefers to avoid CPU inference ---
    if device == "cpu":
        logger.warning("⚠️ DistilBERT inference skipped: running on CPU only.")
        raise RuntimeError("❌ DistilBERT cannot be used without GPU acceleration.")

    logger.info(f"DistilBERT Inference: Using device: {device}")
    # Using a general-purpose model suitable for sentiment
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []

    # Use tqdm for progress tracking during large-scale embedding generation (Scalability)
    for i in tqdm(
        range(0, len(texts), batch_size), desc="Generating DistilBERT Embeddings"
    ):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pool over sequence dimension (ignoring CLS and SEP tokens)
            pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def run_comparison_experiment(
    vectorizer_type: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    n_estimators: int,
    max_depth: int,
    batch_size: int = 32,
) -> None:
    """
    Run experiment for the given vectorizer type using a RandomForest baseline.

    Args:
        vectorizer_type: "TF-IDF" or "DistilBERT".
        ngram_range: N-gram tuple.
        max_features: Max features for TF-IDF.
        n_estimators: RF trees.
        max_depth: RF depth.
        batch_size: DistilBERT batch size.
    """

    # --- Load data (best practice for feature selection: use VAL set for tuning/comparison) ---
    train_df, val_df = load_train_val_data()

    # Text and Labels for Training
    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values

    # Text and Labels for Validation/Evaluation
    X_val_text = val_df["clean_comment"].tolist()
    y_val = val_df["category"].values

    logger.info(f"Data split: Train {train_df.shape[0]}, Val {val_df.shape[0]}")

    # --- Vectorization ---
    X_train: Union[np.ndarray, spmatrix]
    X_val: Union[np.ndarray, spmatrix]
    vectorizer: Any = None  # Will hold the TfidfVectorizer if used

    if vectorizer_type == "TF-IDF":
        logger.info("Generating TF-IDF features...")
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
            lowercase=False,
            min_df=2,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)
        feature_dim = X_train.shape[1]

        run_name = f"TFIDF_{ngram_range[0]}-{ngram_range[1]}gram_{max_features}feat"

    elif vectorizer_type == "DistilBERT":
        logger.info("Generating DistilBERT embeddings...")
        X_train = get_distilbert_embeddings(X_train_text, batch_size=batch_size)
        X_val = get_distilbert_embeddings(X_val_text, batch_size=batch_size)
        feature_dim = X_train.shape[1]  # 768 for distilbert-base-uncased

        run_name = f"DistilBERT_768dim_Batch{batch_size}"

    else:
        raise ValueError("Unsupported vectorizer_type")

    logger.info(f"Using {vectorizer_type} with {feature_dim} features.")

    # --- MLflow Tracking, Training, and Evaluation ---
    with mlflow.start_run(run_name=run_name):
        # 1. Train Model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # 2. Define Params and Tags for Logging
        common_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "feature_dim": feature_dim,
            "random_state": 42,
        }
        tags = {
            "experiment_type": "feature_comparison",
            "model_type": "RandomForestClassifier",
            "vectorizer_type": vectorizer_type,
        }

        if vectorizer_type == "TF-IDF":
            run_params = {
                **common_params,
                "ngram_range": str(ngram_range),  # log as string for consistency
                "max_features": max_features,
            }
        else:  # DistilBERT
            run_params = {
                **common_params,
                "batch_size": batch_size,
                "distilbert_model": "distilbert-base-uncased",
            }

        # 3. Evaluation and Logging
        evaluate_and_log(
            model=model,
            X_val=X_val,
            y_val=y_val,
            run_name=run_name,
            params=run_params,
            tags=tags,
            log_model=False,
        )

        logger.info(
            f"✅ Experiment finished: {run_name} | MLflow Run ID: {mlflow.last_active_run().info.run_id}"
        )


def main() -> None:
    """Parse args and run experiments, using params.yaml as the source of truth."""
    # --- DVC/CLI Parameter Loading ---
    params = load_params()
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF and DistilBERT. Params from params.yaml by default."
    )
    # Define arguments for optional CLI overrides
    parser.add_argument(
        "--ngram_ranges",
        type=str,
        required=False,
        help="Override ngram_ranges from params.yaml.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        required=False,
        help="Override max_features from params.yaml.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Override batch_size from params.yaml.",
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
    parser.add_argument(
        "--use_distilbert",
        type=str2bool,  # Custom str to bool converter
        required=False,
        help="Override use_distilbert from params.yaml.",
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
    setup_experiment("Exp - Feature Comparison (TFIDF vs DistilBERT)", mlflow_uri)

    # --- Parameter Parsing ---
    ngram_ranges = parse_dvc_param(
        final_params["ngram_ranges"], name="ngram_ranges", expected_type=list
    )

    # --- TF-IDF experiments ---
    logger.info("🚀 Starting TF-IDF Experiments")
    for ngram_range in ngram_ranges:
        run_comparison_experiment(
            vectorizer_type="TF-IDF",
            ngram_range=tuple(ngram_range),  # Ensure it's a tuple for the function
            max_features=final_params["max_features"],
            n_estimators=final_params["n_estimators"],
            max_depth=final_params["max_depth"],
            batch_size=final_params[
                "batch_size"
            ],  # Not used for TFIDF, but passed for consistency
        )

    # --- DistilBERT experiment (conditional run) ---
    if final_params["use_distilbert"]:
        try:
            logger.info("🚀 Starting DistilBERT_768dim Experiment")
            run_comparison_experiment(
                vectorizer_type="DistilBERT",
                ngram_range=(1, 1),  # N-gram not applicable but required by function
                max_features=final_params["max_features"],
                n_estimators=final_params["n_estimators"],
                max_depth=final_params["max_depth"],
                batch_size=final_params["batch_size"],
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(
                "⚠️ Skipping DistilBERT experiment: %s. Ensure PyTorch and CUDA are correctly installed.",
                str(e),
            )
    else:
        logger.info(
            "ℹ️ DistilBERT experiment intentionally skipped (use_distilbert=False)."
        )

    logger.info("--- TF-IDF vs. DistilBERT complete. Analyze results in MLflow UI ---")


if __name__ == "__main__":
    main()
