"""
Compare TF-IDF vs. BERT embeddings for sentiment feature engineering.

Loads processed data, generates embeddings, trains RandomForest baselines, logs to MLflow.
Tracks multiple TF-IDF n-gram variants.

The feature_comparison stage should not have an outs section in dvc.yaml,
as its primary purpose is hyperparameter tuning and comparison where results are logged to MLflow,
not saved as local artifacts for DVC versioning.

Usage (DVC command):
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
from typing import Tuple, Optional, Any, Union, List
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix  # For sparse matrix type hint

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.features.helpers.feature_utils import (
    setup_mlflow_run,
    load_train_val_data,
    parse_dvc_param,
    evaluate_and_log,
)

# --- Logging Setup ---
logger = get_logger(__name__, headline="tfidf_vs_bert.py")


def get_bert_embeddings(
    texts: List[str], device: Optional[str] = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled BERT embeddings for texts.

    Args:
        texts (list): Cleaned text samples.
        device (str): "cuda" or "cpu". Auto-detected if None.
        batch_size (int): Batch size for batched inference.

    Returns:
        np.ndarray: Dense embeddings (n_samples, 768).
    """
    # --- Lazy imports (only executed if BERT is actually used) ---
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            "BERT embeddings require `torch`, `transformers`, and `tqdm`. "
            "Please install them via: uv add torch transformers tqdm"
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Skip if no GPU and user prefers to avoid CPU inference ---
    if device == "cpu":
        logger.warning("⚠️ BERT inference skipped: running on CPU only.")
        raise RuntimeError("BERT cannot be used without GPU acceleration.")

    logger.info(f"BERT Inference: Using device: {device}")
    # Using a general-purpose model suitable for sentiment
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []

    # Use tqdm for progress tracking during large-scale embedding generation (Scalability)
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT Embeddings"):
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
        vectorizer_type: "TF-IDF" or "BERT".
        ngram_range: N-gram tuple.
        max_features: Max features for TF-IDF.
        n_estimators: RF trees.
        max_depth: RF depth.
        batch_size: BERT batch size.
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

    elif vectorizer_type == "BERT":
        logger.info("Generating BERT embeddings...")
        X_train = get_bert_embeddings(X_train_text, batch_size=batch_size)
        X_val = get_bert_embeddings(X_val_text, batch_size=batch_size)
        feature_dim = X_train.shape[1]  # 768 for bert-base-uncased

        run_name = f"BERT_768dim_Batch{batch_size}"

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
                "ngram_range": ngram_range,
                "max_features": max_features,
            }
        else:  # BERT
            run_params = {
                **common_params,
                "batch_size": batch_size,
                "bert_model": "bert-base-uncased",
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
        "--max_features",
        type=int,
        default=5000,
        help="Max TF-IDF features.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="BERT batch size.")
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="RF n_estimators."
    )
    parser.add_argument("--max_depth", type=int, default=15, help="RF max_depth.")
    parser.add_argument(
        "--use_bert",
        type=bool,
        default=False,
        help="Whether to run the BERT experiment (True/False).",
    )
    args = parser.parse_args()

    # --- MLflow Setup ---
    setup_mlflow_run(experiment_name="Exp - Feature Comparison (TFIDF vs BERT)")

    # --- Parameter Parsing ---
    # Parse the list of ngram tuples from DVC/CLI
    ngram_ranges = parse_dvc_param(
        args.ngram_ranges, name="ngram_ranges", expected_type=list
    )

    # --- TF-IDF experiments with different n-grams (parameterized) ---
    logger.info("--- Starting TF-IDF Experiments ---")
    for ngram_range in ngram_ranges:
        run_comparison_experiment(
            vectorizer_type="TF-IDF",
            ngram_range=tuple(ngram_range),  # Ensure it's a tuple for the function
            max_features=args.max_features,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            batch_size=args.batch_size,  # Not used for TFIDF, but passed for consistency
        )

    # --- BERT experiment (single conditional run) ---
    if args.use_bert:
        import torch  # Ensure torch is available for BERT

        if "torch" in globals() and torch.cuda.is_available():
            logger.info("--- Starting BERT_768dim Experiment ---")
            run_comparison_experiment(
                vectorizer_type="BERT",
                ngram_range=(1, 1),
                max_features=args.max_features,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                batch_size=args.batch_size,
            )
        else:
            logger.warning(
                "⚠️ Skipping BERT experiment: torch not available or running on CPU."
            )

    logger.info("--- TF-IDF vs. BERT complete. Analyze results in MLflow UI ---")


if __name__ == "__main__":
    main()
