"""
Feature Engineering Module for Sentiment Analysis (DVC-Aware)

Generates and saves reusable feature matrices (X) and labels (y) using parameters
defined in `params.yaml`. Supports both TF-IDF and DistilBERT representations.

Saves feature matrices as compressed NumPy sparse arrays (.npz) and labels
as NumPy arrays (.npy) to the models/features directory.

Usage (DVC - preferred):
    uv run dvc repro feature_engineering

Usage (local cli override only):
    uv run python -m src.features.feature_engineering --max_features 2000

Requirements:
    - Parameters defined in params.yaml under `feature_engineering` and `imbalance_tuning`.
    - Processed data available in data/processed/.
    - `uv sync` must be run for all dependencies.

Outputs:
    models/features
        ├── X_train.npz / y_train.npy
        ├── X_val.npz   / y_val.npy
        ├── X_test.npz  / y_test.npy
        ├── vectorizer.pkl (if TF-IDF)
        └── label_encoder.pkl
"""

import argparse
import pickle
from typing import Any, Dict, Optional, Tuple, Union

import dvc.api
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse, save_npz, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# --- Project Utilities ---
from src.features.helpers.feature_utils import parse_dvc_param
from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR, PROJECT_ROOT, TEST_PATH, TRAIN_PATH, VAL_PATH

# --- Logging Setup ---
logger = get_logger(__name__, headline="feature_engineering.py")


def load_params() -> Dict[str, Any]:
    """
    Load parameters from params.yaml using DVC.
    Merges `feature_engineering` and relevant `imbalance_tuning` sections.
    """
    try:
        logger.info("Loading params via dvc.api")
        all_params = dvc.api.params_show()

        fe_params = all_params.get("feature_engineering", {})
        it_params = all_params.get("imbalance_tuning", {})

        # DVC can load bools as strings, so we ensure it's a proper boolean
        use_distilbert_val = fe_params.get("use_distilbert", False)
        if isinstance(use_distilbert_val, str):
            use_distilbert = use_distilbert_val.lower() == "true"
        else:
            use_distilbert = bool(use_distilbert_val)

        # Create a unified dictionary, mapping names from params.yaml
        params = {
            "use_distilbert": use_distilbert,
            "distilbert_batch_size": fe_params.get("distilbert_batch_size", 32),
            "max_features": it_params.get("best_max_features", 1000),
            "ngram_range": it_params.get("best_ngram_range", "(1,1)"),
        }
        return params
    except Exception as e:
        logger.warning(f"Could not load params via dvc.api: {e}")
        logger.warning("Falling back to script defaults (only for local debugging).")
        return {
            "use_distilbert": False,
            "distilbert_batch_size": 32,
            "max_features": 1000,
            "ngram_range": "(1,1)",
        }


def _get_distilbert_embeddings(
    texts: list, device: Optional[str] = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled DistilBERT embeddings for a list of texts.
    Imports torch and transformers lazily.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "DistilBERT mode requires torch and transformers. "
            "Install with `uv pip install torch transformers`."
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"DistilBERT Inference: Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
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
            pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def _add_derived_features(df: pd.DataFrame) -> np.ndarray:
    """Calculates dense, simple, derived features (length, word ratios)."""
    df_local = df.copy()
    df_local["char_len"] = df_local["clean_comment"].str.len()
    df_local["word_len"] = df_local["clean_comment"].str.split().str.len()

    pos_words = {"good", "great", "love", "like", "positive", "best"}
    neg_words = {"bad", "hate", "worst", "negative", "shit", "fuck"}

    def count_lexicon_ratio(text, lexicon):
        words = text.split()
        return len([w for w in words if w in lexicon]) / max(len(words), 1)

    df_local["pos_ratio"] = df_local["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, pos_words)
    )
    df_local["neg_ratio"] = df_local["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, neg_words)
    )

    return df_local[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values


def engineer_features(
    use_distilbert: bool,
    max_features: int,
    ngram_range: Tuple[int, int],
    distilbert_batch_size: int,
) -> None:
    """
    Generates final feature matrices (X) and labels (y) based on selected parameters.
    """
    # 1. Load Data
    splits = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    dfs = {}
    for name, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data missing: {path}. Run data_preparation first."
            )
        dfs[name] = pd.read_parquet(path)
        if dfs[name]["clean_comment"].empty:
            raise ValueError(f"Clean comments column is empty in {name} split.")
        logger.info(f"{name.capitalize()} set loaded: {dfs[name].shape[0]} samples.")

    train_texts = dfs["train"]["clean_comment"].tolist()
    val_texts = dfs["val"]["clean_comment"].tolist()
    test_texts = dfs["test"]["clean_comment"].tolist()

    y_all = pd.concat([dfs[s]["sentiment_label"] for s in splits])
    le = LabelEncoder()
    le.fit(y_all)
    y_train = le.transform(dfs["train"]["sentiment_label"])
    y_val = le.transform(dfs["val"]["sentiment_label"])
    y_test = le.transform(dfs["test"]["sentiment_label"])

    # 3. Text Feature Generation
    vectorizer: Optional[TfidfVectorizer] = None
    X_train_text: Union[spmatrix, np.ndarray]
    X_val_text: Union[spmatrix, np.ndarray]
    X_test_text: Union[spmatrix, np.ndarray]

    if use_distilbert:
        logger.info("🚀 Generating DistilBERT embeddings (768 dim)...")
        X_train_text = _get_distilbert_embeddings(
            train_texts, batch_size=distilbert_batch_size
        )
        X_val_text = _get_distilbert_embeddings(
            val_texts, batch_size=distilbert_batch_size
        )
        X_test_text = _get_distilbert_embeddings(
            test_texts, batch_size=distilbert_batch_size
        )
    else:
        logger.info(
            f"🚀 Generating TF-IDF features (max_features={max_features}, ngram={ngram_range})..."
        )
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=False,
            min_df=2,
        )
        X_train_text = vectorizer.fit_transform(train_texts)
        X_val_text = vectorizer.transform(val_texts)
        X_test_text = vectorizer.transform(test_texts)

    # 4. Derived Feature Generation
    derived_train = _add_derived_features(dfs["train"])
    derived_val = _add_derived_features(dfs["val"])
    derived_test = _add_derived_features(dfs["test"])

    # 5. Combine Features
    X_sets = []
    for X_text, X_derived in zip(
        [X_train_text, X_val_text, X_test_text],
        [derived_train, derived_val, derived_test],
    ):
        X_combined = (
            hstack([X_text, X_derived])
            if issparse(X_text)
            else csr_matrix(np.hstack([X_text, X_derived]))
        )
        X_sets.append(X_combined)

    # 6. Save Artifacts
    X_train, X_val, X_test = X_sets
    for split, X in zip(splits, X_sets):
        save_npz(FEATURES_DIR / f"X_{split}.npz", X)
    for split, y in zip(splits, [y_train, y_val, y_test]):
        np.save(FEATURES_DIR / f"y_{split}.npy", y)

    with open(FEATURES_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    if vectorizer:
        vectorizer_name = f"tfidf_vectorizer_max_{max_features}.pkl"
        with open(FEATURES_DIR / vectorizer_name, "wb") as f:
            pickle.dump(vectorizer, f)

    feature_type = (
        "DistilBERT" if use_distilbert else f"TF-IDF (max_features={max_features})"
    )
    logger.info(
        f"✅ Features engineered successfully | Type: {feature_type} | "
        f"Shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )


def main() -> None:
    """Parse args and run feature engineering using DVC params as source of truth."""
    params = load_params()
    parser = argparse.ArgumentParser(
        description="Generate final feature set. Params from params.yaml by default."
    )

    parser.add_argument(
        "--use_distilbert",
        type=lambda x: x.lower() == "true",
        required=False,
        help="Override 'use_distilbert' from params.yaml.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        required=False,
        help="Override 'best_max_features' from params.yaml.",
    )
    parser.add_argument(
        "--ngram_range",
        type=str,
        required=False,
        help="Override 'best_ngram_range' from params.yaml.",
    )
    parser.add_argument(
        "--distilbert_batch_size",
        type=int,
        required=False,
        help="Override 'distilbert_batch_size' from params.yaml.",
    )
    args = parser.parse_args()

    # Consolidate parameters (CLI overrides DVC-loaded params)
    final_params = params.copy()
    overridden_keys = []
    for key in final_params:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            final_params[key] = cli_val
            overridden_keys.append(key)

    if overridden_keys:
        logger.warning(
            "CLI overrides detected for: %s. This run may not be reproducible with 'dvc repro'.",
            ", ".join(overridden_keys),
        )

    # Parse ngram_range specifically
    ngram_range = parse_dvc_param(
        final_params["ngram_range"], name="ngram_range", expected_type=tuple
    )
    if ngram_range is None:
        return

    logger.info("--- Feature Engineering Started ---")
    engineer_features(
        use_distilbert=final_params["use_distilbert"],
        max_features=final_params["max_features"],
        ngram_range=ngram_range,
        distilbert_batch_size=final_params["distilbert_batch_size"],
    )


if __name__ == "__main__":
    main()
