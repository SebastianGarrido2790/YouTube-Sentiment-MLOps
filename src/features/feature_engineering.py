"""
Feature Engineering Module for Sentiment Analysis (MLOps-Standards).

Generates and saves reusable feature matrices (X) and labels (y) using parameters
defined in `params.yaml`. Supports both TF-IDF and DistilBERT representations.

Saves feature matrices as compressed NumPy sparse arrays (.npz) and labels
as NumPy arrays (.npy) to the models/features directory.

Usage:
Run the entire pipeline:
    uv run dvc repro               # Uses params.yaml → fully reproducible
Run specific pipeline stage:
    uv run python -m src.features.feature_engineering

Requirements:
    - Parameters defined in params.yaml under `feature_engineering` and `imbalance_tuning`.
    - Processed data available in data/processed/.

Outputs:
    models/features
        ├── X_train.npz / y_train.npy
        ├── X_val.npz   / y_val.npy
        ├── X_test.npz  / y_test.npy
        ├── vectorizer.pkl (if TF-IDF)
        └── label_encoder.pkl
"""

import pickle
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse, save_npz, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# --- Project Utilities ---
from src.config.manager import ConfigurationManager
from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH

# --- Logging Setup ---
logger = get_logger(__name__, headline="feature_engineering.py")


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
        logger.info("💪🏻 Generating DistilBERT embeddings (768 dim)... 💪🏻")
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
            f"💪🏻 Generating TF-IDF features (max_features={max_features}, ngram={ngram_range})... 💪🏻"
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
    if not FEATURES_DIR.exists():
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

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
        f"Features engineered | Type: {feature_type} | "
        f"Shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )


def main() -> None:
    """Run feature engineering using ConfigurationManager as source of truth."""
    logger.info("🚀 Starting Feature Engineering 🚀")

    # --- Parameter Loading via ConfigurationManager ---
    config_manager = ConfigurationManager()

    # Load separate configs (feature_engineering and imbalance_tuning)
    # Rationale: Feature engineering reuses 'best parameters' found in imbalance tuning.
    fe_config = config_manager.get_feature_engineering_config()
    it_config = config_manager.get_imbalance_tuning_config()

    # Parse boolean from string (if necessary, though specific to params.yaml structure)
    # The Pydantic model defines use_distilbert as `str` to catch "False"/"True" strings.
    use_distilbert_val = fe_config.use_distilbert
    if isinstance(use_distilbert_val, str):
        use_distilbert = use_distilbert_val.lower() == "true"
    else:
        use_distilbert = bool(use_distilbert_val)

    # Use parameters from Imbalance Tuning for optimal feature extraction
    max_features = it_config.best_max_features
    # Start as tuple from the list provided in config
    ngram_range = tuple(it_config.best_ngram_range)

    distilbert_batch_size = fe_config.distilbert_batch_size

    logger.info("Configuration Loaded:")
    logger.info(f"  - Use DistilBERT: {use_distilbert}")
    logger.info(f"  - Max Features: {max_features}")
    logger.info(f"  - N-gram Range: {ngram_range}")
    logger.info(f"  - DistilBERT Batch Size: {distilbert_batch_size}")

    engineer_features(
        use_distilbert=use_distilbert,
        max_features=max_features,
        ngram_range=ngram_range,
        distilbert_batch_size=distilbert_batch_size,
    )

    logger.info("✅ Feature Engineering Pipeline Complete ✅")


if __name__ == "__main__":
    main()
