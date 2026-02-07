"""
Prepare a processed dataset from raw Reddit data.

Loads raw CSV, cleans text, engineers labels, performs stratified train/val/test split,
and saves Parquet files to data/processed/.

Usage (preferred):
    uv run dvc repro                 # Uses params.yaml → fully reproducible
Run specific pipeline stage:
    uv run dvc repro data_preparation
"""

import pandas as pd
import re
from typing import Optional
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Project Utilities ---
from src.utils.paths import RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, PROJECT_ROOT
from src.utils.logger import get_logger
from src.config.manager import ConfigurationManager
from src.config.schemas import DataPreparationConfig

# --- Logging Setup ---
logger = get_logger(__name__, headline="make_dataset.py")


def load_params() -> DataPreparationConfig:
    """
    Load parameters from params.yaml using ConfigurationManager.
    """
    try:
        logger.info("Loading params via ConfigurationManager")
        config = ConfigurationManager().get_data_preparation_config()
        return config
    except Exception as e:
        logger.warning(f"Could not load params via ConfigurationManager: {e}")
        logger.warning("Falling back to defaults (only for local debugging).")
        # Return a default config object
        return DataPreparationConfig(test_size=0.15, random_state=42)


def clean_text(text: str, stop_words: Optional[set] = None) -> str:
    """
    Enhanced text cleaning: lowercase, remove non-alphabetic except spaces,
    remove stopwords, strip whitespace. Retains sentiment signals.

    Args:
        text (str): Input text.
        stop_words (set, optional): NLTK stopwords to remove.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if stop_words:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        text = " ".join(tokens)
    return text


def prepare_reddit_dataset(test_size: float, random_state: int) -> None:
    """
    Orchestrate data preparation.

    Args:
        test_size (float): Fraction for test split.
        random_state (int): Seed for reproducibility.
    """
    logger.info("Starting dataset preparation...")
    logger.info(f"Using test_size={test_size}, random_state={random_state}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data missing: {RAW_PATH.relative_to(PROJECT_ROOT)}. Run 'dvc repro data_ingestion' first."
        )

    # Load and initial validation
    logger.info(f"Loading raw data from {RAW_PATH.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(RAW_PATH)
    if df.empty or "clean_comment" not in df.columns or "category" not in df.columns:
        raise ValueError("Invalid raw data structure.")
    logger.info(f"Loaded {len(df)} rows from raw data with shape: {df.shape}.")

    # --- Label normalization for consistency ---
    # Many ML tools (e.g., np.bincount, SMOTE, StratifiedKFold) require non-negative labels.
    unique_labels = sorted(df["category"].unique())
    # This ensures your raw data always has the expected structure
    if unique_labels != [-1, 0, 1]:
        raise ValueError(f"Unexpected category labels: {unique_labels}")
    # Map {-1, 0, 1} → {0, 1, 2}
    df["category_encoded"] = df["category"].map({-1: 0, 0: 1, 1: 2})
    logger.info(
        f"Original label distribution: {dict(df['category'].value_counts().sort_index())}"
    )
    logger.info(
        f"Encoded label distribution: {dict(df['category_encoded'].value_counts().sort_index())}"
    )

    # Cleaning
    stop_words = set(stopwords.words("english"))
    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].apply(lambda x: clean_text(x, stop_words))
    df = df[df["clean_comment"].str.len() > 0]
    logger.info(f"Dataset after cleaning: {len(df)} rows.")

    # Label engineering
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    df["sentiment_label"] = df["category"].map(label_map)

    # Stratified split
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["category"]
    )
    # Calculate the proportion of the remaining data (train_val) to be used
    # validation set, ensuring that the overall validation set size relative
    # the original dataset matches the desired 'test_size'.
    # Example: If test_size=0.15, then 15% of data is for test, 85% for train_val.
    # To get 15% of the *original* data for validation, we need 0.15 / 0.85 train_val.
    val_size = test_size / (1 - test_size)  # ~0.1765 for 15% val from 85%
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val["category"],
    )

    # Save and validate shapes
    outputs = [
        (TRAIN_PATH, train),
        (VAL_PATH, val),
        (TEST_PATH, test),
    ]
    for out_path, split_df in outputs:
        split_df.to_parquet(out_path, index=False)
        if split_df.empty:
            raise ValueError(f"Empty split for {out_path}.")

    # Log splits
    logger.info(
        f"Splits prepared: Train {train.shape[0]}, Val {val.shape[0]}, Test {test.shape[0]}"
    )
    logger.info(
        f"Train class distribution: {train['category'].value_counts().to_dict()}"
    )
    logger.info("✅ Successfully prepared and saved processed datasets. ✅")


def main() -> None:
    """Entry point."""
    # Ensure NLTK data is available
    logger.info("🚀 Starting data preparation process 🚀")
    logger.info("Downloading NLTK data (if not already present)...")
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    logger.info("NLTK data download complete.")

    # Load parameters from params.yaml via ConfigurationManager
    config = load_params()

    prepare_reddit_dataset(config.test_size, config.random_state)


if __name__ == "__main__":
    main()
