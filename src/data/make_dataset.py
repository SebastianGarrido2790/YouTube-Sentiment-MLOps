"""
Prepare processed dataset from raw Reddit data.

Loads raw CSV, cleans text, engineers labels, performs stratified train/val/test split,
and saves Parquet files to data/processed/.

Usage:
    uv run python -m src.data.make_dataset --test_size 0.15 --random_state 42

Requirements:
    - Raw data at data/raw/reddit_comments.csv (from download_dataset.py).
    - uv sync (for pandas, scikit-learn, nltk).

Design Considerations:
- Reliability: Input validation, NaN handling, post-split integrity checks.
- Scalability: Parquet for efficient I/O; vectorized pandas operations.
- Maintainability: Logging, type hints, modular functions; paths relative to script.
- Adaptability: Parameterized splits; extensible cleaning (e.g., add lemmatization).
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if needed (one-time; punkt_tab for modern NLTK)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths relative to project root (three parents from src/data/)
ROOT = Path(__file__).parent.parent.parent
RAW_PATH = ROOT / "data" / "raw" / "reddit_comments.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


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


def prepare_reddit_dataset(test_size: float = 0.15, random_state: int = 42) -> None:
    """
    Orchestrate data preparation.

    Args:
        test_size (float): Fraction for test split.
        random_state (int): Seed for reproducibility.
    """
    logger.info("Starting data preparation and splitting.")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data missing: {RAW_PATH}. Run data ingestion first."
        )

    # Load and initial validation
    df = pd.read_csv(RAW_PATH)
    if df.empty or "clean_comment" not in df.columns or "category" not in df.columns:
        raise ValueError("Invalid raw data structure.")
    logger.info(f"Loaded {len(df)} rows from raw data with shape: {df.shape}.")

    # Cleaning
    stop_words = set(stopwords.words("english"))
    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].apply(lambda x: clean_text(x, stop_words))
    df = df[df["clean_comment"].str.len() > 0]
    logger.info(f"After cleaning: {len(df)} rows.")

    # Label engineering
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    df["sentiment_label"] = df["category"].map(label_map)

    # Ensure processed dir
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Stratified split
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["category"]
    )
    val_size = test_size / (1 - test_size)  # ~0.1765 for 15% val from 85%
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val["category"],
    )

    # Save and validate shapes
    outputs = [
        (PROCESSED_DIR / "train.parquet", train),
        (PROCESSED_DIR / "val.parquet", val),
        (PROCESSED_DIR / "test.parquet", test),
    ]
    for out_path, split_df in outputs:
        split_df.to_parquet(out_path, index=False)
        if split_df.empty:
            raise ValueError(f"Empty split for {out_path}.")

    # Log splits
    logger.info(
        f"Splits prepared: Train {train.shape[0]}, Val {val.shape[0]}, Test {test.shape[0]}"
    )


def main() -> None:
    """Parse args and run preparation."""
    parser = argparse.ArgumentParser(description="Prepare processed dataset.")
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Test split fraction."
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    prepare_reddit_dataset(args.test_size, args.random_state)


if __name__ == "__main__":
    main()
