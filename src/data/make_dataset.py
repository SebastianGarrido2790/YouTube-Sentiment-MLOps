import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove non-alphabetic characters except spaces,
    strip whitespace. Retains core sentiment signals.
    """
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(
        r"[^a-zA-Z\s]", " ", text
    )  # Remove special chars, keep letters/spaces
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text


def prepare_reddit_dataset() -> None:
    """
    Load, clean, and split Reddit dataset into train/val/test sets.
    Saves to data/processed/ as Parquet files.
    """
    raw_path = "../../data/raw/reddit.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run download_dataset.py first."
        )

    # Load data
    df = pd.read_csv(raw_path)

    # Cleaning
    df = df.dropna(subset=["clean_comment"])  # Drop missing comments
    df["clean_comment"] = df["clean_comment"].apply(clean_text)
    df = df[df["clean_comment"].str.len() > 0]  # Remove empty after cleaning

    # Engineer sentiment_label
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    df["sentiment_label"] = df["category"].map(label_map)

    # Ensure directories
    os.makedirs("../../data/processed", exist_ok=True)

    # Stratified split: 70% train, 15% val, 15% test
    train_val, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["category"]
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1765,
        random_state=42,
        stratify=train_val["category"],  # 15/85 ≈ 0.1765
    )

    # Save
    train.to_parquet("../../data/processed/train.parquet", index=False)
    val.to_parquet("../../data/processed/val.parquet", index=False)
    test.to_parquet("../../data/processed/test.parquet", index=False)

    print(
        f"Prepared datasets: Train {train.shape[0]}, Val {val.shape[0]}, Test {test.shape[0]}"
    )


if __name__ == "__main__":
    prepare_reddit_dataset()
