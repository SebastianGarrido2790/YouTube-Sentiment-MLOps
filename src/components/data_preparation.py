"""
Data Preparation Component

Handles dataset cleaning, label encoding, and stratified data splitting
to ensure consistent inputs for training and evaluation.
"""

import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from src.constants import PROJECT_ROOT, RAW_PATH, TEST_PATH, TRAIN_PATH, VAL_PATH
from src.entity.config_entity import DataPreparationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="data_preparation_component")


class DataPreparation:
    """
    Component for cleaning, encoding, and splitting the raw dataset.

    Responsible for text normalization, category-to-numeric mapping,
    and generating stratified snapshots for training, validation, and testing.
    """

    def __init__(self, config: DataPreparationConfig) -> None:
        """
        Initializes the DataPreparation component.

        Args:
            config (DataPreparationConfig): Configuration for splitting and randomness.
        """
        self.config = config

    @staticmethod
    def clean_text(text: str, stop_words: set | None = None) -> str:
        """
        Normalizes text by lowercasing, removing non-alphabetic characters,
        and optionally filtering stop words.

        Args:
            text (str): The raw input text.
            stop_words (set | None): A set of stop words to remove.

        Returns:
            str: The cleaned and normalized text.
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
        return text.strip()

    def prepare_reddit_dataset(self) -> None:
        logger.info("Starting dataset preparation...")
        logger.info(f"Using test_size={self.config.test_size}, random_state={self.config.random_state}")

        if not RAW_PATH.exists():
            raise FileNotFoundError(
                f"Raw data missing: {RAW_PATH.relative_to(PROJECT_ROOT)}. Run 'dvc repro data_ingestion' first."
            )

        logger.info(f"Loading raw data from {RAW_PATH.relative_to(PROJECT_ROOT)}")
        df = pd.read_csv(RAW_PATH)
        if df.empty or "clean_comment" not in df.columns or "category" not in df.columns:
            raise ValueError("Invalid raw data structure.")
        logger.info(f"Loaded {len(df)} rows from raw data with shape: {df.shape}.")

        unique_labels = sorted(df["category"].unique())
        if unique_labels != [-1, 0, 1]:
            raise ValueError(f"Unexpected category labels: {unique_labels}")

        df["category_encoded"] = df["category"].map({-1: 0, 0: 1, 1: 2})
        logger.info(f"Original label distribution: {dict(df['category'].value_counts().sort_index())}")
        logger.info(f"Encoded label distribution: {dict(df['category_encoded'].value_counts().sort_index())}")

        stop_words = set(stopwords.words("english"))
        df = df.dropna(subset=["clean_comment"])
        df["clean_comment"] = df["clean_comment"].apply(lambda x: DataPreparation.clean_text(x, stop_words))
        df = df[df["clean_comment"].str.len() > 0]
        logger.info(f"Dataset after cleaning: {len(df)} rows.")

        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        df["sentiment_label"] = df["category"].map(label_map)

        train_val, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["category"],  # type: ignore
        )
        val_size = self.config.test_size / (1 - self.config.test_size)
        train_test_val = train_test_split(
            train_val,
            test_size=val_size,
            random_state=self.config.random_state,
            stratify=train_val["category"],  # type: ignore
        )
        train = train_test_val[0]  # type: ignore
        val = train_test_val[1]  # type: ignore

        outputs: list[tuple[Path, pd.DataFrame]] = [
            (TRAIN_PATH, train),  # type: ignore
            (VAL_PATH, val),  # type: ignore
            (TEST_PATH, test),  # type: ignore
        ]
        for out_path, split_df in outputs:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            split_df.to_parquet(out_path, index=False)
            if split_df.empty:
                raise ValueError(f"Empty split for {out_path}.")

        logger.info(f"Splits prepared: Train {train.shape[0]}, Val {val.shape[0]}, Test {test.shape[0]}")
        logger.info(f"Train class distribution: {train['category'].value_counts().to_dict()}")  # type: ignore
        logger.info("✅ Successfully prepared and saved processed datasets. ✅")
