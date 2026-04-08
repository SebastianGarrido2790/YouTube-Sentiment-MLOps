"""
Data Preprocessing and Validation Suite

This module contains unit tests for the data cleaning and preprocessing logic.
It validates the normalization of text data, including special character
removal, stopword filtering, and handling of null or empty values.
"""

from typing import Any

import pandas as pd
import pytest

from src.components.data_preparation import DataPreparation


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello World! 123", "hello world"),
        ("Sentim@nt An@lys!s #", "sentim nt an lys s"),
        ("", ""),
        (pd.NA, ""),
        (None, ""),
    ],
    ids=["basic_cleaning", "special_chars", "empty_string", "pd_NA", "None_value"],
)
def test_clean_text_general(text: Any, expected: str):
    """
    Tests general text cleaning scenarios without stopwords.

    This test ensures that the basic cleaning pipeline (lowercasing,
    non-alphabetic character removal) handles various input types including
    strings, pandas NA, and None.

    Args:
        text: Input text to be cleaned.
        expected: The expected normalized output string.
    """
    assert DataPreparation.clean_text(text) == expected


@pytest.mark.parametrize(
    "text,stop_words,expected",
    [
        ("This is a test sentence", {"this", "is", "a"}, "test sentence"),
        ("go to the gym", {"the"}, "gym"),  # "go" and "to" removed due to len <= 2
    ],
    ids=["remove_stopwords", "remove_short_tokens"],
)
def test_clean_text_with_stopwords(text: str, stop_words: set[str] | None, expected: str):
    """
    Tests text cleaning with stopword removal and length filtering.

    This test validates that when a stopword set is provided, the cleaner
    correctly filters out restricted words and also removes tokens with a
    length of 2 or less.

    Args:
        text: Input text to be cleaned.
        stop_words: A set of words to be removed from the text.
        expected: The expected filtered output string.
    """
    assert DataPreparation.clean_text(text, stop_words) == expected


def test_prepare_reddit_dataset(tmp_path, monkeypatch):
    """
    Tests the full prepare_reddit_dataset pipeline.
    Mocks filesystem constants and internal NLTK calls to ensure portability.
    """
    # 1. Setup Mock Workspace
    raw_file = tmp_path / "raw.csv"
    train_file = tmp_path / "train.parquet"
    val_file = tmp_path / "val.parquet"
    test_file = tmp_path / "test.parquet"

    # 2. Create Dummy Raw Data (Increase size for stratification)
    df = pd.DataFrame(
        {
            "clean_comment": [
                "great video",
                "bad idea",
                "neutral stuff",
                "awesome",
                "terrible",
                "meh",
                "good",
                "horrible",
                "okay",
                "fun",
                "scary",
                "boring",
                "helpful",
                "confusing",
                "normal",
            ],
            "category": [1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0],
        }
    )
    df.to_csv(raw_file, index=False)

    # 3. Patch Constants
    monkeypatch.setattr("src.components.data_preparation.RAW_PATH", raw_file)
    monkeypatch.setattr("src.components.data_preparation.TRAIN_PATH", train_file)
    monkeypatch.setattr("src.components.data_preparation.VAL_PATH", val_file)
    monkeypatch.setattr("src.components.data_preparation.TEST_PATH", test_file)
    monkeypatch.setattr("src.components.data_preparation.PROJECT_ROOT", tmp_path)

    # Mock NLTK stopwords
    monkeypatch.setattr("nltk.corpus.stopwords.words", lambda lang: set())

    # 4. Run Preparation
    from src.entity.config_entity import DataPreparationConfig

    config = DataPreparationConfig(test_size=0.2, random_state=42)
    prep = DataPreparation(config)
    prep.prepare_reddit_dataset()

    # 5. Assertions
    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()

    train_df = pd.read_parquet(train_file)
    assert "category_encoded" in train_df.columns
    assert "sentiment_label" in train_df.columns
    # Check encoding -1 -> 0, 0 -> 1, 1 -> 2
    assert train_df[train_df["category"] == 1]["category_encoded"].iloc[0] == 2
