"""
Data Preprocessing and Validation Suite

This module contains unit tests for the data cleaning and preprocessing logic.
It validates the normalization of text data, including special character
removal, stopword filtering, and handling of null or empty values.
"""

from typing import Any

import pandas as pd
import pytest

from src.data.make_dataset import clean_text


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
    assert clean_text(text) == expected


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
    assert clean_text(text, stop_words) == expected
