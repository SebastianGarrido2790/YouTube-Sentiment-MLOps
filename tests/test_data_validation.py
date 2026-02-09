import pytest
import pandas as pd
from src.data.make_dataset import clean_text


def test_clean_text_basic():
    """Test basic text cleaning (lowercase, removal of non-alpha)."""
    text = "Hello World! 123"
    expected = "hello world"
    assert clean_text(text) == expected


def test_clean_text_special_chars():
    """Test removal of special characters."""
    text = "Sentim@nt An@lys!s #"
    expected = "sentim nt an lys s"
    assert clean_text(text) == expected


def test_clean_text_stopwords():
    """Test stopword removal."""
    text = "This is a test sentence"
    stop_words = {"this", "is", "a"}
    expected = "test sentence"
    # Note: 'clean_text' logic splits by space.
    # "This is a test sentence" -> lower -> "this is a test sentence"
    # -> tokens: ["this", "is", "a", "test", "sentence"]
    # -> filter: "test", "sentence" -> join -> "test sentence"
    assert clean_text(text, stop_words) == expected


def test_clean_text_empty_and_nan():
    """Test handling of empty strings and NaN."""
    assert clean_text("") == ""
    assert clean_text(pd.NA) == ""
    assert clean_text(None) == ""


def test_clean_text_short_tokens():
    """Test removal of short tokens when stopwords are provided."""
    # The logic in clean_text is: if stop_words provided, also remove len(t) <= 2
    text = "go to the gym"
    stop_words = {"the"}
    # "go", "to", "gym" -> "go"(2), "to"(2), "gym"(3)
    # expected: "gym"
    expected = "gym"
    assert clean_text(text, stop_words) == expected
