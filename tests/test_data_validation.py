import pytest
import pandas as pd
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
def test_clean_text_general(text, expected):
    """
    Test general text cleaning scenarios.

    Covers:
    - Basic lowercase and non-alpha removal
    - Special character removal
    - Edge cases (empty, NA, None)
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
def test_clean_text_with_stopwords(text, stop_words, expected):
    """
    Test text cleaning with stopword removal enabled.

    Covers:
    - Stopword filtering
    - Short token filtering (len <= 2) when stopwords are provided
    """
    assert clean_text(text, stop_words) == expected
