"""
Unit Tests for Feature Engineering Component

Tests the numerical feature generation logic that replaces the old utility scripts.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.components.feature_engineering import FeatureEngineering
from src.entity.config_entity import FeatureEngineeringConfig


@pytest.fixture
def feature_config():
    return FeatureEngineeringConfig(
        use_distilbert=False,
        distilbert_batch_size=32,
        best_ngram_range="(1,2)",
        best_max_features=100,
    )


@pytest.fixture
def dummy_train_data():
    return pd.DataFrame(
        {
            "clean_comment": ["this is a good great video", "terrible bad worst video", "neutral average"],
            "category_encoded": [2, 0, 1],  # Let's assume Positive=2, Negative=0, Neutral=1
        }
    )


def test_build_derived_features(dummy_train_data):
    """Tests that domain-specific features (char_len, word_len, pos_ratio, neg_ratio) are generated."""
    features = FeatureEngineering.build_derived_features(dummy_train_data)

    assert features.shape == (3, 4)
    # Check "good great" ratios for the first comment
    assert features[0, 2] > 0.0  # pos_ratio
    assert features[0, 3] == 0.0  # neg_ratio


@patch("src.components.feature_engineering.pd.read_parquet")
@patch("src.components.feature_engineering.save_npz")
@patch("src.components.feature_engineering.np.save")
@patch("src.components.feature_engineering.pickle.dump")
def test_build_features_tfidf(
    mock_pickle, mock_np_save, mock_save_npz, mock_read_parquet, feature_config, dummy_train_data
):
    """Tests the execution of the main feature builder using TF-IDF."""
    mock_read_parquet.return_value = dummy_train_data

    fe = FeatureEngineering(feature_config)
    fe.build_features()

    assert mock_save_npz.call_count == 3  # train, val, test
    assert mock_np_save.call_count == 3  # y_train, y_val, y_test
    assert mock_pickle.call_count == 2  # vectorizer, label_encoder
