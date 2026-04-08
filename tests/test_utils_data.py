"""
Unit Tests for Data Loading Utilities

Tests the feature loading, text loading, and imbalance correction (ADASYN) logic.
Ensures that data transformation utilities maintain shape and type consistency.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from src.utils.data_loader import apply_adasyn, load_feature_data, load_text_data


@pytest.fixture
def mock_feature_files(tmp_path, monkeypatch):
    """
    Creates temporary .npz, .npy, and .pkl files to simulate pre-engineered features.
    """
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    # Mock the project constants
    monkeypatch.setattr("src.utils.data_loader.FEATURES_DIR", feature_dir)
    monkeypatch.setattr("src.utils.data_loader.PROCESSED_DATA_DIR", feature_dir)

    # 1. Sparse Mats
    X = csr_matrix([[1, 0], [0, 1], [1, 1]])
    save_npz(feature_dir / "X_train.npz", X)
    save_npz(feature_dir / "X_val.npz", X)
    save_npz(feature_dir / "X_test.npz", X)

    # 2. Labels
    y = np.array([0, 1, 0])
    np.save(feature_dir / "y_train.npy", y)
    np.save(feature_dir / "y_val.npy", y)
    np.save(feature_dir / "y_test.npy", y)

    # 3. Label Encoder
    le = "mock_label_encoder"
    with open(feature_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return feature_dir


def test_load_feature_data(mock_feature_files):
    """
    Tests loading of sparse feature matrices and labels.
    """
    X_train, _, _, y_train, _, _, _ = load_feature_data()

    assert X_train.shape == (3, 2)
    assert y_train.shape == (3,)
    assert isinstance(X_train, csr_matrix)


def test_load_feature_data_missing(tmp_path, monkeypatch):
    """
    Tests that FileNotFoundError is raised when files are missing.
    """
    monkeypatch.setattr("src.utils.data_loader.FEATURES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_feature_data(validate_files=True)


def test_load_text_data(tmp_path, monkeypatch):
    """
    Tests loading of processed parquet text data.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    monkeypatch.setattr("src.utils.data_loader.PROCESSED_DATA_DIR", processed_dir)

    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    df.to_parquet(processed_dir / "train.parquet")
    df.to_parquet(processed_dir / "val.parquet")
    df.to_parquet(processed_dir / "test.parquet")

    train_df, _, _ = load_text_data()
    assert len(train_df) == 2
    assert "text" in train_df.columns


def test_apply_adasyn():
    """
    Tests ADASYN imbalance correction with a simple dataset.
    """
    # Create an imbalanced dataset
    # Class 0: 4 samples
    # Class 1: 30 samples
    # We need at least n_neighbors for ADASYN, so let's make it 6 for class 0
    y = np.array([0] * 6 + [1] * 30)
    X = np.random.rand(36, 10)

    _, y_res = apply_adasyn(X, y)

    # Resampled dataset should be more balanced (not exactly equal, but closer)
    assert len(y_res) > len(y)
    # Check count of class 0
    unique, counts = np.unique(y_res, return_counts=True)
    class_counts = dict(zip(unique, counts, strict=True))
    assert class_counts[0] > 6
