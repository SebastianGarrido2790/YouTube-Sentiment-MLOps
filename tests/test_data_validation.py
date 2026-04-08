"""
Unit Tests for Data Validation Component

Tests the integration with Great Expectations to enforce data contracts.
Ensures that schema validation, null checks, and text length checks are performed.
"""

import os

import pandas as pd
import pytest

from src.components.data_validation import DataValidation
from src.entity.config_entity import DataPathsConfig, DataValidationConfig, OpsPathsConfig, SchemaConfig


@pytest.fixture
def validation_setup(tmp_path):
    """
    Sets up the necessary configs and a dummy CSV for DataValidation tests.
    """
    raw_path = tmp_path / "raw" / "dataset.csv"
    raw_path.parent.mkdir(parents=True)

    # Create valid dummy data
    df = pd.DataFrame({"clean_comment": ["This is great", "I love this", "Amazing video"], "category": [1, 1, 1]})
    df.to_csv(raw_path, index=False)

    config = DataValidationConfig(null_threshold_percent=0.01, min_text_length=5, max_text_length=500)
    ops_paths = OpsPathsConfig(logs_dir="logs", mlruns_dir="mlruns", gx_dir=str(tmp_path / "gx"))
    data_paths = DataPathsConfig(
        raw_dir="data/raw",
        external_dir="data/external",
        processed_dir="data/processed",
        raw_path=str(raw_path),
        train_path="data/train",
        test_path="data/test",
        val_path="data/val",
    )
    schema = SchemaConfig(columns={"clean_comment": "string", "category": "integer"}, target="category")

    return DataValidation(config, ops_paths, data_paths, schema)


def test_validate_raw_data_success(validation_setup):
    """
    Tests that a valid dataset passes validation.
    """
    success = validation_setup.validate_raw_data()
    assert success is True

    # Verify that GX output files were created
    assert os.path.exists(os.path.join(validation_setup.ops_paths.gx_dir, "youtube_sentiment_raw_suite.json"))
    assert os.path.exists(os.path.join(validation_setup.ops_paths.gx_dir, "validation_results.json"))


def test_validate_raw_data_fail_missing_column(validation_setup):
    """
    Tests that validation fails if a required column is missing.
    """
    # Overwrite raw data with missing 'category' column
    df = pd.DataFrame({"clean_comment": ["test"]})
    df.to_csv(validation_setup.data_paths.raw_path, index=False)

    success = validation_setup.validate_raw_data()
    assert success is False


def test_validate_raw_data_fail_null_threshold(validation_setup):
    """
    Tests that validation fails if null percentage exceeds threshold.
    """
    # Overwrite raw data with nulls in 'clean_comment'
    df = pd.DataFrame({"clean_comment": [None, None, "Valid"], "category": [1, 1, 1]})
    df.to_csv(validation_setup.data_paths.raw_path, index=False)

    success = validation_setup.validate_raw_data()
    assert success is False


def test_validate_raw_data_missing_file(validation_setup):
    """
    Tests handling of missing raw data file.
    """
    os.remove(validation_setup.data_paths.raw_path)
    with pytest.raises(FileNotFoundError):
        validation_setup.validate_raw_data()


def test_validate_raw_data_empty_file(validation_setup):
    """
    Tests handling of empty raw data file.
    """
    df = pd.DataFrame()
    df.to_csv(validation_setup.data_paths.raw_path, index=False)
    with pytest.raises(ValueError, match="Raw dataset is empty"):
        validation_setup.validate_raw_data()
