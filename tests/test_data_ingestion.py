"""
Unit Tests for Data Ingestion Component

Tests the downloading and local storage logic for raw datasets.
Mocks external network calls to ensure isolation and speed.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


@pytest.fixture
def ingestion_config(tmp_path):
    """Provides a mock DataIngestionConfig with temporary paths."""
    return DataIngestionConfig(url="http://example.com/data.csv", output_path=str(tmp_path / "raw" / "dataset.csv"))


def test_download_file_success(ingestion_config):
    """
    Validates a successful file download.
    Ensures directories are created and data is written in chunks.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

    # Use patch to mock requests.get
    with patch("requests.get", return_value=mock_response) as mock_get:
        ingester = DataIngestion(ingestion_config)
        ingester.download_file()

        # Verify request call
        mock_get.assert_called_once()
        mock_response.raise_for_status.assert_called_once()

        # Verify file creation
        assert os.path.exists(ingestion_config.output_path)
        with open(ingestion_config.output_path, "rb") as f:
            content = f.read()
            assert content == b"chunk1chunk2"


def test_download_file_http_error(ingestion_config):
    """
    Tests handling of HTTP errors during download.
    Ensures that RequestException is raised and logged.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

    with patch("requests.get", return_value=mock_response):
        ingester = DataIngestion(ingestion_config)
        with pytest.raises(requests.exceptions.RequestException):
            ingester.download_file()


def test_download_file_unexpected_error(ingestion_config):
    """
    Tests handling of unexpected exceptions (e.g., OS errors).
    """
    with patch("requests.get", side_effect=Exception("Unexpected boom")):
        ingester = DataIngestion(ingestion_config)
        with pytest.raises(Exception, match="Unexpected boom"):
            ingester.download_file()
