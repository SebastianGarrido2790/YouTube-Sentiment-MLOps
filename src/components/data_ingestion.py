"""
Component for downloading and ingesting raw datasets.

This module provides the worker component responsible for fetching the
raw data from a remote URL and saving it securely locally.
"""

import os

import requests

from src.entity.config_entity import DataIngestionConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="data_ingestion_component")


class DataIngestion:
    """
    Component handling the logic of data downloading and ingestion.

    Attributes:
        config (DataIngestionConfig): Configuration containing URL and output paths.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion component.

        Args:
            config (DataIngestionConfig): Configuration parameters for the data ingestion process.
        """
        self.config = config

    def download_file(self):
        """
        Download the data file from the configured URL and save it to the output path.

        Creates directory structures if they do not exist and streams the download
        in chunks to avoid memory bottlenecks on large datasets.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        url = self.config.url
        output_path = self.config.output_path

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")

        try:
            logger.info(f"Downloading data from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✅ Successfully downloaded and saved data to: {output_path} ✅")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading data: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
