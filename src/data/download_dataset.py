"""
Download a raw dataset from an external source using DVC parameters.

This script fetches a dataset (e.g., Reddit sentiment CSV) as a proxy for initial
model training and saves it to the `data/raw/` directory.

Usage (preferred):
    uv run dvc repro               # Uses params.yaml → fully reproducible
Run specific pipeline stage:
    uv run dvc repro data_ingestion
"""

import os
import requests

# --- Project Utilities ---
from src.utils.paths import RAW_PATH
from src.utils.logger import get_logger
from src.config.manager import ConfigurationManager
from src.config.schemas import DataIngestionConfig

# --- Logging Setup ---
logger = get_logger(__name__, headline="download_dataset.py")


def load_params() -> DataIngestionConfig:
    """
    Load data ingestion parameters from params.yaml using ConfigurationManager.
    """
    try:
        logger.info("Loading params via ConfigurationManager")
        config = ConfigurationManager().get_data_ingestion_config()
        return config
    except Exception as e:
        logger.warning(f"Could not load params via ConfigurationManager: {e}")
        logger.warning("Falling back to defaults (only for local debugging).")
        # Provide a fallback URL for standalone execution without DVC
        return DataIngestionConfig(
            url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv",
            output_path=str(RAW_PATH),
        )


def download_file(url: str, output_path: str):
    """
    Downloads a file from a URL and saves it to the specified path.

    Args:
        url (str): The public URL of the file to download.
        output_path (str): The local path to save the downloaded file.
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    try:
        # Use requests to download the file content
        logger.info(f"Downloading data from: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Write the content to the file
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


def main():
    """
    Main function to initiate the data download.
    Parameters are sourced exclusively from params.yaml via ConfigurationManager.
    """
    logger.info("🚀 Starting Download Process 🚀")

    # Load parameters from params.yaml
    config = load_params()

    if not config.url:
        logger.error("URL not found in configuration. Aborting.")
        return

    # Download the file
    download_file(config.url, config.output_path)


if __name__ == "__main__":
    main()
