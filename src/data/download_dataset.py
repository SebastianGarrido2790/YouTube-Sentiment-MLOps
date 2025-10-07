"""
Download raw dataset from external source.

This script fetches the Reddit sentiment CSV as a **proxy dataset** for initial model training.
It saves the downloaded file to the `data/raw/` directory.

The script ensures the output directory exists, uses robust HTTP requests with error handling,
and leverages streamed downloads for efficiency.

Usage:
    uv run python -m src.data.download_dataset
    # Or, to specify a different path:
    uv run python -m src.data.download_dataset --output_path data/raw/new_data.csv

Requirements:
    - **uv sync** must be run (for `requests` dependency).

Design Considerations:
- **Reliability**: Uses `requests` with a timeout and raises exceptions on HTTP errors.
- **Scalability**: Uses streamed download for potentially large files.
- **Maintainability**: Clear configuration, standard Python logging, detailed docstrings, and type hints.
- **Adaptability**: Source URL and output path are configurable via command-line arguments, making it simple to swap in a YouTube-collected dataset later.
"""

import argparse
import logging
import os
import requests
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# The URL for the raw Reddit dataset you provided
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
# The target path for the raw data
DEFAULT_OUTPUT_PATH = "data/raw/reddit_comments.csv"

# --- Functions ---


def download_file(url: str, output_path: str):
    """
    Downloads a file from a URL and saves it to the specified path.

    Args:
        url (str): The public URL of the file to download.
        output_path (str): The local path to save the downloaded file.
    """
    logger.info(f"Attempting to download data from: {url}")

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    try:
        # Use requests to download the file content
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Write the content to the file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Successfully downloaded and saved data to: {output_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def main():
    """
    Main function to parse arguments and initiate the data download.
    """
    parser = argparse.ArgumentParser(
        description="Download the raw sentiment analysis dataset."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATA_URL,
        help="URL of the raw dataset to download.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Local path to save the downloaded raw dataset.",
    )

    args = parser.parse_args()

    # Download the file
    download_file(args.url, args.output_path)


if __name__ == "__main__":
    main()
