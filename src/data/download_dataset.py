"""
Download a raw dataset from an external source using DVC parameters.

This script fetches a dataset (e.g., Reddit sentiment CSV) as a proxy for initial
model training and saves it to the `data/raw/` directory. It is designed to be
run as part of a DVC pipeline, ensuring reproducibility.

Usage (preferred):
    uv run dvc repro               # Uses params.yaml → fully reproducible
    Run specific pipeline stage:
    uv run dvc repro data_ingestion

Usage (local cli override only):
    uv run python -m src.data.download_dataset --url <new_url>

Requirements:
    - Raw data URL defined in params.yaml under `data_ingestion`.
    - `uv sync` must be run (for `requests` and `dvc` dependencies).

Design:
    - Parameters (URL, output path) are read from `params.yaml` via `dvc.api`.
    - CLI arguments are optional and only override for quick local testing.
    - All runs are reproducible when using DVC, as overrides are discouraged.
"""

import argparse
import os
from typing import Dict, Any

import requests
import dvc.api

# --- Project Utilities ---
from src.utils.paths import RAW_PATH
from src.utils.logger import get_logger

# --- Logging Setup ---
logger = get_logger(__name__, headline="download_dataset.py")


def load_params() -> Dict[str, Any]:
    """
    Load data ingestion parameters from params.yaml using DVC.
    Falls back gracefully if running outside a DVC pipeline.
    """
    try:
        logger.info("Loading params via dvc.api")
        params = dvc.api.params_show()
        return params["data_ingestion"]
    except Exception as e:
        logger.warning(f"Could not load params via dvc.api: {e}")
        logger.warning("Falling back to defaults (only for local debugging).")
        # Provide a fallback URL for standalone execution without DVC
        return {
            "url": "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv",
            "output_path": str(RAW_PATH),
        }


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

        logger.info(f"✅ Successfully downloaded and saved data to: {output_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def main():
    """
    Main function to parse arguments and initiate the data download.
    Parameters are sourced from params.yaml by default, with CLI overrides.
    """
    # Load parameters from params.yaml via DVC
    params = load_params()
    default_url = params.get("url")
    default_output_path = params.get("output_path", str(RAW_PATH))

    if not default_url:
        logger.error("URL not found in params.yaml or fallback. Aborting.")
        return

    parser = argparse.ArgumentParser(
        description="Download the raw dataset. Parameters come from params.yaml by default."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        help="URL of the dataset to download (default from params.yaml).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help=f"Local path to save the dataset (default: {default_output_path} from params.yaml).",
    )

    args = parser.parse_args()

    # Determine final URL and output path, preferring CLI overrides
    final_url = args.url if args.url is not None else default_url
    final_output_path = (
        args.output_path if args.output_path is not None else default_output_path
    )

    if args.url is not None or args.output_path is not None:
        logger.warning(
            "CLI override detected. This run may not be reproducible with 'dvc repro'. "
            "For reproducible runs, update params.yaml and run 'dvc repro'."
        )

    # Download the file
    logger.info(f"--- Starting download from {final_url} to {final_output_path} ---")
    download_file(final_url, final_output_path)


if __name__ == "__main__":
    main()
