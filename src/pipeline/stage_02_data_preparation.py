"""
Data Preparation Pipeline Stage

Orchestrates the NLTK resource management, directory initialization,
and delegates core dataset cleaning and splitting logic.
"""

import os

import nltk

from src.components.data_preparation import DataPreparation
from src.config.configuration import ConfigurationManager
from src.constants import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="stage_02_data_preparation.py")


class DataPreparationPipeline:
    """
    Orchestrates the data preparation lifecycle.

    This pipeline stage handles NLTK resource management, directory initialization,
    and delegates the core cleaning/splitting logic to the DataPreparation component.
    """

    def __init__(self) -> None:
        """Initializes the DataPreparationPipeline."""
        pass

    def main(self) -> None:
        """
        Executes the data preparation stage.

        Downloads required NLTK resources, ensures output directories exist,
        and triggers the dataset preparation logic.

        Raises:
            Exception: If any part of the preparation pipeline fails.
        """
        logger.info("🚀 Starting data preparation process 🚀")
        logger.info("Downloading NLTK data (if not already present)...")
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        logger.info("NLTK data download complete.")

        # Ensure processed directory exists
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        config = ConfigurationManager().get_data_preparation_config()
        data_preparation = DataPreparation(config=config)
        data_preparation.prepare_reddit_dataset()


if __name__ == "__main__":
    try:
        pipeline = DataPreparationPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
