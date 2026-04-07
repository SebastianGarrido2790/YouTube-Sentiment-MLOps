"""
Execution Conductor: Data Ingestion.

This pipeline stage orchestrates the downloading of the raw dataset.
It acts as a thin wrapper that initializes the DataIngestion component
(the "Business Logic") using configuration parameters tracked by DVC.

Usage (preferred):
    uv run dvc repro
Run specific pipeline stage:
    uv run dvc repro data_ingestion
"""

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.constants import RAW_PATH
from src.entity.config_entity import DataIngestionConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="download_dataset.py")


class DataIngestionPipeline:
    """
    Pipeline stage execution class orchestrating the Data Ingestion phase.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self):
        """Execute the data ingestion step by triggering the worker component."""
        logger.info("🚀 Starting Download Process 🚀")
        try:
            logger.info("Loading params via ConfigurationManager")
            config = ConfigurationManager().get_data_ingestion_config()
        except Exception as e:
            logger.warning(f"Could not load params via ConfigurationManager: {e}")
            logger.warning("Falling back to defaults (only for local debugging).")
            config = DataIngestionConfig(
                url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv",
                output_path=str(RAW_PATH),
            )

        if not config.url:
            logger.error("URL not found in configuration. Aborting.")
            return

        data_ingestion = DataIngestion(config=config)
        data_ingestion.download_file()


if __name__ == "__main__":
    try:
        pipeline = DataIngestionPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
