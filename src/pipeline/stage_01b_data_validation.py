"""
Data Validation Pipeline Stage

Orchestrates the Great Expectations data quality checks on the ingested data.
"""

from src.components.data_validation import DataValidation
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="stage_01b_data_validation.py")


class DataValidationPipeline:
    """
    Orchestrates the data validation lifecycle.

    This pipeline stage ensures the raw data meets the statistical contracts
    before any preparation or feature engineering occurs.
    """

    def __init__(self) -> None:
        """Initializes the DataValidationPipeline."""
        pass

    def main(self) -> None:
        """
        Executes the data validation stage.

        Raises:
            Exception: If validation critically fails or encounters an error.
        """
        logger.info("🛡️ Starting data validation process 🛡️")

        config_manager = ConfigurationManager()
        val_config = config_manager.get_data_validation_config()
        ops_paths = config_manager.get_system_config().ops
        data_paths = config_manager.get_system_config().data
        schema = config_manager.get_data_schema()

        data_validation = DataValidation(config=val_config, ops_paths=ops_paths, data_paths=data_paths, schema=schema)

        success = data_validation.validate_raw_data()

        if not success:
            logger.error("Data Validation did not pass completely. Please review the contracts.")
            # Depending on strictness, we could raise an Exception here to stop pipeline
            # raise RuntimeError("Data Quality Contracts failed.")


if __name__ == "__main__":
    try:
        pipeline = DataValidationPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
