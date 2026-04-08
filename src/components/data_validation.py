"""
Data Validation Component using Great Expectations.

This component is responsible for ensuring the quality and integrity of the
ingested raw dataset using statistical data quality contracts.
"""

import json
import os

import great_expectations as gx
import pandas as pd

from src.entity.config_entity import DataPathsConfig, DataValidationConfig, OpsPathsConfig, SchemaConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidation:
    """
    Component for profiling and validating data quality.
    """

    def __init__(
        self, config: DataValidationConfig, ops_paths: OpsPathsConfig, data_paths: DataPathsConfig, schema: SchemaConfig
    ):
        self.config = config
        self.ops_paths = ops_paths
        self.data_paths = data_paths
        self.schema = schema

        # Ensure contracts directory exists
        os.makedirs(self.ops_paths.gx_dir, exist_ok=True)

    def validate_raw_data(self) -> bool:
        """
        Validates the raw data against the defined Great Expectations suite.
        """
        logger.info("Initializing Great Expectations context...")

        # Load raw data
        raw_data_path = self.data_paths.raw_path
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")

        try:
            df = pd.read_csv(raw_data_path)
        except pd.errors.EmptyDataError:
            raise ValueError("Raw dataset is empty. Validation failed.") from None

        # Fast fail if empty
        if df.empty:
            raise ValueError("Raw dataset is empty. Validation failed.")

        logger.info(f"Loaded dataset with {len(df)} rows for validation.")

        # Create an ephemeral data context
        context = gx.get_context()

        # Instead of configuring a full GX project, we use in-memory approach
        # Create a pandas data source
        data_source = context.data_sources.add_pandas("youtube_sentiment_source")
        data_asset = data_source.add_dataframe_asset("raw_youtube_comments")
        batch_definition = data_asset.add_batch_definition_whole_dataframe("raw_batch")

        # Create or update expectation suite
        suite_name = "youtube_sentiment_raw_suite"
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

        # Add expectations
        logger.info("Defining statistical data quality contracts...")

        # 1. Expected columns must exist
        expected_columns = list(self.schema.columns.keys())
        for col in expected_columns:
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

        # 2. Value distribution (null % thresholds)
        # Using ExpectColumnValuesToNotBeNull with mostly parameter
        mostly = 1.0 - (self.config.null_threshold_percent / 100.0)
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="clean_comment", mostly=mostly))

        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="category"))

        # 3. Text length range checks
        suite.add_expectation(
            gx.expectations.ExpectColumnValueLengthsToBeBetween(
                column="clean_comment",
                min_value=self.config.min_text_length,
                max_value=self.config.max_text_length,
                mostly=mostly,
            )
        )

        # 4. Label balance monitoring (categories must be within known range -1, 0, 1)
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="category",
                value_set=[-1, 0, 1],
                mostly=0.99,  # Allow tiny percentage of potential parsing issues
            )
        )

        # Save suite to Artifacts manually since we use Ephemeral context
        contract_path = os.path.join(self.ops_paths.gx_dir, f"{suite_name}.json")
        with open(contract_path, "w") as f:
            json.dump(suite.to_json_dict(), f, indent=4)
        logger.info(f"Expectation suite saved to: {contract_path}")

        # Run Validation
        logger.info("Running validation against the data...")
        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(name="raw_data_validation", data=batch_definition, suite=suite)
        )

        validation_results = validation_definition.run(batch_parameters={"dataframe": df})

        success = validation_results.success

        # Log results
        result_path = os.path.join(self.ops_paths.gx_dir, "validation_results.json")
        with open(result_path, "w") as f:
            json.dump(validation_results.to_json_dict(), f, indent=4)

        if success:
            logger.info("✅ Data Validation PASSED.")
        else:
            logger.warning("❌ Data Validation FAILED with some expectations.")
            logger.warning(f"Check the results here: {result_path}")

        return success
