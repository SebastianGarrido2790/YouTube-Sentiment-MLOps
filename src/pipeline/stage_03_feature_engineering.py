"""
Execution Conductor: Feature Engineering.

This pipeline stage orchestrates the process of transforming raw text into features.
It delegates the actual NLP processing and TF-IDF matrix creation to the
`FeatureEngineering` component.

Usage:
Run the entire pipeline:
    uv run dvc repro
Run specific pipeline stage:
    uv run python -m src.pipeline.stage_03_feature_engineering
"""

from src.components.feature_engineering import FeatureEngineering
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="stage_03_feature_engineering.py")


class FeatureEngineeringPipeline:
    """
    Pipeline stage execution class orchestrating the text vectorization phase.
    """

    def __init__(self):
        """Initialize the execution pipeline stage."""
        pass

    def main(self):
        """Execute the feature engineering step by triggering the worker component."""
        config = ConfigurationManager().get_feature_engineering_config()
        feature_engineering = FeatureEngineering(config=config)
        feature_engineering.build_features()


if __name__ == "__main__":
    try:
        pipeline = FeatureEngineeringPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        raise e
