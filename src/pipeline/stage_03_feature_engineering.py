from src.components.feature_engineering import FeatureEngineering
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="stage_03_feature_engineering.py")


class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
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
