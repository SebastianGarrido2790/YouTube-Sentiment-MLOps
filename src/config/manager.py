from pathlib import Path
import yaml
from typing import Optional
from src.config.schemas import AppConfig, DataPreparationConfig, DataIngestionConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationManager:
    _instance = None

    def __new__(cls, params_path: str = "params.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, params_path: str = "params.yaml"):
        if self._initialized:
            return

        self.params_path = Path(params_path)
        self.config: Optional[AppConfig] = None
        self._load_params()
        self._initialized = True

    def _load_params(self):
        """Loads parameters from yaml file and validates with Pydantic."""
        if not self.params_path.exists():
            logger.warning(
                f"{self.params_path} not found. Configuration will be incomplete."
            )
            return

        try:
            with open(self.params_path, "r") as f:
                params_dict = yaml.safe_load(f)

            logger.info("Validating configuration against schemas...")
            self.config = AppConfig(**params_dict)
            logger.info("Configuration loaded and validated successfully.")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.data_ingestion

    def get_data_preparation_config(self) -> DataPreparationConfig:
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.data_preparation

    def get_mlflow_config(self) -> str:
        # Re-using the logic from schemas but strictly typed now
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.feature_comparison.mlflow_uri
