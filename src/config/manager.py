"""
Configuration Management Module.

This module provides a singleton `ConfigurationManager` class that centralizes
the loading, validation, and access of project configurations. It serves as
the **Single Source of Truth** for all pipeline stages.

Key Features:
- **Singleton Pattern:** Ensures config is loaded only once per execution.
- **Pydantic Validation:** Uses `src.config.schemas` to enforce deep type checking.
- **Strict Typing:** All accessor methods return specific Pydantic models, enabling IDE autocompletion and error checking.
- **Fail-Fast:** Raises errors immediately if `params.yaml` is missing or invalid.
"""

from pathlib import Path
import yaml
from typing import Optional
from src.config.schemas import (
    AppConfig,
    DataPreparationConfig,
    DataIngestionConfig,
    FeatureComparisonConfig,
    FeatureTuningConfig,
    ImbalanceTuningConfig,
    FeatureEngineeringConfig,
    LogisticBaselineConfig,
    TrainConfig,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationManager:
    """
    Singleton class to manage application configuration.

    This class loads `params.yaml` upon instantiation (or access) and validates it
    against the `AppConfig` Pydantic schema. It provides typed accessor methods
    for each pipeline stage's configuration.

    Usage:
        >>> config = ConfigurationManager()
        >>> data_config = config.get_data_ingestion_config()
        >>> print(data_config.url)
    """

    _instance = None

    def __new__(cls, params_path: str = "params.yaml"):
        """Ensures only one instance of configuration is loaded."""
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, params_path: str = "params.yaml"):
        """
        Initialize the configuration manager.

        Args:
            params_path: Path to the parameter file (default: "params.yaml").
        """
        if self._initialized:
            return

        self.params_path = Path(params_path)
        self.config: Optional[AppConfig] = None
        self._load_params()
        self._initialized = True

    def _load_params(self):
        """
        Internal method to load YAML and validate with Pydantic.

        Raises:
            Exception: If validation fails or file read error occurs.
        """
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
        """Returns the configuration for the Data Ingestion stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.data_ingestion

    def get_data_preparation_config(self) -> DataPreparationConfig:
        """Returns the configuration for the Data Preparation stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.data_preparation

    def get_mlflow_config(self) -> str:
        """
        Returns the MLflow URI string from the configuration.

        Note: This is often used as a fallback if env vars are missing.
        """
        # Re-using the logic from schemas but strictly typed now
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.feature_comparison.mlflow_uri

    def get_feature_comparison_config(self) -> FeatureComparisonConfig:
        """Returns the configuration for the Feature/Vectorization Comparison stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.feature_comparison

    def get_feature_tuning_config(self) -> FeatureTuningConfig:
        """Returns the configuration for the TF-IDF Max Features Tuning stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.feature_tuning

    def get_imbalance_tuning_config(self) -> ImbalanceTuningConfig:
        """Returns the configuration for the Imbalance Tuning stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.imbalance_tuning

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """Returns the configuration for the Feature Engineering stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.feature_engineering

    def get_logistic_baseline_config(self) -> LogisticBaselineConfig:
        """Returns the configuration for the Logistic Regression Baseline."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.train.logistic_baseline

    def get_train_config(self) -> TrainConfig:
        """Returns the master configuration for the training stage."""
        if not self.config:
            raise RuntimeError("Configuration not loaded.")
        return self.config.train
