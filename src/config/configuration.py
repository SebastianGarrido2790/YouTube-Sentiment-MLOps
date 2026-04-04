"""
Configuration Management Module.

This module provides a singleton `ConfigurationManager` class that centralizes
the loading, validation, and access of project configurations. It serves as
the **Single Source of Truth** for all pipelines.
"""

from pathlib import Path

import yaml

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    AppConfig,
    DataIngestionConfig,
    DataPreparationConfig,
    DistilBERTConfig,
    FeatureComparisonConfig,
    FeatureEngineeringConfig,
    FeatureTuningConfig,
    ImbalanceTuningConfig,
    LogisticBaselineConfig,
    ModelEvaluationConfig,
    RegisterConfig,
    SchemaConfig,
    SystemConfig,
    TrainConfig,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationManager:
    """Singleton Configuration Manager."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        params_path: Path = PARAMS_FILE_PATH,
        config_path: Path = CONFIG_FILE_PATH,
        schema_path: Path = SCHEMA_FILE_PATH,
    ):
        if self._initialized:
            return

        self.params_path = Path(params_path)
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)

        self.config: AppConfig | None = None
        self.system: SystemConfig | None = None
        self.data_schema: SchemaConfig | None = None

        self._load_configs()
        self._initialized = True

    def _load_configs(self):
        """Loads and validates all configs."""
        try:
            # 1. Hyperparameters
            if self.params_path.exists():
                with open(self.params_path) as f:
                    self.config = AppConfig(**yaml.safe_load(f))
            else:
                logger.warning(f"{self.params_path} not found.")

            # 2. System Paths
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self.system = SystemConfig(**yaml.safe_load(f))
            else:
                logger.warning(f"{self.config_path} not found.")

            # 3. Data Schema
            if self.schema_path.exists():
                with open(self.schema_path) as f:
                    self.data_schema = SchemaConfig(**yaml.safe_load(f))
            else:
                logger.warning(f"{self.schema_path} not found.")

            logger.info("All configurations loaded and validated successfully.")

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise e

    # --- System & Schema Getters ---
    def get_system_config(self) -> SystemConfig:
        if not self.system:
            raise RuntimeError("System configuration not loaded.")
        return self.system

    def get_data_schema(self) -> SchemaConfig:
        if not self.data_schema:
            raise RuntimeError("Data schema not loaded.")
        return self.data_schema

    # --- Hyperparameter Getters ---
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.data_ingestion

    def get_data_preparation_config(self) -> DataPreparationConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.data_preparation

    def get_mlflow_config(self) -> str:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.feature_comparison.mlflow_uri

    def get_feature_comparison_config(self) -> FeatureComparisonConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.feature_comparison

    def get_feature_tuning_config(self) -> FeatureTuningConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.feature_tuning

    def get_imbalance_tuning_config(self) -> ImbalanceTuningConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.imbalance_tuning

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.feature_engineering

    def get_logistic_baseline_config(self) -> LogisticBaselineConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.train.logistic_baseline

    def get_train_config(self) -> TrainConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.train

    def get_distilbert_config(self) -> DistilBERTConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.train.distilbert

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.model_evaluation

    def get_register_config(self) -> RegisterConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.register_config
