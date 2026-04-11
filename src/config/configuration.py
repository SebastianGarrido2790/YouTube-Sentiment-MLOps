"""
Configuration Management Module.

This module provides a singleton `ConfigurationManager` class that centralizes
the loading, validation, and access of project configurations. It serves as
the Single Source of Truth for all pipelines.
"""

from pathlib import Path

import yaml

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    AgentConfig,
    AppConfig,
    DataIngestionConfig,
    DataPreparationConfig,
    DataValidationConfig,
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
            if not self.params_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.params_path}")
            with open(self.params_path) as f:
                self.config = AppConfig(**yaml.safe_load(f))

            # 2. System Paths
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            with open(self.config_path) as f:
                self.system = SystemConfig(**yaml.safe_load(f))

            # 3. Data Schema
            if not self.schema_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.schema_path}")
            with open(self.schema_path) as f:
                self.data_schema = SchemaConfig(**yaml.safe_load(f))

            logger.info("All configurations loaded and validated successfully.")

        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise e

    # --- System & Schema Getters ---
    def get_params(self) -> AppConfig:
        if not self.config:
            raise RuntimeError("App configuration not loaded.")
        return self.config

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

    def get_data_validation_config(self) -> DataValidationConfig:
        if not self.config:
            raise RuntimeError("App Config not loaded.")
        return self.config.data_validation

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

    def get_agent_config(self) -> AgentConfig:
        """
        Returns the Content Intelligence Analyst Agent configuration.
        Merges tunable parameters from params.yaml with infrastructure from config.yaml.
        """
        if not self.system:
            raise RuntimeError("System configuration not loaded.")
        if not self.config:
            raise RuntimeError("App configuration not loaded.")

        params = self.config.agent
        infra = self.system.agent

        return AgentConfig(
            model_name=params.model_name,
            max_comments=params.max_comments,
            fallback_enabled=params.fallback_enabled,
            fallback_model_name=params.fallback_model_name,
            inference_api_url=infra.inference_api_url,
            insights_api_url=infra.insights_api_url,
            tool_timeout_seconds=infra.tool_timeout_seconds,
        )
