"""
Configuration Manager Validation Suite

Tests the loading and validation of system configurations using Pydantic.
Verifies that parameters, paths, and data contracts are correctly hydrated
from YAML files into immutable entity objects.
"""

from src.config.configuration import ConfigurationManager
from src.entity.config_entity import AppConfig, DataPreparationConfig


def test_configuration_manager_initialization(config_manager: ConfigurationManager):
    """
    Validates that the ConfigurationManager correctly loads all primary
    configuration objects and initializes them as the expected Pydantic entities.
    """
    assert isinstance(config_manager, ConfigurationManager)
    assert isinstance(config_manager.get_params(), AppConfig)


def test_data_preparation_config(config_manager: ConfigurationManager):
    """
    Tests the retrieval of a specific stage configuration (Data Preparation).
    Ensures that default values and overrides from the mock are applied.
    """
    dp_config = config_manager.get_data_preparation_config()

    assert isinstance(dp_config, DataPreparationConfig)
    # Based on our mock in conftest.py
    assert dp_config.test_size == 0.2
    assert dp_config.random_state == 42


def test_invalid_config_path():
    """
    Verifies that the ConfigurationManager raises an error when provided
    with a non-existent configuration path, ensuring early failure for misconfiguration.
    """
    import pytest

    ConfigurationManager._instance = None
    with pytest.raises(FileNotFoundError):
        ConfigurationManager(params_path="non_existent.yaml")
