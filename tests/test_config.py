"""
Configuration Validation Suite

Automated unit tests for the ConfigurationManager and configuration hydration
system. This suite ensures that the Pydantic-based configuration schemas
correctly validate parameters and that the singleton manager loads the
environment as expected.
"""

import pytest

from src.config.configuration import ConfigurationManager
from src.config.schemas import AppConfig, DataPreparationConfig


def test_config_loading(config_manager: ConfigurationManager):
    """
    Tests that the configuration is loaded and validated correctly.

    Arrange:
        - The `config_manager` fixture is initialized with a mock `params.yaml`.

    Act:
        - Access the `config` attribute of the ConfigurationManager.

    Assert:
        - `config` is not None and is an instance of `AppConfig`.
        - Specific values (e.g., `test_size`) match the mock parameters.
    """
    assert config_manager.config is not None
    assert isinstance(config_manager.config, AppConfig)
    assert config_manager.config.data_preparation.test_size == 0.2


def test_get_data_preparation_config(config_manager: ConfigurationManager):
    """
    Tests retrieval of specific configuration sections.

    Arrange:
        - The `config_manager` fixture is initialized with a mock `params.yaml`.

    Act:
        - Call `get_data_preparation_config()`.

    Assert:
        - Returned object is an instance of `DataPreparationConfig`.
        - Attributes (`test_size`, `random_state`) match the mock parameters.
    """
    data_prep_config = config_manager.get_data_preparation_config()
    assert isinstance(data_prep_config, DataPreparationConfig)
    assert data_prep_config.test_size == 0.2
    assert data_prep_config.random_state == 123


def test_missing_config_file():
    """
    Tests the system behavior when the configuration file is missing.

    Arrange:
        - Reset the ConfigurationManager singleton instance.
        - Initialize `ConfigurationManager` with a non-existent path.

    Act:
        - Attempt to access configuration or getter methods.

    Assert:
        - `config` attribute is None.
        - `get_data_preparation_config()` raises a `RuntimeError` due to missing data.
    """
    ConfigurationManager._instance = None
    manager = ConfigurationManager(params_path="non_existent.yaml")
    assert manager.config is None

    with pytest.raises(RuntimeError, match="Configuration not loaded"):
        manager.get_data_preparation_config()
