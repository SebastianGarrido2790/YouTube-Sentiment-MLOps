import pytest
from src.config.manager import ConfigurationManager
from src.config.schemas import AppConfig, DataPreparationConfig


def test_config_loading(config_manager):
    """Test that configuration is loaded and validated correctly."""
    assert config_manager.config is not None
    assert isinstance(config_manager.config, AppConfig)
    assert config_manager.config.data_preparation.test_size == 0.2


def test_get_data_preparation_config(config_manager):
    """Test retrieval of specific config section."""
    data_prep_config = config_manager.get_data_preparation_config()
    assert isinstance(data_prep_config, DataPreparationConfig)
    assert data_prep_config.test_size == 0.2
    assert data_prep_config.random_state == 123


def test_missing_config_file():
    """Test behavior when params.yaml is missing."""
    ConfigurationManager._instance = None
    manager = ConfigurationManager(params_path="non_existent.yaml")
    assert manager.config is None

    with pytest.raises(RuntimeError, match="Configuration not loaded"):
        manager.get_data_preparation_config()
