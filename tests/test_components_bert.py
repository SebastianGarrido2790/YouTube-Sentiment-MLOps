"""
Unit Tests for DistilBERT Training Component

Tests the fine-tuning logic for transformers using mocks.
Ensures the HuggingFace trainer is correctly instantiated.
"""

import sys
from unittest.mock import MagicMock, patch

# Mocking expensive transformer imports before they are called
mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["datasets"] = MagicMock()

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.components.distilbert_training import DistilBERTTraining  # noqa: E402
from src.entity.config_entity import DistilBERTConfig  # noqa: E402


@pytest.fixture
def bert_config():
    return DistilBERTConfig(
        enable=True,
        n_trials=1,
        batch_size=[16, 32],
        lr=[2e-5, 3e-5],
        weight_decay=[0.01, 0.1],
    )


@pytest.fixture
def trainer(bert_config):
    with patch("src.components.distilbert_training.DistilBERTTraining.create_placeholder_artifacts"):
        yield DistilBERTTraining(bert_config)


@patch("src.components.distilbert_training.torch.cuda.is_available", return_value=True)
@patch("src.components.distilbert_training.AutoTokenizer.from_pretrained")
@patch("src.components.distilbert_training.AutoModelForSequenceClassification.from_pretrained")
@patch("src.components.distilbert_training.Trainer")
@patch("src.components.distilbert_training.Dataset.from_pandas")
@patch("src.components.distilbert_training.load_text_data")
@patch("src.components.distilbert_training.mlflow")
def test_train_distilbert_success(
    mock_mlflow, mock_load_data, mock_ds_csv, mock_trainer, mock_model, mock_tok, mock_cuda, trainer
):
    """Tests the full fine-tuning orchestration with mocks."""
    # Mock data loading
    mock_load_data.return_value = (
        pd.DataFrame({"clean_comment": ["test"], "category": [1]}),
        pd.DataFrame({"clean_comment": ["test"], "category": [1]}),
        pd.DataFrame({"clean_comment": ["test"], "category": [1]}),
    )

    # Mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_tok.return_value = mock_tokenizer
    mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": [0]}

    # Mock trainer
    mock_trainer_inst = MagicMock()
    mock_trainer.return_value = mock_trainer_inst
    mock_trainer_inst.evaluate.return_value = {"eval_macro_f1": 0.85}

    trainer.fine_tune()

    assert mock_tok.called
    assert mock_model.called
    assert mock_trainer.called
    assert mock_trainer_inst.train.called
    assert mock_mlflow.transformers.log_model.called
