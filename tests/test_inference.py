"""
Inference Endpoint Validation Suite

Automated integration tests for the YouTube Sentiment Analysis API.
This suite validates sentiment prediction, Aspect-Based Sentiment Analysis (ABSA),
and system health checks using FastAPI's TestClient while mocking heavy
artifacts to enable isolated validation.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Creates a TestClient for the FastAPI app, mocking the model and artifacts.
    This ensures that tests remain deterministic and do not require a live
    MLflow server or large model files.
    """
    # 1. Setup Mock Model
    mock_model = MagicMock()
    # Mocking return value for a batch of 1 sample (numeric index 2 = Positive)
    import numpy as np

    mock_model.predict.return_value = np.array([2])

    # 2. Setup Mock Label Encoder
    mock_le = MagicMock()
    mock_le.inverse_transform.side_effect = lambda x: ["Positive" if p == 2 else "Neutral" for p in x]

    # 3. Setup Mock Vectorizer
    mock_vec = MagicMock()
    mock_vec.transform.return_value = MagicMock()

    # 5. Import app and manually inject mocks into its module namespace
    from src.api import main as api_main

    api_main.model = mock_model
    api_main.vectorizer = mock_vec
    api_main.label_encoder = mock_le

    # 6. Mock joblib and load_production_model to avoid lifespan loading real files
    with (
        patch("src.api.main.joblib.load") as mock_joblib_load,
        patch("src.api.main.load_production_model") as mock_load_prod,
    ):
        # Configure mock_joblib_load to return the right mock based on the path
        def side_effect(path):
            if "vectorizer.pkl" in str(path):
                return mock_vec
            if "label_encoder.pkl" in str(path):
                return mock_le
            return MagicMock()

        mock_joblib_load.side_effect = side_effect
        mock_load_prod.return_value = mock_model

        # 7. Return client WITH lifespan (but mocked artifacts won't hang)
        with TestClient(api_main.app) as tc:
            yield tc


# Sample payloads for general sentiment prediction
PREDICT_TEST_PAYLOADS = [
    ({"texts": ["I love this video! It was super helpful and well explained."]}, "Positive"),
    ({"texts": ["This is terrible, I absolutely hate it."]}, "Positive"),  # Mocked result is always Positive
]


@pytest.mark.parametrize("payload, expected_sentiment", PREDICT_TEST_PAYLOADS)
def test_predict_sentiment(client: TestClient, payload: dict, expected_sentiment: str):
    """
    Tests the /v1/predict endpoint for general sentiment.
    """
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "encoded_labels" in data
    # Note: Mock returns 'Positive' for everything in this test
    assert data["predictions"][0] == "Positive"


def test_predict_absa(client: TestClient):
    """
    Tests the /v1/predict_absa endpoint for aspect-based sentiment.
    """
    # Mocking the ABSA model lazy load inside the endpoint
    with patch("src.components.absa_model.ABSAModel") as mock_absa_cls:
        mock_absa = mock_absa_cls.return_value
        mock_absa.predict.return_value = [
            {"aspect": "video quality", "sentiment": "Positive", "score": 0.9},
            {"aspect": "presenter", "sentiment": "Neutral", "score": 0.5},
        ]

        payload = {
            "text": "The video quality was amazing, but the presenter seemed bored.",
            "aspects": ["video quality", "presenter"],
        }

        response = client.post("/v1/predict_absa", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["aspect"] == "video quality"


def test_health_check(client: TestClient):
    """
    Tests the /v1/health endpoint.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
