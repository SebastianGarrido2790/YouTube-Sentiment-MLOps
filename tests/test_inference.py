"""
Inference Endpoint Validation Suite
==================================

Automated integration tests for the YouTube Sentiment Analysis API.
This suite validates sentiment prediction, Aspect-Based Sentiment Analysis (ABSA),
and system health checks using FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# Sample payloads for general sentiment prediction
PREDICT_TEST_PAYLOADS = [
    ({"texts": ["I love this video! It was super helpful and well explained."]}, "Positive"),
    ({"texts": ["This is terrible, I absolutely hate it."]}, "Negative"),
    ({"texts": ["It is okay, nothing special but not bad either."]}, "Neutral"),
]

# Sample payloads for Aspect-Based Sentiment Analysis (ABSA)
ABSA_TEST_PAYLOADS = [
    {
        "text": "The video quality was amazing, but the presenter seemed bored.",
        "aspects": ["video quality", "presenter"],
    },
    {
        "text": "The food was delicious, but the service was slow.",
        "aspects": ["food", "service"],
    },
]

@pytest.mark.parametrize("payload, expected_sentiment", PREDICT_TEST_PAYLOADS)
def test_predict_sentiment(payload: dict[str, list[str]], expected_sentiment: str):
    """
    Tests the /v1/predict endpoint for general sentiment.
    Note: We use /v1 prefix as per the updated API versioning.
    """
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "encoded_labels" in data
    assert data["predictions"][0] == expected_sentiment

def test_predict_absa():
    """
    Tests the /v1/predict_absa endpoint for aspect-based sentiment.
    """
    for payload in ABSA_TEST_PAYLOADS:
        response = client.post("/v1/predict_absa", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == len(payload["aspects"])
        for item in data:
            assert "aspect" in item
            assert "sentiment" in item
            assert "score" in item

def test_health_check():
    """
    Tests the /v1/health endpoint.
    """
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
