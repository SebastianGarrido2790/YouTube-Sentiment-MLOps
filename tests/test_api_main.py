"""
Unit Tests for FastAPI Main Entry Point

Tests the inference API, health checks, and metrics endpoints.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Tests the /health endpoint."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("src.api.main.model")
@patch("src.api.main.vectorizer")
@patch("src.api.main.label_encoder")
@patch("src.api.main.build_derived_features")
def test_predict_sentiment_endpoint(mock_build, mock_le, mock_vec, mock_model):
    """Tests the /predict endpoint."""
    mock_model.predict.return_value = [2]
    mock_le.inverse_transform.return_value = ["Positive"]

    # Mock vectorizer output
    import numpy as np
    from scipy.sparse import csr_matrix

    mock_vec.transform.return_value = csr_matrix([[1, 0]])
    mock_build.return_value = np.array([[2, 3, 0.5, 0.1]])

    # Request
    request_data = {"texts": ["I love this project!"]}
    response = client.post("/v1/predict", json=request_data)

    # Assertions
    assert response.status_code == 200
    assert response.json()["predictions"] == ["Positive"]
    assert response.json()["numeric_labels"] == [1]


def test_predict_empty_text():
    """Tests /predict with invalid schema type."""
    response = client.post("/v1/predict", json={"text": "should be array of texts"})
    assert response.status_code == 422  # Pydantic validation error
