"""
Integration Tests for the Agent API Endpoint.

Tests the FastAPI Agent router (POST /v1/agent/analyze) using TestClient
with mocked Agent execution. Validates request schema enforcement,
response structure, and error code mapping.

Run with:
    uv run pytest tests/test_agent_api.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.entity.agent_schemas import (
    AnalystReport,
    SentimentBreakdown,
)


def _make_mock_report() -> AnalystReport:
    """Factory for a valid AnalystReport fixture."""
    return AnalystReport(
        video_id="dQw4w9WgXcQ",
        video_title="Never Gonna Give You Up",
        sentiment_breakdown=SentimentBreakdown(
            positive_pct=0.65,
            neutral_pct=0.20,
            negative_pct=0.15,
            dominant_sentiment="Positive",
            total_analyzed=100,
            raw_predictions=["Positive"] * 65 + ["Neutral"] * 20 + ["Negative"] * 15,
        ),
        data_quality_passed=True,
        model_version="v3-champion",
        executive_summary="The audience responded overwhelmingly positively to this content.",
        key_insights=[
            "65% of commenters expressed positive sentiment.",
            "Negative comments cluster around audio quality concerns.",
        ],
        strategic_recommendation="Address audio quality in your next upload to convert negative commenters.",
        confidence_score=0.82,
    )


@pytest.fixture()
def client():
    """
    Creates a TestClient for the FastAPI app with mocked artifacts and lifespan.
    Uses the same pattern as test_inference.py to prevent the 56-minute hang
    caused by loading real model files during the lifespan startup on Windows.
    """
    from unittest.mock import MagicMock, patch

    import numpy as np

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2])

    mock_le = MagicMock()
    mock_le.inverse_transform.side_effect = lambda x: ["Positive" if p == 2 else "Neutral" for p in x]

    mock_vec = MagicMock()
    mock_vec.transform.return_value = MagicMock()

    from src.api import main as api_main

    api_main.model = mock_model
    api_main.vectorizer = mock_vec
    api_main.label_encoder = mock_le

    with (
        patch("src.api.main.joblib.load") as mock_joblib_load,
        patch("src.api.main.load_production_model") as mock_load_prod,
    ):

        def side_effect(path):
            if "vectorizer.pkl" in str(path):
                return mock_vec
            if "label_encoder.pkl" in str(path):
                return mock_le
            return MagicMock()

        mock_joblib_load.side_effect = side_effect
        mock_load_prod.return_value = mock_model

        with TestClient(api_main.app, raise_server_exceptions=False) as tc:
            yield tc


class TestAgentAnalyzeEndpoint:
    """Tests for POST /v1/agent/analyze."""

    def test_valid_request_returns_analyst_report(self, client):
        mock_report = _make_mock_report()

        with (
            patch("src.api.agent_api.ConfigurationManager") as mock_cm,
            patch("src.api.agent_api.run_content_analyst", new_callable=AsyncMock) as mock_run,
        ):
            mock_cm.return_value.get_agent_config.return_value = MagicMock()
            mock_run.return_value = mock_report

            response = client.post(
                "/v1/agent/analyze",
                json={"video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ", "max_comments": 50},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "dQw4w9WgXcQ"
        assert data["data_quality_passed"] is True
        assert "executive_summary" in data
        assert "key_insights" in data
        assert "strategic_recommendation" in data
        assert "confidence_score" in data

    def test_invalid_url_too_short_returns_422(self, client):
        response = client.post(
            "/v1/agent/analyze",
            json={"video_url": "short", "max_comments": 50},
        )
        assert response.status_code == 422

    def test_max_comments_below_minimum_returns_422(self, client):
        response = client.post(
            "/v1/agent/analyze",
            json={"video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ", "max_comments": 5},
        )
        assert response.status_code == 422

    def test_max_comments_above_maximum_returns_422(self, client):
        response = client.post(
            "/v1/agent/analyze",
            json={"video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ", "max_comments": 999},
        )
        assert response.status_code == 422

    def test_missing_video_url_returns_422(self, client):
        response = client.post(
            "/v1/agent/analyze",
            json={"max_comments": 50},
        )
        assert response.status_code == 422

    def test_extra_forbidden_field_returns_422(self, client):
        response = client.post(
            "/v1/agent/analyze",
            json={
                "video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
                "max_comments": 50,
                "forbidden_field": "hacked",
            },
        )
        assert response.status_code == 422

    def test_missing_gemini_key_returns_503(self, client):
        with (
            patch("src.api.agent_api.ConfigurationManager") as mock_cm,
            patch(
                "src.api.agent_api.run_content_analyst",
                new_callable=AsyncMock,
                side_effect=OSError("GEMINI_API_KEY is not set"),
            ),
        ):
            mock_cm.return_value.get_agent_config.return_value = MagicMock()
            response = client.post(
                "/v1/agent/analyze",
                json={"video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert response.status_code == 503
        assert "GEMINI_API_KEY" in response.json()["detail"]

    def test_downstream_api_unavailable_returns_502(self, client):
        with (
            patch("src.api.agent_api.ConfigurationManager") as mock_cm,
            patch(
                "src.api.agent_api.run_content_analyst",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Cannot connect to Inference API"),
            ),
        ):
            mock_cm.return_value.get_agent_config.return_value = MagicMock()
            response = client.post(
                "/v1/agent/analyze",
                json={"video_url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert response.status_code == 502
