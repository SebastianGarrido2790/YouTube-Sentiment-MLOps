"""
Unit Tests for Agentic Tool Layer.

These tests validate each deterministic tool in isolation using mocked
HTTP responses and no real API calls. Custom Exceptions are used to
ensure that tool failures raise the appropriate domain exception.

Run with:
    uv run pytest tests/test_agent_tools.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.entity.agent_schemas import CommentBatch, DataQualityReport, SentimentBreakdown
from src.tools.data_quality_tool import (
    NULL_THRESHOLD,
    DataQualityToolError,
    check_data_quality,
)
from src.tools.sentiment_tool import InferenceAPIError, analyze_sentiment
from src.tools.youtube_tool import YouTubeToolError, _extract_video_id, fetch_youtube_comments

# =============================================================================
# YouTube Tool Tests
# =============================================================================


class TestExtractVideoId:
    """Unit tests for the video ID extraction helper."""

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_youtu_be_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(YouTubeToolError, match="Could not extract"):
            _extract_video_id("https://example.com/not-a-video")


@pytest.mark.asyncio
class TestFetchYouTubeComments:
    """Integration tests for the YouTube comment fetcher (mocked HTTP)."""

    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        with pytest.raises(YouTubeToolError, match="YOUTUBE_API_KEY"):
            await fetch_youtube_comments("https://youtube.com/watch?v=abc1234abcd")

    async def test_successful_fetch_returns_batch(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")

        mock_video_resp = MagicMock()
        mock_video_resp.json.return_value = {"items": [{"snippet": {"title": "Test Video"}}]}
        mock_video_resp.raise_for_status = MagicMock()

        mock_comments_resp = MagicMock()
        mock_comments_resp.raise_for_status = MagicMock()
        mock_comments_resp.json.return_value = {
            "items": [{"snippet": {"topLevelComment": {"snippet": {"textDisplay": f"Comment {i}"}}}} for i in range(5)],
            "nextPageToken": None,
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=[mock_video_resp, mock_comments_resp])

        with patch("src.tools.youtube_tool.httpx.AsyncClient", return_value=mock_client):
            batch = await fetch_youtube_comments("https://youtube.com/watch?v=dQw4w9WgXcQ", max_comments=5)

        assert isinstance(batch, CommentBatch)
        assert batch.video_id == "dQw4w9WgXcQ"
        assert batch.video_title == "Test Video"
        assert batch.comment_count == 5
        assert len(batch.comments) == 5


# =============================================================================
# Data Quality Tool Tests
# =============================================================================


class TestCheckDataQuality:
    """Unit tests for the deterministic data quality gate."""

    def test_clean_batch_passes(self):
        comments = [f"This is a proper comment number {i}" for i in range(20)]
        report = check_data_quality(comments)
        assert isinstance(report, DataQualityReport)
        assert report.passed is True
        assert report.failure_reasons == []

    def test_empty_list_raises(self):
        with pytest.raises(DataQualityToolError, match="empty"):
            check_data_quality([])

    def test_high_null_ratio_fails(self):
        # 50% empty → exceeds NULL_THRESHOLD (20%)
        comments = [""] * 10 + ["Real comment"] * 10
        report = check_data_quality(comments)
        assert report.passed is False
        assert any("null" in r.lower() or "empty" in r.lower() for r in report.failure_reasons)

    def test_null_ratio_at_threshold_passes(self):
        total = 20
        null_count = int(total * NULL_THRESHOLD)  # Exactly at threshold
        comments = [""] * null_count + ["Real comment"] * (total - null_count)
        report = check_data_quality(comments)
        # At exactly the threshold → should still pass (strict >)
        assert report.passed is True

    def test_too_small_batch_fails(self):
        comments = ["hi", "ok"]
        report = check_data_quality(comments)
        assert report.passed is False
        assert any("batch size" in r.lower() for r in report.failure_reasons)

    def test_returns_correct_null_ratio(self):
        comments = [""] * 2 + ["real"] * 8  # 20% null
        report = check_data_quality(comments)
        assert abs(report.null_ratio - 0.2) < 0.001


# =============================================================================
# Sentiment Tool Tests
# =============================================================================


@pytest.mark.asyncio
class TestAnalyzeSentiment:
    """Unit tests for the Inference API wrapper (mocked HTTP)."""

    async def test_empty_list_raises(self):
        with pytest.raises(InferenceAPIError, match="empty"):
            await analyze_sentiment([])

    async def test_successful_prediction_returns_breakdown(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"predictions": ["Positive", "Positive", "Negative", "Neutral", "Positive"]}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.tools.sentiment_tool.httpx.AsyncClient", return_value=mock_client):
            breakdown = await analyze_sentiment(
                ["c1", "c2", "c3", "c4", "c5"],
                inference_api_url="http://127.0.0.1:8000",
            )

        assert isinstance(breakdown, SentimentBreakdown)
        assert breakdown.total_analyzed == 5
        assert breakdown.positive_pct == pytest.approx(0.6, abs=0.01)
        assert breakdown.negative_pct == pytest.approx(0.2, abs=0.01)
        assert breakdown.neutral_pct == pytest.approx(0.2, abs=0.01)
        assert breakdown.dominant_sentiment == "Positive"

    async def test_connect_error_raises_inference_api_error(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with (
            patch("src.tools.sentiment_tool.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(InferenceAPIError, match="Cannot connect"),
        ):
            await analyze_sentiment(["test comment"])
