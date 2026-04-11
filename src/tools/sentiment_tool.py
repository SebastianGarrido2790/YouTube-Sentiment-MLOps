"""
Sentiment Analysis Tool — Deterministic Inference API Wrapper.

This tool wraps the FastAPI Inference Service (`/v1/predict`) to classify
a batch of YouTube comments. It is the primary "hand" for quantitative
sentiment analysis in the agentic workflow.

Rules:
1. The LLM NEVER classifies sentiment. This tool does.
2. Tools are microservices. Communicates with the Inference API via HTTP.
3. Raises InferenceAPIError with rich metadata.

The returned SentimentBreakdown is purely deterministic — the ML model's output,
not the LLM's approximation.

Usage:
    from src.tools.sentiment_tool import analyze_sentiment
    breakdown = await analyze_sentiment(comments, inference_api_url="http://127.0.0.1:8000")
"""

from collections import Counter

import httpx

from src.entity.agent_schemas import SentimentBreakdown
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain Exception
# ---------------------------------------------------------------------------


class InferenceAPIError(Exception):
    """Raised when the Inference API is unreachable or returns an error."""


# ---------------------------------------------------------------------------
# Public Tool Function
# ---------------------------------------------------------------------------


async def analyze_sentiment(
    comments: list[str],
    inference_api_url: str = "http://127.0.0.1:8000",
    timeout: int = 30,
) -> SentimentBreakdown:
    """
    Classifies a batch of YouTube comments using the production ML model.

    Sends raw comment text to the Inference API's /v1/predict endpoint.
    All sentiment classification is performed by the deterministic ML model —
    the Agent MUST call this tool and NEVER estimate sentiment itself.

    Args:
        comments: List of raw or pre-processed comment strings to classify.
        inference_api_url: Base URL of the running Inference API service.
        timeout: Request timeout in seconds.

    Returns:
        SentimentBreakdown with percentage splits and dominant sentiment label.

    Raises:
        InferenceAPIError: If the API is unavailable, returns a non-200 status,
            or the response schema is unexpected.
    """
    if not comments:
        raise InferenceAPIError("Cannot analyze an empty comment list.")

    endpoint = f"{inference_api_url.rstrip('/')}/v1/predict"
    logger.info(f"📡 Calling Inference API: POST {endpoint} ({len(comments)} comments)")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                json={"texts": comments},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
    except httpx.ConnectError as e:
        raise InferenceAPIError(
            f"Cannot connect to Inference API at '{inference_api_url}'. "
            "Ensure the service is running: uv run python -m uvicorn src.api.main:app --port 8000"
        ) from e
    except httpx.TimeoutException as e:
        raise InferenceAPIError(
            f"Inference API timed out after {timeout}s. The model may still be loading or the batch is too large."
        ) from e
    except httpx.HTTPStatusError as e:
        raise InferenceAPIError(f"Inference API returned HTTP {e.response.status_code}: {e.response.text}") from e

    try:
        data = response.json()
        raw_predictions = data["predictions"]

        # --- Robust Label Mapping (Rule: Tools must be deterministic) ---
        # Map integer labels (0,1,2) or numeric-strings to categorical names
        # Standard mapping: 0=Negative, 1=Neutral, 2=Positive
        LABEL_MAP = {
            0: "Negative",
            "0": "Negative",
            1: "Neutral",
            "1": "Neutral",
            2: "Positive",
            "2": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "positive": "Positive",
        }

        predictions: list[str] = [
            LABEL_MAP.get(p, str(p))
            if not isinstance(p, str) or p.lower() not in ["positive", "neutral", "negative"]
            else p.capitalize()
            for p in raw_predictions
        ]
    except (KeyError, ValueError, TypeError) as e:
        raise InferenceAPIError(f"Unexpected response schema from Inference API: {response.text}") from e

    # --- Deterministic aggregation (no LLM math) ---
    total = len(predictions)
    counts = Counter(predictions)

    positive_pct = round(counts.get("Positive", 0) / total, 4)
    neutral_pct = round(counts.get("Neutral", 0) / total, 4)
    negative_pct = round(counts.get("Negative", 0) / total, 4)

    dominant_sentiment = max(counts, key=lambda k: counts[k]) if counts else "Neutral"

    logger.info(
        f"✅ Sentiment analysis complete: "
        f"+{positive_pct:.0%} / ~{neutral_pct:.0%} / -{negative_pct:.0%} "
        f"(dominant: {dominant_sentiment})"
    )

    return SentimentBreakdown(
        positive_pct=positive_pct,
        neutral_pct=neutral_pct,
        negative_pct=negative_pct,
        dominant_sentiment=dominant_sentiment,
        total_analyzed=total,
        raw_predictions=predictions,
    )
