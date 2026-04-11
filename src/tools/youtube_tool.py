"""
YouTube Data Tool — Deterministic Comment Fetching.

This module wraps the YouTube Data API v3 to fetch top-level comments
from a YouTube video. It is a purely deterministic tool:
- Input validation via Pydantic schemas.
- Returns a strongly-typed CommentBatch.
- Raises YouTubeToolError on any failure so the Agent can self-correct.

Rules:
1. Tools are microservices. This tool NEVER calls an LLM.
2. Raises domain-specific exceptions for observability.

Usage:
    from src.tools.youtube_tool import fetch_youtube_comments
    batch = await fetch_youtube_comments("https://youtube.com/watch?v=dQw4w9WgXcQ", max_comments=100)
"""

import os
import re

import httpx

from src.entity.agent_schemas import CommentBatch
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain Exception
# ---------------------------------------------------------------------------


class YouTubeToolError(Exception):
    """Raised when the YouTube Data API v3 call fails or data is invalid."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_ID_PATTERN = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})")


def _extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.

    Args:
        url: Full YouTube video URL in any common format.

    Returns:
        11-character video ID string.

    Raises:
        YouTubeToolError: If no valid video ID can be parsed.
    """
    match = _VIDEO_ID_PATTERN.search(url)
    if not match:
        raise YouTubeToolError(
            f"Could not extract a valid YouTube video ID from URL: '{url}'. "
            "Ensure the URL is a standard youtube.com/watch?v= or youtu.be/ link."
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# Public Tool Function
# ---------------------------------------------------------------------------


async def fetch_youtube_comments(video_url: str, max_comments: int = 100) -> CommentBatch:
    """
    Fetches top-level comments from a YouTube video via the Data API v3.

    This is a DETERMINISTIC tool. It contacts the YouTube API and returns a
    validated CommentBatch. The Agent must call this FIRST before any
    analysis can begin.

    Args:
        video_url: Full YouTube video URL.
        max_comments: Maximum number of comments to retrieve (10-500).

    Returns:
        CommentBatch with the video ID, title, and raw comment strings.

    Raises:
        YouTubeToolError: If the API key is missing, the video is not found,
            comments are disabled, or the API returns an unexpected error.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise YouTubeToolError(
            "YOUTUBE_API_KEY is not set in the environment. Add it to your .env file to enable comment fetching."
        )

    video_id = _extract_video_id(video_url)
    comments: list[str] = []
    video_title = "Unknown"
    next_page_token: str | None = None
    base_url = "https://www.googleapis.com/youtube/v3"

    logger.info(f"🎬 Fetching comments for video ID: {video_id} (max={max_comments})")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Fetch video title for context
        try:
            video_resp = await client.get(
                f"{base_url}/videos",
                params={"part": "snippet", "id": video_id, "key": api_key},
            )
            video_resp.raise_for_status()
            items = video_resp.json().get("items", [])
            if items:
                video_title = items[0]["snippet"]["title"]
        except httpx.HTTPError as e:
            logger.warning(f"⚠️ Could not fetch video title: {e}")

        # 2. Paginate through comments
        while len(comments) < max_comments:
            params: dict[str, str | int] = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(comments)),
                "textFormat": "plainText",
                "key": api_key,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = await client.get(f"{base_url}/commentThreads", params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                if "commentsDisabled" in error_body:
                    raise YouTubeToolError(f"Comments are disabled for video '{video_id}'.") from e
                if "videoNotFound" in error_body or e.response.status_code == 404:
                    raise YouTubeToolError(f"Video not found: '{video_id}'. Check the URL.") from e
                raise YouTubeToolError(f"YouTube API error (HTTP {e.response.status_code}): {error_body}") from e
            except httpx.HTTPError as e:
                raise YouTubeToolError(f"Network error contacting YouTube API: {e}") from e

            data = resp.json()
            for item in data.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                if text and text.strip():
                    comments.append(text.strip())

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

    if not comments:
        raise YouTubeToolError(
            f"No retrievable comments found for video '{video_id}'. "
            "The video may have no comments, or they may be restricted."
        )

    logger.info(f"✅ Retrieved {len(comments)} comments for '{video_title}'")

    return CommentBatch(
        video_id=video_id,
        video_title=video_title,
        comments=comments[:max_comments],
        comment_count=len(comments[:max_comments]),
    )
