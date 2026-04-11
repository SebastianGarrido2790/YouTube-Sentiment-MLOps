"""
Agent Data Contracts for the Content Intelligence Analyst.

This module defines Pydantic schemas that enforce strict I/O contracts
for the agentic layer. Every input to and output from the Agent MUST
be validated against these schemas before tool execution or user delivery.

These schemas follow Structured Output Enforcement and
Tools as Microservices rules by providing typed boundaries between
the probabilistic LLM layer and the deterministic tool layer.
"""

from pydantic import BaseModel, ConfigDict, Field


class AnalysisRequest(BaseModel):
    """
    Validated input contract for triggering a content analysis.

    This is the entry point for any consumer (Chrome Extension, API client,
    or direct call) requesting a full analyst report on a YouTube video.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_url: str = Field(
        ...,
        description="Full YouTube video URL (e.g., https://youtube.com/watch?v=...)",
        min_length=10,
    )
    max_comments: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of comments to fetch and analyze.",
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code for comment filtering.",
    )


class CommentBatch(BaseModel):
    """
    Internal contract for a validated batch of raw YouTube comments.
    Produced by the YouTube Tool and consumed by the Sentiment Tool.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    video_id: str = Field(..., description="Extracted YouTube video ID.")
    video_title: str = Field(default="Unknown", description="Video title for context.")
    comments: list[str] = Field(..., min_length=1, description="Raw comment strings.")
    comment_count: int = Field(..., ge=1, description="Number of comments retrieved.")


class DataQualityReport(BaseModel):
    """
    Structured result from the Great Expectations data quality gate.
    If passed=False, the Agent MUST halt and surface this to the user.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(..., description="True if all data quality checks pass.")
    null_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of null/empty comments.")
    short_comment_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of very short comments.")
    failure_reasons: list[str] = Field(default_factory=list, description="Descriptions of any failed checks.")


class SentimentBreakdown(BaseModel):
    """
    Deterministic sentiment classification output from the Inference API.
    The LLM NEVER computes these values — they are produced by the ML model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    positive_pct: float = Field(..., ge=0.0, le=1.0, description="Fraction of positive comments.")
    neutral_pct: float = Field(..., ge=0.0, le=1.0, description="Fraction of neutral comments.")
    negative_pct: float = Field(..., ge=0.0, le=1.0, description="Fraction of negative comments.")
    dominant_sentiment: str = Field(..., description="The majority sentiment class label.")
    total_analyzed: int = Field(..., ge=1, description="Total number of comments analyzed.")
    raw_predictions: list[str] = Field(default_factory=list, description="Per-comment prediction labels.")


class AnalystReport(BaseModel):
    """
    Final structured output of the Content Intelligence Analyst Agent.

    This is the authoritative deliverable for the Chrome Extension and API clients.
    All fields are required except where a default is provided.
    The LLM synthesizes ONLY the qualitative narrative fields
    (executive_summary, key_insights, strategic_recommendation).
    All quantitative fields are populated deterministically by tools.
    """

    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(..., description="Analyzed YouTube video ID.")
    video_title: str = Field(default="Unknown", description="Video title for context.")
    sentiment_breakdown: SentimentBreakdown = Field(..., description="Quantitative sentiment distribution.")
    data_quality_passed: bool = Field(..., description="Whether the GX data quality gate passed.")
    model_version: str = Field(default="unknown", description="MLflow model version used for inference.")
    executive_summary: str = Field(
        ...,
        description="2-3 sentence business narrative synthesized by the LLM.",
        min_length=20,
    )
    key_insights: list[str] = Field(
        ...,
        min_length=2,
        description="Bullet-point findings synthesized by the LLM (min 2 items).",
    )
    strategic_recommendation: str = Field(
        ...,
        description="Single actionable directive for the content creator.",
        min_length=10,
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed confidence in the synthesis (0.0-1.0).",
    )
