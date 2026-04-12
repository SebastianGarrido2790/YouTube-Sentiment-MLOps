"""
Content Intelligence Analyst Agent.

This module defines the core pydantic-ai Agent that orchestrates the
Hybrid Agentic MLOps workflow. It embodies the Separation of Concerns:

- The Brain (this Agent): reasoning, routing, and business synthesis.
- The Hands (tools in src/tools/): deterministic data fetching and ML inference.

Architecture Decision:
    pydantic-ai was chosen over LangChain/LangGraph for this implementation
    because it provides first-class Pydantic v2 integration, strict type safety
    compatible with pyright strict mode, and a clean API. The Agent's result type is
    `AnalystReport`, enforcing structured output at the framework level.

    The LLM (Gemini Flash) is responsible ONLY for:
    1. Orchestrating the tool call sequence.
    2. Synthesizing qualitative narrative fields (executive_summary,
       key_insights, strategic_recommendation).

    The LLM is NEVER responsible for:
    - Computing sentiment percentages (→ sentiment_tool.py)
    - Fetching YouTube data (→ youtube_tool.py)
    - Evaluating data quality (→ data_quality_tool.py)

Usage:
    from src.agents.content_analyst import run_content_analyst
    report = await run_content_analyst("https://youtube.com/watch?v=...")
"""

import os

from pydantic_ai import Agent, RunContext

from src.agents.prompts.system_prompt import ACTIVE_SYSTEM_PROMPT
from src.entity.agent_schemas import (
    AnalysisRequest,
    AnalystReport,
    CommentBatch,
    DataQualityReport,
    SentimentBreakdown,
)
from src.entity.config_entity import AgentConfig
from src.tools.data_quality_tool import check_data_quality
from src.tools.sentiment_tool import InferenceAPIError, analyze_sentiment
from src.tools.youtube_tool import YouTubeToolError, fetch_youtube_comments
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Agent Dependencies (passed via RunContext)
# ---------------------------------------------------------------------------


class AgentDeps:
    """
    Dependency injection container for the Content Intelligence Analyst Agent.

    Carries runtime configuration into tool functions, avoiding global state.
    Following pydantic-ai's dependency injection pattern.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config


# ---------------------------------------------------------------------------
# Agent Definition
# ---------------------------------------------------------------------------


def build_agent(config: AgentConfig, model_override: str | None = None) -> Agent[AgentDeps, AnalystReport]:
    """
    Constructs and returns a configured Content Intelligence Analyst Agent.

    Args:
        config: Validated AgentConfig.
        model_override: Optional model name to use instead of config.model_name.

    Returns:
        A pydantic-ai Agent.
    """
    model_to_use = model_override or config.model_name

    # Validate environment for chosen provider
    if "google" in model_to_use and not os.getenv("GEMINI_API_KEY"):
        raise OSError(
            "❌ GEMINI_API_KEY is not set. Add it to your .env file. "
            "Get a key at: https://aistudio.google.com/app/apikey"
        )
    if "huggingface" in model_to_use and not os.getenv("HF_TOKEN"):
        raise OSError(
            "❌ HF_TOKEN is not set. A Hugging Face token is required for the fallback model. Add it to your .env file."
        )
    if "groq" in model_to_use and not os.getenv("GROQ_API_KEY"):
        raise OSError(
            "❌ GROQ_API_KEY is not set. A Groq API key is required for the fallback model. "
            "Add it to your .env file from: https://console.groq.com/keys"
        )

    agent: Agent[AgentDeps, AnalystReport] = Agent(
        model=model_to_use,
        deps_type=AgentDeps,
        output_type=AnalystReport,
        system_prompt=ACTIVE_SYSTEM_PROMPT,
        retries=3,
    )

    # -----------------------------------------------------------------------
    # Tool Registrations
    # All tools are deterministic. The Agent decides WHEN to call them;
    # the tools decide HOW to execute.
    # -----------------------------------------------------------------------

    @agent.tool
    async def fetch_youtube_comments_tool(ctx: RunContext[AgentDeps], video_url: str) -> CommentBatch:
        """
        Fetches raw YouTube comments from a video URL via the YouTube Data API v3.

        Call this FIRST before any other tool. Provide the full video URL.
        Returns a CommentBatch with the video ID, title, and comment list.
        Raises an error if comments are disabled, the video is not found,
        or the API key is missing.

        Args:
            ctx: Agent run context with injected dependencies.
            video_url: Full YouTube video URL.

        Returns:
            CommentBatch with raw comment strings.
        """
        try:
            return await fetch_youtube_comments(
                video_url=video_url,
                max_comments=ctx.deps.config.max_comments,
            )
        except YouTubeToolError as e:
            logger.error(f"❌ YouTube tool failed: {e}")
            raise

    @agent.tool
    def check_data_quality_tool(ctx: RunContext[AgentDeps], comments: list[str]) -> DataQualityReport:
        """
        Validates the fetched comment batch against statistical quality contracts.

        Call this SECOND, immediately after fetch_youtube_comments_tool.
        If the returned DataQualityReport has passed=False, you MUST halt
        and include the failure_reasons in your executive_summary.
        Do NOT proceed to analyze_sentiment_tool if quality has failed.

        Args:
            ctx: Agent run context.
            comments: Raw comment strings from fetch_youtube_comments_tool.

        Returns:
            DataQualityReport with pass/fail status and failure reasons.
        """
        return check_data_quality(comments)

    @agent.tool
    async def analyze_sentiment_tool(ctx: RunContext[AgentDeps], comments: list[str]) -> SentimentBreakdown:
        """
        Classifies YouTube comments using the production ML sentiment model.

        Call this THIRD, only after the data quality gate has passed.
        This tool contacts the Inference API — NEVER estimate sentiment yourself.
        Returns a SentimentBreakdown with percentage splits by sentiment class.

        Args:
            ctx: Agent run context.
            comments: Validated comment strings to classify.

        Returns:
            SentimentBreakdown with positive/neutral/negative percentages.

        Raises:
            InferenceAPIError: If the Inference API is not running or unreachable.
        """
        try:
            return await analyze_sentiment(
                comments=comments,
                inference_api_url=ctx.deps.config.inference_api_url,
                timeout=ctx.deps.config.tool_timeout_seconds,
            )
        except InferenceAPIError as e:
            logger.error(f"❌ Sentiment tool failed: {e}")
            raise

    return agent


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------


async def run_content_analyst(
    request: AnalysisRequest,
    config: AgentConfig,
) -> AnalystReport:
    """
    Executes the Content Intelligence Analyst Agent for a given video URL.

    This is the single entry point for the Agent API endpoint and tests.
    Builds the agent, injects dependencies, and runs the full agentic workflow:
    1. Fetch YouTube comments
    2. Validate data quality
    3. Classify sentiment via the Inference API
    4. Synthesize and return the AnalystReport

    Args:
        request: Validated AnalysisRequest with video URL and parameters.
        config: AgentConfig from ConfigurationManager.

    Returns:
        AnalystReport — structured business intelligence for the content creator.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not configured.
        pydantic_ai.exceptions.UnexpectedModelBehavior: If the LLM fails to
            produce a valid AnalystReport after retries.
    """
    logger.info(f"🧠 Content Intelligence Analyst starting for: {request.video_url}")

    agent = build_agent(config)
    deps = AgentDeps(config=config)

    user_message = (
        f"Analyze the YouTube video at this URL: {request.video_url}\n"
        f"Fetch up to {request.max_comments} comments and produce a full AnalystReport."
    )

    try:
        result = await agent.run(user_message, deps=deps)
    except Exception as e:
        error_msg = str(e)
        # Check for Quota Exceeded (429) or Request Too Large (413)
        is_quota_error = any(kw in error_msg for kw in ["429", "RESOURCE_EXHAUSTED", "quota"])
        is_size_error = "413" in error_msg or "rate_limit_exceeded" in error_msg

        if config.fallback_enabled and (is_quota_error or is_size_error):
            logger.warning(
                f"⚠️ Provider limit hit ({'Size' if is_size_error else 'Quota'}). "
                f"Switching to fallback ({config.fallback_model_name}) with truncated payload (max=40)."
            )

            # 1. Create a reduced config to force tools FETCHING less data
            # AgentConfig is frozen, so we use model_copy
            reduced_config = config.model_copy(update={"max_comments": 40})

            # 2. Update the user message to reflect the new instruction
            truncated_message = (
                f"Analyze the YouTube video at this URL: {request.video_url}\n"
                f"Fetch up to 40 comments (truncated for reliability) and produce a full AnalystReport."
            )

            # 3. Rebuild agent and deps with fallback settings
            fallback_agent = build_agent(reduced_config, model_override=config.fallback_model_name)
            fallback_deps = AgentDeps(config=reduced_config)

            result = await fallback_agent.run(truncated_message, deps=fallback_deps)
        else:
            logger.error(f"❌ Agent execution failed: {e}")
            raise e

    logger.info("✅ Content Intelligence Analyst completed successfully.")
    return result.output
