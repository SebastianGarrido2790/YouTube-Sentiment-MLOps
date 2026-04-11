"""
Agent API Router — Content Intelligence Analyst Endpoint.

This FastAPI router exposes the pydantic-ai Content Intelligence Analyst
Agent as a versioned REST endpoint. It integrates into the existing
/v1 router architecture of the Inference API (src/api/main.py).

The endpoint enforces the full Agentic workflow:
1. Input validation via AnalysisRequest (Pydantic, extra="forbid").
2. Agent execution orchestrating all deterministic tool calls.
3. Structured output via AnalystReport — no free-form text.

Usage (once registered in main.py):
    POST http://127.0.0.1:8000/v1/agent/analyze
    Body: { "video_url": "https://youtube.com/watch?v=...", "max_comments": 100 }
"""

from fastapi import APIRouter, HTTPException

from src.agents.content_analyst import run_content_analyst
from src.config.configuration import ConfigurationManager
from src.entity.agent_schemas import AnalysisRequest, AnalystReport
from src.utils.logger import get_logger

logger = get_logger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])


@agent_router.post(
    "/analyze",
    response_model=AnalystReport,
    summary="Content Intelligence Analysis",
    description=(
        "Runs the Content Intelligence Analyst Agent on a YouTube video. "
        "Fetches comments, validates data quality, classifies sentiment via the ML model, "
        "and synthesizes a structured AnalystReport with business narrative."
    ),
)
async def analyze_video(request: AnalysisRequest) -> AnalystReport:
    """
    Triggers full agentic analysis for a YouTube video.

    The Agent orchestrates three deterministic tools in sequence:
    1. YouTube Data API v3 (comment fetching)
    2. Data Quality Gate (GX-inspired validation)
    3. Inference API (ML sentiment classification)

    The LLM synthesizes only the qualitative narrative fields.
    All quantitative values are computed deterministically.

    Args:
        request: Validated AnalysisRequest with video URL and parameters.

    Returns:
        AnalystReport with sentiment breakdown, key insights, and strategic recommendation.

    Raises:
        HTTPException 503: If the GEMINI_API_KEY is missing.
        HTTPException 422: If the request body fails schema validation.
        HTTPException 502: If a downstream tool (YouTube API / Inference API) fails.
        HTTPException 500: For unexpected internal errors.
    """
    logger.info(f"🌐 Agent API: Received analysis request for '{request.video_url}'")

    try:
        config = ConfigurationManager().get_agent_config()
    except RuntimeError as e:
        logger.error(f"❌ ConfigurationManager not initialized: {e}")
        raise HTTPException(
            status_code=500,
            detail="ConfigurationManager is not initialized. Ensure startup completed.",
        ) from e

    try:
        report = await run_content_analyst(request=request, config=config)
        logger.info(f"✅ Agent analysis complete for video: {report.video_id}")
        return report

    except OSError as e:
        logger.error(f"❌ Missing environment variable: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e

    except Exception as e:
        # Distinguish tool failures (likely Bad Gateway) from internal crashes
        error_msg = str(e)
        if any(kw in error_msg for kw in ["API", "connect", "timeout", "unreachable"]):
            logger.error(f"❌ Downstream tool failure: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"A downstream service is unavailable: {error_msg}",
            ) from e

        logger.error(f"❌ Unexpected agent error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {error_msg}",
        ) from e
