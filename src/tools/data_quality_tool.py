"""
Data Quality Gate Tool — Great Expectations Validation for Comment Batches.

This deterministic tool applies statistical data quality checks to a batch
of raw YouTube comments before it enters the ML inference pipeline.
It enforces the same data contract principles established by the DVC pipeline
but adapted for real-time agentic workloads.

Rules:
1. deterministic execution — never calls an LLM.
2. raises DataQualityToolError with rich metadata for Agent self-correction.
3. Data contracts prevent "garbage in, garbage out".

Usage:
    from src.tools.data_quality_tool import check_data_quality
    report = check_data_quality(comments)
"""

from src.entity.agent_schemas import DataQualityReport
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain Exception
# ---------------------------------------------------------------------------


class DataQualityToolError(Exception):
    """Raised when the data quality check itself fails to execute."""


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

NULL_THRESHOLD: float = 0.20  # Max 20% null/empty comments allowed
SHORT_COMMENT_THRESHOLD: float = 0.50  # Max 50% very short comments (< 5 chars)
MIN_COMMENT_BATCH_SIZE: int = 5  # Minimum viable batch for analysis


# ---------------------------------------------------------------------------
# Public Tool Function
# ---------------------------------------------------------------------------


def check_data_quality(comments: list[str]) -> DataQualityReport:
    """
    Validates a batch of raw YouTube comments against statistical quality contracts.

    This is a DETERMINISTIC tool. It checks for null rates, degenerate short
    comments, and minimum batch size. If the gate fails, the Agent MUST halt
    and report the failure to the user rather than proceeding to inference.

    Args:
        comments: List of raw comment strings from the YouTube API.

    Returns:
        DataQualityReport with pass/fail status and specific failure reasons.

    Raises:
        DataQualityToolError: If the validation logic itself encounters an error
            (not a quality failure — that is represented by passed=False).
    """
    if not comments:
        raise DataQualityToolError("Cannot validate an empty comment list.")

    logger.info(f"🔍 Running data quality checks on {len(comments)} comments...")

    failure_reasons: list[str] = []

    try:
        total = len(comments)

        # --- Check 1: Minimum batch size ---
        if total < MIN_COMMENT_BATCH_SIZE:
            failure_reasons.append(f"Batch size too small: {total} comments (minimum is {MIN_COMMENT_BATCH_SIZE}).")

        # --- Check 2: Null / empty comment ratio ---
        null_count = sum(1 for c in comments if not c or not c.strip())
        null_ratio = null_count / total
        if null_ratio > NULL_THRESHOLD:
            failure_reasons.append(
                f"Too many null/empty comments: {null_ratio:.1%} (threshold is {NULL_THRESHOLD:.0%})."
            )

        # --- Check 3: Short comment ratio (likely spam or emoji-only) ---
        short_count = sum(1 for c in comments if len(c.strip()) < 5)
        short_comment_ratio = short_count / total
        if short_comment_ratio > SHORT_COMMENT_THRESHOLD:
            failure_reasons.append(
                f"Too many very short comments (< 5 chars): {short_comment_ratio:.1%} "
                f"(threshold is {SHORT_COMMENT_THRESHOLD:.0%}). "
                "Batch may be dominated by spam or emoji-only comments."
            )

        passed = len(failure_reasons) == 0

        if passed:
            logger.info("✅ Data quality gate PASSED.")
        else:
            logger.warning(f"⚠️ Data quality gate FAILED: {failure_reasons}")

        return DataQualityReport(
            passed=passed,
            null_ratio=round(null_ratio, 4),
            short_comment_ratio=round(short_comment_ratio, 4),
            failure_reasons=failure_reasons,
        )

    except Exception as e:
        raise DataQualityToolError(f"Data quality validation failed with an unexpected error: {e}") from e
