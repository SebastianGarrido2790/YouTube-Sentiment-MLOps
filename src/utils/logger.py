"""
Centralized logging configuration for the entire project.

Usage:
    from src.utils.logger import get_logger, log_spacer

    # Initialize (Headline is optional, adds visual separator in files)
    logger = get_logger(__name__, headline="Data Ingestion")

    # Regular logging
    logger.info("Started data download...")
    logger.warning("Found 5 empty comments, skipping.")
    logger.error("Failed to connect to MLflow registry!")

    # Visual spacer for file readability (skipped in JSON mode)
    log_spacer()

    # NOTE: Set env var JSON_LOGS=1 to switch to structured JSON output.
"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Break circular dependency by defining LOGS_DIR locally
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "running_logs.log"


def get_logger(name: str | None = None, headline: str | None = None) -> logging.Logger:
    """
    Returns a configured logger with consistent formatting.
    Adds an optional headline section to separate logs per script.

    Ensures that:
        - Only one handler is attached (prevents duplicates)
        - Log messages include timestamps and module names
        - Works safely across multi-module projects

    Args:
        name (Optional[str]): Optional logger name, typically __name__.
        headline (Optional[str]): Optional headline for visual separation
            (e.g., script name).

    Returns:
        logging.Logger: Configured logger instance (using RichHandler if available).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger already configured
    if not logger.handlers:
        import os

        import json_log_formatter

        use_json = os.environ.get("JSON_LOGS", "false").lower() in ("1", "true")

        if use_json:
            file_formatter = json_log_formatter.JSONFormatter()
            console_formatter = file_formatter
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M",
            )
            console_formatter = logging.Formatter("%(message)s")

        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5_000_000,  # 5 MB per file
            backupCount=5,  # Keep up to 5 old logs
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)

        if use_json:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
        else:
            try:
                from rich.logging import RichHandler

                console_handler = RichHandler(rich_tracebacks=True, markup=True)
                console_handler.setFormatter(console_formatter)
            except ImportError:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.propagate = False

        # Only add visual separators for human-readable logs
        if not use_json:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write("\n\n")

            if headline:
                headline_text = (
                    f"========================= START: {headline} "
                    f"({datetime.now():%Y-%m-%d %H:%M}) =========================\n"
                )
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(headline_text)
        elif headline:
            # For JSON logs, just log the headline as a standard event
            logger.info(f"START: {headline}")

    return logger


def log_spacer() -> None:
    """
    Appends a raw newline to the log file to provide visual spacing
    without the log formatter prefix (timestamp/levelname).
    Skipped if JSON output is active.
    """
    import os

    if os.environ.get("JSON_LOGS", "false").lower() not in ("1", "true"):
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n")
