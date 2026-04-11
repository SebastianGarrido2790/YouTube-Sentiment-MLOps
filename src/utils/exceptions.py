"""
Custom exception handling for this project.

This module defines a custom exception class that captures detailed error information,
including the file name and line number where the exception occurred.
This is critical for MLOps pipelines to quickly debug failures in automated
workflows and prevent silent failures.

It also supports AgentOps metrics by allowing the storage of
plan-specific metadata and failures.
"""

from types import ModuleType
from typing import Any


def error_message_detail(error: Exception | str, error_detail: ModuleType) -> str:
    """
    Extracts the detailed error message including file name and line number.

    Args:
        error (Exception | str): The exception or error message.
        error_detail (ModuleType): The sys module to access execution info.

    Returns:
        str: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    # Safety check to prevent crashes in edge cases where the traceback
    # might be incomplete.
    if exc_tb is not None and exc_tb.tb_frame is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    error_message = (
        f"Error occurred in python script: [{file_name}] line number: [{line_number}] error message: [{error!s}]"
    )

    return error_message


class CustomExceptionError(Exception):
    """
    Custom Exception class to provide detailed traceback information within the message.
    Supports Agentic MLOps metadata.
    """

    def __init__(
        self,
        error_message: Exception | str,
        error_detail: ModuleType,
        agent_metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize the CustomExceptionError.

        Args:
            error_message (Exception | str): The original error message
                or exception object.
            error_detail (ModuleType): The sys module to capture stack trace.
            agent_metadata (dict, optional): Agentic metadata for AgentOps tracking
                (e.g., plan_id, tool_name, retry_count).
        """
        self.detailed_message = error_message_detail(error=error_message, error_detail=error_detail)
        self.agent_metadata = agent_metadata or {}

        # Enrich detailed message with agent metadata if present
        if self.agent_metadata:
            self.detailed_message += f" | AgentOps: {self.agent_metadata}"

        super().__init__(self.detailed_message)

    def __str__(self) -> str:
        return self.detailed_message
