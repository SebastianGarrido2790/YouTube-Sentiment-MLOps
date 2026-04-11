"""
API Data Contracts for the Orchestrator Service.

This module defines the schemas for pipeline status tracking and
AgentOps metrics enforcement.
"""

from pydantic import BaseModel, Field


class AgentOpsMetrics(BaseModel):
    """
    Advanced AgentOps metrics for system auditing.
    """

    total_plans_executed: int = Field(default=0, description="Total number of pipeline runs triggered.")
    failed_tool_calls: int = Field(default=0, description="Number of individual tool (stage) failures.")
    plan_success_rate: float = Field(default=0.0, description="Fraction of plans that completed successfully.")
    tool_call_accuracy: float = Field(default=0.0, description="Success rate of individual tool calls.")
    avg_retry_latency: float = Field(default=0.0, description="Average time spent on retries in seconds.")


class PipelineStatus(BaseModel):
    """
    Schema for tracking the lifecycle of an asynchronous pipeline run.
    """

    run_id: str = Field(..., description="Unique UUID for the pipeline run.")
    status: str = Field(..., description="Current status (pending, running, completed, failed).")
    current_stage: str = Field(default="None", description="The stage currently being executed.")
    error: str | None = Field(default=None, description="Error message if the pipeline failed.")
    metrics: AgentOpsMetrics | None = Field(
        default=None, description="Snapshotted metrics at the time of status check."
    )
