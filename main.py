"""
Main Orchestrator Microservice for the YouTube Sentiment Analysis Project.

This module refactors the traditional script-based orchestration into a
production-ready FastAPI microservice. It monitors and exposes
AgentOps metrics to track system reliability.

Usage:
    uv run python main.py
"""

import sys
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.entity.api_entity import AgentOpsMetrics, PipelineStatus
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_01b_data_validation import DataValidationPipeline
from src.pipeline.stage_02_data_preparation import DataPreparationPipeline
from src.pipeline.stage_03_feature_engineering import FeatureEngineeringPipeline
from src.pipeline.stage_04c_distilbert_training import DistilBERTTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_06_register_model import ModelRegistrationPipeline
from src.utils.exceptions import CustomExceptionError
from src.utils.logger import get_logger, log_spacer

logger = get_logger(__name__, headline="Orchestrator Service")

app = FastAPI(
    title="YouTube Sentiment Analysis: Orchestrator Service",
    description="Agentic MLOps Orchestrator serving pipeline control and AgentOps metrics.",
    version="2.0.0",
)

# In-memory storage for AgentOps and Pipeline tracking
PIPELINE_REGISTRY: dict[str, PipelineStatus] = {}
AGENT_METRICS = AgentOpsMetrics()


def run_training_pipeline(run_id: str):
    """
    Executes the full FTI pipeline stages while recording AgentOps metrics.
    """
    global AGENT_METRICS
    status = PIPELINE_REGISTRY[run_id]
    status.status = "running"

    # Define the DAG of stages
    stages = [
        ("Data Ingestion", DataIngestionPipeline()),
        ("Data Validation", DataValidationPipeline()),
        ("Data Preparation", DataPreparationPipeline()),
        ("Feature Engineering", FeatureEngineeringPipeline()),
        ("Model Training", DistilBERTTrainingPipeline()),
        ("Model Evaluation", ModelEvaluationPipeline()),
        ("Model Registration", ModelRegistrationPipeline()),
    ]

    successful_tools = 0
    total_tools = len(stages)
    total_retries = 0
    total_retry_latency = 0.0
    AGENT_METRICS.total_plans_executed += 1

    try:
        for stage_name, pipeline in stages:
            status.current_stage = stage_name
            logger.info(f"Executing Tool: {stage_name}")

            # Simulate Retry Logic for Agentic Healing / Robustness
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    pipeline.main()
                    successful_tools += 1
                    break
                except Exception as e:
                    if attempt < max_retries:
                        total_retries += 1
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {stage_name}")

                        # Track retry latency
                        retry_start = time.time()
                        time.sleep(1)  # Simulated backoff
                        total_retry_latency += time.time() - retry_start
                    else:
                        raise e

            log_spacer()

        status.status = "completed"
        status.current_stage = "None"

    except Exception as e:
        status.status = "failed"
        status.error = str(e)
        AGENT_METRICS.failed_tool_calls += 1

        # Capture enriched metadata for AgentOps auditing
        custom_err = CustomExceptionError(
            error_message=e,
            error_detail=sys,
            agent_metadata={
                "run_id": run_id,
                "failed_stage": status.current_stage,
                "total_successful_stages": successful_tools,
                "retries_attempted": total_retries,
            },
        )
        logger.error(custom_err.detailed_message)

    finally:
        # Finalize AgentOps Metrics
        completed_runs = sum(1 for p in PIPELINE_REGISTRY.values() if p.status == "completed")
        if AGENT_METRICS.total_plans_executed > 0:
            AGENT_METRICS.plan_success_rate = completed_runs / AGENT_METRICS.total_plans_executed

        # Tool Call Accuracy across all stages
        AGENT_METRICS.tool_call_accuracy = successful_tools / total_tools

        # Compute average retry latency
        if total_retries > 0:
            AGENT_METRICS.avg_retry_latency = total_retry_latency / total_retries

        status.metrics = AGENT_METRICS
        logger.info(f"Pipeline Run {run_id} finished with status: {status.status}")


@app.get("/health", tags=["System"])
def health():
    return {"status": "online", "service": "orchestrator", "project": "youtube-sentiment-analysis"}


@app.post("/v1/train", response_model=PipelineStatus, tags=["Orchestration"])
def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers the full MLOps pipeline as a background process.
    """
    run_id = str(uuid.uuid4())
    status = PipelineStatus(run_id=run_id, status="pending")
    PIPELINE_REGISTRY[run_id] = status

    background_tasks.add_task(run_training_pipeline, run_id)

    return status


@app.get("/v1/status/{run_id}", response_model=PipelineStatus, tags=["Orchestration"])
def get_status(run_id: str):
    """
    Retrieves the current status of a specific pipeline run.
    """
    if run_id not in PIPELINE_REGISTRY:
        raise HTTPException(status_code=404, detail="Run ID not found.")
    return PIPELINE_REGISTRY[run_id]


@app.get("/v1/metrics", response_model=AgentOpsMetrics, tags=["AgentOps"])
def get_metrics():
    """
    Exposes AgentOps metrics for system auditing.
    """
    return AGENT_METRICS


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
