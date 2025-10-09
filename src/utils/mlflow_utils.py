"""
Utility functions for MLflow configuration across modules.
"""

import os
import yaml
from dotenv import load_dotenv
from pathlib import Path


def get_mlflow_uri(params_path: str = "params.yaml") -> str:
    """
    Returns the MLflow Tracking URI.

    Priority:
        1. Environment variable MLFLOW_TRACKING_URI (from .env or system env)
        2. 'feature_comparison.mlflow_uri' in params.yaml (for local reproducibility)

    The 'ENV' variable defines which environment configuration to use:
        - ENV=production → use only environment variable
        - ENV=staging → use environment variable
        - ENV=local (default) → fallback to params.yaml

    Args:
        params_path (str): Path to params.yaml (default: root-level).

    Returns:
        str: MLflow URI.
    """
    load_dotenv()  # Load variables from .env if available

    env = os.getenv("ENV", "local").lower()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not mlflow_uri and env == "local":
        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f)
                mlflow_uri = params["feature_comparison"]["mlflow_uri"]
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(
                f"MLflow URI not found in {params_path} under 'feature_comparison.mlflow_uri'. "
                f"Define it in .env (MLFLOW_TRACKING_URI) for non-local ENV."
            ) from e

    if not mlflow_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI not defined for current environment (ENV={}).".format(
                env
            )
        )

    return mlflow_uri
