"""
Inference utilities for sentiment analysis prediction services.

Provides:
- Model loading from MLflow Model Registry (with local fallback).
- Derived feature engineering (e.g., lengths, lexicon ratios) for consistent preprocessing.

Usage:
    from app.inference_utils import load_production_model, build_derived_features
"""

import joblib
import mlflow
import pandas as pd
import numpy as np
import json
from typing import Any, Set
from scipy.sparse import issparse

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import ADVANCED_DIR, PROJECT_ROOT, EVAL_DIR

# --- Logging Setup ---
logger = get_logger(__name__, headline="inference_utils.py")


# =====================================================================
# Load Best Model Name (Dynamic Discovery)
# =====================================================================
def load_champion_model_name():
    """
    Loads the champion model name from 'best_model_run_info.json' for dynamic MLflow loading.

    Crucially, it prepends 'youtube_sentiment_' to the model name (e.g., 'xgboost'
    becomes 'youtube_sentiment_xgboost') to match the name registered by
    src.models.register_model.py.

    Returns the full dynamic model name or a hardcoded fallback if the file is not found.
    """
    info_path = EVAL_DIR / "best_model_run_info.json"
    # Fallback name matches the common name used by the registration script
    default_model_name = "youtube_sentiment_xgboost"

    if not info_path.exists():
        logger.warning(
            f"Champion model info not found at {info_path.relative_to(PROJECT_ROOT)}. | "
            f"Falling back to hardcoded model name: {default_model_name}."
        )
        return default_model_name
    try:
        with open(info_path, "r") as f:
            data = json.load(f)
            model_base_name = data["model_name"]  # e.g., 'xgboost'

            # Apply the prefix to match the MLflow Registered Model name
            if not model_base_name.startswith("youtube_sentiment_"):
                full_model_name = f"youtube_sentiment_{model_base_name}"
            else:
                full_model_name = model_base_name

            logger.info(f"Dynamically loaded champion model name: {full_model_name}")
            return full_model_name
    except Exception as e:
        logger.error(
            f"Error loading champion model info: {e}. Falling back to {default_model_name}"
        )
        return default_model_name


# =====================================================================
# Main Model Loading Utility
# =====================================================================
def load_production_model(alias_name: str = "Production") -> Any:
    """
    Loads the trained model object, prioritizing MLflow Model Registry with the
    '@Production' alias, and falling back to a local DVC-tracked PKL file.

    The model name is dynamically retrieved from the 'best_model_run_info.json'
    file created during the 'model_evaluation' stage.

    MLflow Priority:
        1. Dynamically retrieve the full model name (e.g., 'youtube_sentiment_xgboost').
        2. Try to load the model artifact from MLflow Model Registry using the
           retrieved name and the '@Production' alias.

    Local Fallback:
        If MLflow loading fails, fall back to loading the locally DVC-tracked
        'xgboost_model.pkl'.

    Returns:
        Any: The loaded model instance (mlflow.pyfunc.PyFuncModel or scikit-learn model).

    Raises:
        RuntimeError: If model loading fails from both sources.
    """
    # === STEP 1: Dynamically determine model_name ===
    # This function will return the full name (e.g., 'youtube_sentiment_xgboost')
    model_name = load_champion_model_name()

    # 1. Attempt MLflow Model Registry Load
    try:
        mlflow_uri = get_mlflow_uri()
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = f"models:/{model_name}@{alias_name}"
        logger.info(f"Attempting to load model from MLflow URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        logger.info(
            f"✅ Loaded model from MLflow registry → {model_name}@{alias_name} "
            f"(Tracking URI: {mlflow_uri})"
        )

        return model

    except Exception as e:
        # 2. Local Fallback Load
        # Updated to the expected champion model's local PKL file name
        model_path = ADVANCED_DIR / "xgboost_model.pkl"

        try:
            # Load the local model artifact
            model = joblib.load(model_path)
            logger.warning(
                f"⚠️ MLflow registry unavailable or alias not found for **{model_name}**. | "
                f"Loaded local XGBoost model from {model_path.relative_to(PROJECT_ROOT)}. | "
                f"Original MLflow error: {e}"
            )
            return model

        except FileNotFoundError:
            logger.error(
                f"❌ Local model fallback failed. Model file not found at "
                f"{model_path.relative_to(PROJECT_ROOT)}."
            )
            raise RuntimeError(
                "Failed to load model from both MLflow and local filesystem. | "
                "Ensure MLflow is running or 'xgboost_model.pkl' is DVC-pulled."
            )


def build_derived_features(df: pd.DataFrame) -> np.ndarray:
    """
    Recreate simple derived features used during training.

    Features:
        - char_len: Character length of cleaned comment.
        - word_len: Word count of cleaned comment.
        - pos_ratio: Ratio of positive lexicon words.
        - neg_ratio: Ratio of negative lexicon words.

    Args:
        df: DataFrame with 'clean_comment' column (preprocessed text).

    Returns:
        np.ndarray: 2D array of shape (n_samples, 4) with derived features.
    """
    df = df.copy()
    df["char_len"] = df["clean_comment"].str.len()
    df["word_len"] = df["clean_comment"].str.split().str.len()

    pos_words: Set[str] = {"good", "great", "love", "like", "positive", "best"}
    neg_words: Set[str] = {"bad", "hate", "worst", "negative", "shit", "fuck"}

    def count_lexicon_ratio(text: str, lexicon: Set[str]) -> float:
        words = text.split()
        return len([w for w in words if w in lexicon]) / max(len(words), 1)

    df["pos_ratio"] = df["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, pos_words)
    )
    df["neg_ratio"] = df["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, neg_words)
    )

    # Return as a numpy array of features
    return df[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values


def safe_to_list(x: Any) -> list:
    """
    Safely convert different data types (NumPy arrays, SciPy sparse matrices)
    to a standard Python list. Handles basic lists, NumPy arrays, and sparse matrices.

    Args:
        x: The input data to convert. Can be a list, np.ndarray, or scipy.sparse matrix.

    Returns:
        A Python list representation of the input.
    """
    if isinstance(x, list):
        return x
    if issparse(x):
        # Converts sparse matrix to dense NumPy array, then to list
        return x.toarray().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    # Fallback for single, non-list items
    return [x]
