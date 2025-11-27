"""
FastAPI Inference Service for YouTube Sentiment Analysis.
===========================================================

This unified service provides two main endpoints:
1.  `/predict`: For general sentiment analysis (Positive, Negative, Neutral).
2.  `/predict_absa`: For Aspect-Based Sentiment Analysis (ABSA).

It loads all required models and artifacts at startup.

Usage (local):
Ensure MLflow server is running if loading from registry:
    uv run mlflow server --host 127.0.0.1 --port 5000
Then run:
    uv run python -m app.main
Or via Uvicorn:
    uv run uvicorn app.main:app --reload --port 8000

Test with:
1.  Sentiment Prediction:
    curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"texts": ["I love this video! It was super helpful."]}'

2.  Aspect-Based Sentiment:
    curl -X POST "http://127.0.0.1:8000/predict_absa" `
     -H "Content-Type: application/json" `
     -d '{ "text": "The video quality was amazing, but the presenter seemed bored.",
     "aspects": ["video quality", "presenter"] }'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scipy.sparse import hstack
import joblib
import numpy as np
from typing import List

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR
from app.inference_utils import (
    load_production_model,
    build_derived_features,
    safe_to_list,
)
from src.models.absa_model import ABSAModel

logger = get_logger(__name__, headline="main_API_endpoints.py")

app = FastAPI(title="YouTube Sentiment Analysis API", version="1.0")

# ============================================================
# Artifact Loading (Run on startup only)
# ============================================================
# This block handles loading all artifacts. The service will fail
# to launch if any critical files are missing.

try:
    # Load the main sentiment prediction model (MLflow or local fallback)
    model = load_production_model()

    # Load preprocessing artifacts for the main model
    vectorizer = joblib.load(FEATURES_DIR / "tfidf_vectorizer_max_1000.pkl")
    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")
    logger.info("✅ Loaded TF-IDF vectorizer and label encoder successfully.")

    # Load the ABSA model
    absa_model = ABSAModel()
    logger.info("✅ Loaded ABSA model successfully.")

except Exception as e:
    logger.error(f"❌ FATAL: Service cannot start. Artifact loading failed. Error: {e}")
    # Re-raise the exception to prevent the application from starting
    raise


# ============================================================
# Request & Response Schemas
# ============================================================
class PredictRequest(BaseModel):
    texts: list[str]


class ABSAPredictRequest(BaseModel):
    text: str
    aspects: List[str]


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    score: float


# ============================================================
# General Sentiment Prediction Endpoint
# ============================================================
@app.post("/predict")
def predict(data: PredictRequest):
    """
    Predicts the overall sentiment of a list of comments.
    """
    try:
        df_input = pd.DataFrame({"clean_comment": data.texts})

        # --- Feature Engineering ---
        X_tfidf = vectorizer.transform(df_input["clean_comment"])
        X_derived = build_derived_features(df_input)
        X = hstack([X_tfidf, X_derived])

        # --- Prediction ---
        raw_preds = model.predict(X)

        # Post-process predictions to get class labels
        if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            preds = np.argmax(raw_preds, axis=1)
        else:
            preds = raw_preds

        # Decode numeric labels to string labels (e.g., "Positive")
        decoded_preds = label_encoder.inverse_transform(preds)

        logger.info(f"✅ Prediction completed for {len(data.texts)} samples.")

        return {
            "predictions": safe_to_list(decoded_preds),
            "encoded_labels": safe_to_list(preds),
            "feature_shape": list(X.shape),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Aspect-Based Sentiment Analysis (ABSA) Endpoint
# ============================================================
@app.post("/predict_absa", response_model=List[AspectSentiment])
def predict_absa(data: ABSAPredictRequest):
    """
    Performs Aspect-Based Sentiment Analysis on a single text.
    """
    try:
        analysis = absa_model.predict(data.text, data.aspects)
        logger.info(f"✅ ABSA prediction completed for text: '{data.text[:50]}...'")
        return analysis
    except Exception as e:
        logger.error(f"ABSA prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# System Health Endpoint
# ============================================================
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "YouTube Sentiment Analysis API is running."}


# ============================================================
# Main Application Runner
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting FastAPI inference server...")
    logger.info("➡️  Access API docs at: http://127.0.0.1:8000/docs")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
