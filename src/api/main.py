"""
FastAPI Inference Service for YouTube Sentiment Analysis.
===========================================================

This unified service provides two main endpoints:
1.  `/predict`: For general sentiment analysis (Positive, Negative, Neutral).
2.  `/predict_absa`: For Aspect-Based Sentiment Analysis (ABSA).

It loads all required models and artifacts at startup.

Usage (local):
Ensure MLflow server is running if loading from registry:
        uv run python -m mlflow server --backend-store-uri sqlite:///mlflow_system.db \
        --default-artifact-root ./mlruns_system --host 127.0.0.1 --port 5000
Then run:
    uv run python -m src.api.main
Or via Uvicorn:
    uv run python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

Test with:
1.  Sentiment Prediction:
    curl -X POST "http://127.0.0.1:8000/v1/predict" `
     -H "Content-Type: application/json" `
     -d '{"texts": ["I love this video! It was super helpful."]}'

2.  Aspect-Based Sentiment:
    curl -X POST "http://127.0.0.1:8000/v1/predict_absa" `
     -H "Content-Type: application/json" `
     -d '{ "text": "The video quality was amazing, but the presenter seemed bored.",
     "aspects": ["video quality", "presenter"] }'
"""

from contextlib import asynccontextmanager
from typing import Any, cast

import joblib
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack

from src.api.agent_api import agent_router
from src.api.inference_utils import (
    build_derived_features,
    load_production_model,
    preprocess_text_inference,
    safe_to_list,
)
from src.constants import FEATURES_DIR
from src.utils.logger import get_logger

# Sentiment Mapping (assuming model outputs 0=Negative, 1=Neutral, 2=Positive)
SENTIMENT_MAP = {
    0: -1,
    1: 0,
    2: 1,
}

logger = get_logger(__name__, headline="main_API_endpoints.py")

app = FastAPI(title="YouTube Sentiment Analysis API", version="1.0")

# ============================================================
# CORS Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global Artifact Placeholders
# ============================================================
model = None
vectorizer = None
label_encoder = None
absa_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle.
    Loads critical artifacts (model, vectorizer, encoder) before handling requests.
    """
    global model, vectorizer, label_encoder
    logger.info("⏳ Initializing FastAPI unified service (lifespan startup)...")
    try:
        # Load main artifacts
        model = load_production_model()
        vectorizer = joblib.load(FEATURES_DIR / "vectorizer.pkl")
        label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")
        logger.info("✅ Core artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"⚠️ Warning: Non-fatal artifact load failure during startup: {e}")
        logger.info("The service will continue starting, but /predict may fail until artifacts are restored.")

    yield
    logger.info("🛑 Shutting down service.")


app = FastAPI(title="YouTube Sentiment Analysis API", version="1.0", lifespan=lifespan)


# ============================================================
# Request & Response Schemas
# ============================================================
class PredictRequest(BaseModel):
    texts: list[str]


class ABSAPredictRequest(BaseModel):
    text: str
    aspects: list[str]


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    score: float


# ============================================================
# API Versioning - v1 Router
# ============================================================
v1_router = APIRouter(prefix="/v1")


@v1_router.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint to confirm the API is running."""
    return {"status": "healthy", "message": "YouTube Sentiment Analysis API is running."}


@v1_router.post("/predict")
def predict(data: PredictRequest):
    """
    Predicts the overall sentiment of a list of comments.
    """
    if not all([model, vectorizer, label_encoder]):
        logger.error("❌ Prediction aborted: Artifacts (model, vectorizer, encoder) are not loaded.")
        raise HTTPException(status_code=503, detail="Service uninitialized: Artifacts missing.")

    try:
        # --- Text Preprocessing (Consistent with Training) ---
        df_input = preprocess_text_inference(data.texts)

        # --- Feature Engineering ---
        x_tfidf = vectorizer.transform(df_input["clean_comment"])
        x_derived = build_derived_features(df_input)
        x_combined = hstack([x_tfidf, x_derived])

        # --- Prediction ---
        raw_preds = model.predict(x_combined)

        # Post-process predictions to get class labels (supporting both sklearn and lightgbm wrappers)
        if hasattr(raw_preds, "ndim") and raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            preds = np.argmax(raw_preds, axis=1)
        else:
            preds = raw_preds

        # Map to original numeric (-1,0,1)
        preds_numeric = [SENTIMENT_MAP.get(int(p), 0) for p in preds]

        # Decode numeric labels to string labels (e.g., "Positive")
        decoded_preds = label_encoder.inverse_transform(preds)

        logger.info(f"✅ Prediction completed for {len(data.texts)} samples.")

        return {
            "predictions": safe_to_list(decoded_preds),
            "encoded_labels": safe_to_list(preds),
            "numeric_labels": preds_numeric,
            "feature_shape": list(cast(Any, x_combined).shape) if hasattr(x_combined, "shape") else [],
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================
# Aspect-Based Sentiment Analysis (ABSA) Endpoint
# ============================================================
# Global ABSA model placeholder
absa_model = None


@v1_router.post("/predict_absa", response_model=list[AspectSentiment])
def predict_absa(data: ABSAPredictRequest):
    """
    Performs Aspect-Based Sentiment Analysis on a single text.
    Lazy-loads the ABSA model on first request to prevent startup hangs.
    """
    global absa_model
    try:
        # Lazy initialization
        if absa_model is None:
            logger.info("⏳ Initializing ABSA model (this may take a moment)...")
            from src.components.absa_model import ABSAModel

            absa_model = ABSAModel()
            logger.info("✅ ABSA model loaded successfully.")

        analysis = absa_model.predict(data.text, data.aspects)
        logger.info(f"✅ ABSA prediction completed for text: '{data.text[:50]}...'")
        return analysis
    except Exception as e:
        logger.error(f"ABSA prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


v1_router.include_router(agent_router)
app.include_router(v1_router)


# ============================================================
# Main Application Runner
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting FastAPI inference server... 🚀")
    logger.info("➡️ Access API docs at: http://127.0.0.1:8000/docs")

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
