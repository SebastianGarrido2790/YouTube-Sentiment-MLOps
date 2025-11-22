"""
FastAPI Inference Service for YouTube Sentiment Analysis.

Loads the latest Production model from MLflow Model Registry (via alias-based loading)
or falls back to a local LightGBM model.

Added Aspect-Based Sentiment Analysis (ABSA) endpoint using a Hugging Face model.

Usage (local):
Ensure MLflow server is running if loading from registry:
    uv run mlflow server --host 127.0.0.1 --port 5000
Then run:
    uv run python -m app.predict_model_absa
Or via Uvicorn:
    uv run uvicorn app.predict_model_absa:app --reload --port 8000

Aspect-Based Sentiment Analysis (ABSA) Endpoint:
    curl -X POST "http://127.0.0.1:8000/predict_absa" `
     -H "Content-Type: application/json" `
     -d '{
           "text": "The video quality was amazing, but the presenter seemed bored.",
           "aspects": ["video quality", "presenter"]
         }'

Response Example:
    [
    {
        "aspect": "video quality",
        "sentiment": "positive",
        "score": 0.99...
    },
    {
        "aspect": "presenter",
        "sentiment": "negative",
        "score": 0.97...
    }
    ]
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

logger = get_logger(__name__, headline="predict_model_absa.py")

app = FastAPI(title="YouTube Sentiment ABSA Prediction API", version="1.0")

# ============================================================
# Artifact Loading (Run on startup only)
# ============================================================
# This block handles the loading of all necessary artifacts when the application starts.
# If any of these critical files are missing, the service will fail to launch.

try:
    # Load the main sentiment prediction model (MLflow or local fallback)
    model = load_production_model()

    # Load preprocessing artifacts
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
# Prediction Endpoints
# ============================================================
@app.post("/predict")
def predict(data: PredictRequest):
    """
    Predicts the sentiment of a list of comments.
    - Vectorizes text using TF-IDF.
    - Appends derived features (e.g., text length, lexicon ratios).
    - Returns sentiment predictions (positive, negative, neutral).
    """
    try:
        df_input = pd.DataFrame({"clean_comment": data.texts})

        # --- Feature Engineering ---
        # 1. Vectorize text with TF-IDF
        X_tfidf = vectorizer.transform(df_input["clean_comment"])
        # 2. Add derived features
        X_derived = build_derived_features(df_input)
        # 3. Combine feature sets
        X = hstack([X_tfidf, X_derived])

        # --- Prediction ---
        # The MLflow pyfunc model handles internal format conversions (Pandas/NumPy/SciPy)
        # Handles the conversion to XGBoost's internal DMatrix format internally.
        raw_preds = model.predict(X)

        # Post-process predictions to get class labels
        if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            # Output is probabilities/scores -> get the argmax
            preds = np.argmax(raw_preds, axis=1)
        else:
            # Output is already class labels
            preds = raw_preds

        # Decode numeric labels back to original string labels (e.g., "Positive")
        decoded_preds = label_encoder.inverse_transform(preds)

        logger.info(f"✅ Prediction completed for {len(data.texts)} samples.")

        return {
            "predictions": safe_to_list(decoded_preds),  # Human-readable labels
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
    Performs Aspect-Based Sentiment Analysis (ABSA) on a single text.
    Identifies the sentiment towards specific aspects within the text.
    """
    if absa_model is None:
        raise HTTPException(
            status_code=501,
            detail="ABSA model is not available. The service was started without it.",
        )
    try:
        analysis = absa_model.predict(data.text, data.aspects)
        logger.info(f"✅ ABSA prediction completed for text: '{data.text[:50]}...'")
        return analysis
    except Exception as e:
        logger.error(f"ABSA prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# System Endpoints
# ============================================================
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "YouTube Sentiment API is running"}


# ============================================================
# Main Application Runner
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting FastAPI inference server...")
    logger.info("➡️  Access API docs at: http://127.0.0.1:8000/docs")
    logger.info("➡️  Access ReDoc at: http://127.0.0.1:8000/redoc")

    uvicorn.run(
        "app.predict_model_absa:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
