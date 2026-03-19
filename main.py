"""
main.py
-------
FastAPI back-end for the Sentiment Analysis web app.

Endpoints:
  GET  /           → Serves the HTML front-end
  POST /predict    → Returns sentiment prediction + probabilities
  GET  /health     → Health check for deployment platforms

Run locally:
  uvicorn main:app --reload
  → http://127.0.0.1:8000

Docs (auto-generated):
  → http://127.0.0.1:8000/docs
"""

import os
import pickle
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── CRITICAL: import TextPreprocessor so pickle can deserialise model.pkl
# Both train_model.py and main.py import from the same module,
# so pickle always finds the class at the same path.
from preprocessor import TextPreprocessor  # noqa: F401  (used implicitly by pickle)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment using TF-IDF + the best auto-selected classifier.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model.pkl"

# Binary labels (IMDB). Neutral kept for backwards compatibility with v2 model.
LABEL_MAP = {0: "Negative", 1: "Positive", 2: "Neutral"}
EMOJI_MAP = {0: "😞",        1: "😊",        2: "😐"}

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅  Model loaded successfully.")
else:
    print("⚠️  model.pkl not found. Run  python train_model.py  first.")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="I absolutely loved this product!")

class PredictResponse(BaseModel):
    sentiment:     str
    emoji:         str
    confidence:    float
    probabilities: dict
    word_count:    int


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run train_model.py first.")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        proba     = model.predict_proba([text])[0]
        pred_idx  = int(np.argmax(proba))
        confidence = float(proba[pred_idx])

        # Build probability dict — works for both 2-class and 3-class models
        probabilities = {
            LABEL_MAP[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return PredictResponse(
            sentiment     = LABEL_MAP[pred_idx],
            emoji         = EMOJI_MAP[pred_idx],
            confidence    = round(confidence, 4),
            probabilities = probabilities,
            word_count    = len(text.split()),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
