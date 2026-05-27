"""
EduPredict AI — FastAPI REST API
Serves the multi-output Deep Learning model
(risk classification + exam score regression).
"""

import os
import pickle
import uuid as uuid_lib

import numpy as np
import pandas as pd

from contextlib import asynccontextmanager
from typing import List, Optional

import tensorflow as tf
from tensorflow import keras

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from analyze import router as analyze_router
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_v2")

MODEL_PATH        = os.path.join(MODEL_DIR, "edupredict_multioutput.keras")
SCALER_PATH       = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENC_PATH    = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")
RISK_MAP_PATH     = os.path.join(MODEL_DIR, "risk_map.pkl")


# ─────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────

_state = {}


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and artifacts...")

    try:
        required_files = [
            MODEL_PATH, SCALER_PATH, LABEL_ENC_PATH,
            FEATURE_COLS_PATH, RISK_MAP_PATH,
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")

        _state["model"] = keras.models.load_model(MODEL_PATH)

        with open(SCALER_PATH, "rb") as f:
            _state["scaler"] = pickle.load(f)

        with open(LABEL_ENC_PATH, "rb") as f:
            _state["label_encoders"] = pickle.load(f)

        with open(FEATURE_COLS_PATH, "rb") as f:
            _state["feature_cols"] = pickle.load(f)

        with open(RISK_MAP_PATH, "rb") as f:
            _state["risk_map"] = pickle.load(f)

        _state["inv_risk_map"] = {v: k for k, v in _state["risk_map"].items()}

        print("✅ Model & artifacts loaded successfully.")

    except Exception as e:
        print(f"❌ Failed loading artifacts: {e}")

    yield

    _state.clear()


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="EduPredict AI",
    description="Deep Learning API for Academic Risk Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(analyze_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Schemas — Input
# ─────────────────────────────────────────────────────────────

class StudentFeatures(BaseModel):
    """
    Pure ML feature payload — no identity fields.
    Shared between single and batch requests.
    """
    Hours_Studied: int = Field(..., ge=0, le=48)
    Attendance: float = Field(..., ge=0, le=100)
    Parental_Involvement: str
    Access_to_Resources: str
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Previous_Scores: float = Field(..., ge=0, le=100)
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int = Field(..., ge=0)
    Family_Income: str
    Teacher_Quality: str
    Peer_Influence: str
    Physical_Activity: int = Field(..., ge=0)
    Parental_Education_Level: str

    @validator(
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Family_Income",
        "Teacher_Quality",
    )
    def validate_levels(cls, v):
        allowed = ["Low", "Medium", "High"]
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v

    @validator("Internet_Access")
    def validate_internet(cls, v):
        if v not in ["Yes", "No"]:
            raise ValueError("Must be Yes or No")
        return v

    @validator("Peer_Influence")
    def validate_peer(cls, v):
        allowed = ["Negative", "Neutral", "Positive"]
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v


class StudentRecord(BaseModel):
    """
    A single student entry for prediction.

    Identity:
      - student_id  : required, opaque identifier (e.g. "STU-001", "12345")
      - name        : optional, echoed back in the response for display

    Separating identity from features keeps the ML pipeline clean —
    features go to the model, identity travels alongside.
    """
    student_id: str = Field(..., example="STU-001")
    # name: Optional[str] = Field(None, example="Airin Sastra")
    features: StudentFeatures


class PredictRequest(BaseModel):
    """Single-student prediction request."""
    student: StudentRecord


class BatchPredictRequest(BaseModel):
    """Batch prediction request. Identical structure to single, just wrapped in a list."""
    students: List[StudentRecord]


# ─────────────────────────────────────────────────────────────
# Schemas — Output
# ─────────────────────────────────────────────────────────────

class StudentMeta(BaseModel):
    """Identity fields echoed back in every prediction result."""
    student_id: str
    # name: Optional[str] = None


class PredictionOutput(BaseModel):
    risk_category: str
    confidence: float
    predicted_exam_score: float


class ProbabilitiesOutput(BaseModel):
    """
    Formatted probabilities (0–100 scale) always included.
    Raw probabilities (0–1 scale) only when debug=true is passed.
    """
    Low: float
    Medium: float
    High: float


class PredictionResult(BaseModel):
    """
    Unified result shape — identical whether from /predict or /predict/batch.
    Frontend can always do: data.results.map(...)
    """
    request_id: str
    student: StudentMeta
    prediction: PredictionOutput
    probabilities: ProbabilitiesOutput
    raw_probabilities: Optional[ProbabilitiesOutput] = None  # only when debug=true


class PredictResponse(BaseModel):
    success: bool = True
    message: str
    count: int
    results: List[PredictionResult]


# ─────────────────────────────────────────────────────────────
# Inference Helper
# ─────────────────────────────────────────────────────────────

def _run_inference(record: StudentRecord, debug: bool = False) -> PredictionResult:

    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model          = _state["model"]
    scaler         = _state["scaler"]
    label_encoders = _state["label_encoders"]
    feature_cols   = _state["feature_cols"]
    inv_risk_map   = _state["inv_risk_map"]

    df = pd.DataFrame([record.features.dict()])

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    X        = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    pred_cls_prob, pred_reg = model.predict(X_scaled, verbose=0)

    cls_idx   = int(np.argmax(pred_cls_prob[0]))
    raw_probs = {
        inv_risk_map[i]: float(pred_cls_prob[0][i])
        for i in range(3)
    }
    fmt_probs = {k: round(v * 100, 2) for k, v in raw_probs.items()}

    return PredictionResult(
        request_id=f"pred_{uuid_lib.uuid4().hex[:8]}",
        student=StudentMeta(
            student_id=record.student_id,
            # name=record.name,
        ),
        prediction=PredictionOutput(
            risk_category=inv_risk_map[cls_idx],
            confidence=round(float(pred_cls_prob[0][cls_idx]) * 100, 2),
            predicted_exam_score=round(float(pred_reg[0][0] * 100), 2),
        ),
        probabilities=ProbabilitiesOutput(**fmt_probs),
        raw_probabilities=ProbabilitiesOutput(**raw_probs) if debug else None,
    )


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "EduPredict AI API is running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok" if "model" in _state else "degraded",
        "model_loaded": "model" in _state,
        "tensorflow_version": tf.__version__,
    }


@app.get("/api/v1/model/info")
def model_info():

    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model = _state["model"]

    return {
        "model_name": model.name,
        "input_shape": str(model.input_shape),
        "output_heads": [out.name for out in model.outputs],
        "feature_cols": _state["feature_cols"],
        "risk_classes": _state["risk_map"],
        "total_params": model.count_params(),
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    debug: bool = Query(False, description="Include raw_probabilities (0–1 scale) in response"),
):
    result = _run_inference(payload.student, debug=debug)

    return PredictResponse(
        success=True,
        message="Prediction completed",
        count=1,
        results=[result],
    )


@app.post("/api/v1/predict/batch", response_model=PredictResponse)
def predict_batch(
    payload: BatchPredictRequest,
    debug: bool = Query(False, description="Include raw_probabilities (0–1 scale) in response"),
):
    results = [
        _run_inference(student, debug=debug)
        for student in payload.students
    ]

    return PredictResponse(
        success=True,
        message=f"Batch prediction completed for {len(results)} student(s)",
        count=len(results),
        results=results,
    )
