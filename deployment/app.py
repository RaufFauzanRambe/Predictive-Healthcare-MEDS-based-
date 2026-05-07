"""
app.py - FastAPI application for serving healthcare prediction models.

Provides RESTful API endpoints for real-time patient readmission
prediction using a trained model checkpoint. Supports both single
and batch predictions with risk level classification.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import Predictor
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Pydantic Models for Request/Response ────────────────────────────────────

class PatientFeatures(BaseModel):
    """Input features for a single patient prediction."""
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    bmi: Optional[float] = Field(None, ge=10, le=60, description="Body mass index")
    blood_pressure_systolic: Optional[float] = Field(None, ge=60, le=250, description="Systolic BP (mmHg)")
    blood_pressure_diastolic: Optional[float] = Field(None, ge=30, le=150, description="Diastolic BP (mmHg)")
    heart_rate: Optional[float] = Field(None, ge=30, le=200, description="Heart rate (bpm)")
    respiratory_rate: Optional[float] = Field(None, ge=8, le=40, description="Respiratory rate")
    oxygen_saturation: Optional[float] = Field(None, ge=70, le=100, description="O2 saturation (%)")
    temperature: Optional[float] = Field(None, ge=90, le=110, description="Temperature (F)")
    white_blood_cell_count: Optional[float] = Field(None, ge=0, le=50, description="WBC count")
    hemoglobin: Optional[float] = Field(None, ge=3, le=20, description="Hemoglobin (g/dL)")
    platelet_count: Optional[float] = Field(None, ge=10, le=600, description="Platelet count")
    creatinine: Optional[float] = Field(None, ge=0.1, le=15, description="Creatinine (mg/dL)")
    glucose: Optional[float] = Field(None, ge=30, le=600, description="Glucose (mg/dL)")
    length_of_stay: Optional[float] = Field(None, ge=0, le=365, description="Length of stay (days)")
    num_prior_admissions: Optional[float] = Field(None, ge=0, le=50, description="Number of prior admissions")
    num_medications: Optional[float] = Field(None, ge=0, le=50, description="Number of medications")
    num_procedures: Optional[float] = Field(None, ge=0, le=20, description="Number of procedures")
    num_diagnoses: Optional[float] = Field(None, ge=0, le=30, description="Number of diagnoses")
    has_diabetes: Optional[int] = Field(None, ge=0, le=1, description="Diabetes indicator")
    has_hypertension: Optional[int] = Field(None, ge=0, le=1, description="Hypertension indicator")
    has_chd: Optional[int] = Field(None, ge=0, le=1, description="Coronary heart disease indicator")
    has_copd: Optional[int] = Field(None, ge=0, le=1, description="COPD indicator")
    has_renal_disease: Optional[int] = Field(None, ge=0, le=1, description="Renal disease indicator")
    smoker: Optional[int] = Field(None, ge=0, le=1, description="Smoker indicator")


class PredictionResponse(BaseModel):
    """Response for a single patient prediction."""
    prediction: int = Field(..., description="Binary prediction (0=no readmission, 1=readmission)")
    probability: float = Field(..., description="Predicted probability of readmission")
    risk_level: str = Field(..., description="Risk classification: low, moderate, or high")
    logit: float = Field(..., description="Raw model logit output")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions on multiple patients."""
    patients: list[PatientFeatures] = Field(..., description="List of patient feature sets")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]
    summary: dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Response with model metadata."""
    model_type: str
    device: str
    threshold: float
    feature_count: Optional[int] = None
    parameter_count: Optional[int] = None


# ─── Feature Processing ──────────────────────────────────────────────────────

# Default feature ordering matching the model's training data
FEATURE_ORDER = [
    "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
    "heart_rate", "respiratory_rate", "oxygen_saturation", "temperature",
    "white_blood_cell_count", "hemoglobin", "platelet_count", "creatinine",
    "glucose", "length_of_stay", "num_prior_admissions", "num_medications",
    "num_procedures", "num_diagnoses", "has_diabetes", "has_hypertension",
    "has_chd", "has_copd", "has_renal_disease", "smoker",
]

# Default values for missing features (clinical normals)
DEFAULT_VALUES = {
    "bmi": 26.0,
    "blood_pressure_systolic": 120.0,
    "blood_pressure_diastolic": 80.0,
    "heart_rate": 75.0,
    "respiratory_rate": 16.0,
    "oxygen_saturation": 97.0,
    "temperature": 98.6,
    "white_blood_cell_count": 7.0,
    "hemoglobin": 13.5,
    "platelet_count": 250.0,
    "creatinine": 1.0,
    "glucose": 100.0,
    "length_of_stay": 3.0,
    "num_prior_admissions": 0.0,
    "num_medications": 5.0,
    "num_procedures": 0.0,
    "num_diagnoses": 2.0,
    "has_diabetes": 0,
    "has_hypertension": 0,
    "has_chd": 0,
    "has_copd": 0,
    "has_renal_disease": 0,
    "smoker": 0,
}


def features_to_array(patient: PatientFeatures) -> np.ndarray:
    """
    Convert PatientFeatures to a numpy array matching model input format.

    Fills missing values with clinical defaults and arranges features
    in the expected order for the trained model.

    Args:
        patient: PatientFeatures object from API request.

    Returns:
        1D numpy array of feature values.
    """
    feature_dict = patient.model_dump()
    values = []

    for feat in FEATURE_ORDER:
        val = feature_dict.get(feat)
        if val is None:
            val = DEFAULT_VALUES.get(feat, 0.0)
        values.append(float(val))

    return np.array(values, dtype=np.float32)


# ─── FastAPI Application ─────────────────────────────────────────────────────

app = FastAPI(
    title="Predictive Healthcare MEDS API",
    description="RESTful API for healthcare readmission prediction",
    version="1.0.0",
)

# Global predictor instance (loaded at startup)
_predictor: Optional[Predictor] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load the model on application startup."""
    global _predictor

    import os
    checkpoint = os.environ.get("MODEL_CHECKPOINT", "results/models/lstm_final.pt")
    model_config_path = os.environ.get("MODEL_CONFIG", "configs/model_config.yaml")

    try:
        model_config = load_config(model_config_path)
        _predictor = Predictor(
            checkpoint_path=checkpoint,
            model_config=model_config,
        )
        logger.info(f"Model loaded from {checkpoint}")
    except FileNotFoundError:
        logger.warning(f"No model checkpoint found at {checkpoint}. "
                       f"API will return errors until a model is loaded.")
        _predictor = None


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "predictive-healthcare-meds",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientFeatures) -> PredictionResponse:
    """
    Predict 30-day readmission risk for a single patient.

    Accepts patient demographic and clinical features, returns a
    binary prediction along with the probability and risk level.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        x = features_to_array(patient)
        result = _predictor.predict_single(x)

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["risk_level"],
            logit=result["logit"],
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict readmission risk for multiple patients in a single request.

    Processes a batch of patient feature sets and returns individual
    predictions along with aggregate summary statistics.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert all patients to feature array
        X = np.stack([features_to_array(p) for p in request.patients])

        # Run batch prediction
        results = _predictor.predict(X, return_probs=True)

        # Build individual responses
        predictions = []
        for i in range(len(results["predictions"])):
            predictions.append(PredictionResponse(
                prediction=int(results["predictions"][i]),
                probability=float(results["probabilities"][i]),
                risk_level=_predictor._classify_risk(results["probabilities"][i]),
                logit=float(results["logits"][i]),
            ))

        # Summary statistics
        probs = results["probabilities"]
        summary = {
            "total_patients": len(predictions),
            "predicted_readmission": int(results["predictions"].sum()),
            "mean_probability": float(probs.mean()),
            "high_risk_count": int((probs >= 0.6).sum()),
            "moderate_risk_count": int(((probs >= 0.3) & (probs < 0.6)).sum()),
            "low_risk_count": int((probs < 0.3).sum()),
        }

        return BatchPredictionResponse(predictions=predictions, summary=summary)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Get information about the currently loaded model."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_type=type(_predictor.model).__name__,
        device=str(_predictor.device),
        threshold=_predictor.threshold,
        parameter_count=sum(p.numel() for p in _predictor.model.parameters()),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
