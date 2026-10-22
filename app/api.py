"""FastAPI production service for real-time transaction fraud detection.

Endpoints:
- GET /health: Health check and model metadata.
- POST /predict: Evaluates transaction risk, probability, tiered decision,
  and local SHAP explainability reasons.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, REPORTS_DIR
from src.features import FraudFeaturePipeline

# Path anchors
MODEL_PATH = MODELS_DIR / "best_lightgbm.joblib"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
FINAL_METRICS_PATH = REPORTS_DIR / "final_metrics.json"

# In-memory global artifacts
model = None
feature_pipeline: Optional[FraudFeaturePipeline] = None
tree_explainer = None
optimal_threshold: float = 0.0543


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, feature pipeline, and SHAP explainer on startup."""
    global model, feature_pipeline, tree_explainer, optimal_threshold

    print("Loading ML artifacts into memory...")
    if not MODEL_PATH.is_file() or not PIPELINE_PATH.is_file():
        raise RuntimeError("ML model artifacts missing. Run pipeline first.")

    model = joblib.load(MODEL_PATH)
    feature_pipeline = FraudFeaturePipeline.load(PIPELINE_PATH)

    # Initialize fast TreeExplainer for local feature attribution
    tree_explainer = shap.TreeExplainer(model)

    # Read verified threshold from final metrics if available
    if FINAL_METRICS_PATH.is_file():
        import json
        with open(FINAL_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
            optimal_threshold = metrics_data.get("optimal_cost_threshold_from_val", 0.0543)

    print(f"Artifacts successfully loaded. Operating threshold: {optimal_threshold:.4f}")
    yield
    print("Shutting down API service.")


app = FastAPI(
    title="Transaction Risk Engine API",
    description="Real-time ML fraud detection pipeline with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    """Input transaction payload with validation."""
    TransactionID: Optional[int] = Field(default=1000001, description="Unique transaction identifier")
    TransactionDT: float = Field(default=86400.0, description="Seconds elapsed from reference epoch")
    TransactionAmt: float = Field(gt=0, default=150.0, description="Monetary transaction amount in USD")
    ProductCD: str = Field(default="W", description="Product code (e.g. W, C, H, R, S)")
    card1: int = Field(default=10000, description="Payment card bank/issuer identifier")
    card2: Optional[float] = Field(default=111.0, description="Card issuer code")
    card3: Optional[float] = Field(default=150.0, description="Card country code")
    card4: Optional[str] = Field(default="visa", description="Card brand (visa, mastercard, discover, etc.)")
    card5: Optional[float] = Field(default=226.0, description="Card category code")
    card6: Optional[str] = Field(default="debit", description="Card type (credit, debit)")
    addr1: Optional[float] = Field(default=315.0, description="Billing region / zip code prefix")
    addr2: Optional[float] = Field(default=87.0, description="Billing country code")
    P_emaildomain: Optional[str] = Field(default="gmail.com", description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(default="gmail.com", description="Recipient email domain")

    # Additional optional signal counters
    C1: Optional[float] = Field(default=1.0, description="Transaction velocity counter 1")
    C13: Optional[float] = Field(default=1.0, description="Transaction velocity counter 13")
    C14: Optional[float] = Field(default=1.0, description="Transaction velocity counter 14")
    D1: Optional[float] = Field(default=0.0, description="Time delta since previous transaction")
    D2: Optional[float] = Field(default=0.0, description="Time delta 2")

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "TransactionID": 9999991,
                "TransactionDT": 12500000,
                "TransactionAmt": 185.50,
                "ProductCD": "W",
                "card1": 13500,
                "card2": 224.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "credit",
                "addr1": 299.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
            }
        },
    )


class ContributingFeature(BaseModel):
    feature: str
    impact_score: float
    description: str


class PredictionResponse(BaseModel):
    transaction_id: Optional[int]
    fraud_probability: float
    is_fraud: bool
    decision: str
    risk_level: str
    operating_threshold: float
    top_contributing_features: List[ContributingFeature]


def describe_feature_impact(feat_name: str, impact: float) -> str:
    """Provide domain explanation for SHAP feature attribution."""
    if "TransactionAmt" in feat_name:
        return "Transaction dollar amount significantly increases fraud risk."
    elif "card1" in feat_name or "card_id" in feat_name:
        return "Card transaction frequency and spend pattern deviation."
    elif "C13" in feat_name or "C1" in feat_name:
        return "High rapid transaction velocity on this card/account."
    elif "email" in feat_name:
        return "Email domain interaction and purchaser risk score."
    elif "hour_of_day" in feat_name:
        return "Late night / atypical transaction hour."
    elif "has_identity" in feat_name:
        return "Absence or anomaly in digital browser/device telemetry."
    return f"Elevated behavioral signal on feature {feat_name}."


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint providing model metadata."""
    return {
        "status": "healthy",
        "service": "Transaction Risk Engine",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "operating_threshold": optimal_threshold,
        "model_type": "LightGBM Classifier",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_fraud(transaction: TransactionRequest):
    """Score a transaction in real-time, returning probability and SHAP reasons."""
    if model is None or feature_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Server initializing.",
        )

    try:
        raw_dict = transaction.model_dump()
        tx_id = raw_dict.get("TransactionID")

        # 1. Transform record using production pipeline
        X_single = feature_pipeline.transform_single(raw_dict)

        # 2. Predict posterior fraud probability
        prob = float(model.predict_proba(X_single)[:, 1][0])
        is_fraud = bool(prob >= optimal_threshold)

        # 3. Multi-tiered operational policy
        if prob >= 0.50:
            risk_level = "HIGH"
            decision = "DECLINE"
        elif prob >= optimal_threshold:
            risk_level = "MEDIUM"
            decision = "CHALLENGE"  # 3D-Secure OTP challenge
        else:
            risk_level = "LOW"
            decision = "APPROVE"

        # 4. Local SHAP Attribution (Top 3 drivers)
        top_reasons: List[ContributingFeature] = []
        if tree_explainer is not None:
            shap_vals = tree_explainer.shap_values(X_single)
            # Handle binary class output format
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                vals = shap_vals[0, :, 1]
            else:
                vals = shap_vals[0]

            # Top 3 positive contributors toward fraud
            sorted_indices = np.argsort(vals)[::-1]
            top_3_indices = [idx for idx in sorted_indices if vals[idx] > 0][:3]

            feature_names = X_single.columns.tolist()
            for idx in top_3_indices:
                fn = feature_names[idx]
                score = round(float(vals[idx]), 4)
                desc = describe_feature_impact(fn, score)
                top_reasons.append(ContributingFeature(feature=fn, impact_score=score, description=desc))

        return PredictionResponse(
            transaction_id=tx_id,
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            decision=decision,
            risk_level=risk_level,
            operating_threshold=round(optimal_threshold, 4),
            top_contributing_features=top_reasons,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
