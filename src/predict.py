"""
Project: Customer Churn Prediction System
Module : predict.py

Responsibility
--------------
Serves predictions for individual customers (used by the Streamlit app and
the API-style ``predict_customer`` helper).

Flow
----
raw customer dict -> feature engineering (same as training) -> scaling
-> Random Forest -> churn probability + risk band.

Artifacts required (produced by modules 3-4)
--------------------------------------------
    models/random_forest.pkl         model + feature schema
    models/scaler.pkl                fitted StandardScaler
    models/feature_config.joblib     one-hot / binary column maps

Run
---
    python -m src.predict
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import build_feature_row
from .train_model import apply_scaler, load_model_artifact

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
CONFIG_PATH = PROJECT_ROOT / "models" / "feature_config.joblib"

# Probability bands mapped to business-friendly risk levels.
RISK_LOW, RISK_MEDIUM = 0.30, 0.60


# ---------------------------------------------------------------------------
# Artifact loading (cached so the Streamlit app loads them only once)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_predictor() -> dict[str, Any]:
    """Load and cache model, scaler and feature config."""
    artifact = load_model_artifact(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = joblib.load(CONFIG_PATH)

    if list(artifact["feature_names"]) != config["feature_columns"]:
        raise ValueError(
            "Feature schema mismatch between model artifact and config. "
            "Re-run feature_engineering and train_model."
        )

    logger.info(
        "Predictor loaded | model=%s, features=%d",
        type(artifact["model"]).__name__,
        len(artifact["feature_names"]),
    )
    return {
        "model": artifact["model"],
        "scaler": scaler,
        "config": config,
        "scaled_columns": artifact["scaled_columns"],
    }


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def probability_to_risk(probability: float) -> str:
    """
    Map a churn probability to an interpretable risk level.

    Bands:
        <  0.30  -> Low
        0.30-0.60 -> Medium
        >= 0.60  -> High
    """
    if probability < RISK_LOW:
        return "Low"
    if probability < RISK_MEDIUM:
        return "Medium"
    return "High"


def predict_customer(customer: dict[str, Any]) -> dict[str, Any]:
    """
    Predict churn for a single raw customer record.

    Parameters
    ----------
    customer : dict
        Raw customer data. Keys must match the cleaned dataset columns
        (e.g. ``gender``, ``tenure``, ``MonthlyCharges``, ...). The
        ``Churn`` key, if present, is ignored.

    Returns
    -------
    dict
        ``churn_probability``, ``churn_risk``, ``predicted_class`` and the
        input record echoed back under ``customer``.
    """
    predictor = load_predictor()
    model, scaler, config = predictor["model"], predictor["scaler"], predictor["config"]

    # Drop the target if present - it has no place in a live prediction.
    raw = {k: v for k, v in customer.items() if k != config["target_column"]}

    row = pd.DataFrame([raw])
    X = build_feature_row(row, config)
    X_scaled = apply_scaler(scaler, X, predictor["scaled_columns"])

    probability = float(model.predict_proba(X_scaled)[:, 1][0])
    predicted_class = int(probability >= 0.5)

    return {
        "churn_probability": round(probability, 4),
        "churn_risk": probability_to_risk(probability),
        "predicted_class": predicted_class,
        "customer": raw,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for a batch of customers in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer records (one row per customer).

    Returns
    -------
    pd.DataFrame
        Original frame augmented with ``churn_probability`` and ``churn_risk``.
    """
    predictor = load_predictor()
    model, scaler, config = predictor["model"], predictor["scaler"], predictor["config"]

    input_cols = [c for c in df.columns if c != config["target_column"]]
    X = build_feature_row(df[input_cols], config)
    X_scaled = apply_scaler(scaler, X, predictor["scaled_columns"])

    probabilities = model.predict_proba(X_scaled)[:, 1]
    return df.assign(
        churn_probability=np.round(probabilities, 4),
        churn_risk=[probability_to_risk(p) for p in probabilities],
    )


if __name__ == "__main__":
    # Quick run:  python -m src.predict
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    sample = {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No internet service",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 53.85,
        "TotalCharges": 108.15,
    }
    result = predict_customer(sample)
    print(f"Churn probability: {result['churn_probability']:.4f}")
    print(f"Risk level      : {result['churn_risk']}")
    print(f"Predicted class : {result['predicted_class']} (1=churn)")
