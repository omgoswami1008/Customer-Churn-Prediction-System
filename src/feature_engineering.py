"""
Project: Customer Churn Prediction System
Module : feature_engineering.py

Responsibility
--------------
Converts the cleaned data into the final model-ready feature matrix.

1. Simplifies service columns: ``No internet service`` -> ``No`` and
   ``No phone service`` -> ``No`` (semantically "not subscribed").
2. Creates new, interpretable features:
   * ``num_addon_services``   - how many internet add-ons a customer pays for.
   * ``avg_monthly_charges``  - lifetime average spend ``TotalCharges / tenure``.
   * ``tenure_group``         - banded tenure, used for EDA / analysis only.
3. Encodes categoricals deterministically so new single-row input can be
   transformed with exactly the same columns:
   * binary Yes/No columns  -> 1/0
   * multi-class columns    -> one-hot (all categories kept)
4. Persists the engineered frame and a ``feature_config`` that later stages
   (training, prediction, Streamlit app) reuse.

Artifacts produced
------------------
    data/processed/features.csv          full engineered frame (incl. Churn)
    models/feature_config.joblib         column schema required to re-encode
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from .data_loader import load_processed_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
CONFIG_PATH = PROJECT_ROOT / "models" / "feature_config.joblib"

# ---------------------------------------------------------------------------
# Column maps
# ---------------------------------------------------------------------------

# Columns whose "No X service" values mean "not subscribed".
PHONE_SERVICE_DEPENDENTS = ["MultipleLines"]
INTERNET_SERVICE_DEPENDENTS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Binary columns encoded as 1/0. "gender" is Male/Female, the rest are Yes/No.
BINARY_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Multi-class columns encoded with one-hot.
MULTI_CLASS_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]

# Numeric columns kept as-is.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COLUMN = "Churn"

# Add-on services used to count subscriptions (must be simplified to Yes/No first).
ADDON_COLUMNS = INTERNET_SERVICE_DEPENDENTS

# Tenure bands, used for EDA / analysis only (not part of the model matrix).
TENURE_BINS = [0, 6, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-6", "6-12", "12-24", "24-36", "36-48", "48-60", "60-72"]

BINARY_VALUE_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}


# ---------------------------------------------------------------------------
# Feature transforms
# ---------------------------------------------------------------------------

def simplify_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'No internet/phone service' to 'No' so they become binary Yes/No."""
    df = df.copy()
    for col in INTERNET_SERVICE_DEPENDENTS:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")
    for col in PHONE_SERVICE_DEPENDENTS:
        if col in df.columns:
            df[col] = df[col].replace("No phone service", "No")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new interpretable features."""
    df = df.copy()

    # Number of paid internet add-ons (0..6).
    if ADDON_COLUMNS:
        df["num_addon_services"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1)

    # Lifetime average monthly spend; undefined (0/0) for brand-new customers.
    df["avg_monthly_charges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"].replace(0, np.nan),
        0.0,
    )
    df["avg_monthly_charges"] = (
        df["avg_monthly_charges"].fillna(df["MonthlyCharges"]).clip(lower=0.0)
    )

    # Banded tenure for EDA / reporting only.
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )

    return df


def _encode_binary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(BINARY_VALUE_MAP).fillna(0).astype(int)
    return df


def _encode_one_hot(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return df.copy()
    return pd.get_dummies(df, columns=columns, prefix="", prefix_sep="").copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Transform a cleaned dataframe into a model-ready feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data as produced by ``preprocessing.clean_dataframe``.

    Returns
    -------
    (X, y, config) : tuple
        ``X``      - feature matrix for modelling.
        ``y``      - target series (1 = churn).
        ``config`` - schema dict persisted for reuse in prediction.

    Notes
    -----
    One-hot encoding is deterministic (``pd.get_dummies``), so a single new
    row can be re-encoded and aligned to the saved ``feature_columns``.
    """
    df = simplify_service_columns(df)
    df = add_derived_features(df)
    df = _encode_binary(df, BINARY_COLUMNS)
    df = _encode_one_hot(df, MULTI_CLASS_COLUMNS)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
    if "tenure_group" not in df.columns:
        raise ValueError("Derived feature 'tenure_group' missing.")

    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    # Columns excluded from the model matrix (target + EDA-only features).
    analysis_columns = [TARGET_COLUMN, "tenure_group"]
    feature_columns = [c for c in df.columns if c not in analysis_columns]

    X = df[feature_columns]

    config = {
        "binary_columns": BINARY_COLUMNS,
        "multi_class_columns": MULTI_CLASS_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "feature_columns": list(feature_columns),
        "target_column": TARGET_COLUMN,
    }

    logger.info("Feature engineering done | X=%s y=%s", X.shape, y.shape)
    return X, y, config


def align_to_config(X: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Align a (possibly single-row) feature frame to the saved feature columns.

    This guarantees prediction input has exactly the same column order and
    set as the training data. Missing one-hot categories are filled with 0.

    Parameters
    ----------
    X : pd.DataFrame
        Frame produced by applying the same transforms to new data.
    config : dict
        Saved feature schema (from ``engineer_features``).

    Returns
    -------
    pd.DataFrame
        Frame re-indexed to ``config["feature_columns"]``.
    """
    missing = [c for c in config["feature_columns"] if c not in X.columns]
    if missing:
        logger.info("Filling %d one-hot column(s) absent in new input.", len(missing))
        X = X.copy()
        for col in missing:
            X[col] = 0
    return X[config["feature_columns"]]


def build_feature_row(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Transform one (or several) raw customer row(s) into an aligned feature matrix.

    Reuses the exact same transforms used at training time, so a single new
    customer produces a row compatible with the saved model schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer data (as entered in the Streamlit form / API).
    config : dict
        Saved feature schema (from ``engineer_features``).

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to ``config["feature_columns"]``.
    """
    transformed = simplify_service_columns(df)
    transformed = add_derived_features(transformed)
    transformed = _encode_binary(transformed, config["binary_columns"])
    transformed = _encode_one_hot(transformed, config["multi_class_columns"])
    return align_to_config(transformed, config)


def save_engineered_data(df: pd.DataFrame, path: Path | str = FEATURES_PATH) -> Path:
    """Persist the full engineered frame (including Churn and tenure_group)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved engineered data: %s", dest)
    return dest


def save_feature_config(config: dict[str, Any], path: Path | str = CONFIG_PATH) -> Path:
    """Persist the feature schema for reuse by prediction / the app."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(config, dest)
    logger.info("Saved feature config: %s", dest)
    return dest


def run_feature_engineering() -> Path:
    """
    End-to-end feature engineering: clean CSV -> encoded features.csv + config.

    ``features.csv`` holds the fully *encoded* matrix (binary ints + one-hot
    columns) together with the target ``Churn`` and the EDA-only
    ``tenure_group``, so the training module can consume it directly.

    Returns
    -------
    Path
        Path to the generated ``features.csv``.
    """
    clean = load_processed_data()

    X, y, config = engineer_features(clean)

    # Rebuild the full encoded frame (incl. target + tenure_group) to save.
    full = simplify_service_columns(clean)
    full = add_derived_features(full)
    full = _encode_binary(full, config["binary_columns"])
    full = _encode_one_hot(full, config["multi_class_columns"])
    save_engineered_data(full)

    save_feature_config(config)
    logger.info(
        "Feature matrix ready: X=%s | churn positives=%d (%.1f%%)",
        X.shape,
        int(y.sum()),
        100 * float(y.mean()),
    )
    return FEATURES_PATH


if __name__ == "__main__":
    # Quick run:  python -m src.feature_engineering
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    out = run_feature_engineering()
    print(f"\nDone -> {out}")
