"""
Project: Customer Churn Prediction System
Module : preprocessing.py

Responsibility
--------------
Transforms the raw dataset into a clean, modelling-ready frame:

1. Drops non-predictive columns (``customerID``) and duplicate rows.
2. Fixes data types (numeric coercion for ``TotalCharges`` / ``tenure``).
3. Normalises string values (strips whitespace, catches empty cells).
4. Standardises categorical encodings (``SeniorCitizen``: 0/1 -> No/Yes).
5. Handles missing values with an explainable imputation strategy.
6. Validates the target column and writes ``data/processed/clean_data.csv``.

Pipeline output
---------------
    data/processed/clean_data.csv   <- loaded later by feature_engineering.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import DEFAULT_PROCESSED_PATH, load_raw_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Columns that carry no predictive signal and must be dropped.
NON_PREDICTIVE_COLUMNS = ["customerID"]

# The target column.
TARGET_COLUMN = "Churn"

# SeniorCitizen is stored as 0/1 but semantically is a Yes/No flag.
SENIOR_CITIZEN_MAP = {0: "No", 1: "Yes"}

# Columns that are numeric by nature and may need coercion.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def _strip_and_catch_empties(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on object columns and convert empty strings to NaN."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Coerce listed columns to numeric.

    Handles the known quirk of this dataset where ``TotalCharges`` can be
    stored as text with blank cells. Unparseable values become NaN and are
    imputed in a later step.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise semantic encodings (e.g. SeniorCitizen 0/1 -> No/Yes)."""
    df = df.copy()
    if "SeniorCitizen" in df.columns:
        present = set(df["SeniorCitizen"].dropna().unique())
        if present.issubset({0, 1}):
            df["SeniorCitizen"] = df["SeniorCitizen"].map(SENIOR_CITIZEN_MAP)
    return df


def _drop_non_predictive(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers and columns with no predictive value."""
    drop_cols = [c for c in NON_PREDICTIVE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning("Removed %d duplicate row(s).", removed)
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numeric values.

    Strategy (explainable, domain-aware):
    * ``TotalCharges`` is the sum of billings so far. For a customer of
      ``tenure`` months charged ``MonthlyCharges``, the expected lifetime
      spend is ``tenure * MonthlyCharges``. We use that where possible and
      fall back to the column median.
    * ``tenure`` / ``MonthlyCharges`` fall back to their median.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        mask = df["TotalCharges"].isna()
        if mask.any():
            inferred = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
            df.loc[mask & inferred.notna(), "TotalCharges"] = inferred[mask & inferred.notna()]
            df["TotalCharges"] = df["TotalCharges"].fillna(
                df["TotalCharges"].median()
            )
            logger.info(
                "Imputed %d missing TotalCharges values (tenure * monthly_charges / median).",
                int(mask.sum()),
            )

    for col in ["tenure", "MonthlyCharges"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed missing %s values with median.", col)

    return df


def _validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the target is present and contains only Yes/No."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing after cleaning."
        )
    bad = df[~df[TARGET_COLUMN].isin(["Yes", "No"])]
    if not bad.empty:
        raise ValueError(
            f"Target contains unexpected values: {bad[TARGET_COLUMN].unique()}"
        )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning pipeline to a raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telecom churn data.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with dtypes fixed, strings normalised, missing values
        imputed, and the target validated.
    """
    logger.info("Preprocessing started | incoming shape=%s", df.shape)

    df = _drop_non_predictive(df)
    df = _strip_and_catch_empties(df)
    df = _coerce_numeric(df, NUMERIC_COLUMNS)
    df = _standardise_categoricals(df)
    df = _remove_duplicates(df)
    df = _impute_missing(df)
    df = _validate_target(df)

    logger.info("Preprocessing finished | out shape=%s", df.shape)
    return df


def save_processed_data(df: pd.DataFrame, path: Optional[Path | str] = None) -> Path:
    """
    Persist the cleaned frame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    path : Optional[Path | str], default=None
        Destination path; defaults to ``data/processed/clean_data.csv``.

    Returns
    -------
    Path
        The path the file was written to.
    """
    dest = Path(path) if path else DEFAULT_PROCESSED_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved processed data: %s", dest)
    return dest


def run_preprocessing() -> Path:
    """
    End-to-end preprocessing entry point: raw CSV -> clean CSV.

    Returns
    -------
    Path
        Path to the generated ``clean_data.csv``.
    """
    raw = load_raw_data()
    clean = clean_dataframe(raw)
    return save_processed_data(clean)


if __name__ == "__main__":
    # Quick run:  python -m src.preprocessing
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    out = run_preprocessing()
    print(f"\nDone -> {out}")
