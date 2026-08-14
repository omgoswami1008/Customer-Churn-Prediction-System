"""
Project: Customer Churn Prediction System
Module : data_loader.py

Responsibility
--------------
Handles all data loading concerns for the project:

1. Locates and reads the raw telecom churn dataset.
2. Validates the file exists and is non-empty.
3. Returns a clean :class:`pandas.DataFrame` ready for downstream
   preprocessing and modelling.

Design notes
------------
- Paths are resolved relative to the project root, so the module works
  no matter which directory it is imported from (CLI, notebook, Streamlit).
- Validation failures raise explicit exceptions with actionable messages
  instead of failing silently, which keeps the pipeline safe to chain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Set up a module-level logger. Consumed by the app / scripts / notebook.
logger = logging.getLogger(__name__)

# Relative path from this file (src/) to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default dataset location inside the project.
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telecom_churn.csv"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"


def load_raw_data(filepath: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load the raw telecom churn dataset from disk.

    Parameters
    ----------
    filepath : Optional[Path | str], default=None
        Full path to the raw CSV file. When ``None`` the default location
        ``<project_root>/data/raw/telecom_churn.csv`` is used.

    Returns
    -------
    pd.DataFrame
        The raw dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the given path.
    ValueError
        If the dataset file exists but is empty.
    """
    path = Path(filepath) if filepath else DEFAULT_RAW_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}\n"
            "Expected default: data/raw/telecom_churn.csv"
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Raw dataset is empty: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded raw dataset: %s | shape=%s", path.name, df.shape)
    return df


def load_processed_data(filepath: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load the cleaned / feature-engineered dataset.

    This is intended to be used *after* the pipeline has run once, so that
    notebooks and the Streamlit app can quickly reuse the prepared data
    without re-running preprocessing every time.

    Parameters
    ----------
    filepath : Optional[Path | str], default=None
        Full path to the processed CSV file. When ``None`` the default
        location ``<project_root>/data/processed/clean_data.csv`` is used.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    Raises
    ------
    FileNotFoundError
        If ``clean_data.csv`` has not been generated yet.
    """
    path = Path(filepath) if filepath else DEFAULT_PROCESSED_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}\n"
            "Run `python src/preprocessing.py` to generate it first."
        )

    df = pd.read_csv(path)
    logger.info("Loaded processed dataset: %s | shape=%s", path.name, df.shape)
    return df


def quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact frame describing the dataset.

    Useful for a fast first look during EDA. For each column it reports
    the dtype, the number of non-null values and the number of missing
    values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarise.

    Returns
    -------
    pd.DataFrame
        A summary frame indexed by column name.
    """
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.count(),
            "nulls": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
        }
    )
    return summary


if __name__ == "__main__":
    # Allow running as a script for a quick sanity check:
    #   python src/data_loader.py
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    raw = load_raw_data()
    print(quick_summary(raw).to_string())
