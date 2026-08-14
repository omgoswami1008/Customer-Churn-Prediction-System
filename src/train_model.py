"""
Project: Customer Churn Prediction System
Module : train_model.py

Responsibility
--------------
Trains and persists the production churn classifier.

Pipeline
--------
1. Load the engineered feature matrix (``data/processed/features.csv``).
2. Stratified train/test split (target is imbalanced: ~26% churn).
3. Standardise numeric columns (kept for pipeline completeness / any future
   linear model; Random Forest itself is scale-invariant).
4. Tune a Random Forest with 5-fold cross-validation, scored on ROC-AUC
   (robust to imbalance) and ``class_weight='balanced'``.
5. Persist model + scaler + split so evaluation and prediction reuse the
   exact same artifacts.

Artifacts produced
------------------
    models/random_forest.pkl        dict{model, feature_names, scaled_columns}
    models/scaler.pkl               fitted StandardScaler
    data/processed/data_splits.joblib  X/y train+test (unscaled)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "data_splits.joblib"

TARGET_COLUMN = "Churn"
EDA_ONLY_COLUMNS = ["tenure_group"]

# Random Forest is scale-invariant; we still scale these for pipeline
# completeness and to support any future linear model.
SCALED_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_addon_services",
    "avg_monthly_charges",
]

# Hyperparameter search space (kept compact so tuning runs in minutes).
PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered (encoded) features and target from ``features.csv``.

    Returns
    -------
    (X, y) : tuple
        Feature matrix and churn target (0/1).
    """
    features_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            "Feature matrix not found. Run feature engineering first: "
            "python -m src.feature_engineering"
        )

    df = pd.read_csv(features_path)
    drop_cols = [c for c in [TARGET_COLUMN, *EDA_ONLY_COLUMNS] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)

    if not X.columns.is_unique:
        raise ValueError("Duplicate feature columns detected in feature matrix.")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        raise ValueError(f"Non-numeric feature columns after encoding: {non_numeric}")

    logger.info("Feature matrix: X=%s | positives=%d (%.1f%%)",
                X.shape, int(y.sum()), 100 * float(y.mean()))
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split, preserving the churn ratio in both folds.

    The split is persisted to ``data_splits.joblib`` so evaluation always
    uses the exact same holdout the model was validated against.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split: train=%s test=%s | churn train=%.1f%% test=%.1f%%",
                X_train.shape, X_test.shape,
                100 * float(y_train.mean()), 100 * float(y_test.mean()))
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(X_train: pd.DataFrame, columns: list[str]) -> StandardScaler:
    """
    Fit a StandardScaler on the numeric columns of the training set only.

    Fitting only on train prevents data leakage from the test set.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[columns])
    logger.info("Fitted StandardScaler on %d numeric columns.", len(columns))
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply a fitted scaler to the numeric columns of a feature frame.

    Non-scaled columns are left untouched; column order is preserved.
    """
    X_out = X.copy()
    X_out[columns] = scaler.transform(X_out[columns])
    return X_out


# ---------------------------------------------------------------------------
# Model selection & training
# ---------------------------------------------------------------------------

def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Grid-search a Random Forest via 5-fold stratified CV on ROC-AUC.

    Returns
    -------
    RandomForestClassifier
        The best estimator, already refit on the full training set.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        estimator=base,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    logger.info("Best params: %s", grid.best_params_)
    logger.info("Best CV ROC-AUC: %.4f (+/- %.4f)",
                grid.best_score_, 2 * grid.cv_results_["std_test_score"][grid.best_index_])
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model_artifact(
    model: RandomForestClassifier,
    feature_names: list[str],
    scaled_columns: list[str],
    path: Path | str = MODEL_PATH,
) -> Path:
    """Persist the trained model together with its input schema."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names, "scaled_columns": scaled_columns},
        dest,
    )
    logger.info("Saved model artifact: %s", dest)
    return dest


def save_scaler(scaler: StandardScaler, path: Path | str = SCALER_PATH) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, dest)
    logger.info("Saved scaler: %s", dest)
    return dest


def save_splits(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    path: Path | str = SPLITS_PATH,
) -> Path:
    """Persist the exact split so evaluation reuses the same holdout."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        },
        dest,
    )
    logger.info("Saved data splits: %s", dest)
    return dest


def load_model_artifact(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    """Load the model artifact dict for prediction / evaluation."""
    dest = Path(path)
    if not dest.exists():
        raise FileNotFoundError(f"Model artifact not found at {dest}. Run train_model first.")
    return joblib.load(dest)


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------

def train() -> dict[str, Any]:
    """Run the full training pipeline and persist all artifacts."""
    X, y = load_feature_matrix()
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = fit_scaler(X_train, SCALED_COLUMNS)

    X_train_scaled = apply_scaler(scaler, X_train, SCALED_COLUMNS)
    X_test_scaled = apply_scaler(scaler, X_test, SCALED_COLUMNS)

    model = tune_random_forest(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    logger.info("Holdout sanity check | ROC-AUC: %.4f", roc_auc_score(y_test, y_proba))

    save_model_artifact(model, list(X.columns), SCALED_COLUMNS)
    save_scaler(scaler)
    save_splits(X_train, X_test, y_train, y_test)

    summary = {
        "best_params": getattr(model, "best_params_", model.get_params()),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_features": X.shape[1],
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }
    return summary


if __name__ == "__main__":
    # Quick run:  python -m src.train_model
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    summary = train()
    print("\n=== TRAINING SUMMARY ===")
    for key, value in summary.items():
        print(f"  {key:>14s}: {value}")
