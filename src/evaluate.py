"""
Project: Customer Churn Prediction System
Module : evaluate.py

Responsibility
--------------
Provides the post-training evaluation suite for the churn model.

1. Reconstructs the exact holdout used during training (``data_splits.joblib``).
2. Computes classification metrics at a default (0.5) and an *optimal*
   decision threshold (Youden's J, i.e. max ``TPR - FPR``).
3. Produces professional figures:
   * confusion matrix
   * ROC curve
   * Precision-Recall curve
   * top-N feature importances
4. Persists figures to ``reports/figures/`` and bundles everything into
   ``reports/model_report.pdf``.

Run
---
    python -m src.evaluate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: works from CLI, notebooks, servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .train_model import apply_scaler, load_model_artifact

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "data_splits.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PDF_PATH = PROJECT_ROOT / "reports" / "model_report.pdf"

TOP_N_FEATURES = 15

# Style constants so every figure looks consistent.
sns.set_theme(style="whitegrid", context="notebook", palette="Set2")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_evaluation_data() -> tuple[pd.DataFrame, pd.Series, Any]:
    """
    Load the model artifact, scaler and exact training split.

    Returns
    -------
    (X_test, y_test, model) : tuple
        Scaled holdout features, true labels and the trained classifier.
    """
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            "Data splits not found. Train the model first: python -m src.train_model"
        )

    splits = joblib.load(SPLITS_PATH)
    scaler = joblib.load(SCALER_PATH)
    artifact = load_model_artifact()

    X_test = apply_scaler(scaler, splits["X_test"], artifact["scaled_columns"])
    y_test = splits["y_test"]

    logger.info("Evaluation data ready | X_test=%s", X_test.shape)
    return X_test, y_test, artifact["model"]


# ---------------------------------------------------------------------------
# Metrics & threshold selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Choose the decision threshold maximising Youden's J (TPR - FPR).

    Returns
    -------
    (threshold, youden_index) : tuple
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def compute_metrics(
    y_true: pd.Series, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return the standard classification metrics at a given threshold."""
    pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1_score": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, threshold: float) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    ax.set_title(f"ROC Curve (AUC = {roc_auc_score(y_true, y_proba):.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: pd.Series, y_proba: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"Precision-Recall Curve (AP = {average_precision_score(y_true, y_proba):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    return fig


def plot_feature_importance(model: Any, feature_names: list[str]) -> plt.Figure:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(TOP_N_FEATURES)

    fig, ax = plt.subplots(figsize=(7, 6))
    top.plot(kind="barh", color=sns.color_palette("viridis", len(top)), ax=ax)
    ax.set_title(f"Top {TOP_N_FEATURES} Feature Importances (Random Forest)")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Persistence & report
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def build_pdf(
    figures: list[plt.Figure],
    metrics: dict[str, float],
    report_text: str,
    path: Path | str = PDF_PATH,
) -> Path:
    """Bundle figures and metrics into a single PDF report."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(dest) as pdf:
        # Title page with model summary.
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = ["Customer Churn Prediction - Model Report", "", ""]
        lines += [f"{k.replace('_', ' ').title():>14s}: {v:.4f}" for k, v in metrics.items()]
        lines += ["", "", "Holdout classification report", "=" * 40]
        lines += report_text.splitlines()
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        for f in figures:
            pdf.savefig(f)
            plt.close(f)

    logger.info("Saved PDF report: %s", dest)
    return dest


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------

def evaluate() -> dict[str, Any]:
    """Run the full evaluation and persist figures + PDF report."""
    X_test, y_test, model = load_evaluation_data()
    artifact = load_model_artifact()
    y_proba = model.predict_proba(X_test)[:, 1]

    opt_threshold, _ = find_optimal_threshold(y_test, y_proba)

    metrics_default = compute_metrics(y_test, y_proba, 0.5)
    metrics_opt = compute_metrics(y_test, y_proba, opt_threshold)

    pred_opt = (y_proba >= opt_threshold).astype(int)
    report_text = classification_report(y_test, pred_opt, digits=4)

    logger.info("Default (0.5) metrics: %s", {k: round(v, 4) for k, v in metrics_default.items()})
    logger.info("Optimal threshold %.2f metrics: %s", opt_threshold,
                {k: round(v, 4) for k, v in metrics_opt.items()})

    figures = [
        plot_confusion_matrix(y_test, pred_opt, opt_threshold),
        plot_roc_curve(y_test, y_proba),
        plot_pr_curve(y_test, y_proba),
        plot_feature_importance(model, artifact["feature_names"]),
    ]

    save_figure(figures[0], "confusion_matrix.png")
    save_figure(figures[1], "roc_curve.png")
    save_figure(figures[2], "pr_curve.png")
    save_figure(figures[3], "feature_importance.png")

    build_pdf(figures, metrics_opt, report_text)

    return {
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_opt,
        "optimal_threshold": opt_threshold,
        "report_path": str(PDF_PATH),
        "figures_dir": str(FIGURES_DIR),
    }


if __name__ == "__main__":
    # Quick run:  python -m src.evaluate
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    result = evaluate()
    print("\n=== EVALUATION SUMMARY (optimal threshold) ===")
    for k, v in result["metrics_optimal"].items():
        print(f"  {k:>14s}: {v:.4f}")
    print(f"\n  optimal threshold: {result['optimal_threshold']:.3f}")
    print(f"  report: {result['report_path']}")
