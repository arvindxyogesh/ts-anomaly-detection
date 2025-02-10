"""Anomaly detection evaluation: precision, recall, F1, PR-AUC."""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)


def evaluate_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    name: str = "",
) -> Dict[str, float]:
    """Return precision, recall, F1, ROC-AUC, PR-AUC for a detector."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "detector": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0, 4),
        "pr_auc":    round(average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
