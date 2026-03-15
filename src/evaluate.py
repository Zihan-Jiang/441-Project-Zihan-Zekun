from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def evaluate_classifier(model, X, y) -> dict:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "y_true": np.array(y),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_prob)
        metrics["avg_precision"] = average_precision_score(y, y_prob)

        fpr, tpr, _ = roc_curve(y, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y, y_prob)

        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["pr_precision_curve"] = pr_precision
        metrics["pr_recall_curve"] = pr_recall
    else:
        metrics["roc_auc"] = np.nan
        metrics["avg_precision"] = np.nan
        metrics["fpr"] = None
        metrics["tpr"] = None
        metrics["pr_precision_curve"] = None
        metrics["pr_recall_curve"] = None

    return metrics