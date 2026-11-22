"""Evaluation metrics for regression (AQI) and 'Poor or worse' classification."""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

from config import AQI_CAT_EDGES, POOR_THRESHOLD


def category(a):
    return np.digitize(a, AQI_CAT_EDGES[1:-1], right=True)   # 0..5


def reg_metrics(y, p, persist_mae=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    p95, p99 = np.quantile(y, [0.95, 0.99])
    out = {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "bias": float(np.mean(p - y)),
        "MAE_top5pct": mean_absolute_error(y[y >= p95], p[y >= p95]),
        "MAE_top1pct": mean_absolute_error(y[y >= p99], p[y >= p99]),
        "bias_top5pct": float(np.mean(p[y >= p95] - y[y >= p95])),
        "cat_acc": float(np.mean(category(y) == category(p))),
        "cat_within1": float(np.mean(np.abs(category(y) - category(p)) <= 1)),
    }
    if persist_mae:
        out["skill_vs_persistence_pct"] = 100 * (1 - out["MAE"] / persist_mae)
    return {k: float(v) for k, v in out.items()}


def cls_metrics(y_true, score, threshold):
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "F1": float(f1_score(y, pred)),
        "PR_AUC": float(average_precision_score(y, score)),
        "ROC_AUC": float(roc_auc_score(y, score)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "threshold": float(threshold),
        "base_rate": float(y.mean()),
    }


def best_f1_threshold(y_true, score):
    """Threshold maximising F1 (chosen on validation data, then frozen for test)."""
    qs = np.unique(np.quantile(score, np.linspace(0.3, 0.97, 200)))
    f1s = [f1_score(y_true, score >= t) for t in qs]
    return float(qs[int(np.argmax(f1s))])


REG_THRESHOLD = POOR_THRESHOLD
