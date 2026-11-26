"""Evaluation metrics for regression (AQI) and 'Poor or worse' classification.

Decision semantics (kept separate on purpose):
  * AQI value  -> class : aqi.is_poor(value)  i.e. AQI > 200   (the target definition)
  * score/prob -> class : score >= threshold                     (threshold chosen on validation)
'AP' is sklearn's average_precision_score (step-wise area under the precision-recall curve).
"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score,
                             r2_score, recall_score, roc_auc_score)

from aqi import category_index, is_poor, is_severe


def category(a):
    """0..5 for Good..Severe, -1 for missing (see aqi.category_index)."""
    return category_index(a)


def reg_metrics(y, p, persist_mae=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    p95, p99 = np.quantile(y, [0.95, 0.99])
    sev = is_severe(y)
    out = {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "bias": float(np.mean(p - y)),
        "MAE_top5pct": mean_absolute_error(y[y >= p95], p[y >= p95]),
        "MAE_top1pct": mean_absolute_error(y[y >= p99], p[y >= p99]),
        "bias_top5pct": float(np.mean(p[y >= p95] - y[y >= p95])),
        "MAE_severe": mean_absolute_error(y[sev], p[sev]) if sev.any() else np.nan,
        "bias_severe": float(np.mean(p[sev] - y[sev])) if sev.any() else np.nan,
        "severe_recall": float(np.mean(is_severe(p)[sev])) if sev.any() else np.nan,
        "cat_acc": float(np.mean(category(y) == category(p))),
        "cat_within1": float(np.mean(np.abs(category(y) - category(p)) <= 1)),
    }
    if persist_mae:
        out["skill_vs_persistence_pct"] = 100 * (1 - out["MAE"] / persist_mae)
    return {k: float(v) for k, v in out.items()}


def _cls_from_pred(y, pred, score, threshold, probability):
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "AP": float(average_precision_score(y, score)),
        "ROC_AUC": float(roc_auc_score(y, score)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "false_alert_rate": float(pred[y == 0].mean()) if (y == 0).any() else np.nan,
        "threshold": float(threshold),
        "base_rate": float(y.mean()),
    }
    if probability:
        out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    return out


def cls_metrics_aqi(y_true, aqi_pred, threshold=None):
    """Classification derived from an AQI forecast: Poor+ if forecast > threshold (default 200,
    the target definition; a validation-tuned AQI threshold may be passed)."""
    y = np.asarray(y_true).astype(int)
    a = np.asarray(aqi_pred, float)
    pred = (is_poor(a) if threshold is None else (a > threshold)).astype(int)
    return _cls_from_pred(y, pred, a, 200 if threshold is None else threshold, probability=False)


def cls_metrics_prob(y_true, score, threshold):
    """Classification from a classifier score: Poor+ if score >= threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    return _cls_from_pred(y, (s >= threshold).astype(int), s, threshold, probability=True)


def best_f1_threshold(y_true, score, strict=False):
    """Threshold maximising F1 over ALL distinct validation scores (exact PR curve).
    strict=False -> rule 'score >= t' (probabilities); strict=True -> rule 'score > t' (AQI values)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, float)
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12), 0)
    t = float(thr[int(np.argmax(f1))])
    if strict:      # 'score >= t' == 'score > t_prev' : the largest distinct score below t
        below = np.unique(s[s < t])
        t = float(below[-1]) if len(below) else t - 1e-6
    return t


def onset_metrics(y_today_measured, y_next, alert):
    """Early warning of an ONSET: AQI(D) measured and <= 200, AQI(D+1) > 200."""
    today = np.asarray(y_today_measured, float)
    nxt = np.asarray(y_next, float)
    a = np.asarray(alert, bool)
    onset = np.isfinite(today) & ~is_poor(today) & is_poor(nxt)
    stay_clean = np.isfinite(today) & ~is_poor(today) & ~is_poor(nxt)
    return {"onsets": int(onset.sum()),
            "onset_recall": float(a[onset].mean()) if onset.any() else np.nan,
            "false_alert_on_clean_days": float(a[stay_clean].mean()) if stay_clean.any() else np.nan}


def block_bootstrap(blocks, stat_fns, B=1000, seed=42):
    """Paired block bootstrap. `blocks`: array of block ids per row (e.g. ISO week: keeps the
    strong same-day correlation between stations and short-range autocorrelation together).
    `stat_fns`: {name: f(row_index_array) -> float}. Returns {name: (lo, hi)} 95 % intervals.
    The same resampled blocks are used for every statistic, so differences are paired."""
    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks)
    uniq, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    res = {k: [] for k in stat_fns}
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        for k, f in stat_fns.items():
            res[k].append(f(idx))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in res.items()}
