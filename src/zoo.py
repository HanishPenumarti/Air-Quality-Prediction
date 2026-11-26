"""Model zoo shared by the benchmark and every analysis, so all of them build identical models.

One protocol for every model family:
  * class balancing is computed from the rows actually being fitted (fold-specific)
  * PCA = StandardScaler + PCA(95 % variance) fitted inside every fit (fold-specific dimension)
  * boosting models (AdaBoost, Gradient Boosting, XGBoost) choose their number of rounds on a
    CHRONOLOGICAL validation block via staged predictions / evaluation history - no random
    internal holdout, no early stopping in the final refit (the chosen count is used)
"""
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from config import SEED

NJ = 12                       # threads for the multi-threaded learners (RF, XGBoost)
MODEL_NAMES = ["DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]
BOOSTERS = {"AdaBoost", "GradBoost", "XGBoost"}
THREADS = {"DecisionTree": 1, "RandomForest": NJ, "AdaBoost": 1, "GradBoost": 1, "XGBoost": NJ}
PCA_VARIANCE = 0.95
MAX_ROUNDS = {"AdaBoost": 100, "GradBoost": 400, "XGBoost": 1500}
LEARNING_RATE = {"AdaBoost": 0.1, "GradBoost": 0.1, "XGBoost": 0.05}

# loss / objective is a tuned hyper-parameter for the regressors (selected on validation MAE)
GRIDS = {
    "DecisionTree": dict(max_depth=[8, 12, 16], min_samples_leaf=[50, 200]),
    "RandomForest": dict(max_depth=[16, 24], min_samples_leaf=[10], max_features=[0.33]),
    "AdaBoost": dict(depth=[6, 9], loss=["linear", "square"]),
    "GradBoost": dict(max_depth=[4, 6], loss=["squared_error", "huber", "absolute_error"]),
    "XGBoost": dict(max_depth=[6, 8], min_child_weight=[5, 20],
                    objective=["reg:squarederror", "reg:absoluteerror"]),
}
REG_ONLY = {"loss", "objective"}


def grid(name, task):
    g = {k: v for k, v in GRIDS[name].items() if task == "reg" or k not in REG_ONLY}
    keys = list(g)
    import itertools
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def make(name, task, p, n_rounds=None):
    reg = task == "reg"
    if name == "DecisionTree":
        cls = DecisionTreeRegressor if reg else DecisionTreeClassifier
        return cls(random_state=SEED, **p, **({} if reg else dict(class_weight="balanced")))
    if name == "RandomForest":
        cls = RandomForestRegressor if reg else RandomForestClassifier
        return cls(n_estimators=150, n_jobs=NJ, random_state=SEED, **p,
                   **({} if reg else dict(class_weight="balanced_subsample")))
    n = n_rounds or MAX_ROUNDS.get(name)
    if name == "AdaBoost":
        base = (DecisionTreeRegressor if reg else DecisionTreeClassifier)(max_depth=p["depth"], min_samples_leaf=50,
                                                                         random_state=SEED)
        if reg:
            return AdaBoostRegressor(estimator=base, n_estimators=n, learning_rate=LEARNING_RATE[name],
                                     loss=p.get("loss", "linear"), random_state=SEED)
        return AdaBoostClassifier(estimator=base, n_estimators=n, learning_rate=LEARNING_RATE[name], random_state=SEED)
    if name == "GradBoost":
        kw = dict(n_estimators=n, learning_rate=LEARNING_RATE[name], subsample=0.7, max_features=0.5,
                  max_depth=p["max_depth"], random_state=SEED, n_iter_no_change=None)
        if reg:
            return GradientBoostingRegressor(loss=p.get("loss", "squared_error"), **kw)
        return GradientBoostingClassifier(**kw)
    if name == "XGBoost":
        kw = dict(n_estimators=n, learning_rate=LEARNING_RATE[name], subsample=0.8, colsample_bytree=0.7,
                  max_depth=p["max_depth"], min_child_weight=p["min_child_weight"], tree_method="hist",
                  n_jobs=NJ, random_state=SEED)
        if reg:
            return XGBRegressor(objective=p.get("objective", "reg:squarederror"), **kw)
        return XGBClassifier(objective="binary:logistic", **kw)
    raise ValueError(name)


def build(name, task, kind, p, n_rounds=None):
    est = make(name, task, p, n_rounds)
    if kind == "pca":
        return make_pipeline(StandardScaler(), PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED),
                             est)
    return est


def final_estimator(model):
    return model[-1] if hasattr(model, "steps") else model


def predict(model, task, X):
    return model.predict(X) if task == "reg" else model.predict_proba(X)[:, 1]


def val_score(task, y, p):
    """Lower is better: MAE for regression, -AP for classification."""
    return float(np.mean(np.abs(p - y))) if task == "reg" else -float(average_precision_score(y, p))


def _weighted_median(P, w):
    """AdaBoostRegressor prediction from the first k rounds (same rule as sklearn)."""
    idx = np.argsort(P, axis=1)
    cdf = np.cumsum(w[idx], axis=1)
    med = np.argmax(cdf >= 0.5 * cdf[:, -1][:, None], axis=1)
    rows = np.arange(len(P))
    return P[rows, idx[rows, med]]


def fit_eval(name, task, kind, p, X, y, Xv=None, yv=None, n_rounds=None):
    """Fit one model; if a validation block is given, also return the validation score of every
    boosting stage (array; a single value for non-boosting models) and the predictions at the
    best stage. Returns (model, stage_scores | None, best_n_rounds | None, val_pred | None)."""
    model = build(name, task, kind, p, n_rounds)
    est = final_estimator(model)
    if kind == "pca":
        pre = model[:-1].fit(X)                      # same step objects as inside `model`
        Xt, Xvt = pre.transform(X), (pre.transform(Xv) if Xv is not None else None)
    else:
        Xt, Xvt = X, Xv
    kw = {}
    if task == "cls":
        if name in ("AdaBoost", "GradBoost"):
            kw["sample_weight"] = compute_sample_weight("balanced", y)
        if name == "XGBoost":
            est.set_params(scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())))
    if name == "XGBoost" and Xvt is not None:
        metric = "mae" if task == "reg" else "aucpr"
        est.set_params(eval_metric=metric, early_stopping_rounds=100)
        kw.update(eval_set=[(Xvt, yv)], verbose=False)
    est.fit(Xt, y, **kw)
    if Xvt is None:
        return model, None, None, None
    if name == "XGBoost":
        hist = np.asarray(est.evals_result()["validation_0"][metric], float)
        stages = hist if task == "reg" else -hist
    elif name == "AdaBoost" and task == "reg":
        # sklearn's AdaBoostRegressor.staged_predict recomputes a weighted median over ALL previous
        # rounds at every stage (quadratic cost). Predict each round once and evaluate every 10th.
        P = np.column_stack([e.predict(Xvt) for e in est.estimators_])
        w = np.asarray(est.estimator_weights_[:P.shape[1]], float)
        grid_n = sorted(set(list(range(10, P.shape[1] + 1, 10)) + [P.shape[1]]))
        stages, preds = [], []
        for k in grid_n:
            pv = _weighted_median(P[:, :k], w[:k])
            stages.append(val_score(task, yv, pv))
            preds.append(pv)
        i = int(np.argmin(stages))
        return model, np.asarray(stages), grid_n[i], preds[i]
    elif name in ("AdaBoost", "GradBoost"):
        gen = est.staged_predict(Xvt) if task == "reg" else (q[:, 1] for q in est.staged_predict_proba(Xvt))
        stages, best_pred, best = [], None, np.inf
        for pv in gen:
            s = val_score(task, yv, pv)
            stages.append(s)
            if s < best:
                best, best_pred = s, pv
        stages = np.asarray(stages)
    else:
        pv = predict(est, task, Xvt)
        return model, np.array([val_score(task, yv, pv)]), None, pv
    n_best = int(np.argmin(stages) + 1)
    if name == "XGBoost":
        pv = est.predict(Xvt, iteration_range=(0, n_best)) if task == "reg" else \
            est.predict_proba(Xvt, iteration_range=(0, n_best))[:, 1]
        est.set_params(early_stopping_rounds=None)
    else:
        pv = best_pred
    return model, stages, n_best, pv


def fit_final(name, task, kind, p, X, y, n_rounds=None):
    model, *_ = fit_eval(name, task, kind, p, X, y, n_rounds=n_rounds)
    return model


def n_components(model):
    return int(model[1].n_components_) if hasattr(model, "steps") else None


def effective_params(model):
    from runinfo import json_safe
    return json.dumps(json_safe(final_estimator(model).get_params()))


def single_thread(model):
    """For 1-row inference (UI, latency benchmark): thread start-up dominates tiny predictions."""
    est = final_estimator(model)
    if "n_jobs" in est.get_params():
        est.set_params(n_jobs=1)
    return model
