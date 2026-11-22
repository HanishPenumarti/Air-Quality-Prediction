"""Step 7 - benchmark all prescribed models on the same splits and target.

Models : Persistence & seasonal-climatology baselines, Decision Tree, Random Forest,
         AdaBoost, Gradient Boosting (sklearn), XGBoost - each on raw features and on
         PCA components (StandardScaler + PCA fitted on the training period only).
Tasks  : regression of AQI(D+1); direct classification of AQI(D+1) > 200; and the
         classification DERIVED from the regression forecast.
Tuning : walk-forward validation with 2 origins (train<=2019 -> 2020, train<=2020 -> 2021),
         small grids, on a 40% row subsample of each training window.
Final  : refit on train+val (<= 2021) with the chosen params; evaluate ONCE on test
         (2022-01 .. 2023-03). Classification threshold chosen on 2021 (model fitted <= 2020).

Usage: python src/models.py [--only XGBoost ...]
Outputs: reports/benchmark.csv, benchmark_meta.json, tuning_log.csv, data/processed/preds_test.parquet, models/*.joblib
"""
import argparse
import io
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, MODELS, POOR_THRESHOLD, PROC, REPORTS, SEED, TRAIN_END, VAL_END  # noqa: E402
from metrics import best_f1_threshold, cls_metrics, reg_metrics  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402

NJ = 12

# ------------------------------------------------------------------ model zoo
GRIDS = {
    "DecisionTree": dict(max_depth=[8, 12, 16], min_samples_leaf=[50, 200]),
    "RandomForest": dict(max_depth=[16, 24], min_samples_leaf=[10], max_features=[0.33]),
    "AdaBoost": dict(depth=[6, 9], learning_rate=[0.05, 0.3]),
    "GradBoost": dict(max_depth=[4, 6], learning_rate=[0.1]),
    "XGBoost": dict(max_depth=[6, 8], learning_rate=[0.05], min_child_weight=[5, 20]),
}


def make(name, task, p, n_pos_w=1.0):
    reg = task == "reg"
    if name == "DecisionTree":
        cls = DecisionTreeRegressor if reg else DecisionTreeClassifier
        kw = {} if reg else dict(class_weight="balanced")
        return cls(random_state=SEED, **p, **kw)
    if name == "RandomForest":
        cls = RandomForestRegressor if reg else RandomForestClassifier
        kw = {} if reg else dict(class_weight="balanced_subsample")
        return cls(n_estimators=150, n_jobs=NJ, random_state=SEED, **p, **kw)
    if name == "AdaBoost":
        base = (DecisionTreeRegressor if reg else DecisionTreeClassifier)(max_depth=p["depth"], min_samples_leaf=50)
        cls = AdaBoostRegressor if reg else AdaBoostClassifier
        kw = dict(loss="linear") if reg else {}
        return cls(estimator=base, n_estimators=100, learning_rate=p["learning_rate"], random_state=SEED, **kw)
    if name == "GradBoost":
        cls = GradientBoostingRegressor if reg else GradientBoostingClassifier
        kw = dict(loss="huber") if reg else {}
        return cls(n_estimators=250, subsample=0.7, max_features=0.5, random_state=SEED,
                   n_iter_no_change=15, validation_fraction=0.1, **p, **kw)
    if name == "XGBoost":
        cls = XGBRegressor if reg else XGBClassifier
        kw = {} if reg else dict(scale_pos_weight=n_pos_w, eval_metric="aucpr")
        return cls(n_estimators=800, subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                   n_jobs=NJ, random_state=SEED, **p, **kw)
    raise ValueError(name)


def fit(model, name, task, X, y, Xv=None, yv=None):
    """Fit with class balancing where the estimator has no class_weight; XGB early-stops on (Xv,yv)."""
    kw = {}
    if task == "cls" and name in ("AdaBoost", "GradBoost"):
        kw["sample_weight"] = compute_sample_weight("balanced", y)
    if name == "XGBoost" and Xv is not None:
        model.set_params(early_stopping_rounds=50)
        kw.update(eval_set=[(Xv, yv)], verbose=False)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if name == "XGBoost" and hasattr(model, "steps") and Xv is not None:
        # pipeline: transform eval set manually
        pre = make_pipeline(*[s for _, s in model.steps[:-1]]).fit(X)
        est.fit(pre.transform(X), y, **{k: v for k, v in kw.items() if k != "eval_set"},
                eval_set=[(pre.transform(Xv), yv)])
        model.steps[:-1] = pre.steps
        return model
    if hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__{k}": v for k, v in kw.items()})
    else:
        model.fit(X, y, **kw)
    return model


def predict(model, task, X):
    return model.predict(X) if task == "reg" else model.predict_proba(X)[:, 1]


def model_bytes(model):
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.tell()


def with_pca(model, n_comp):
    return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=SEED), model)


# ------------------------------------------------------------------ main
def main(only=None):
    tag = ""
    rng = np.random.default_rng(SEED)
    F, feats, groups = load()
    t = F.target_date
    imp_final = Imputer("ffill_clim", feats).fit(F, (t <= VAL_END).values)
    X_all = imp_final.transform(F).astype("float32")
    cols = list(X_all.columns)
    y_reg = F.y_aqi.values
    y_cls = F.y_poor.values.astype(int)
    m_test = (t > VAL_END).values
    m_trval = ~m_test

    # walk-forward folds for tuning (imputer refitted per fold on its own training window)
    folds = []
    for tr_end, va_end in [("2019-12-31", "2020-12-31"), (TRAIN_END, VAL_END)]:
        tr = (t <= tr_end).values
        va = ((t > tr_end) & (t <= va_end)).values
        Xf = Imputer("ffill_clim", feats).fit(F, tr).transform(F).astype("float32").values
        sub = np.flatnonzero(tr)
        sub = rng.choice(sub, int(0.4 * len(sub)), replace=False)
        folds.append((Xf, sub, np.flatnonzero(va)))
    print(f"features: {len(cols)}  rows train+val: {m_trval.sum():,}  test: {m_test.sum():,}", flush=True)

    # PCA (fit on training period only) - explained variance
    sc = StandardScaler().fit(X_all.values[m_trval])
    pca_full = PCA(random_state=SEED).fit(sc.transform(X_all.values[m_trval]))
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(1, len(cum) + 1), pca_full.explained_variance_ratio_ * 100, color=SERIES[0], width=0.8,
           label="per component")
    ax.plot(range(1, len(cum) + 1), cum * 100, color=SERIES[1], label="cumulative")
    ax.axhline(95, color=TEXT2, ls=":", lw=1)
    ax.axvline(n95, color=TEXT2, ls=":", lw=1)
    ax.text(n95 + 1, 50, f"{n95} components\n= 95% variance", fontsize=8, color=TEXT2)
    ax.set_xlabel("principal component")
    ax.set_ylabel("% variance explained")
    ax.set_title(f"PCA on {len(cols)} standardised features (fit on training period)")
    ax.legend(loc="center right")
    save(fig, FIG / f"pca_explained_variance{tag}.png")

    # ---- baselines -----------------------------------------------------------------
    results, preds = [], pd.DataFrame({"station": F.station[m_test].values, "date": F.date[m_test].values,
                                       "target_date": t[m_test].values, "city": F.city[m_test].values,
                                       "state": F.state[m_test].values, "y": y_reg[m_test]})
    persist = F.aqi_l0.where(F.miss_aqi_l0 == 0, F.aqi_last_valid).fillna(F.clim_aqi_month).values
    seasonal = F.clim_aqi_month.fillna(np.nanmedian(F.clim_aqi_month)).values
    p_mae = float(np.mean(np.abs(persist[m_test] - y_reg[m_test])))
    va21 = ((t > TRAIN_END) & (t <= VAL_END)).values
    for name, p in [("Persistence", persist), ("Seasonal climatology", seasonal)]:
        r = dict(model=name, features="-", task="regression", **reg_metrics(y_reg[m_test], p[m_test], p_mae))
        results.append(r)
        results.append(dict(model=name, features="-", task="derived_cls",
                            **cls_metrics(y_cls[m_test], p[m_test], POOR_THRESHOLD)))
        preds[f"{name}|reg"] = p[m_test]
    print(pd.DataFrame(results)[["model", "task", "MAE", "F1"]], flush=True)

    # ---- learned models ----------------------------------------------------------------
    tuning_log = []
    names = only or list(GRIDS)
    for name in names:
        for task in ["reg", "cls"]:
            y = y_reg if task == "reg" else y_cls
            npw = float((y_cls[m_trval] == 0).sum() / max(1, (y_cls[m_trval] == 1).sum()))
            # -- tuning (raw features), walk-forward
            grid = [dict(zip(GRIDS[name], v)) for v in itertools.product(*GRIDS[name].values())]
            best, best_score = None, np.inf
            for p in grid:
                scores = []
                for Xf, tr_idx, va_idx in folds:
                    mdl = fit(make(name, task, p, npw), name, task, Xf[tr_idx], y[tr_idx], Xf[va_idx], y[va_idx])
                    pv = predict(mdl, task, Xf[va_idx])
                    if task == "reg":
                        scores.append(np.mean(np.abs(pv - y[va_idx])))
                    else:
                        from sklearn.metrics import average_precision_score
                        scores.append(-average_precision_score(y[va_idx], pv))
                s = float(np.mean(scores))
                tuning_log.append(dict(model=name, task=task, params=p, score=s))
                print(f"  tune {name:12s} {task} {p} -> {s:.4f}", flush=True)
                if s < best_score:
                    best, best_score = p, s

            for feat_kind in ["raw", "pca"]:
                Xtr, Xte = X_all.values[m_trval], X_all.values[m_test]
                base = make(name, task, best, npw)
                if name == "XGBoost":   # use # rounds found by early stopping on the 2021 fold
                    probe = fit(make(name, task, best, npw), name, task, *_fold_xy(folds[1], y),
                                ) if feat_kind == "raw" else None
                    if probe is not None:
                        n_best = int(probe.best_iteration + 1)
                    base.set_params(n_estimators=n_best)
                model = with_pca(base, n95) if feat_kind == "pca" else base

                # threshold for classification: fit on <=2020, choose on 2021 (frozen for test)
                thr = POOR_THRESHOLD
                if task == "cls":
                    Xf, tr_idx, va_idx = folds[1]
                    mm = with_pca(make(name, task, best, npw), n95) if feat_kind == "pca" else make(name, task, best, npw)
                    if name == "XGBoost":
                        (mm.steps[-1][1] if hasattr(mm, "steps") else mm).set_params(n_estimators=n_best)
                    mm = fit(mm, name, task, Xf[tr_idx], y[tr_idx])
                    thr = best_f1_threshold(y[va_idx], predict(mm, task, Xf[va_idx]))

                t0 = time.perf_counter()
                model = fit(model, name, task, Xtr, y[m_trval])
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                pt = predict(model, task, Xte)
                pred_s = time.perf_counter() - t0
                size = model_bytes(model)
                key = f"{name}|{task}|{feat_kind}"
                preds[key] = pt
                eff = dict(fit_s=fit_s, predict_s=pred_s, predict_ms_per_1k=1000 * pred_s / len(Xte) * 1000,
                           size_mb=size / 1e6, params=json.dumps(best))
                if task == "reg":
                    results.append(dict(model=name, features=feat_kind, task="regression",
                                        **reg_metrics(y_reg[m_test], pt, p_mae), **eff))
                    results.append(dict(model=name, features=feat_kind, task="derived_cls",
                                        **cls_metrics(y_cls[m_test], pt, POOR_THRESHOLD)))
                else:
                    results.append(dict(model=name, features=feat_kind, task="direct_cls",
                                        **cls_metrics(y_cls[m_test], pt, thr), **eff))
                joblib.dump(model, MODELS / f"{name}_{task}_{feat_kind}{tag}.joblib", compress=3)
                last = results[-1] if task == "cls" else results[-2]
                print(f"{key:32s} " + (f"MAE {last['MAE']:.2f}" if task == "reg" else f"F1 {last['F1']:.3f}")
                      + f"  fit {fit_s:.0f}s  size {size / 1e6:.1f}MB", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(REPORTS / f"benchmark{tag}.csv", index=False)
    meta = dict(n_features=len(cols), features=cols, pca_n95=n95, pca_cumvar=cum.tolist(),
                n_trainval=int(m_trval.sum()), n_test=int(m_test.sum()), persistence_mae=p_mae,
                imputer_median=imp_final.median_.to_dict())
    (REPORTS / f"benchmark_meta{tag}.json").write_text(json.dumps(meta, indent=1, default=float))
    pd.DataFrame(tuning_log).to_csv(REPORTS / f"tuning_log{tag}.csv", index=False)
    preds.to_parquet(PROC / f"preds_test{tag}.parquet", index=False)
    joblib.dump(imp_final, MODELS / f"imputer{tag}.joblib")
    show = ["model", "features", "task", "MAE", "RMSE", "R2", "MAE_top5pct", "F1", "PR_AUC", "fit_s"]
    print(res[[c for c in show if c in res]].to_string())


def _fold_xy(fold, y):
    Xf, tr, va = fold
    return Xf[tr], y[tr], Xf[va], y[va]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(only=a.only)
