"""Step 5 - benchmark all prescribed models on the same splits and target.

Models : persistence & seasonal-climatology baselines, Decision Tree, Random Forest, AdaBoost,
         Gradient Boosting (sklearn), XGBoost - each on raw features and on PCA components.
Tasks  : regression of AQI(D+1); direct classification of AQI(D+1) > 200; and the classification
         DERIVED from the regression forecast (at the fixed 200 rule and at a validation-tuned rule).
Protocol (identical for every model and both feature kinds):
  1. screening  : walk-forward folds (train <= 2019 -> validate 2020, train <= 2020 -> validate 2021).
                  The imputer, class weights and PCA are fitted inside each fold on that fold's
                  training rows. Grids are screened on a 40 % sample of each fold's training rows
                  (stratified by station x year). Boosters pick their best round on the
                  validation year; the score is the mean over the two folds.
  2. finalist   : the chosen setting is refitted on ALL rows <= 2020 and validated on 2021 ->
                  number of rounds, classification thresholds (F1) and the validation score that
                  selects the production model (validation only, never the test set).
  3. final      : refit on <= 2021 with those settings; evaluated on test 2022-01 .. 2023-03.
Outputs: reports/benchmark.csv, benchmark_ci.csv, benchmark_meta.json, tuning_log.csv,
         run_manifest.json, data/processed/preds_test.parquet, models/*.joblib, models/final_params.json,
         models/thresholds.json, models/production.json
Usage  : python src/models.py                      full benchmark
         python src/models.py --only XGBoost       partial run -> written to reports/partial, models/partial
         python src/models.py --jobs 6             parallel screening jobs for single-threaded models
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import (FIG, MODELS, PERSIST_MAX_AGE, PROC, REPORTS, SEED, TRAIN_END, TUNE_FOLDS,  # noqa: E402
                    VAL_END)
from metrics import (best_f1_threshold, block_bootstrap, cls_metrics_aqi, cls_metrics_prob,  # noqa: E402
                     reg_metrics)
from prep import Imputer, load  # noqa: E402
from runinfo import file_sha256, write_manifest  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
import zoo  # noqa: E402

KINDS = ["raw", "pca"]


def stratified_sample(F, idx, frac, rng):
    """`frac` of the rows in idx, sampled within every station x year (keeps coverage)."""
    if frac >= 1:
        return np.asarray(idx)
    df = pd.DataFrame({"i": idx, "g": F.station.values[idx] + "_" + F.date.dt.year.values[idx].astype(str),
                       "r": rng.random(len(idx))}).sort_values(["g", "r"])
    n = df.groupby("g").g.transform("size").values
    k = df.groupby("g").cumcount().values
    return np.sort(df.i.values[k < np.maximum(1, np.round(frac * n))])


def model_bytes(model):
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.tell()


def persistence(F, clim):
    """Tomorrow = today's MEASURED AQI; if missing, the last measured AQI if at most
    PERSIST_MAX_AGE days old; otherwise the seasonal climatology (always finite)."""
    p = F.aqi_l0.where(F.miss_aqi_l0 == 0)
    p = p.fillna(F.aqi_last_valid.where(F.days_since_valid <= PERSIST_MAX_AGE))
    return p.fillna(pd.Series(clim, index=F.index)).values.astype(float)


def screen_one(name, task, kind, p, Xtr, ytr, Xva, yva):
    _, stages, n_best, _ = zoo.fit_eval(name, task, kind, p, Xtr, ytr, Xva, yva)
    return float(np.min(stages)), n_best


def pca_figure(X, mask, ncols):
    sc = StandardScaler().fit(X[mask])
    pca = PCA(random_state=SEED).fit(sc.transform(X[mask]))
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(1, len(cum) + 1), ev * 100, color=SERIES[0], width=0.8, label="per component")
    ax.plot(range(1, len(cum) + 1), cum * 100, color=SERIES[1], label="cumulative")
    ax.axhline(95, color=TEXT2, ls=":", lw=1)
    ax.axvline(n95, color=TEXT2, ls=":", lw=1)
    ax.text(n95 + 1, 50, f"{n95} components\n= 95% variance", fontsize=8, color=TEXT2)
    ax.set_xlabel("principal component")
    ax.set_ylabel("% variance explained")
    ax.set_title(f"PCA on {ncols} standardised features (fit on <= 2021)")
    ax.legend(loc="center right")
    save(fig, FIG / "pca_explained_variance.png")
    return n95, cum


# ------------------------------------------------------------------ main
def main(only=None, jobs=6, screen_frac=0.4):
    partial = bool(only)
    out_rep = REPORTS / "partial" if partial else REPORTS
    out_mod = MODELS / "partial" if partial else MODELS
    out_rep.mkdir(exist_ok=True, parents=True)
    out_mod.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(SEED)

    F, feats, groups = load()
    t = F.target_date
    y_reg = F.y_aqi.values.astype(float)
    y_cls = F.y_poor.values.astype(int)
    m_test = (t > VAL_END).values
    m_trval = ~m_test
    m_tr20 = (t <= TRAIN_END).values
    m_va21 = ((t > TRAIN_END) & (t <= VAL_END)).values

    imp_final = Imputer("clim", feats).fit(F, m_trval)
    X_all = imp_final.transform(F).astype("float32")
    cols = list(X_all.columns)
    X_all = X_all.values
    imp20 = Imputer("clim", feats).fit(F, m_tr20)
    X20 = imp20.transform(F).astype("float32").values          # finalist fit <= 2020, validate 2021

    folds = []
    for tr_end, va_end in TUNE_FOLDS:
        tr = (t <= tr_end).values
        va = ((t > tr_end) & (t <= va_end)).values
        Xf = Imputer("clim", feats).fit(F, tr).transform(F).astype("float32").values
        folds.append((Xf, stratified_sample(F, np.flatnonzero(tr), screen_frac, rng), np.flatnonzero(va)))
    print(f"features: {len(cols)}  rows train+val: {m_trval.sum():,}  test: {m_test.sum():,}", flush=True)
    n95, cum = pca_figure(X_all, m_trval, len(cols))

    # ---- baselines (fitted with the same cutoffs as the models) ---------------------
    seasonal = imp_final.target_climatology(F).values.astype(float)
    persist = persistence(F, seasonal)
    seasonal20 = imp20.target_climatology(F).values.astype(float)
    persist20 = persistence(F, seasonal20)
    p_mae = float(np.mean(np.abs(persist[m_test] - y_reg[m_test])))
    results = []
    preds = pd.DataFrame({"station": F.station[m_test].values, "date": F.date[m_test].values,
                          "target_date": t[m_test].values, "city": F.city[m_test].values,
                          "city_key": F.city_key[m_test].values, "state": F.state[m_test].values,
                          "y": y_reg[m_test], "y_poor": y_cls[m_test]})
    thresholds = {}
    for name, p, p20 in [("Persistence", persist, persist20), ("Seasonal climatology", seasonal, seasonal20)]:
        thr_aqi = best_f1_threshold(y_cls[m_va21], p20[m_va21], strict=True)
        results += [dict(model=name, features="-", task="regression", **reg_metrics(y_reg[m_test], p[m_test], p_mae),
                         val_score=float(np.mean(np.abs(p20[m_va21] - y_reg[m_va21])))),
                    dict(model=name, features="-", task="derived_cls", **cls_metrics_aqi(y_cls[m_test], p[m_test])),
                    dict(model=name, features="-", task="derived_cls_tuned",
                         **cls_metrics_aqi(y_cls[m_test], p[m_test], thr_aqi))]
        preds[f"{name}|reg"] = p[m_test]
        thresholds[f"{name}|-"] = dict(derived_aqi=thr_aqi)
    print(pd.DataFrame(results)[["model", "task", "MAE", "F1"]], flush=True)

    # ---- learned models ----------------------------------------------------------------
    tuning_log, final_params, val_scores = [], {}, {}
    for name in (only or zoo.MODEL_NAMES):
        for task in ["reg", "cls"]:
            y = y_reg if task == "reg" else y_cls
            for kind in KINDS:
                # 1. screening on the two walk-forward folds
                gp = zoo.grid(name, task)
                todo = [(pi, fi) for pi in range(len(gp)) for fi in range(len(folds))]

                def args(pi, fi):   # slices are created lazily, one job at a time
                    Xf, tr, va = folds[fi]
                    return (name, task, kind, gp[pi], Xf[tr], y[tr], Xf[va], y[va])
                if zoo.THREADS[name] == 1 and jobs > 1:
                    out = Parallel(n_jobs=jobs)(delayed(screen_one)(*args(pi, fi)) for pi, fi in todo)
                else:
                    out = [screen_one(*args(pi, fi)) for pi, fi in todo]
                sc = {}
                for (pi, fi), (s, nb) in zip(todo, out):
                    sc.setdefault(pi, []).append(s)
                    tuning_log.append(dict(model=name, task=task, features=kind, params=json.dumps(gp[pi]),
                                           fold=fi + 1, val_score=s, best_rounds=nb))
                best_pi = min(sc, key=lambda k: np.mean(sc[k]))
                best = gp[best_pi]
                print(f"  {name} {task} {kind}: best {best} (mean val score {np.mean(sc[best_pi]):.4f})", flush=True)

                # 2. finalist on all rows <= 2020, validated on 2021
                m20, stages, n_best, pv = zoo.fit_eval(name, task, kind, best, X20[m_tr20], y[m_tr20],
                                                       X20[m_va21], y[m_va21])
                vs = float(np.min(stages))
                val_scores[f"{name}|{task}|{kind}"] = vs
                thr = {}
                if task == "cls":
                    thr["direct"] = best_f1_threshold(y_cls[m_va21], pv)
                else:
                    thr["derived_aqi"] = best_f1_threshold(y_cls[m_va21], pv, strict=True)
                thresholds[f"{name}|{task}|{kind}"] = thr
                del m20

                # 3. final fit on <= 2021, test
                t0, c0 = time.perf_counter(), time.process_time()
                model = zoo.fit_final(name, task, kind, best, X_all[m_trval], y[m_trval], n_rounds=n_best)
                fit_s, fit_cpu = time.perf_counter() - t0, time.process_time() - c0
                t0 = time.perf_counter()
                pt = zoo.predict(model, task, X_all[m_test])
                pred_s = time.perf_counter() - t0
                size = model_bytes(model)
                key = f"{name}|{task}|{kind}"
                preds[key] = pt
                eff = dict(fit_s=fit_s, fit_cpu_s=fit_cpu, threads=zoo.THREADS[name], predict_s=pred_s,
                           predict_ms_per_1k=1e6 * pred_s / m_test.sum(), size_mb=size / 1e6,
                           params=json.dumps(best), n_rounds=n_best, pca_components=zoo.n_components(model),
                           val_score=vs, effective_params=zoo.effective_params(model))
                if task == "reg":
                    results += [dict(model=name, features=kind, task="regression",
                                     **reg_metrics(y_reg[m_test], pt, p_mae), **eff),
                                dict(model=name, features=kind, task="derived_cls", **cls_metrics_aqi(y_cls[m_test], pt)),
                                dict(model=name, features=kind, task="derived_cls_tuned",
                                     **cls_metrics_aqi(y_cls[m_test], pt, thr["derived_aqi"]))]
                    msg = f"MAE {results[-3]['MAE']:.2f}"
                else:
                    results.append(dict(model=name, features=kind, task="direct_cls",
                                        **cls_metrics_prob(y_cls[m_test], pt, thr["direct"]), **eff))
                    msg = f"F1 {results[-1]['F1']:.3f}  AP {results[-1]['AP']:.3f}"
                joblib.dump(model, out_mod / f"{name}_{task}_{kind}.joblib", compress=3)
                final_params.setdefault(name, {}).setdefault(task, {})[kind] = dict(params=best, n_rounds=n_best)
                print(f"{key:28s} {msg}  rounds {n_best}  fit {fit_s:.0f}s (cpu {fit_cpu:.0f}s)  "
                      f"size {size / 1e6:.1f}MB", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(out_rep / "benchmark.csv", index=False)
    pd.DataFrame(tuning_log).to_csv(out_rep / "tuning_log.csv", index=False)
    preds.to_parquet((PROC / "partial_preds_test.parquet") if partial else (PROC / "preds_test.parquet"), index=False)
    joblib.dump(imp_final, out_mod / "imputer.joblib")
    (out_mod / "final_params.json").write_text(json.dumps(final_params, indent=1))
    (out_mod / "thresholds.json").write_text(json.dumps(thresholds, indent=1))

    # production model = best VALIDATION score (2021), never chosen on the test set
    reg_keys = {k: v for k, v in val_scores.items() if "|reg|" in k}
    cls_keys = {k: v for k, v in val_scores.items() if "|cls|" in k}
    prod = dict(regressor=min(reg_keys, key=reg_keys.get) if reg_keys else None,
                classifier=min(cls_keys, key=cls_keys.get) if cls_keys else None,
                rule="lowest 2021 validation MAE (regression) / highest 2021 validation AP (classifier), "
                     "model fitted on <= 2020", validation_scores=val_scores)
    (out_mod / "production.json").write_text(json.dumps(prod, indent=1))

    ci = confidence_intervals(preds, res)
    ci.to_csv(out_rep / "benchmark_ci.csv", index=False)
    meta = dict(n_features=len(cols), features=cols, pca_n95_trainval=n95, pca_cumvar=cum.tolist(),
                n_trainval=int(m_trval.sum()), n_test=int(m_test.sum()), persistence_mae=p_mae,
                screen_fraction=screen_frac, production=prod)
    (out_rep / "benchmark_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    write_manifest(out_rep / "run_manifest.json", step="models", partial=partial, models=only or zoo.MODEL_NAMES,
                   cutoffs=dict(train_end=TRAIN_END, val_end=VAL_END, tune_folds=TUNE_FOLDS), seed=SEED,
                   final_params=final_params, thresholds=thresholds,
                   data=dict(features_parquet=file_sha256(PROC / "features.parquet"), n_rows=len(F)))
    show = ["model", "features", "task", "MAE", "RMSE", "MAE_severe", "F1", "AP", "fit_s", "fit_cpu_s"]
    print(res[[c for c in show if c in res]].to_string())
    print("production:", prod["regressor"], "/", prod["classifier"])


def confidence_intervals(preds, res, B=1000):
    """95 % paired block-bootstrap intervals (blocks = ISO weeks of the target date) for test MAE,
    MAE difference vs persistence, and (direct classifiers) F1 and AP."""
    from sklearn.metrics import average_precision_score, f1_score
    wk = pd.to_datetime(preds.target_date).dt.strftime("%G-%V").values
    y, yc = preds.y.values, preds.y_poor.values
    fns = {}
    per = preds["Persistence|reg"].values
    for c in preds.columns:
        if c.endswith("|reg") or "|reg|" in c:
            p = preds[c].values
            fns[f"{c}|MAE"] = lambda i, p=p: np.mean(np.abs(p[i] - y[i]))
            fns[f"{c}|dMAE_vs_persistence"] = lambda i, p=p: np.mean(np.abs(p[i] - y[i]) - np.abs(per[i] - y[i]))
        if "|cls|" in c:
            p = preds[c].values
            thr = res[(res.model == c.split("|")[0]) & (res.features == c.split("|")[2]) &
                      (res.task == "direct_cls")].threshold.iloc[0]
            fns[f"{c}|F1"] = lambda i, p=p, thr=thr: f1_score(yc[i], p[i] >= thr)
            fns[f"{c}|AP"] = lambda i, p=p: average_precision_score(yc[i], p[i])
    ci = block_bootstrap(wk, fns, B=B, seed=SEED)
    rows = []
    for k, (lo, hi) in ci.items():
        col, metric = k.rsplit("|", 1)
        rows.append(dict(prediction=col, metric=metric, value=float(fns[k](np.arange(len(y)))), lo95=lo, hi95=hi))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=zoo.MODEL_NAMES)
    ap.add_argument("--jobs", type=int, default=6, help="parallel screening jobs for single-threaded models")
    ap.add_argument("--screen-frac", type=float, default=0.4)
    a = ap.parse_args()
    main(only=a.only, jobs=a.jobs, screen_frac=a.screen_frac)
