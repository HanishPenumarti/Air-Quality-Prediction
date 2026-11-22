"""Step 6 - compare imputation strategies (PDF: 'compare at least two ... and justify').

Part 1  Reconstruction: hide 8% of KNOWN daily PM2.5 values (2019-2022) in contiguous gaps
        of 1-7 days (mimics real outages) and rebuild them with each method; MAE by gap length.
Part 2  Downstream: same XGBoost model trained with each causal feature-imputation strategy;
        forecast MAE / F1 on the 2021 validation year.

Outputs: reports/imputation_results.json, figures imp_01/imp_02
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, PROC, REPORTS, SEED  # noqa: E402
from prep import Imputer, load, split_masks  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402


def reconstruction():
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(PROC / "daily.parquet", columns=["station", "city", "date", "PM25_mean"])
    d = d[(d.date >= "2019-01-01") & (d.date <= "2022-12-31")].sort_values(["station", "date"]).reset_index(drop=True)
    truth = d.PM25_mean.copy()
    known = np.flatnonzero(truth.notna().values)
    starts = rng.choice(known, size=int(0.08 * len(known) / 4), replace=False)
    mask = np.zeros(len(d), bool)
    gap_len = np.zeros(len(d), int)
    for s in starts:
        L = rng.integers(1, 8)
        e = min(s + L, len(d))
        same = d.station.values[s:e] == d.station.values[s]
        idx = np.arange(s, e)[same]
        mask[idx] = True
        gap_len[idx] = L
    mask &= truth.notna().values
    x = truth.mask(mask)
    st = d.station
    month = d.date.dt.month
    res = {}
    res["ffill (causal)"] = x.groupby(st).ffill()
    res["linear interp (uses future)"] = x.groupby(st).transform(lambda s: s.interpolate(limit_area="inside"))
    clim = x.groupby([st, month]).transform("median")
    res["station-month climatology"] = clim
    # spatial: mean of other stations in the same city on that day
    s_ = x.groupby([d.city, d.date]).transform("sum")
    n_ = x.groupby([d.city, d.date]).transform("count")
    res["same-city neighbours"] = (s_ - x.fillna(0)) / (n_ - x.notna().astype(int))
    res["hybrid: neighbour>ffill>clim"] = res["same-city neighbours"].fillna(res["ffill (causal)"]).fillna(clim)
    rows = []
    for name, est in res.items():
        e = (est - truth).abs()[mask]
        cov = est[mask].notna().mean()
        for L in range(1, 8):
            m = mask & (gap_len == L)
            rows.append(dict(method=name, gap=L, mae=float((est - truth).abs()[m].mean())))
        rows.append(dict(method=name, gap="all", mae=float(e.mean()), coverage=float(cov)))
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for k, name in enumerate(res):
        rr = r[(r.method == name) & (r.gap != "all")]
        ax.plot(rr.gap.astype(int), rr.mae, marker="o", ms=4, label=name, color=SERIES[k])
    ax.set_xlabel("gap length (days)")
    ax.set_ylabel("MAE of reconstructed daily PM2.5 (ug/m3)")
    ax.set_title("Imputation reconstruction error vs gap length (2019-2022, masked known values)")
    ax.legend(fontsize=7)
    save(fig, FIG / "imp_01_reconstruction.png")
    return r[r.gap == "all"][["method", "mae", "coverage"]].to_dict("records")


def downstream():
    import xgboost as xgb
    from sklearn.metrics import f1_score, mean_absolute_error
    F, feats, _ = load()
    m = split_masks(F)
    out = []
    for strat in ["native", "median", "ffill", "clim", "ffill_clim"]:
        imp = Imputer(strat, feats).fit(F, m["train"])
        X = imp.transform(F)
        model = xgb.XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.06, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=12, random_state=SEED)
        model.fit(X[m["train"]], F.y_aqi[m["train"]])
        p = model.predict(X[m["val"]])
        y = F.y_aqi[m["val"]]
        out.append(dict(strategy=strat, val_mae=float(mean_absolute_error(y, p)),
                        val_f1_poor=float(f1_score(y > 200, p > 200))))
        print(out[-1], flush=True)
    r = pd.DataFrame(out)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh(r.strategy, r.val_mae, color=SERIES[0], height=0.55)
    for i, v in enumerate(r.val_mae):
        ax.text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=8, color=TEXT2)
    ax.set_xlim(r.val_mae.min() * 0.9, r.val_mae.max() * 1.05)
    ax.set_xlabel("validation MAE (AQI points), XGBoost")
    ax.set_title("Downstream forecast error by imputation strategy")
    save(fig, FIG / "imp_02_downstream.png")
    return out


if __name__ == "__main__":
    res = {"reconstruction": reconstruction()}
    print(json.dumps(res, indent=2))
    res["downstream"] = downstream()
    (REPORTS / "imputation_results.json").write_text(json.dumps(res, indent=2))
