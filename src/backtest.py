"""Rolling-origin backtest: is the benchmark ranking stable across years?

For each origin year Y in config.BACKTEST_YEARS: fit on target dates <= 31 Dec (Y-1), test on year Y
(Y = 2022 runs to the end of the data, Mar 2023 - identical to the main benchmark, whose test
predictions are reused). The imputer/climatology is refitted for every origin. All five models
(raw features, regression and direct classification) use the benchmark's tuned settings and
round counts (models/final_params.json). Those settings were chosen on 2020-2021 validation,
so the 2019 and 2020 origins are slightly optimistic for the learned models - stated, not hidden.

Outputs: reports/backtest.csv (+ 95 % block-bootstrap intervals), figure bt_01_backtest.png
Usage  : python src/backtest.py [--jobs 4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_YEARS, FIG, MODELS, PROC, REPORTS, SEED  # noqa: E402
from metrics import block_bootstrap, cls_metrics_aqi, reg_metrics  # noqa: E402
from models import persistence  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
import zoo  # noqa: E402


def fit_predict(name, task, p, n, Xtr, ytr, Xte):
    m = zoo.fit_final(name, task, "raw", p, Xtr, ytr, n)
    return zoo.predict(m, task, Xte)


def main(jobs=4):
    F, feats, _ = load()
    fp = json.loads((MODELS / "final_params.json").read_text())
    t = F.target_date
    y = F.y_aqi.values.astype(float)
    yc = F.y_poor.values.astype(int)
    bench = pd.read_parquet(PROC / "preds_test.parquet")
    rows, ci_rows = [], []
    for Y in BACKTEST_YEARS:
        tr = (t <= f"{Y - 1}-12-31").values
        te = ((t > f"{Y - 1}-12-31") & (t <= (f"{Y}-12-31" if Y < 2022 else "2099-12-31"))).values
        imp = Imputer("clim", feats).fit(F, tr)
        clim = imp.target_climatology(F).values.astype(float)
        preds = {"Persistence": persistence(F, clim)[te], "Seasonal climatology": clim[te]}
        scores = {}
        if Y == max(BACKTEST_YEARS):
            b = bench.set_index(["station", "date"]).reindex(pd.MultiIndex.from_arrays([F.station[te], F.date[te]]))
            for name in zoo.MODEL_NAMES:
                preds[name] = b[f"{name}|reg|raw"].values
                scores[name] = b[f"{name}|cls|raw"].values
        else:
            X = imp.transform(F).astype("float32").values
            jobs_list = [(name, task) for name in zoo.MODEL_NAMES for task in ["reg", "cls"]]
            single = [(nm, tk) for nm, tk in jobs_list if zoo.THREADS[nm] == 1]
            multi = [(nm, tk) for nm, tk in jobs_list if zoo.THREADS[nm] > 1]
            yy = {"reg": y, "cls": yc}

            def args(nm, tk):           # arrays passed as arguments -> memory-mapped by joblib
                q = fp[nm][tk]["raw"]
                return nm, tk, q["params"], q["n_rounds"], X[tr], yy[tk][tr], X[te]
            out = Parallel(n_jobs=jobs)(delayed(fit_predict)(*args(nm, tk)) for nm, tk in single) if jobs > 1 \
                else [fit_predict(*args(nm, tk)) for nm, tk in single]
            out += [fit_predict(*args(nm, tk)) for nm, tk in multi]
            for (nm, tk), pr in zip(single + multi, out):
                (preds if tk == "reg" else scores)[nm] = pr
        yt, yct = y[te], yc[te]
        p_mae = float(np.mean(np.abs(preds["Persistence"] - yt)))
        for name, pr in preds.items():
            r = reg_metrics(yt, pr, p_mae)
            c = cls_metrics_aqi(yct, pr)
            rows.append(dict(origin=Y, model=name, n_test=int(te.sum()), n_train=int(tr.sum()),
                             MAE=r["MAE"], RMSE=r["RMSE"], MAE_top5pct=r["MAE_top5pct"], MAE_severe=r["MAE_severe"],
                             skill_vs_persistence_pct=r["skill_vs_persistence_pct"], derived_F1=c["F1"],
                             direct_AP=float(average_precision_score(yct, scores[name])) if name in scores else np.nan))
        wk = pd.to_datetime(t[te]).dt.strftime("%G-%V").values
        per = preds["Persistence"]
        fns = {}
        for name, pr in preds.items():
            fns[f"{name}|MAE"] = lambda i, pr=pr: np.mean(np.abs(pr[i] - yt[i]))
            fns[f"{name}|dMAE"] = lambda i, pr=pr: np.mean(np.abs(pr[i] - yt[i]) - np.abs(per[i] - yt[i]))
        for k, (lo, hi) in block_bootstrap(wk, fns, B=500, seed=SEED).items():
            nm, met = k.rsplit("|", 1)
            ci_rows.append(dict(origin=Y, model=nm, metric=met, lo95=lo, hi95=hi))
        print(pd.DataFrame(rows)[lambda d: d.origin == Y][["model", "MAE", "skill_vs_persistence_pct",
                                                           "derived_F1", "direct_AP"]].round(3).to_string(), flush=True)
    r = pd.DataFrame(rows)
    ci = pd.DataFrame(ci_rows).pivot_table(index=["origin", "model"], columns="metric", values=["lo95", "hi95"])
    ci.columns = [f"{m}_{b}" for b, m in ci.columns]
    r = r.merge(ci.reset_index(), on=["origin", "model"], how="left")
    r.to_csv(REPORTS / "backtest.csv", index=False)
    plot(r)


def plot(r):
    names = ["Persistence", "Seasonal climatology"] + zoo.MODEL_NAMES
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for k, nm in enumerate(names):
        s = r[r.model == nm].sort_values("origin")
        ls = "--" if nm in ("Persistence", "Seasonal climatology") else "-"
        col = "#8a8a8a" if nm == "Persistence" else ("#c3c2b7" if ls == "--" else SERIES[k - 2])
        axes[0].plot(s.origin, s.MAE, marker="o", ms=4, ls=ls, color=col, label=nm)
        if nm in zoo.MODEL_NAMES:
            axes[1].errorbar(s.origin + (k - 4) * 0.06, s.skill_vs_persistence_pct, fmt="o-", ms=4, color=col,
                             label=nm)
    axes[0].set_title("Test MAE per origin year")
    axes[0].set_ylabel("MAE (AQI points)")
    axes[1].axhline(0, color=TEXT2, lw=1)
    axes[1].set_title("Skill vs persistence (% MAE reduction)")
    for a in axes:
        a.set_xticks(sorted(r.origin.unique()), [f"{o}" if o < 2022 else "2022-23" for o in sorted(r.origin.unique())])
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, FIG / "bt_01_backtest.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    main(ap.parse_args().jobs)
