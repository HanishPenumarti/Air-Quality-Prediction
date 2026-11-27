"""Paired model-vs-model differences on the test period, with 95 % block-bootstrap intervals.

benchmark_ci.csv gives each model's interval and its difference from persistence. Two models'
marginal intervals overlapping does not show that they are equal (their errors are strongly
correlated), so the differences that matter for the ranking are bootstrapped directly here, with
the same blocks (ISO weeks of the target date) and seed as models.py.

    regression      : MAE(a) - MAE(b) for neighbouring models in the ranking, and raw - PCA per model
    classification  : AP and F1 of the XGBoost direct classifier minus each other direct classifier
    direct vs derived: F1 of each booster's direct classifier minus its derived rule (AQI > 200 and tuned)

Negative dMAE / positive dAP, dF1 favour the first named prediction.
Output: reports/benchmark_pairwise.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, PROC, REPORTS, SEED  # noqa: E402
from metrics import block_bootstrap  # noqa: E402

RANKED = ["XGBoost", "GradBoost", "RandomForest", "AdaBoost", "DecisionTree"]


def main(B=1000):
    p = pd.read_parquet(PROC / "preds_test.parquet")
    thr = json.loads((MODELS / "thresholds.json").read_text())
    wk = pd.to_datetime(p.target_date).dt.strftime("%G-%V").values
    y, yc = p.y.values, p.y_poor.values
    ae = {c: np.abs(p[c].values - y) for c in p.columns if "|reg" in c}

    fns, rows = {}, []
    # order the regressors by their 2021 validation score (never by test results)
    val = json.loads((MODELS / "production.json").read_text())["validation_scores"]
    reg = sorted([m for m in RANKED if f"{m}|reg|raw" in p], key=lambda m: val[f"{m}|reg|raw"])
    for a, b in zip(reg, reg[1:]):
        fns[("dMAE", f"{a}|reg|raw", f"{b}|reg|raw")] = None
    for m in reg:
        if f"{m}|reg|pca" in p:
            fns[("dMAE", f"{m}|reg|raw", f"{m}|reg|pca")] = None
    cls = [m for m in RANKED if f"{m}|cls|raw" in p]
    best = min(cls, key=lambda m: val[f"{m}|cls|raw"])
    for m in cls:
        if m != best:
            fns[("dAP", f"{best}|cls|raw", f"{m}|cls|raw")] = None
            fns[("dF1", f"{best}|cls|raw", f"{m}|cls|raw")] = None
    for m in cls:
        if f"{m}|reg|raw" in p:
            fns[("dF1", f"{m}|cls|raw", f"{m}|reg|raw >200")] = None
            fns[("dF1", f"{m}|cls|raw", f"{m}|reg|raw tuned")] = None

    def decision(name):
        """0/1 Poor-or-worse decisions of a prediction column under its rule."""
        if name.endswith(" >200"):
            return p[name[:-5]].values > 200
        if name.endswith(" tuned"):
            return p[name[:-6]].values >= thr[name[:-6]]["derived_aqi"]
        return p[name].values >= thr[name]["direct"]

    stat = {}
    for key in fns:
        kind, a, b = key
        if kind == "dMAE":
            stat[key] = lambda i, da=ae[a], db=ae[b]: float(np.mean(da[i] - db[i]))
        elif kind == "dAP":
            stat[key] = lambda i, sa=p[a].values, sb=p[b].values: (average_precision_score(yc[i], sa[i])
                                                                    - average_precision_score(yc[i], sb[i]))
        else:
            stat[key] = lambda i, da=decision(a), db=decision(b): f1_score(yc[i], da[i]) - f1_score(yc[i], db[i])
    names = {f"{k[0]}|{k[1]}|{k[2]}": f for k, f in stat.items()}
    ci = block_bootstrap(wk, names, B=B, seed=SEED)
    full = np.arange(len(y))
    for k, f in stat.items():
        lo, hi = ci[f"{k[0]}|{k[1]}|{k[2]}"]
        rows.append(dict(metric=k[0], first=k[1], second=k[2], diff=f(full), lo95=lo, hi95=hi,
                         excludes_0=bool(lo > 0 or hi < 0)))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "benchmark_pairwise.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
