"""Analysis A - Feature analysis & ablation studies.

For DecisionTree, RandomForest and XGBoost (same chronological split as the benchmark:
fit <= 2021, evaluate 2022-01..2023-03):
  1. leave-one-group-out  : drop each feature group in turn
  2. single-group / incremental views: history only, + spatial, + weather ...
  3. dimensionality reduction: PCA (k components) vs. feature selection (top-k by XGBoost gain)
Outputs: reports/ablation.csv, figures abl_*.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, POOR_THRESHOLD, REPORTS, SEED, VAL_END  # noqa: E402
from metrics import reg_metrics  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

GROUP_LABEL = {"lag_aqi": "AQI history", "lag_poll": "pollutant history", "missing": "missingness flags",
               "spatial": "neighbouring stations", "met_site": "on-site weather",
               "fire": "fires (FIRMS)", "calendar": "calendar/festivals", "static": "station context"}


def models():
    """Fixed, benchmark-like settings (RF slightly smaller to keep 3x~30 runs tractable)."""
    return {
        "DecisionTree": lambda: DecisionTreeRegressor(max_depth=12, min_samples_leaf=200, random_state=SEED),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=80, max_depth=20, min_samples_leaf=10,
                                                      max_features=0.33, n_jobs=12, random_state=SEED),
        "XGBoost": lambda: XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.7, min_child_weight=10, tree_method="hist",
                                        n_jobs=12, random_state=SEED),
    }


def run(F, X, cols, label, tr, te, y, out, pipe=None):
    for mname, mk in models().items():
        m = mk() if pipe is None else make_pipeline(*pipe(), mk())
        t0 = time.perf_counter()
        m.fit(X[tr][:, cols] if cols is not None else X[tr], y[tr])
        fit_s = time.perf_counter() - t0
        p = m.predict(X[te][:, cols] if cols is not None else X[te])
        r = reg_metrics(y[te], p)
        out.append(dict(experiment=label, model=mname, n_features=len(cols) if cols is not None else X.shape[1],
                        MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"], MAE_top5pct=r["MAE_top5pct"],
                        F1_poor=f1_score(y[te] > POOR_THRESHOLD, p > POOR_THRESHOLD), fit_s=fit_s))
        print(f"{label:40s} {mname:12s} MAE {r['MAE']:.2f}", flush=True)


def main():
    F, _, groups = load()
    t = F.target_date
    tr, te = (t <= VAL_END).values, (t > VAL_END).values
    y = F.y_aqi.values
    all_feats = [c for g, cs in groups.items() for c in cs]
    Xdf = Imputer("ffill_clim", all_feats).fit(F, tr).transform(F)
    Xdf = Xdf[[c for c in Xdf.columns if not c.startswith("isna_")] + [c for c in Xdf.columns if c.startswith("isna_")]]
    X = Xdf.values.astype("float32")
    name2i = {c: i for i, c in enumerate(Xdf.columns)}
    gidx = {g: [name2i[c] for c in cs] for g, cs in groups.items()}
    gidx["missing"] = gidx["missing"] + [name2i[c] for c in Xdf.columns if c.startswith("isna_")]
    strict = list(groups)
    full = sorted(i for g in strict for i in gidx[g])
    out = []

    run(F, X, full, "ALL features", tr, te, y, out)
    for g in strict:
        cols = sorted(i for gg in strict if gg != g for i in gidx[gg])
        run(F, X, cols, f"- without {GROUP_LABEL[g]}", tr, te, y, out)
    # incremental views
    inc = [("AQI history only", ["lag_aqi"]),
           ("+ pollutant history", ["lag_aqi", "lag_poll", "missing"]),
           ("+ neighbouring stations", ["lag_aqi", "lag_poll", "missing", "spatial"]),
           ("+ calendar & context", ["lag_aqi", "lag_poll", "missing", "spatial", "calendar", "static"]),
           ("+ fires", ["lag_aqi", "lag_poll", "missing", "spatial", "calendar", "static", "fire"])]
    for lab, gs in inc:
        run(F, X, sorted(i for g in gs if g in gidx for i in gidx[g]), "inc: " + lab, tr, te, y, out)

    # PCA vs feature selection at equal k
    ranker = models()["XGBoost"]().fit(X[tr][:, full], y[tr])
    gain = ranker.get_booster().get_score(importance_type="gain")
    imp = np.array([gain.get(f"f{j}", 0.0) for j in range(len(full))])
    order = [full[j] for j in np.argsort(-imp)]
    for k in [5, 10, 20, 40]:
        run(F, X, sorted(order[:k]), f"select top-{k} (XGB gain)", tr, te, y, out)
        run(F, X, full, f"PCA {k} components", tr, te, y, out,
            pipe=lambda k=k: [StandardScaler(), PCA(n_components=k, random_state=SEED)])

    r = pd.DataFrame(out)
    r.to_csv(REPORTS / "ablation.csv", index=False)
    (REPORTS / "feature_ranking_xgb_gain.json").write_text(json.dumps(
        [list(Xdf.columns)[i] for i in order], indent=1))
    plots(r)


def plots(r):
    base = r[r.experiment == "ALL features"].set_index("model").MAE
    # leave-one-group-out
    lo = r[r.experiment.str.startswith("- without")].copy()
    lo["dMAE"] = lo.MAE - lo.model.map(base)
    piv = lo.pivot(index="experiment", columns="model", values="dMAE")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(piv) + 1))
    yy = np.arange(len(piv))
    for k, m in enumerate(piv.columns):
        ax.barh(yy + (k - 1) * 0.27, piv[m], height=0.25, color=SERIES[k], label=m)
    ax.axvline(0, color=TEXT2, lw=1)
    ax.set_yticks(yy, [s.replace("- without ", "") for s in piv.index])
    ax.set_xlabel("increase in test MAE when the group is removed (AQI points)")
    ax.set_title("Leave-one-group-out ablation (positive = group helps)")
    ax.legend()
    save(fig, FIG / "abl_01_leave_one_group_out.png")

    # PCA vs selection
    sel = r[r.experiment.str.startswith(("select top", "PCA "))].copy()
    sel["k"] = sel.experiment.str.extract(r"(\d+)").astype(int)
    sel["method"] = np.where(sel.experiment.str.startswith("PCA"), "PCA", "top-k selection")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for a, m in zip(axes, ["DecisionTree", "RandomForest", "XGBoost"]):
        for k, meth in enumerate(["top-k selection", "PCA"]):
            s = sel[(sel.model == m) & (sel.method == meth)].sort_values("k")
            a.plot(s.k, s.MAE, marker="o", ms=5, color=SERIES[k], label=meth)
        a.axhline(base[m], color=TEXT2, ls=":", lw=1)
        a.text(40, base[m], " all features", fontsize=7, color=TEXT2, va="bottom", ha="right")
        a.set_title(m)
        a.set_xlabel("number of features / components")
    axes[0].set_ylabel("test MAE")
    axes[0].legend()
    save(fig, FIG / "abl_02_pca_vs_selection.png")

    # incremental
    inc = r[r.experiment.str.startswith(("inc:", "ALL"))].copy()
    order = ["inc: AQI history only", "inc: + pollutant history", "inc: + neighbouring stations",
             "inc: + calendar & context", "inc: + fires", "ALL features"]
    piv = inc.pivot(index="experiment", columns="model", values="MAE").reindex(order).dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for k, m in enumerate(piv.columns):
        ax.plot(range(len(piv)), piv[m], marker="o", ms=5, color=SERIES[k], label=m)
    ax.set_xticks(range(len(piv)), [s.replace("inc: ", "").replace("ALL features", "+ on-site weather (= all)")
                                    for s in piv.index], rotation=20, ha="right")
    ax.set_ylabel("test MAE")
    ax.set_title("Adding feature groups one at a time")
    ax.legend()
    save(fig, FIG / "abl_03_incremental.png")


if __name__ == "__main__":
    main()
