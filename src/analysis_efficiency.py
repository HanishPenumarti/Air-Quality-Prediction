"""Analysis C - Model efficiency: training time, inference time, model size, and the
accuracy-vs-cost trade-off, for every benchmarked model (raw and PCA features).

  1. from the benchmark run: fit time, batch inference time, serialized size (joblib, compressed)
  2. scaling experiment: fit time & test MAE when trained on 10/25/50/100% of the training rows
     (random rows, same for all models; 100% = benchmark values)
  3. single-forecast latency (1 row, as in the UI), median of 50 calls
Outputs: reports/efficiency.csv, reports/efficiency_scaling.csv, figures eff_*.png
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, MODELS, REPORTS, SEED, VAL_END  # noqa: E402
from prep import load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402

MODEL_NAMES = ["DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]


def main():
    b = pd.read_csv(REPORTS / "benchmark.csv")
    reg = b[(b.task == "regression") & b.fit_s.notna()].copy()
    cls = b[(b.task == "direct_cls") & b.fit_s.notna()].copy()

    # --- single-row latency + reload check
    F, feats, _ = load()
    imp = joblib.load(MODELS / "imputer.joblib")
    X = imp.transform(F).astype("float32").values
    t = F.target_date
    tr, te = (t <= VAL_END).values, (t > VAL_END).values
    y = F.y_aqi.values
    one = X[te][:1]
    lat = []
    for _, r in reg.iterrows():
        m = joblib.load(MODELS / f"{r.model}_reg_{r.features}.joblib")
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            m.predict(one)
            ts.append(time.perf_counter() - t0)
        lat.append(1000 * np.median(ts))
    reg["latency_1row_ms"] = lat
    eff = pd.concat([reg.assign(kind="regression"), cls.assign(kind="classification")])
    eff[["model", "features", "kind", "MAE", "F1", "PR_AUC", "fit_s", "predict_ms_per_1k", "size_mb",
         "latency_1row_ms"]].to_csv(REPORTS / "efficiency.csv", index=False)

    # --- scaling experiment (raw features, regression)
    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(tr)
    rows = []
    for frac in [0.1, 0.25, 0.5]:
        sub = rng.choice(idx, int(frac * len(idx)), replace=False)
        for name in MODEL_NAMES:
            m = clone(joblib.load(MODELS / f"{name}_reg_raw.joblib"))
            t0 = time.perf_counter()
            m.fit(X[sub], y[sub])
            fs = time.perf_counter() - t0
            mae = float(np.mean(np.abs(m.predict(X[te]) - y[te])))
            rows.append(dict(model=name, frac=frac, n_rows=len(sub), fit_s=fs, MAE=mae))
            print(rows[-1], flush=True)
    for _, r in reg[reg.features == "raw"].iterrows():
        rows.append(dict(model=r.model, frac=1.0, n_rows=int(tr.sum()), fit_s=r.fit_s, MAE=r.MAE))
    sc = pd.DataFrame(rows)
    sc.to_csv(REPORTS / "efficiency_scaling.csv", index=False)
    plots(reg, sc)


def plots(reg, sc):
    colors = {m: SERIES[i] for i, m in enumerate(MODEL_NAMES)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for feat, mk in [("raw", "o"), ("pca", "s")]:
        s = reg[reg.features == feat]
        for _, r in s.iterrows():
            for a, xcol in zip(axes, ["fit_s", "predict_ms_per_1k", "size_mb"]):
                a.scatter(r[xcol], r.MAE, color=colors[r.model], marker=mk, s=60,
                          edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    for a, lab in zip(axes, ["training time (s, log)", "inference time per 1000 rows (ms, log)",
                             "model size on disk (MB, log)"]):
        a.set_xscale("log")
        a.set_xlabel(lab)
    axes[0].set_ylabel("test MAE (lower is better)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in colors.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=TEXT2, label="raw features"),
                plt.Line2D([], [], marker="s", ls="", color=TEXT2, label="PCA features")]
    axes[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, frameon=False)
    fig.suptitle("Accuracy vs computational cost (each marker = one model)", fontweight="bold")
    save(fig, FIG / "eff_01_accuracy_vs_cost.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for m in MODEL_NAMES:
        s = sc[sc.model == m].sort_values("n_rows")
        axes[0].plot(s.n_rows, s.fit_s, marker="o", ms=4, color=colors[m], label=m)
        axes[1].plot(s.n_rows, s.MAE, marker="o", ms=4, color=colors[m], label=m)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("training time (s)")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("test MAE")
    for a in axes:
        a.set_xlabel("training rows")
    axes[1].legend()
    fig.suptitle("Scaling with training-set size", fontweight="bold")
    save(fig, FIG / "eff_02_scaling.png")


if __name__ == "__main__":
    main()
