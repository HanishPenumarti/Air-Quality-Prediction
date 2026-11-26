"""Analysis C - Model efficiency: training time, inference time, model size, and the
accuracy-vs-cost trade-off, for every benchmarked model (raw and PCA features).

What is measured (and what is not):
  1. ESTIMATOR fit time of the final model (from the benchmark run): wall-clock AND CPU seconds.
     RF and XGBoost use 12 threads, DT/AdaBoost/GBM are single-threaded in sklearn, so CPU-seconds
     is the fairer "work" comparison and wall time the "waiting" comparison. Tuning time and
     feature building are NOT included.
  2. Inference latency, repeated 30 times (median and IQR), warm and cold (first call after
     loading from disk), for 1 row (one station, the UI case, 1 thread) and for one day's batch
     of ~450 stations; plus an END-TO-END request (imputer transform + predict) for that batch.
  3. Learner-sample curve: fit time and validation MAE when the learner sees 10/25/50/100 % of the
     <= 2020 rows (station x year stratified; the imputer is refitted on the same subset; 2 seeds;
     the 100 % control is re-run in the same session). Timings are sequential (no parallel jobs).
Hardware, thread counts and package versions are written to reports/efficiency_env.json.
Outputs: reports/efficiency.csv, reports/efficiency_scaling.csv, reports/efficiency_env.json, figures eff_*.png
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, MODELS, REPORTS, SEED, TRAIN_END, VAL_END  # noqa: E402
from models import stratified_sample  # noqa: E402
from prep import Imputer, load  # noqa: E402
from runinfo import environment  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
import zoo  # noqa: E402

MODEL_NAMES = zoo.MODEL_NAMES
REPEATS = 30


def timeit(fn, n=REPEATS):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = 1000 * np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 25)), float(np.percentile(ts, 75))


def main():
    b = pd.read_csv(REPORTS / "benchmark.csv")
    reg = b[(b.task == "regression") & b.fit_s.notna()].copy()
    cls = b[(b.task == "direct_cls") & b.fit_s.notna()].copy()

    F, feats, _ = load()
    imp = joblib.load(MODELS / "imputer.joblib")
    t = F.target_date
    te = (t > VAL_END).values
    day = F.date[te].value_counts().idxmax()               # busiest test day ~ one daily batch
    Fday = F[(F.date == day).values]
    Xday = imp.transform(Fday).astype("float32").values
    one = Xday[:1]

    lat = []
    for _, r in reg.iterrows():
        f = MODELS / f"{r.model}_reg_{r.features}.joblib"
        t0 = time.perf_counter()
        m = joblib.load(f)
        m.predict(one)
        cold = 1000 * (time.perf_counter() - t0)
        zoo.single_thread(m)
        row1 = timeit(lambda: m.predict(one))
        m_b = joblib.load(f)
        batch = timeit(lambda: m_b.predict(Xday))
        e2e = timeit(lambda: m_b.predict(imp.transform(Fday).astype("float32").values), n=10)
        lat.append(dict(model=r.model, features=r.features, cold_load_and_1row_ms=cold,
                        row1_ms_median=row1[0], row1_ms_iqr=f"{row1[1]:.2f}-{row1[2]:.2f}",
                        batch_rows=len(Xday), batch_ms_median=batch[0], batch_ms_iqr=f"{batch[1]:.1f}-{batch[2]:.1f}",
                        end_to_end_batch_ms_median=e2e[0]))
        print(lat[-1], flush=True)
    lat = pd.DataFrame(lat)
    reg = reg.merge(lat, on=["model", "features"], how="left")
    eff = pd.concat([reg.assign(kind="regression"), cls.assign(kind="classification")])
    keep = ["model", "features", "kind", "MAE", "F1", "AP", "fit_s", "fit_cpu_s", "threads", "n_rounds",
            "predict_ms_per_1k", "size_mb", "cold_load_and_1row_ms", "row1_ms_median", "row1_ms_iqr",
            "batch_rows", "batch_ms_median", "batch_ms_iqr", "end_to_end_batch_ms_median"]
    eff[[c for c in keep if c in eff]].to_csv(REPORTS / "efficiency.csv", index=False)
    (REPORTS / "efficiency_env.json").write_text(json.dumps(dict(environment(), threads=zoo.THREADS,
                                                                 timing_repeats=REPEATS), indent=1))

    # --- learner-sample curve (raw features, regression), validation 2021 --------------
    fp = json.loads((MODELS / "final_params.json").read_text())
    tr = (t <= TRAIN_END).values
    va = ((t > TRAIN_END) & (t <= VAL_END)).values
    y = F.y_aqi.values.astype(float)
    rows = []
    for seed in [SEED, SEED + 1]:
        rng = np.random.default_rng(seed)
        for frac in [0.1, 0.25, 0.5, 1.0]:
            if frac == 1.0 and seed != SEED:
                continue
            sub = stratified_sample(F, np.flatnonzero(tr), frac, rng)
            msk = np.zeros(len(F), bool)
            msk[sub] = True
            X = Imputer("clim", feats).fit(F, msk).transform(F).astype("float32").values
            for name in MODEL_NAMES:
                q = fp[name]["reg"]["raw"]
                t0, c0 = time.perf_counter(), time.process_time()
                m = zoo.fit_final(name, "reg", "raw", q["params"], X[sub], y[sub], q["n_rounds"])
                fs, fc = time.perf_counter() - t0, time.process_time() - c0
                mae = float(np.mean(np.abs(m.predict(X[va]) - y[va])))
                rows.append(dict(model=name, seed=seed, frac=frac, n_rows=len(sub), fit_s=fs, fit_cpu_s=fc, val_MAE=mae))
                print(rows[-1], flush=True)
    sc = pd.DataFrame(rows)
    sc.to_csv(REPORTS / "efficiency_scaling.csv", index=False)
    plots(reg, sc)


def plots(reg, sc):
    colors = {m: SERIES[i] for i, m in enumerate(MODEL_NAMES)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for feat, mk in [("raw", "o"), ("pca", "s")]:
        s = reg[reg.features == feat]
        for _, r in s.iterrows():
            for a, xcol in zip(axes, ["fit_cpu_s", "predict_ms_per_1k", "size_mb"]):
                if xcol in r and pd.notna(r[xcol]):
                    a.scatter(r[xcol], r.MAE, color=colors[r.model], marker=mk, s=60,
                              edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    for a, lab in zip(axes, ["estimator training CPU time (s, log)", "batch inference per 1000 rows (ms, log)",
                             "model size on disk (MB, log)"]):
        a.set_xscale("log")
        a.set_xlabel(lab)
    axes[0].set_ylabel("test MAE (lower is better)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in colors.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", color=TEXT2, label="raw features"),
                plt.Line2D([], [], marker="s", ls="", color=TEXT2, label="PCA features")]
    axes[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, frameon=False)
    fig.suptitle("Accuracy vs estimator cost (each marker = one model)", fontweight="bold")
    save(fig, FIG / "eff_01_accuracy_vs_cost.png")

    g = sc.groupby(["model", "frac"]).agg(n_rows=("n_rows", "mean"), fit_cpu_s=("fit_cpu_s", "mean"),
                                          mae=("val_MAE", "mean"), lo=("val_MAE", "min"), hi=("val_MAE", "max"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for m in MODEL_NAMES:
        if m not in g.index.get_level_values(0):
            continue
        s = g.loc[m].sort_index()
        axes[0].plot(s.n_rows, s.fit_cpu_s, marker="o", ms=4, color=colors[m], label=m)
        axes[1].plot(s.n_rows, s.mae, marker="o", ms=4, color=colors[m], label=m)
        axes[1].fill_between(s.n_rows, s.lo, s.hi, color=colors[m], alpha=0.15, lw=0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("estimator training CPU time (s)")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("validation MAE (2021); band = 2 seeds")
    for a in axes:
        a.set_xlabel("training rows seen by the learner")
    axes[1].legend(fontsize=7)
    fig.suptitle("Learner-sample curve (imputer refitted on each subset)", fontweight="bold")
    save(fig, FIG / "eff_02_scaling.png")


if __name__ == "__main__":
    main()
