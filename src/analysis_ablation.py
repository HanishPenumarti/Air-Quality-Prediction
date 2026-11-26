"""Analysis A - Feature analysis & ablation studies, for ALL five model families.

Design (stated before running):
  * every model uses its benchmark settings (models/final_params.json: tuned parameters + rounds)
  * fit on all rows <= 2020, evaluate on the 2021 VALIDATION year - the test period is not used,
    so nothing here influences, or is influenced by, the final test numbers
  * these are COLUMN-GROUP ablations: removing a group removes its columns from the model, but
    information from the same source can still reach the model through the causal fill of the
    history (e.g. same-city neighbours fill gaps in a station's own AQI lags)
  * each raw-missingness flag belongs to its source group (isna_city_aqi -> spatial,
    isna_site_WS -> on-site weather, isna_PM25/PM10/aqi_l1 -> missingness)
Experiments
  1. leave-one-group-out (8)            4. PCA with k components vs top-k features (k = 5, 10, 20, 40);
  2. single group only (8)                 ranking by XGBoost gain on <= 2020 only
  3. incremental, ONE group per step    5. grouped permutation importance of the full model (3 repeats)
Outputs: reports/ablation.csv, reports/permutation_importance.csv, reports/feature_ranking_xgb_gain.json,
         figures abl_01..abl_04
Usage  : python src/analysis_ablation.py [--jobs 6] [--frac 1.0] [--models XGBoost RandomForest ...]
         (--frac < 1 trains on a station x year stratified sample of the <= 2020 rows, to save time)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from aqi import is_poor  # noqa: E402
from config import FIG, MODELS, REPORTS, SEED, TRAIN_END, VAL_END  # noqa: E402
from metrics import reg_metrics  # noqa: E402
from models import stratified_sample  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
import zoo  # noqa: E402

GROUP_LABEL = {"lag_aqi": "AQI history", "lag_poll": "pollutant history", "missing": "missingness flags",
               "spatial": "neighbouring stations", "met_site": "on-site weather",
               "fire": "fires (FIRMS)", "calendar": "calendar/festivals", "static": "station context"}
INC_ORDER = ["lag_aqi", "lag_poll", "missing", "spatial", "met_site", "calendar", "static", "fire"]


def fit_score(name, p, n, Xtr, ytr, Xva, yva, k_pca=None, keep=False):
    est = zoo.make(name, "reg", p, n)
    m = make_pipeline(StandardScaler(), PCA(n_components=k_pca, random_state=SEED), est) if k_pca else est
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    pv = m.predict(Xva)
    r = reg_metrics(yva, pv)
    return dict(MAE=r["MAE"], RMSE=r["RMSE"], R2=r["R2"], MAE_top5pct=r["MAE_top5pct"], MAE_severe=r["MAE_severe"],
                F1_poor=f1_score(is_poor(yva), is_poor(pv)), fit_s=fit_s), (m if keep else None)


def main(jobs=6, frac=1.0, model_names=None):
    model_names = model_names or zoo.MODEL_NAMES
    F, feats, groups = load()
    fp = json.loads((MODELS / "final_params.json").read_text())
    t = F.target_date
    tr = (t <= TRAIN_END).values
    va = ((t > TRAIN_END) & (t <= VAL_END)).values
    rng = np.random.default_rng(SEED)
    tr_idx = stratified_sample(F, np.flatnonzero(tr), frac, rng)
    va_idx = np.flatnonzero(va)
    y = F.y_aqi.values.astype(float)
    Xdf = Imputer("clim", feats).fit(F, tr).transform(F)
    cols = list(Xdf.columns)
    X = Xdf.values.astype("float32")
    gidx = {g: [cols.index(c) for c in cs if c in cols] for g, cs in groups.items()}
    full = sorted(i for g in gidx for i in gidx[g])

    # ---- experiment list (column index sets) -------------------------------------------
    exps = [("ALL features", full, None)]
    for g in gidx:
        exps.append((f"- without {GROUP_LABEL[g]}", sorted(i for gg in gidx if gg != g for i in gidx[gg]), None))
    for g in gidx:
        exps.append((f"only {GROUP_LABEL[g]}", sorted(gidx[g]), None))
    acc = []
    for g in INC_ORDER[:-1]:          # the last step equals ALL features
        acc = acc + [g]
        exps.append(("inc: " + " + ".join(GROUP_LABEL[x] for x in acc), sorted(i for x in acc for i in gidx[x]), None))
    ranker = zoo.make("XGBoost", "reg", fp["XGBoost"]["reg"]["raw"]["params"], fp["XGBoost"]["reg"]["raw"]["n_rounds"])
    ranker.fit(X[tr_idx][:, full], y[tr_idx])
    gain = ranker.get_booster().get_score(importance_type="gain")
    order = [full[j] for j in np.argsort(-np.array([gain.get(f"f{j}", 0.0) for j in range(len(full))]))]
    (REPORTS / "feature_ranking_xgb_gain.json").write_text(json.dumps([cols[i] for i in order], indent=1))
    for k in [5, 10, 20, 40]:
        exps.append((f"select top-{k} (XGB gain)", sorted(order[:k]), None))
        exps.append((f"PCA {k} components", full, k))

    # ---- run: single-threaded families in parallel, multi-threaded ones sequentially ----
    jobs_list = [(e, nm) for e in exps for nm in model_names]
    Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]

    def call(e, nm):
        lab, cs, kp = e
        q = fp[nm]["reg"]["raw"]
        return delayed(fit_score)(nm, q["params"], q["n_rounds"], Xtr[:, cs], ytr, Xva[:, cs], yva, kp,
                                  lab == "ALL features")
    single = [(e, nm) for e, nm in jobs_list if zoo.THREADS[nm] == 1]
    multi = [(e, nm) for e, nm in jobs_list if zoo.THREADS[nm] > 1]
    res = Parallel(n_jobs=jobs, verbose=5)(call(e, nm) for e, nm in single) if single else []
    for e, nm in multi:
        f, a, kw = call(e, nm)
        res.append(f(*a, **kw))
    out, full_models = [], {}
    for (e, nm), (r, mdl) in zip(single + multi, res):
        lab, cs, kp = e
        out.append(dict(experiment=lab, model=nm, n_features=kp if kp else len(cs), **r))
        if lab == "ALL features":
            full_models[nm] = mdl
    r = pd.DataFrame(out)
    r["split"] = f"fit <= {TRAIN_END}, evaluated on 2021 validation"
    r["train_fraction"] = frac
    r.to_csv(REPORTS / "ablation.csv", index=False)

    # ---- grouped permutation importance (full models, validation year) ---------------
    prng = np.random.default_rng(SEED)
    rows = []
    for nm, mdl in full_models.items():
        base = float(np.mean(np.abs(mdl.predict(Xva[:, full]) - yva)))
        for g, gi in gidx.items():
            pos = [full.index(i) for i in gi]
            for rep in range(3):
                Xp = Xva[:, full].copy()
                Xp[:, pos] = Xp[prng.permutation(len(Xp))][:, pos]
                rows.append(dict(model=nm, group=GROUP_LABEL[g], repeat=rep,
                                 dMAE=float(np.mean(np.abs(mdl.predict(Xp) - yva))) - base))
    pi = pd.DataFrame(rows).groupby(["model", "group"]).dMAE.agg(["mean", "std"]).reset_index()
    pi.to_csv(REPORTS / "permutation_importance.csv", index=False)
    plots(r, pi)


def plots(r, pi=None):
    names = [m for m in zoo.MODEL_NAMES if m in set(r.model)]
    colors = {m: SERIES[zoo.MODEL_NAMES.index(m)] for m in names}
    base = r[r.experiment == "ALL features"].set_index("model").MAE
    w = 0.8 / len(names)

    def bars(df, fname, title, xlabel):
        piv = df.pivot(index="experiment", columns="model", values="v")[names]
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(piv) + 1.2))
        yy = np.arange(len(piv))
        for k, m in enumerate(names):
            ax.barh(yy + (k - (len(names) - 1) / 2) * w, piv[m], height=w * 0.9, color=colors[m], label=m)
        ax.axvline(0, color=TEXT2, lw=1)
        ax.set_yticks(yy, piv.index)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
        save(fig, FIG / fname)

    lo = r[r.experiment.str.startswith("- without")].copy()
    lo["v"] = lo.MAE - lo.model.map(base)
    lo["experiment"] = lo.experiment.str.replace("- without ", "")
    bars(lo, "abl_01_leave_one_group_out.png", "Leave-one-group-out (validation 2021; positive = group helps)",
         "increase in MAE when the column group is removed (AQI points)")

    sel = r[r.experiment.str.startswith(("select top", "PCA "))].copy()
    sel["k"] = sel.experiment.str.extract(r"(\d+)").astype(int)
    sel["method"] = np.where(sel.experiment.str.startswith("PCA"), "PCA (k components)", "top-k features")
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3.3), sharey=True)
    for a, m in zip(np.atleast_1d(axes), names):
        for k, meth in enumerate(["top-k features", "PCA (k components)"]):
            s = sel[(sel.model == m) & (sel.method == meth)].sort_values("k")
            a.plot(s.k, s.MAE, marker="o", ms=5, color=SERIES[k], label=meth)
        a.axhline(base[m], color=TEXT2, ls=":", lw=1)
        a.set_title(m)
        a.set_xlabel("k")
    np.atleast_1d(axes)[0].set_ylabel("validation MAE")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    save(fig, FIG / "abl_02_pca_vs_selection.png")

    inc = r[r.experiment.str.startswith("inc:") | (r.experiment == "ALL features")]
    order = list(dict.fromkeys(inc[inc.experiment.str.startswith("inc:")].experiment)) + ["ALL features"]
    piv = inc.pivot(index="experiment", columns="model", values="MAE").reindex(order)[names]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for m in names:
        ax.plot(range(len(piv)), piv[m], marker="o", ms=5, color=colors[m], label=m)
    labels = ["AQI history"] + ["+ " + s.split(" + ")[-1] for s in order[1:-1]] + [f"+ {GROUP_LABEL[INC_ORDER[-1]]} (= all)"]
    ax.set_xticks(range(len(piv)), labels, rotation=20, ha="right")
    ax.set_ylabel("validation MAE")
    ax.set_title("Adding one feature group at a time")
    ax.legend(fontsize=7)
    save(fig, FIG / "abl_03_incremental.png")

    if pi is not None and len(pi):
        p2 = pi.rename(columns={"group": "experiment", "mean": "v"})
        bars(p2, "abl_04_permutation.png", "Grouped permutation importance (validation 2021)",
             "increase in MAE when the group's columns are shuffled together")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--models", nargs="*", choices=zoo.MODEL_NAMES)
    a = ap.parse_args()
    main(a.jobs, a.frac, a.models)
