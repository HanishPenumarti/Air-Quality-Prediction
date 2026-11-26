"""Analysis E - Error analysis: where and why do the models fail?

Test period 2022-01 .. 2023-03 (pre-specified evaluation; nothing here selects a model).
All regression models (raw features) + persistence, and the direct classifiers.

Regression slices
  * season/month, weekday, region, station (pooled AND station-/city-averaged scores)
  * known stations (present in training) vs stations NEW in the test period
  * TRUE category and PREDICTED category and TODAY's category: conditioning on the true value alone
    makes any mean-reverting model look bad at the top, so all three views are shown
  * size of the day-to-day change, only where AQI(D) and AQI(D+1) are both MEASURED
    (days whose AQI(D) was missing are reported separately)
  * Diwali window, data gaps
Classification ('Poor or worse' early warning)
  * confusion counts, recall/precision/false alerts by season, region and true category
  * recall on Severe days, ONSET detection (AQI(D) <= 200 measured -> AQI(D+1) > 200)
  * calibration (reliability) of the classifier scores + Brier score
Episodes: chosen by a stated rule - (a) the station with the most Severe test days, (b) the
station whose XGBoost MAE is the median among IGP stations with >= 200 test days (representative).
Note: the target is a DAILY AQI, so hour-of-day is not applicable; weekday is used instead.
Outputs: reports/errors_*.csv, figures err_01..err_07, err_09_calibration.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aqi import category_index, is_poor, is_severe  # noqa: E402
from config import AQI_CATEGORIES, FIG, MODELS, PROC, REPORTS, VAL_END, region_of  # noqa: E402
from viz import AQI_COLORS, SERIES, TEXT2, plt, save  # noqa: E402

REG_MODELS = ["Persistence", "DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]
CLS_MODELS = ["DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]
CAT = np.array(AQI_CATEGORIES + ["Unknown"])


def season_of(m):
    return np.select([np.isin(m, [11, 12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8, 9])],
                     ["Winter (Nov-Feb)", "Pre-monsoon (Mar-May)", "Monsoon (Jun-Sep)"], "Post-monsoon (Oct)")


def load_preds():
    p = pd.read_parquet(PROC / "preds_test.parquet")
    F = pd.read_parquet(PROC / "features.parquet",
                        columns=["station", "date", "aqi_l0", "miss_aqi_l0", "days_from_diwali", "lat", "lon"])
    p = p.merge(F, on=["station", "date"], how="left", validate="one_to_one")
    ren = {"Persistence|reg": "Persistence"}
    ren.update({f"{m}|reg|raw": m for m in REG_MODELS[1:]})
    ren.update({f"{m}|cls|raw": f"{m}_score" for m in CLS_MODELS})
    p = p.rename(columns=ren)
    p["region"] = p.state.map(region_of)
    p["month"] = pd.to_datetime(p.target_date).dt.month
    p["season"] = season_of(p.month.values)
    p["true_cat"] = pd.Categorical(CAT[category_index(p.y)], AQI_CATEGORIES, ordered=True)
    p["today"] = p.aqi_l0.where(p.miss_aqi_l0 == 0)                    # MEASURED AQI(D) only
    p["today_cat"] = pd.Categorical(np.where(p.today.notna(), CAT[category_index(p.today)], "AQI(D) missing"),
                                    AQI_CATEGORIES + ["AQI(D) missing"], ordered=True)
    p["change"] = (p.y - p.today).abs()                                 # NaN where AQI(D) not measured
    Ff = pd.read_parquet(PROC / "features.parquet", columns=["station", "target_date"])
    seen = set(Ff[Ff.target_date <= VAL_END].station)
    p["station_type"] = np.where(p.station.isin(seen), "known station", "new station (no training rows)")
    for m in REG_MODELS:
        p[f"err_{m}"] = p[m] - p.y
    return p


def mae_by(p, key, models=REG_MODELS):
    g = p.groupby(key, observed=True)
    out = pd.DataFrame({m: g[f"err_{m}"].apply(lambda e: e.abs().mean()) for m in models})
    out["n"] = g.size()
    return out


def bias_by(p, key, models=REG_MODELS):
    g = p.groupby(key, observed=True)
    out = pd.DataFrame({m: g[f"err_{m}"].mean() for m in models})
    out["n"] = g.size()
    return out


def lines(ax, df, title, xlabel, ylabel="MAE (AQI points)"):
    if df.empty:
        ax.set_title(title + " (no data)")
        return
    for k, m in enumerate(REG_MODELS):
        ls = "--" if m == "Persistence" else "-"
        col = "#8a8a8a" if m == "Persistence" else SERIES[k - 1]
        ax.plot(range(len(df)), df[m], marker="o", ms=4, ls=ls, color=col, label=m)
    ax.set_xticks(range(len(df)), [str(i) for i in df.index], rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def thresholds():
    f = MODELS / "thresholds.json"
    th = json.loads(f.read_text()) if f.exists() else {}
    return {m: th.get(f"{m}|cls|raw", {}).get("direct", 0.5) for m in CLS_MODELS}


def classification_slices(p, th):
    rows = []
    for m in CLS_MODELS:
        if f"{m}_score" not in p:
            continue
        alert = p[f"{m}_score"].values >= th[m]
        derived = is_poor(p[m].values)
        for key in ["season", "region", "true_cat", "station_type"]:
            for grp, idx in p.groupby(key, observed=True).indices.items():
                yy, aa = p.y_poor.values[idx].astype(bool), alert[idx]
                rows.append(dict(model=m, slice=key, group=str(grp), n=len(idx), positives=int(yy.sum()),
                                 TP=int((yy & aa).sum()), FN=int((yy & ~aa).sum()), FP=int((~yy & aa).sum()),
                                 TN=int((~yy & ~aa).sum())))
        sev = is_severe(p.y.values)
        today = p.today.values
        onset = np.isfinite(today) & ~is_poor(today) & is_poor(p.y.values)
        clean = np.isfinite(today) & ~is_poor(today) & ~is_poor(p.y.values)
        for lab, a in [("direct classifier", alert), ("derived (forecast AQI > 200)", derived)]:
            rows.append(dict(model=m, slice="warning", group=lab, n=len(p), positives=int(p.y_poor.sum()),
                             recall_poor=float(a[p.y_poor.values == 1].mean()),
                             recall_on_severe_days=float(a[sev].mean()) if sev.any() else np.nan,
                             onsets=int(onset.sum()), onset_recall=float(a[onset].mean()) if onset.any() else np.nan,
                             false_alert_rate=float(a[p.y_poor.values == 0].mean()),
                             false_alert_on_clean_days=float(a[clean].mean()) if clean.any() else np.nan))
    c = pd.DataFrame(rows)
    has = c.TP.notna() if "TP" in c else pd.Series(False, index=c.index)
    c.loc[has, "recall"] = c.TP / (c.TP + c.FN).replace(0, np.nan)
    c.loc[has, "precision"] = c.TP / (c.TP + c.FP).replace(0, np.nan)
    return c


def calibration(p, th):
    from sklearn.metrics import brier_score_loss
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], color=TEXT2, lw=1, ls=":")
    rows = []
    for k, m in enumerate(CLS_MODELS):
        if f"{m}_score" not in p:
            continue
        s, yy = p[f"{m}_score"].values, p.y_poor.values
        bins = np.clip((s * 10).astype(int), 0, 9)
        g = pd.DataFrame({"b": bins, "s": s, "y": yy}).groupby("b").agg(s=("s", "mean"), y=("y", "mean"), n=("y", "size"))
        ax.plot(g.s, g.y, marker="o", ms=4, color=SERIES[k], label=m)
        rows.append(dict(model=m, brier=float(brier_score_loss(yy, np.clip(s, 0, 1))), threshold=th[m]))
    ax.set_xlabel("mean classifier score (bin)")
    ax.set_ylabel("observed share Poor-or-worse")
    ax.set_title("Calibration of the direct classifiers (test)")
    ax.legend(fontsize=7)
    save(fig, FIG / "err_09_calibration.png")
    return pd.DataFrame(rows)


def episode(ax, p, stn, title):
    s = p[p.station == stn].sort_values("target_date")
    s = s[(pd.to_datetime(s.target_date) >= "2022-10-01") & (pd.to_datetime(s.target_date) <= "2023-01-31")]
    if s.empty:
        ax.set_title(title + " (no winter data)")
        return
    for c, (lo, hi) in zip(AQI_CATEGORIES, zip([0, 50, 100, 200, 300, 400], [50, 100, 200, 300, 400, 500])):
        ax.axhspan(lo, hi, color=AQI_COLORS[c], alpha=0.12, lw=0)
    ax.plot(pd.to_datetime(s.target_date), s.y, color="#0b0b0b", lw=2, label="observed")
    for m, col in [("Persistence", "#8a8a8a"), ("RandomForest", SERIES[1]), ("XGBoost", SERIES[4])]:
        ax.plot(pd.to_datetime(s.target_date), s[m], color=col, lw=1.5, ls="--" if m == "Persistence" else "-",
                label=m)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("AQI")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)


def main():
    p = load_preds()
    out = {}
    out["month"] = mae_by(p, "month")
    out["weekday"] = mae_by(p, pd.to_datetime(p.target_date).dt.dayofweek)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    lines(axes[0], out["month"], "Error by target month", "month")
    out["weekday"].index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(out["weekday"])]
    lines(axes[1], out["weekday"], "Error by target weekday", "")
    axes[1].legend(fontsize=7)
    save(fig, FIG / "err_01_month_weekday.png")

    out["region"] = mae_by(p, "region").sort_values("XGBoost")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    yy = np.arange(len(out["region"]))
    w = 0.8 / len(REG_MODELS)
    for k, m in enumerate(REG_MODELS):
        ax.barh(yy + (k - 2.5) * w, out["region"][m], height=w, label=m,
                color="#8a8a8a" if m == "Persistence" else SERIES[k - 1])
    ax.set_yticks(yy, [f"{r} (n={n:,})" for r, n in zip(out["region"].index, out["region"].n)])
    ax.set_xlabel("MAE (AQI points)")
    ax.set_title("Error by region")
    ax.legend(fontsize=7, ncol=2)
    save(fig, FIG / "err_02_region.png")

    # pollution level: true vs predicted vs today's category
    out["true_cat_mae"] = mae_by(p, "true_cat")
    out["true_cat_bias"] = bias_by(p, "true_cat")
    p["xgb_pred_cat"] = pd.Categorical(CAT[category_index(p.XGBoost)], AQI_CATEGORIES, ordered=True)
    out["pred_cat_bias"] = bias_by(p, "xgb_pred_cat")
    out["today_cat_mae"] = mae_by(p, "today_cat")
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    lines(axes[0], out["true_cat_mae"], "MAE by TRUE category of D+1", "")
    lines(axes[1], out["true_cat_bias"], "Bias by TRUE category (conditions on outcome)", "", "mean error")
    lines(axes[2], out["pred_cat_bias"], "Bias by XGBoost FORECAST category", "", "mean error")
    for a in axes[1:]:
        a.axhline(0, color=TEXT2, lw=1)
    axes[2].legend(fontsize=7)
    save(fig, FIG / "err_03_by_category.png")

    # day-to-day change: only where AQI(D) and AQI(D+1) are both measured
    bins = [-1, 25, 50, 100, 150, 1000]
    labels = ["<25", "25-50", "50-100", "100-150", ">150"]
    p["change_bin"] = pd.cut(p.change, bins, labels=labels).astype(object).where(p.change.notna(), "AQI(D) missing")
    order = labels + ["AQI(D) missing"]
    out["change"] = mae_by(p, "change_bin").reindex(order)
    share = p.change_bin.value_counts(normalize=True).reindex(order) * 100
    fig, ax = plt.subplots(figsize=(9, 3.5))
    lines(ax, out["change"], "Error vs |AQI(D+1) - AQI(D)| (both measured)", "change (AQI points)")
    ax.set_xticks(range(len(order)), [f"{l}\n({s:.0f}% of days)" for l, s in zip(order, share.fillna(0))])
    ax.legend(fontsize=7)
    save(fig, FIG / "err_04_change_size.png")

    p["diwali"] = np.where(p.days_from_diwali.abs() <= 2, "Diwali +/-2 d", "other days")
    p["gap"] = np.where(p.miss_aqi_l0 == 1, "AQI(D) missing", "AQI(D) measured")
    out["diwali"] = mae_by(p, "diwali")
    out["gap"] = mae_by(p, "gap")
    out["station_type"] = mae_by(p, "station_type")

    # stations: pooled vs station- and city-averaged scores
    st = p.groupby(["station", "city", "region"]).agg(
        n=("y", "size"), mean_aqi=("y", "mean"), XGB_MAE=("err_XGBoost", lambda e: e.abs().mean()),
        persist_MAE=("err_Persistence", lambda e: e.abs().mean()), lat=("lat", "first"), lon=("lon", "first"),
        miss_rate=("miss_aqi_l0", "mean")).reset_index()
    st = st[st.n >= 60]
    st["skill_pct"] = 100 * (1 - st.XGB_MAE / st.persist_MAE)
    st.sort_values("XGB_MAE", ascending=False).to_csv(REPORTS / "errors_by_station.csv", index=False)
    avg = pd.DataFrame({m: [p[f"err_{m}"].abs().mean(),
                            p.groupby("station")[f"err_{m}"].apply(lambda e: e.abs().mean()).mean(),
                            p.groupby("city_key")[f"err_{m}"].apply(lambda e: e.abs().mean()).mean()]
                        for m in REG_MODELS}, index=["pooled (station-days)", "station-averaged", "city-averaged"])
    out["averaging"] = avg
    if len(st):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sc = axes[0].scatter(st.mean_aqi, st.XGB_MAE, c=st.miss_rate * 100, cmap="Blues", s=22, edgecolor=TEXT2,
                             linewidth=0.3)
        fig.colorbar(sc, ax=axes[0], label="% days with AQI(D) missing")
        axes[0].set_xlabel("station mean AQI (test)")
        axes[0].set_ylabel("XGBoost MAE")
        axes[0].set_title("Station error grows with pollution level")
        axes[1].hist(st.skill_pct, bins=40, color=SERIES[0], edgecolor="#fcfcfb")
        axes[1].axvline(0, color=TEXT2, lw=1)
        axes[1].set_xlabel("XGBoost skill vs persistence (% MAE reduction)")
        axes[1].set_ylabel("stations")
        axes[1].set_title(f"{(st.skill_pct > 0).mean() * 100:.0f}% of stations beat persistence")
        save(fig, FIG / "err_05_stations.png")

    E = p[[f"err_{m}" for m in REG_MODELS]]
    E.columns = REG_MODELS
    corr = E.corr()
    out["error_corr"] = corr
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(REG_MODELS)), REG_MODELS, rotation=35, ha="right")
    ax.set_yticks(range(len(REG_MODELS)), REG_MODELS)
    ax.grid(False)
    for i in range(len(REG_MODELS)):
        for j in range(len(REG_MODELS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if corr.values[i, j] > 0.85 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of forecast errors between models")
    save(fig, FIG / "err_06_error_correlation.png")

    # predicted vs actual + two episodes chosen by a stated rule
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4, 1.4]))
    hb = axes[0].hexbin(p.y, p.XGBoost, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    axes[0].plot([0, 500], [0, 500], color=TEXT2, lw=1)
    axes[0].set_xlabel("observed AQI (D+1)")
    axes[0].set_ylabel("XGBoost forecast")
    axes[0].set_title("Predicted vs actual (test)")
    fig.colorbar(hb, ax=axes[0], label="log10 count")
    sev = p[is_severe(p.y.values)].groupby("station").size().sort_values(ascending=False)
    if len(sev):
        episode(axes[1], p, sev.index[0], f"(a) Most Severe days: {sev.index[0]} ({p[p.station == sev.index[0]].city.iloc[0]})")
    igp = st[(st.region == "North (IGP)") & (st.n >= 200)].sort_values("XGB_MAE")
    if len(igp):
        rep = igp.iloc[len(igp) // 2]
        episode(axes[2], p, rep.station, f"(b) Median-error IGP station: {rep.station} ({rep.city})")
    save(fig, FIG / "err_07_pred_vs_actual_episode.png")

    th = thresholds()
    cl = classification_slices(p, th)
    cl.to_csv(REPORTS / "errors_classification.csv", index=False)
    calibration(p, th).to_csv(REPORTS / "errors_calibration.csv", index=False)

    for k, v in out.items():
        v.to_csv(REPORTS / f"errors_{k}.csv")
    print({k: v.round(1).to_dict() for k, v in out.items() if k in ("true_cat_mae", "change", "station_type")})
    print(cl[cl.slice == "warning"].round(3).to_string())


if __name__ == "__main__":
    main()
