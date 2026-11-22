"""Analysis E - Error analysis: where and why do the models fail?

Test period 2022-01 .. 2023-03, all regression models (raw features) + persistence.
Slices: month/season, region, station, true AQI category (esp. Severe), day-of-week,
day-to-day change size (sudden episodes vs steady days), Diwali window, data gaps.
Note: the target is a DAILY AQI, so "hour of day" is not applicable; issue weekday is used.
Outputs: reports/errors_*.csv, figures err_*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import AQI_CATEGORIES, FIG, PROC, REPORTS  # noqa: E402
from eda import region_of  # noqa: E402
from metrics import category  # noqa: E402
from viz import AQI_COLORS, SERIES, TEXT2, plt, save  # noqa: E402

MODELS = ["Persistence", "DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]


def load_preds(tag=""):
    p = pd.read_parquet(PROC / f"preds_test{tag}.parquet")
    F = pd.read_parquet(PROC / "features.parquet",
                        columns=["station", "date", "aqi_l0", "miss_aqi_l0", "days_from_diwali", "lat", "lon"])
    p = p.merge(F, on=["station", "date"], how="left")
    ren = {"Persistence|reg": "Persistence"}
    ren.update({f"{m}|reg|raw": m for m in MODELS[1:]})
    p = p.rename(columns=ren)
    p["region"] = p.state.map(region_of)
    p["month"] = pd.to_datetime(p.target_date).dt.month
    p["true_cat"] = pd.Categorical(np.array(AQI_CATEGORIES)[category(p.y)], AQI_CATEGORIES, ordered=True)
    p["change"] = (p.y - p.aqi_l0).abs()
    for m in MODELS:
        p[f"err_{m}"] = p[m] - p.y
    return p


def mae_by(p, key):
    return pd.DataFrame({m: p.groupby(key, observed=True)[f"err_{m}"].apply(lambda e: e.abs().mean())
                         for m in MODELS})


def bias_by(p, key):
    return pd.DataFrame({m: p.groupby(key, observed=True)[f"err_{m}"].mean() for m in MODELS})


def lines(ax, df, title, xlabel, ylabel="MAE (AQI points)"):
    for k, m in enumerate(MODELS):
        ls = "--" if m == "Persistence" else "-"
        ax.plot(range(len(df)), df[m], marker="o", ms=4, ls=ls, color=SERIES[k], label=m)
    ax.set_xticks(range(len(df)), [str(i) for i in df.index], rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    p = load_preds()
    out = {}
    # 1. season & weekday ------------------------------------------------------------
    out["month"] = mae_by(p, "month")
    out["weekday"] = mae_by(p, pd.to_datetime(p.target_date).dt.dayofweek)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    lines(axes[0], out["month"], "Error by target month", "month")
    out["weekday"].index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines(axes[1], out["weekday"], "Error by target weekday", "")
    axes[1].legend(fontsize=7)
    save(fig, FIG / "err_01_month_weekday.png")

    # 2. region ----------------------------------------------------------------------
    out["region"] = mae_by(p, "region").sort_values("XGBoost")
    nr = p.groupby("region").size()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    yy = np.arange(len(out["region"]))
    w = 0.8 / len(MODELS)
    for k, m in enumerate(MODELS):
        ax.barh(yy + (k - 2.5) * w, out["region"][m], height=w, color=SERIES[k], label=m)
    ax.set_yticks(yy, [f"{r} (n={nr[r]:,})" for r in out["region"].index])
    ax.set_xlabel("MAE (AQI points)")
    ax.set_title("Error by region")
    ax.legend(fontsize=7, ncol=2)
    save(fig, FIG / "err_02_region.png")

    # 3. pollution level -------------------------------------------------------------
    out["true_cat_mae"] = mae_by(p, "true_cat")
    out["true_cat_bias"] = bias_by(p, "true_cat")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    lines(axes[0], out["true_cat_mae"], "MAE by TRUE AQI category", "")
    lines(axes[1], out["true_cat_bias"], "Bias (forecast - truth) by TRUE category", "", "mean error")
    axes[1].axhline(0, color=TEXT2, lw=1)
    axes[1].legend(fontsize=7)
    save(fig, FIG / "err_03_by_category.png")

    # 4. sudden change vs steady -------------------------------------------------------
    bins = [-1, 25, 50, 100, 150, 1000]
    labels = ["<25", "25-50", "50-100", "100-150", ">150"]
    p["change_bin"] = pd.cut(p.change, bins, labels=labels)
    out["change"] = mae_by(p, "change_bin")
    share = p.change_bin.value_counts(normalize=True).reindex(labels) * 100
    fig, ax = plt.subplots(figsize=(8, 3.5))
    lines(ax, out["change"], "Error vs size of the day-to-day AQI change |AQI(D+1) - AQI(D)|", "change (AQI points)")
    ax.set_xticks(range(len(labels)), [f"{l}\n({s:.0f}% of days)" for l, s in zip(labels, share)])
    ax.legend(fontsize=7)
    save(fig, FIG / "err_04_change_size.png")

    # 5. Diwali & data gaps -------------------------------------------------------------
    p["diwali"] = np.where(p.days_from_diwali.abs() <= 2, "Diwali +/-2 d", "other days")
    p["gap"] = np.where(p.miss_aqi_l0 == 1, "AQI(D) missing", "AQI(D) measured")
    out["diwali"] = mae_by(p, "diwali")
    out["gap"] = mae_by(p, "gap")

    # 6. stations ------------------------------------------------------------------------
    st = p.groupby(["station", "city", "region"]).agg(
        n=("y", "size"), mean_aqi=("y", "mean"), XGB_MAE=("err_XGBoost", lambda e: e.abs().mean()),
        persist_MAE=("err_Persistence", lambda e: e.abs().mean()), lat=("lat", "first"), lon=("lon", "first"),
        miss_rate=("miss_aqi_l0", "mean")).reset_index()
    st = st[st.n >= 60]
    st["skill_pct"] = 100 * (1 - st.XGB_MAE / st.persist_MAE)
    st.sort_values("XGB_MAE", ascending=False).to_csv(REPORTS / "errors_by_station.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sc = axes[0].scatter(st.mean_aqi, st.XGB_MAE, c=st.miss_rate * 100, cmap="Blues", s=22,
                         edgecolor=TEXT2, linewidth=0.3)
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

    # 7. error correlation between models -------------------------------------------------
    E = p[[f"err_{m}" for m in MODELS]]
    E.columns = MODELS
    corr = E.corr()
    out["error_corr"] = corr
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=35, ha="right")
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.grid(False)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if corr.values[i, j] > 0.85 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of forecast errors between models")
    save(fig, FIG / "err_06_error_correlation.png")

    # 8. predicted vs actual + episode time-series ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    hb = axes[0].hexbin(p.y, p.XGBoost, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    axes[0].plot([0, 500], [0, 500], color=TEXT2, lw=1)
    for e in [100, 200, 300, 400]:
        axes[0].axvline(e, color="#d9d8d4", lw=0.6)
        axes[0].axhline(e, color="#d9d8d4", lw=0.6)
    axes[0].set_xlabel("observed AQI (D+1)")
    axes[0].set_ylabel("XGBoost forecast")
    axes[0].set_title("Predicted vs actual (test)")
    fig.colorbar(hb, ax=axes[0], label="log10 count")
    # episode: Delhi-area station with the most Severe days in the test set
    sev = p[p.y > 400].groupby("station").size().sort_values(ascending=False)
    stn = sev.index[0]
    s = p[p.station == stn].sort_values("target_date")
    s = s[(pd.to_datetime(s.target_date) >= "2022-10-01") & (pd.to_datetime(s.target_date) <= "2023-01-31")]
    ax = axes[1]
    for c, (lo, hi) in zip(AQI_CATEGORIES, zip([0, 50, 100, 200, 300, 400], [50, 100, 200, 300, 400, 500])):
        ax.axhspan(lo, hi, color=AQI_COLORS[c], alpha=0.12, lw=0)
    ax.plot(pd.to_datetime(s.target_date), s.y, color="#0b0b0b", lw=2, label="observed")
    for m, col in [("Persistence", "#8a8a8a"), ("RandomForest", SERIES[1]), ("XGBoost", SERIES[4])]:
        ax.plot(pd.to_datetime(s.target_date), s[m], color=col, lw=1.5,
                ls="--" if m == "Persistence" else "-", label=m)
    ax.set_title(f"Winter episode at {stn} ({s.city.iloc[0]}): forecasts lag sudden jumps")
    ax.set_ylabel("AQI")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    save(fig, FIG / "err_07_pred_vs_actual_episode.png")

    for k, v in out.items():
        v.to_csv(REPORTS / f"errors_{k}.csv")
    print({k: v.round(1).to_dict() for k, v in out.items() if k in ("true_cat_mae", "change", "diwali", "gap")})


if __name__ == "__main__":
    main()
