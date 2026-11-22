"""Step 4 - exploratory data analysis. Writes figures to reports/figures and key numbers
to reports/eda_summary.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import AQI_CATEGORIES, EXT, FIG, POLLUTANTS, PROC, REPORTS  # noqa: E402
from external import DIWALI  # noqa: E402
from viz import AQI_COLORS, SEQ, SERIES, TEXT2, plt, save  # noqa: E402

REGION = {
    "Delhi": "North (IGP)", "Uttar Pradesh": "North (IGP)", "Haryana": "North (IGP)",
    "Punjab": "North (IGP)", "Bihar": "North (IGP)", "Chandigarh": "North (IGP)",
    "West Bengal": "East", "Jharkhand": "East", "Odisha": "East",
    "Rajasthan": "West", "Gujarat": "West", "Maharashtra": "West",
    "Madhya Pradesh": "Central", "Chhattisgarh": "Central",
    "Karnataka": "South", "Tamil Nadu": "South", "Kerala": "South", "Telangana": "South",
    "Andhra Pradesh": "South", "Puducherry": "South",
}


def region_of(state):
    return REGION.get(state, "North-East & Hills")


def main():
    d = pd.read_parquet(PROC / "daily.parquet")
    d["region"] = d["state"].map(region_of)
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year
    s = {}
    s["station_days"] = len(d)
    s["valid_aqi_days"] = int(d.AQI.notna().sum())
    s["stations"] = int(d.station.nunique())
    s["cities"] = int(d.city.nunique())
    s["date_range"] = [str(d.date.min().date()), str(d.date.max().date())]

    # 1. network growth -------------------------------------------------------------
    act = d[d.PM25_nh > 0].groupby(d.date.dt.to_period("M")).station.nunique()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(act.index.to_timestamp(), act.values, color=SERIES[0])
    ax.set_title("Active monitoring stations per month (reporting PM2.5)")
    ax.set_ylabel("stations")
    save(fig, FIG / "eda_01_network_growth.png")

    # 2. missingness: pollutant x year (station-days of active stations) -------------
    active = d[d[[f"{p}_nh" for p in POLLUTANTS]].sum(axis=1) > 0]
    miss = pd.DataFrame({p: active[f"{p}_mean"].isna().groupby(active.year).mean() for p in POLLUTANTS}).T
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(miss.values * 100, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(SEQ),
                   vmin=0, vmax=100)
    ax.set_xticks(range(miss.shape[1]), miss.columns)
    ax.set_yticks(range(miss.shape[0]), miss.index)
    ax.grid(False)
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            v = miss.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 55 else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="% station-days missing (daily mean invalid)")
    ax.set_title("Missingness by pollutant and year (stations that were reporting anything)")
    save(fig, FIG / "eda_02_missing_pollutant_year.png")
    s["missing_pct_overall"] = (active[[f"{p}_mean" for p in POLLUTANTS]].isna().mean() * 100).round(1).to_dict()

    # 3. missingness by month & by station -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for k, p in enumerate(["PM25", "PM10", "O3", "CO"]):
        mm = active[f"{p}_mean"].isna().groupby(active.month).mean() * 100
        axes[0].plot(mm.index, mm.values, marker="o", ms=4, label=p, color=SERIES[k])
    axes[0].set_xticks(range(1, 13))
    axes[0].set_title("Missingness by month")
    axes[0].set_ylabel("% station-days missing")
    axes[0].legend()
    st_miss = active.groupby("station")["PM25_mean"].apply(lambda x: x.isna().mean() * 100)
    axes[1].hist(st_miss, bins=30, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1)
    axes[1].set_title("PM2.5 missingness per station")
    axes[1].set_xlabel("% of station's days missing")
    axes[1].set_ylabel("stations")
    save(fig, FIG / "eda_03_missing_month_station.png")

    # 4. is missingness informative?  P(station PM2.5 missing) vs city AQI (other stations)
    city_aqi = d.groupby(["city", "date"]).AQI.agg(["sum", "count"]).rename(columns={"sum": "cs", "count": "cn"})
    x = d.join(city_aqi, on=["city", "date"])
    own = x.AQI.fillna(0)
    x["city_aqi_others"] = (x.cs - own) / (x.cn - x.AQI.notna().astype(int))
    x = x[np.isfinite(x.city_aqi_others)]
    x["pm_missing"] = x.PM25_mean.isna()
    bins = [0, 50, 100, 200, 300, 400, 1000]
    grp = x.groupby(pd.cut(x.city_aqi_others, bins, labels=AQI_CATEGORIES), observed=True).pm_missing.mean() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(grp.index.astype(str), grp.values, color=[AQI_COLORS[c] for c in grp.index], width=0.6)
    for i, v in enumerate(grp.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8, color=TEXT2)
    ax.set_title("Station PM2.5 missing vs air quality at OTHER stations in the same city")
    ax.set_ylabel("% station-days PM2.5 missing")
    save(fig, FIG / "eda_04_missing_vs_pollution.png")
    s["pm25_missing_by_city_aqi_cat"] = grp.round(2).to_dict()

    # 5. seasonal cycle by region ------------------------------------------------------
    reg = d.groupby(["region", "month"]).PM25_mean.mean().unstack(0)
    order = reg.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, r in enumerate(order):
        ax.plot(reg.index, reg[r], marker="o", ms=4, label=r, color=SERIES[k])
    ax.axhline(60, color=TEXT2, lw=1, ls="--")
    ax.text(12.2, 60, "NAAQS 24-h\n60 ug/m3", fontsize=7, color=TEXT2, va="center")
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("mean daily PM2.5 (ug/m3)")
    ax.set_title("Seasonal cycle of PM2.5 by region: winter peak, monsoon trough")
    ax.legend(ncol=2)
    save(fig, FIG / "eda_05_seasonal_region.png")

    # 6. diurnal cycle (hourly data) ----------------------------------------------------
    h = pd.read_parquet(PROC / "hourly_clean.parquet", columns=["time", "PM25", "NO2", "O3"])
    h = h[h.PM25.notna() | h.O3.notna()]
    mon = h.time.dt.month
    season = np.select([mon.isin([12, 1, 2]), mon.isin([3, 4, 5]), mon.isin([6, 7, 8, 9])],
                       ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)"], "Post-monsoon (ON)")
    di = h.groupby([season, h.time.dt.hour])[["PM25", "O3", "NO2"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    for j, v in enumerate(["PM25", "NO2", "O3"]):
        for k, sname in enumerate(["Winter (DJF)", "Post-monsoon (ON)", "Pre-monsoon (MAM)", "Monsoon (JJAS)"]):
            axes[j].plot(di.loc[sname].index, di.loc[sname][v], label=sname, color=SERIES[k])
        axes[j].set_title(f"Diurnal cycle - {v}")
        axes[j].set_xlabel("hour of day (IST)")
        axes[j].set_xticks([0, 6, 12, 18, 23])
    axes[0].set_ylabel("ug/m3")
    axes[2].legend()
    save(fig, FIG / "eda_06_diurnal.png")
    del h

    # 7. class balance -----------------------------------------------------------------
    v = d[d.AQI.notna()]
    cat_m = pd.crosstab(v.month, v.AQI_cat, normalize="index")[AQI_CATEGORIES] * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
    tot = v.AQI_cat.value_counts(normalize=True)[AQI_CATEGORIES] * 100
    axes[0].barh(tot.index, tot.values, color=[AQI_COLORS[c] for c in tot.index], height=0.6)
    for i, val in enumerate(tot.values):
        axes[0].text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=8, color=TEXT2)
    axes[0].invert_yaxis()
    axes[0].set_title("AQI category share (all valid station-days)")
    bottom = np.zeros(12)
    for c in AQI_CATEGORIES:
        axes[1].bar(cat_m.index, cat_m[c], bottom=bottom, color=AQI_COLORS[c], label=c,
                    edgecolor="#fcfcfb", linewidth=1, width=0.8)
        bottom += cat_m[c].values
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("AQI category mix by month")
    axes[1].set_ylabel("% of station-days")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, FIG / "eda_07_class_balance.png")
    s["aqi_cat_share_pct"] = tot.round(2).to_dict()
    s["poor_or_worse_pct"] = round(float((v.AQI > 200).mean() * 100), 2)
    s["poor_or_worse_pct_by_region"] = (v.groupby("region").AQI.apply(lambda a: (a > 200).mean() * 100)).round(1).to_dict()

    # 8. AQI distribution + driver pollutant ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].hist(v.AQI, bins=100, color=SERIES[0])
    for edge in [50, 100, 200, 300, 400]:
        axes[0].axvline(edge, color=TEXT2, lw=0.8, ls=":")
    axes[0].set_title(f"AQI distribution (skew {v.AQI.skew():.2f})")
    axes[0].set_xlabel("AQI")
    drv = v.AQI_driver.value_counts(normalize=True) * 100
    drv_poor = v[v.AQI > 200].AQI_driver.value_counts(normalize=True) * 100
    dd = pd.DataFrame({"all days": drv, "Poor+ days": drv_poor}).fillna(0).sort_values("all days")
    yy = np.arange(len(dd))
    axes[1].barh(yy - 0.2, dd["all days"], height=0.38, color=SERIES[0], label="all days")
    axes[1].barh(yy + 0.2, dd["Poor+ days"], height=0.38, color=SERIES[1], label="Poor or worse days")
    axes[1].set_yticks(yy, dd.index)
    axes[1].set_title("Which pollutant sets the AQI?")
    axes[1].set_xlabel("% of days")
    axes[1].legend()
    save(fig, FIG / "eda_08_aqi_dist_driver.png")
    s["aqi_driver_pct_all"] = drv.round(1).to_dict()
    s["aqi_driver_pct_poor"] = drv_poor.round(1).to_dict()
    s["aqi_quantiles"] = v.AQI.quantile([.5, .9, .95, .99]).round(1).to_dict()

    # 9. spatial correlation vs distance ---------------------------------------------------
    geo_path = EXT / "stations_geo.csv"
    if geo_path.exists():
        geo = pd.read_csv(geo_path).set_index("station")
        piv = d[d.date >= "2019-01-01"].pivot_table(index="date", columns="station", values="PM25_mean")
        piv = piv.loc[:, piv.notna().sum() > 365]
        cor = piv.corr(min_periods=200)
        st = cor.columns
        la, lo = geo.loc[st, "lat"].values, geo.loc[st, "lon"].values
        R = 6371
        p1, p2 = np.radians(la)[:, None], np.radians(la)[None, :]
        dl = np.radians(lo)[:, None] - np.radians(lo)[None, :]
        dist = R * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))
        iu = np.triu_indices(len(st), 1)
        dfc = pd.DataFrame({"dist": dist[iu], "r": cor.values[iu]}).dropna()
        dfc["bin"] = pd.cut(dfc.dist, [-1, 1, 50, 100, 250, 500, 1000, 3000],
                            labels=["same city", "<50", "50-100", "100-250", "250-500", "500-1000", ">1000"])
        b = dfc.groupby("bin", observed=True).r.median()
        fig, ax = plt.subplots(figsize=(7, 3.3))
        ax.bar(b.index.astype(str), b.values, color=SERIES[0], width=0.6)
        for i, val in enumerate(b.values):
            ax.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=8, color=TEXT2)
        ax.set_xlabel("distance between station pair (km)")
        ax.set_ylabel("median Pearson r (daily PM2.5)")
        ax.set_title("Spatial dependence: nearby stations move together")
        save(fig, FIG / "eda_09_spatial_corr.png")
        s["spatial_corr_by_distance"] = b.round(3).to_dict()

        # station map coloured by mean AQI
        m = d.groupby("station").AQI.mean().to_frame("aqi").join(geo[["lat", "lon"]]).dropna()
        fig, ax = plt.subplots(figsize=(6, 6.5))
        sc = ax.scatter(m.lon, m.lat, c=m.aqi, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ),
                        s=28, edgecolor="#fcfcfb", linewidth=0.8)
        fig.colorbar(sc, ax=ax, label="mean AQI", shrink=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"{len(m)} stations, mean AQI (city-level coordinates)")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        save(fig, FIG / "eda_10_station_map.png")

    # 10. Diwali effect -------------------------------------------------------------------
    rows = []
    north = d[d.region == "North (IGP)"]
    daily_mean = north.groupby("date").PM25_mean.mean()
    for dw in pd.to_datetime(DIWALI):
        for k in range(-10, 11):
            day = dw + pd.Timedelta(days=k)
            if day in daily_mean.index:
                rows.append(dict(offset=k, year=dw.year, pm=daily_mean[day]))
    dwl = pd.DataFrame(rows)
    rel = dwl.pivot_table(index="offset", columns="year", values="pm")
    rel = rel / rel.loc[-10:-4].mean()
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.plot(rel.index, rel.median(axis=1), color=SERIES[1], marker="o", ms=4)
    ax.fill_between(rel.index, rel.quantile(.25, axis=1), rel.quantile(.75, axis=1), color=SERIES[1], alpha=0.15)
    ax.axvline(0, color=TEXT2, ls=":", lw=1)
    ax.set_xlabel("days relative to Diwali")
    ax.set_ylabel("PM2.5 / pre-Diwali baseline")
    ax.set_title("Diwali firecracker spike, North India (median & IQR over years)")
    save(fig, FIG / "eda_11_diwali.png")
    s["diwali_day_plus1_ratio_median"] = round(float(rel.loc[1].median()), 2)

    # 11. QA summary ---------------------------------------------------------------------
    qa = pd.read_csv(PROC / "qa_log.csv")
    qq = qa.groupby(["var", "rule"]).n_removed.sum().unstack(fill_value=0)
    qq = qq.loc[qq.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    left = np.zeros(len(qq))
    for k, r in enumerate(qq.columns):
        ax.barh(qq.index, qq[r], left=left, color=SERIES[k], label=r, edgecolor="#fcfcfb", linewidth=1)
        left += qq[r].values
    ax.set_xlabel("hourly values set to missing")
    ax.set_title("Sensor-fault QA: values removed by rule")
    ax.legend()
    save(fig, FIG / "eda_12_qa_removed.png")
    s["qa_removed_by_rule"] = qa.groupby("rule").n_removed.sum().to_dict()

    (REPORTS / "eda_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
