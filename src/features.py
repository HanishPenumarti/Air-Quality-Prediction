"""Step 5 - build the forecasting feature matrix.

Forecast set-up
---------------
  * one row = (station s, issue day D). The forecast is issued at 24:00 IST of day D.
  * target   = official CPCB AQI of day D+1   (regression)
               AQI(D+1) > 200  "Poor or worse" (classification)
  * every feature uses information up to and including day D only, EXCEPT calendar
    features of D+1 (known in advance).
  * weather = OBSERVED on-site CPCB sensors up to day D (available live at issue time).
    Weather of the target day D+1 is never used: it would be observed future weather (leakage).

All lags are calendar-day lags (series re-indexed onto a complete daily grid per station),
and all rolling windows end at D (window [D-k+1, D]) - never at D+1.

Output: data/processed/features.parquet  + data/processed/feature_groups.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXT, POOR_THRESHOLD, PROC, TRAIN_END  # noqa: E402
from eda import region_of  # noqa: E402
from external import DIWALI  # noqa: E402

POLL_L0 = ["PM25_mean", "PM10_mean", "NO2_mean", "SO2_mean", "CO_8h", "O3_8h", "NH3_mean",
           "NO_mean", "NOx_mean", "Benzene_mean", "Toluene_mean"]
INTRADAY = ["PM25_max", "PM25_eve", "PM10_max", "PM10_eve", "NO2_eve", "CO_eve"]
SITE = ["site_WS", "site_RH", "site_AT", "site_SR", "site_BP", "site_RF"]
LOCKDOWN = ("2020-03-25", "2020-05-31")   # national COVID-19 lockdown phases 1-4


def complete_grid(d):
    parts = []
    for st, g in d.groupby("station", sort=False):
        idx = pd.date_range(g.date.min(), g.date.max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g["station"] = st
        parts.append(g.rename_axis("date").reset_index())
    out = pd.concat(parts, ignore_index=True)
    for c in ["state", "city"]:
        out[c] = out.groupby("station")[c].transform(lambda s: s.ffill().bfill())
    return out


def hybrid_impute(d, cols):
    """Causal daily-level imputation chosen in the imputation experiment (src/imputation.py):
    1. mean of OTHER stations in the same city on the same day   (spatial, r~0.90)
    2. forward-fill within station, at most 3 days                (temporal, past only)
    3. station x month median from the TRAINING period            (seasonal climatology)
    Used only to build input features - the target AQI(D+1) is never imputed."""
    out = d[cols].copy()
    trn = d.date <= TRAIN_END
    month = d.date.dt.month
    for c in cols:
        x = d[c]
        s_ = x.groupby([d.city, d.date]).transform("sum")
        n_ = x.groupby([d.city, d.date]).transform("count")
        neigh = ((s_ - x.fillna(0)) / (n_ - x.notna().astype(int))).replace([np.inf, -np.inf], np.nan)
        v = x.fillna(neigh)
        v = v.groupby(d.station).ffill(limit=3)
        clim = x[trn].groupby([d.station[trn], month[trn]]).median()
        cl = pd.Series(clim.reindex(list(zip(d.station, month))).values, index=d.index)
        out[c] = v.fillna(cl)
    return out


def fire_counts(geo):
    """MODIS fire detections near each station per day (vegetation fires, confidence>=30)."""
    from sklearn.neighbors import BallTree
    f = pd.read_parquet(EXT / "firms_modis_india.parquet")
    f = f[(f.confidence >= 30) & (f.type == 0)]
    f["date"] = pd.to_datetime(f.acq_date)
    st = geo.dropna(subset=["lat"]).reset_index(drop=True)
    tree = BallTree(np.radians(st[["lat", "lon"]].values), metric="haversine")
    R = 6371.0
    rows = []
    for _, chunk in f.groupby(f.date.dt.year):
        X = np.radians(chunk[["latitude", "longitude"]].values)
        ind, dist = tree.query_radius(X, r=300 / R, return_distance=True)
        lens = np.array([len(i) for i in ind])
        rep = np.repeat(np.arange(len(chunk)), lens)
        if not len(rep):
            continue
        pairs = pd.DataFrame({"station": st.station.values[np.concatenate(ind)],
                              "date": chunk.date.values[rep],
                              "dist": np.concatenate(dist) * R,
                              "frp": chunk.frp.values[rep]})
        pairs["n100"] = pairs.dist <= 100
        agg = pairs.groupby(["station", "date"]).agg(fire_300km=("dist", "size"), fire_100km=("n100", "sum"),
                                                     frp_300km=("frp", "sum"))
        rows.append(agg)
    fc = pd.concat(rows).groupby(level=[0, 1]).sum().reset_index()
    # regional stubble-burning signal: all fires in Punjab + Haryana box
    pbhr = f[(f.latitude.between(27.6, 32.5)) & (f.longitude.between(73.8, 77.6))]
    reg = pbhr.groupby("date").size().rename("fire_PB_HR").reset_index()
    return fc, reg


def main():
    d = pd.read_parquet(PROC / "daily.parquet")
    geo = pd.read_csv(EXT / "stations_geo.csv")[["station", "lat", "lon", "elevation"]]
    d = complete_grid(d)
    d = d.sort_values(["station", "date"]).reset_index(drop=True)
    g = d.groupby("station", sort=False)
    F = pd.DataFrame({"station": d.station, "city": d.city, "state": d.state, "date": d.date})
    groups = {}

    def add(group, name, values):
        F[name] = values
        groups.setdefault(group, []).append(name)

    # ---------------- target (D+1) -----------------------------------------------------
    F["target_date"] = d.date + pd.Timedelta(days=1)
    F["y_aqi"] = g.AQI.shift(-1)
    F["y_poor"] = (F.y_aqi > POOR_THRESHOLD).astype("float32").where(F.y_aqi.notna())
    F["y_pm25"] = g.PM25_mean.shift(-1)          # kept for diagnostics only, never a feature

    # ---------------- causal imputation of the history used for lag features -----------
    di = hybrid_impute(d, ["AQI"] + POLL_L0 + INTRADAY)
    gi = di.groupby(d.station, sort=False)
    aqi = d.AQI          # raw (for missingness features)
    aqi_i = di.AQI       # imputed (for lag features)

    # ---------------- target history (AQI lags) ---------------------------------------
    for k in range(7):
        add("lag_aqi", f"aqi_l{k}", gi.AQI.shift(k))
    for w in (3, 7, 14, 30):
        add("lag_aqi", f"aqi_rm{w}", gi.AQI.transform(lambda s: s.rolling(w, min_periods=max(1, w // 3)).mean()))
    add("lag_aqi", "aqi_rsd7", gi.AQI.transform(lambda s: s.rolling(7, min_periods=3).std()))
    add("lag_aqi", "aqi_rmax7", gi.AQI.transform(lambda s: s.rolling(7, min_periods=3).max()))
    add("lag_aqi", "aqi_diff1", aqi_i - gi.AQI.shift(1))
    add("lag_aqi", "poor_days7", (aqi_i > POOR_THRESHOLD).astype(float).where(aqi_i.notna())
        .groupby(d.station).transform(lambda s: s.rolling(7, min_periods=1).sum()))
    add("lag_aqi", "aqi_last_valid", g.AQI.transform(lambda s: s.ffill()))   # last MEASURED AQI

    # ---------------- other pollutant lags + intraday shape of day D -------------------
    for c in POLL_L0:
        add("lag_poll", f"{c}_l0", di[c])
    for c in ["PM25_mean", "PM10_mean", "NO2_mean", "CO_8h"]:
        add("lag_poll", f"{c}_l1", gi[c].shift(1))
    for c in ["PM25_mean", "PM10_mean"]:
        add("lag_poll", f"{c}_rm7", gi[c].transform(lambda s: s.rolling(7, min_periods=3).mean()))
    for c in INTRADAY:
        add("lag_poll", f"{c}_l0", di[c])
    add("lag_poll", "pm_ratio_l0", di.PM25_mean / di.PM10_mean)

    # ---------------- missingness / data quality --------------------------------------
    add("missing", "PM25_nh_l0", d.PM25_nh.fillna(0))
    add("missing", "PM10_nh_l0", d.PM10_nh.fillna(0))
    add("missing", "aqi_nsub_l0", d.AQI_n_sub.fillna(0))
    add("missing", "miss_aqi_l0", aqi.isna().astype("int8"))
    add("missing", "miss_aqi_7d", aqi.isna().astype(float).groupby(d.station)
        .transform(lambda s: s.rolling(7, min_periods=1).sum()))
    grp_valid = aqi.notna().groupby(d.station).cumsum()
    days_since = d.groupby([d.station, grp_valid]).cumcount()
    add("missing", "days_since_valid", days_since.where(grp_valid > 0).astype(float))

    # ---------------- spatial neighbours (other stations, same day D) -----------------
    for key, col, nm in [("city", "AQI", "city_aqi_l0"), ("city", "PM25_mean", "city_pm25_l0"),
                         ("state", "AQI", "state_aqi_l0")]:
        s_ = d.groupby([key, "date"])[col].transform("sum")
        n_ = d.groupby([key, "date"])[col].transform("count")
        own = d[col].notna().astype(int)
        add("spatial", nm, ((s_ - d[col].fillna(0)) / (n_ - own)).replace([np.inf, -np.inf], np.nan))
    add("spatial", "city_nst_l0", d.groupby(["city", "date"]).AQI.transform("count") - aqi.notna().astype(int))
    add("spatial", "city_aqi_rm3", F.groupby("station").city_aqi_l0
        .transform(lambda s: s.rolling(3, min_periods=1).mean()))

    # ---------------- on-site meteorology (day D) ------------------------------------
    for c in SITE:
        add("met_site", f"{c}_l0", d[c])
    add("met_site", "site_WD_sin_l0", np.sin(np.radians(d.site_WD)))
    add("met_site", "site_WD_cos_l0", np.cos(np.radians(d.site_WD)))

    # ---------------- fires ----------------------------------------------------------
    fc, reg = fire_counts(geo)
    m = F[["station", "date"]].merge(fc, on=["station", "date"], how="left")
    for c in ["fire_100km", "fire_300km", "frp_300km"]:
        add("fire", f"{c}_l0", m[c].fillna(0).values)
    add("fire", "fire_300km_rm3", pd.Series(m.fire_300km.fillna(0).values).groupby(F.station.values)
        .transform(lambda s: s.rolling(3, min_periods=1).sum()).values)
    r = F[["date"]].merge(reg, on="date", how="left").fire_PB_HR.fillna(0).values
    add("fire", "fire_PB_HR_l0", r)
    add("fire", "fire_PB_HR_rm7", pd.Series(r).groupby(F.station.values)
        .transform(lambda s: s.rolling(7, min_periods=1).mean()).values)
    # fires only observed since FIRMS archive start - before 2010-11 MODIS is still complete

    # ---------------- calendar of target day D+1 (known in advance) -------------------
    t = F.target_date
    doy = t.dt.dayofyear
    add("calendar", "doy_sin", np.sin(2 * np.pi * doy / 365.25))
    add("calendar", "doy_cos", np.cos(2 * np.pi * doy / 365.25))
    add("calendar", "month", t.dt.month)
    add("calendar", "dow", t.dt.dayofweek)
    add("calendar", "weekend", (t.dt.dayofweek >= 5).astype("int8"))
    dw = pd.to_datetime(DIWALI)
    delta = np.stack([(t - x).dt.days.values for x in dw])
    nearest = delta[np.abs(delta).argmin(axis=0), np.arange(len(t))]
    add("calendar", "days_from_diwali", np.clip(nearest, -30, 30))
    add("calendar", "diwali_window", (np.abs(nearest) <= 2).astype("int8"))
    add("calendar", "lockdown", t.between(*LOCKDOWN).astype("int8"))

    # ---------------- static station context ------------------------------------------
    F = F.merge(geo, on="station", how="left")
    for c in ["lat", "lon", "elevation"]:
        groups.setdefault("static", []).append(c)
    regions = sorted({region_of(s) for s in F.state.unique()})
    add("static", "region_code", F.state.map(region_of).map({r: i for i, r in enumerate(regions)}))
    # station x month climatology of AQI, computed from TRAINING period only
    trn = d[d.date <= TRAIN_END]
    clim_sm = trn.groupby(["station", trn.date.dt.month]).AQI.mean()
    clim_cm = trn.groupby(["city", trn.date.dt.month]).AQI.mean()
    clim_rm = trn.assign(region=trn.state.map(region_of)).groupby(["region", trn.date.dt.month]).AQI.mean()
    mo = F.target_date.dt.month
    c1 = pd.Series(clim_sm.reindex(list(zip(F.station, mo))).values)
    c2 = pd.Series(clim_cm.reindex(list(zip(F.city, mo))).values)
    c3 = pd.Series(clim_rm.reindex(list(zip(F.state.map(region_of), mo))).values)
    add("static", "clim_aqi_month", c1.fillna(c2).fillna(c3).values)

    # ---------------- finalize --------------------------------------------------------
    num = F.select_dtypes("float64").columns
    F[num] = F[num].astype("float32")
    # full grid (incl. days whose next-day AQI is unknown) -> lets the UI forecast any day
    F[F.aqi_last_valid.notna()].reset_index(drop=True).to_parquet(PROC / "features_full.parquet", index=False)
    F = F[F.y_aqi.notna()].reset_index(drop=True)
    F.to_parquet(PROC / "features.parquet", index=False)
    (PROC / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    n_feat = sum(len(v) for v in groups.values())
    print(f"rows: {len(F):,}   features: {n_feat}")
    for k, v in groups.items():
        print(f"  {k:10s} {len(v):3d}")
    print("Poor-or-worse share:", round(float(F.y_poor.mean()), 3))


if __name__ == "__main__":
    main()
