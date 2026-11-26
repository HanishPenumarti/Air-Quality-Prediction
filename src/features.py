"""Step 3 - build the forecasting feature matrix (issue-time features only).

Forecast set-up
---------------
  * one row = (station s, issue day D). The forecast is issued at 24:00 IST of day D.
  * target   = AQI of day D+1 computed with the CPCB breakpoints (regression)
               AQI(D+1) > 200  "Poor or worse" (classification)
  * every feature uses information available by 24:00 of D, EXCEPT calendar features of D+1
    (known in advance).
  * weather = OBSERVED on-site CPCB sensors up to day D. Weather of the target day is never used.

What this script does NOT do: nothing here is fitted on a training period. Statistics that need a
training period (station x month climatology fallbacks, the `clim_aqi_month` feature) are fitted
per fold / per hold-out by prep.Imputer, so the walk-forward folds, the final model and the
spatial hold-outs each use only their own training data.

Missing history is filled causally on the COMPLETE daily calendar grid of each station
(`causal_fill`): (1) mean of OTHER stations in the same (state, city) on the same day, then
(2) forward-fill of at most 3 calendar days. Remaining gaps stay NaN for prep.Imputer.
Raw-missingness flags (isna_*) are taken BEFORE any fill.

All features are row-wise functions of data up to D, so the same station-day gets the same
features whether or not its next-day target is known (training rows == UI rows).

Outputs: data/processed/features_full.parquet (every station-day on the grid; used by the UI),
features.parquet (rows whose target AQI(D+1) is known; used for modelling),
feature_groups.json, grid.parquet (raw inputs for re-computing spatial features in hold-outs),
reports/coverage.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aqi import is_poor  # noqa: E402
from config import EXT, FIRMS_LATENCY_H, PROC, REPORTS, city_key, region_of  # noqa: E402
from external import DIWALI  # noqa: E402

POLL_L0 = ["PM25_mean", "PM10_mean", "NO2_mean", "SO2_mean", "CO_8h", "O3_8h", "NH3_mean",
           "NO_mean", "NOx_mean", "Benzene_mean", "Toluene_mean"]
INTRADAY = ["PM25_max", "PM25_eve", "PM10_max", "PM10_eve", "NO2_eve", "CO_eve"]
SITE = ["site_WS", "site_RH", "site_AT", "site_SR", "site_BP", "site_RF"]
LOCKDOWN = ("2020-03-25", "2020-05-31")   # national COVID-19 lockdown phases 1-4
FILL_DAYS = 3
FOLD_FITTED = ["clim_aqi_month"]          # listed in feature groups, computed by prep.Imputer
REGIONS = ["Central", "East", "North (IGP)", "North-East & Hills", "South", "West"]


# ------------------------------------------------------------------ grid & causal fill
def complete_grid(d):
    """One row per station per calendar day between its first and last record."""
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    parts = []
    for st, g in d.groupby("station", sort=True):
        idx = pd.date_range(g.date.min(), g.date.max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g["station"] = st
        parts.append(g.rename_axis("date").reset_index())
    out = pd.concat(parts, ignore_index=True)
    for c in ["state", "city"]:
        out[c] = out.groupby("station")[c].transform(lambda s: s.ffill().bfill())
    out["city_key"] = [city_key(s, c) for s, c in zip(out.state, out.city)]
    return out.sort_values(["station", "date"]).reset_index(drop=True)


def neighbour_mean(d, col, key="city_key", donor=None):
    """Mean of OTHER stations' raw `col` in the same group on the same day (NaN if none).
    `donor` (bool array) restricts which rows may contribute (used by spatial hold-outs)."""
    x = d[col] if donor is None else d[col].where(np.asarray(donor))
    grp = [d[key], d["date"]]
    s_ = x.groupby(grp).transform("sum")
    n_ = x.groupby(grp).transform("count")
    own = x.notna()
    return ((s_ - x.fillna(0)) / (n_ - own.astype(int))).replace([np.inf, -np.inf], np.nan), n_ - own.astype(int)


def causal_fill(d, cols, how="hybrid", days=FILL_DAYS):
    """Issue-time imputation of the history used for lag features (d must be a complete grid).
    how: hybrid (neighbours -> ffill <= `days` calendar days) | neighbour | ffill | none."""
    out = d[cols].copy()
    if how == "none":
        return out
    for c in cols:
        v = d[c]
        if how in ("hybrid", "neighbour"):
            v = v.fillna(neighbour_mean(d, c)[0])
        if how in ("hybrid", "ffill"):
            v = v.groupby(d.station.values).ffill(limit=days)
        out[c] = v
    return out


# ------------------------------------------------------------------ spatial features
def spatial_features(d, donor=None):
    """Neighbour context on day D (other stations only). `donor` excludes stations from being
    used as donors (e.g. a held-out city must not feed the training rows of other cities)."""
    out = pd.DataFrame(index=d.index)
    out["city_aqi_l0"], n_city = neighbour_mean(d, "AQI", "city_key", donor)
    out["city_pm25_l0"], _ = neighbour_mean(d, "PM25_mean", "city_key", donor)
    out["state_aqi_l0"], _ = neighbour_mean(d, "AQI", "state", donor)
    out["city_nst_l0"] = n_city.astype(float)
    out["city_aqi_rm3"] = out.city_aqi_l0.groupby(d.station.values).transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    out["isna_city_aqi_l0"] = out.city_aqi_l0.isna().astype("int8")
    return out


SPATIAL_COLS = ["city_aqi_l0", "city_pm25_l0", "state_aqi_l0", "city_nst_l0", "city_aqi_rm3", "isna_city_aqi_l0"]


# ------------------------------------------------------------------ fires
def fire_events(f):
    """FIRMS detections -> the IST calendar day on which each detection is AVAILABLE.
    acq_date/acq_time are UTC; IST = UTC + 5:30; near-real-time data arrive ~3 h after the
    overpass, so a 22:30 IST night detection only counts for the NEXT issue day.
    The archive's `type` field exists only in the standard-quality product (not near-real-time),
    so it is NOT used; vegetation and static sources are both counted (confidence >= 30)."""
    f = f.dropna(subset=["latitude", "longitude", "acq_date", "acq_time"])
    f = f[f.latitude.between(-90, 90) & f.longitude.between(-180, 180)]
    f = f[pd.to_numeric(f.confidence, errors="coerce") >= 30].copy()
    at = f.acq_time.astype(int)
    t_utc = pd.to_datetime(f.acq_date) + pd.to_timedelta(at // 100, unit="h") + pd.to_timedelta(at % 100, unit="m")
    f["date"] = (t_utc + pd.Timedelta(hours=5.5 + FIRMS_LATENCY_H)).dt.floor("D")
    return f


def fire_counts(geo, f):
    """Per station and issue day: detections within 100/300 km and FRP within 300 km; plus the
    regional Punjab + Haryana total. Stations without coordinates get no rows (-> NaN later)."""
    from sklearn.neighbors import BallTree
    cols = ["station", "date", "fire_300km", "fire_100km", "frp_300km"]
    st = geo.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    rows = []
    if len(st) and len(f):
        tree = BallTree(np.radians(st[["lat", "lon"]].values), metric="haversine")
        R = 6371.0
        for _, chunk in f.groupby(f.date.dt.year):
            ind, dist = tree.query_radius(np.radians(chunk[["latitude", "longitude"]].values),
                                          r=300 / R, return_distance=True)
            lens = np.array([len(i) for i in ind])
            if not lens.sum():
                continue
            rep = np.repeat(np.arange(len(chunk)), lens)
            pairs = pd.DataFrame({"station": st.station.values[np.concatenate(ind)],
                                  "date": chunk.date.values[rep], "dist": np.concatenate(dist) * R,
                                  "frp": chunk.frp.values[rep]})
            pairs["n100"] = pairs.dist <= 100
            rows.append(pairs.groupby(["station", "date"]).agg(
                fire_300km=("dist", "size"), fire_100km=("n100", "sum"), frp_300km=("frp", "sum")))
    fc = (pd.concat(rows).groupby(level=[0, 1]).sum().reset_index() if rows
          else pd.DataFrame(columns=cols))
    pbhr = f[(f.latitude.between(27.6, 32.5)) & (f.longitude.between(73.8, 77.6))]
    reg = pbhr.groupby("date").size().rename("fire_PB_HR").reset_index()
    return fc, reg, set(st.station)


def load_fires(geo):
    p = EXT / "firms_modis_india.parquet"
    if not p.exists():
        raise SystemExit("data/external/firms_modis_india.parquet missing - run: python src/external.py firms")
    return fire_counts(geo, fire_events(pd.read_parquet(p)))


# ------------------------------------------------------------------ build
def build(d, geo, fires, fill="hybrid"):
    """d: daily.parquet frame; geo: stations_geo; fires: output of load_fires/fire_counts.
    Returns (F, groups) on the complete daily grid (all rows, target may be NaN)."""
    d = complete_grid(d)
    st_ = d.station.values
    g = d.groupby(st_, sort=False)
    C, groups = {}, {}

    def add(group, name, values):
        C[name] = np.asarray(values)
        groups.setdefault(group, []).append(name)

    # ---------------- target (D+1) ----------------------------------------------------
    target_date = d.date + pd.Timedelta(days=1)
    y = g.AQI.shift(-1)
    targets = pd.DataFrame({"target_date": target_date, "y_aqi": y,
                            "y_poor": pd.Series(is_poor(y), index=d.index).astype("float32").where(y.notna()),
                            "y_pm25": g.PM25_mean.shift(-1)})      # diagnostics only, never a feature

    # ---------------- causal imputation of the history used for lag features -----------
    di = causal_fill(d, ["AQI"] + POLL_L0 + INTRADAY, how=fill)
    gi = di.groupby(st_, sort=False)
    aqi = d.AQI          # raw (for missingness features)
    aqi_i = di.AQI       # filled (for lag features)

    # ---------------- AQI history -------------------------------------------------------
    for k in range(7):
        add("lag_aqi", f"aqi_l{k}", gi.AQI.shift(k))
    for w in (3, 7, 14, 30):
        add("lag_aqi", f"aqi_rm{w}", gi.AQI.transform(lambda s: s.rolling(w, min_periods=max(1, w // 3)).mean()))
    add("lag_aqi", "aqi_rsd7", gi.AQI.transform(lambda s: s.rolling(7, min_periods=3).std()))
    add("lag_aqi", "aqi_rmax7", gi.AQI.transform(lambda s: s.rolling(7, min_periods=3).max()))
    add("lag_aqi", "aqi_diff1", aqi_i - gi.AQI.shift(1))
    poor = pd.Series(is_poor(aqi_i), index=d.index).astype(float).where(aqi_i.notna())
    add("lag_aqi", "poor_days7", poor.groupby(st_).transform(lambda s: s.rolling(7, min_periods=1).sum()))
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

    # ---------------- missingness / data quality (RAW values, before any fill) ---------
    add("missing", "PM25_nh_l0", d.PM25_nh.fillna(0))
    add("missing", "PM10_nh_l0", d.PM10_nh.fillna(0))
    add("missing", "aqi_nsub_l0", d.AQI_n_sub.fillna(0))
    add("missing", "miss_aqi_l0", aqi.isna().astype("int8"))
    add("missing", "miss_aqi_7d", aqi.isna().astype(float).groupby(st_).transform(lambda s: s.rolling(7, min_periods=1).sum()))
    grp_valid = aqi.notna().groupby(st_).cumsum()
    days_since = d.groupby([d.station, grp_valid]).cumcount()
    add("missing", "days_since_valid", days_since.where(grp_valid > 0).astype(float))
    add("missing", "isna_PM25_mean_l0", d.PM25_mean.isna().astype("int8"))
    add("missing", "isna_PM10_mean_l0", d.PM10_mean.isna().astype("int8"))
    add("missing", "isna_aqi_l1", aqi.groupby(st_).shift(1).isna().astype("int8"))

    # ---------------- spatial neighbours (other stations, same day D) -----------------
    sp = spatial_features(d)
    for c in SPATIAL_COLS:
        add("spatial", c, sp[c])

    # ---------------- on-site meteorology (day D, observed) ---------------------------
    for c in SITE:
        add("met_site", f"{c}_l0", d[c])
    add("met_site", "site_WD_sin_l0", np.sin(np.radians(d.site_WD)))
    add("met_site", "site_WD_cos_l0", np.cos(np.radians(d.site_WD)))
    add("met_site", "isna_site_WS_l0", d.site_WS.isna().astype("int8"))

    # ---------------- fires (available by 24:00 IST of D) ------------------------------
    fc, reg, located = fires
    key = pd.DataFrame({"station": d.station, "date": d.date})
    m = key.merge(fc, on=["station", "date"], how="left")
    known = d.station.isin(located).values
    for c in ["fire_100km", "fire_300km", "frp_300km"]:
        add("fire", f"{c}_l0", np.where(known, m[c].astype(float).fillna(0).values, np.nan))
    f300 = pd.Series(C["fire_300km_l0"], index=d.index)
    add("fire", "fire_300km_rm3", f300.groupby(st_).transform(lambda s: s.rolling(3, min_periods=1).sum()))
    r = key[["date"]].merge(reg, on="date", how="left").fire_PB_HR.astype(float).fillna(0)
    r.index = d.index
    add("fire", "fire_PB_HR_l0", r)
    add("fire", "fire_PB_HR_rm7", r.groupby(st_).transform(lambda s: s.rolling(7, min_periods=1).mean()))

    # ---------------- calendar of target day D+1 (known in advance) -------------------
    t = target_date
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
    gm = d[["station"]].merge(geo[["station", "lat", "lon", "elevation"]], on="station", how="left")
    for c in ["lat", "lon", "elevation"]:
        add("static", c, gm[c].values)
    add("static", "region_code", d.state.map(region_of).map({r_: i for i, r_ in enumerate(REGIONS)}))
    groups["static"] += FOLD_FITTED      # computed per fold by prep.Imputer

    ids = pd.DataFrame({"station": d.station, "city": d.city, "state": d.state, "city_key": d.city_key,
                        "date": d.date})
    F = pd.concat([ids, targets, pd.DataFrame(C, index=d.index)], axis=1)
    num = F.select_dtypes("float64").columns
    F[num] = F[num].astype("float32")
    return F, groups, d


def main():
    d0 = pd.read_parquet(PROC / "daily.parquet")
    geo = pd.read_csv(EXT / "stations_geo.csv")
    F, groups, d = build(d0, geo, load_fires(geo))
    d[["station", "state", "city_key", "date", "AQI", "PM25_mean"]].to_parquet(PROC / "grid.parquet", index=False)

    # every station-day (the UI can forecast any of them) / rows usable for modelling (target known)
    F_full = F.reset_index(drop=True)
    F_full.to_parquet(PROC / "features_full.parquet", index=False)
    Fm = F[F.y_aqi.notna()].reset_index(drop=True)
    Fm.to_parquet(PROC / "features.parquet", index=False)
    (PROC / "feature_groups.json").write_text(json.dumps(groups, indent=2))

    # coverage: how many station-days survive each stage (prediction vs evaluation availability)
    cov = pd.DataFrame({"year": F.date.dt.year, "grid": 1, "aqi_today": F.miss_aqi_l0 == 0,
                        "forecastable": F.aqi_last_valid.notna(), "target_known": F.y_aqi.notna()})
    cov = cov.groupby("year").sum().astype(int)
    cov.loc["total"] = cov.sum()
    cov.to_csv(REPORTS / "coverage.csv")

    n_feat = sum(len(v) for v in groups.values())
    print(f"rows: {len(Fm):,} with a known target, {len(F_full):,} on the station-day grid; features: {n_feat}")
    for k, v in groups.items():
        print(f"  {k:10s} {len(v):3d}")
    print("Poor-or-worse share:", round(float(Fm.y_poor.mean()), 3))
    print(cov.to_string())


if __name__ == "__main__":
    main()
