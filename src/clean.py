"""Step 1 - clean raw CPCB hourly files and aggregate to station-day.

For every station file:
  1. harmonise column names and units to canonical CPCB units
  2. sensor-fault QA (all rules use only data from the SAME calendar day, so no
     future information leaks into features):
       - impossible values  (negative, zero PM, outside physical range)
       - PM2.5 > 1.5 x PM10 (physically inconsistent pair)
       - stuck sensor       (>= 8 identical consecutive hourly readings)
       - isolated spike     (> 4x day median AND > floor AND both neighbours < half)
  3. daily aggregation following CPCB rules (>= 16 valid hours, 8-h max for CO/O3)
  4. CPCB AQI sub-indices, AQI and category for every station-day

Outputs (data/processed/):
  hourly_clean.parquet  - key hourly variables (for EDA: diurnal cycles etc.)
  daily.parquet         - one row per station-day
  qa_log.csv            - number of values removed by each rule, per station/variable
"""
import re
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import (AQI_BREAKS, AQI_CAT_EDGES, AQI_CATEGORIES, AQI_INDEX, AQI_POLLUTANTS,
                    MET_ONSITE, MIN_HOURS_PER_DAY, POLLUTANTS, PROC, RAW, SPIKE_FLOOR,
                    STUCK_RUN_HOURS, VALID_RANGE)

# ------------------------------------------------------------------ name / unit maps
NAME_MAP = {"PM2.5": "PM25", "PM10": "PM10", "NO2": "NO2", "SO2": "SO2", "CO": "CO",
            "Ozone": "O3", "NH3": "NH3", "NO": "NO", "NOx": "NOx", "Benzene": "Benzene",
            "Toluene": "Toluene", "WS": "WS", "WD": "WD", "RH": "RH", "AT": "AT",
            "Temp": "AT", "SR": "SR", "BP": "BP", "RF": "RF"}

# multiplier to convert (variable, unit) -> canonical unit. 25 degC, 1 atm.
UNIT_FACTOR = {
    ("CO", "mg/m3"): 1.0, ("CO", "mg/Nm3"): 1.0, ("CO", "ug/m3"): 1e-3, ("CO", "ng/m3"): 1e-6,
    ("NOx", "ppb"): 1.0, ("NOx", "ppm"): 1000.0, ("NOx", "ug/m3"): 1 / 1.88,
    ("NO", "ug/m3"): 1.0, ("NO", "mg/m3"): 1000.0, ("NO", "ppb"): 1.23, ("NO", "ppm"): 1230.0,
    ("O3", "ug/m3"): 1.0, ("O3", "ppb"): 1.96,
    ("NH3", "ug/m3"): 1.0, ("NH3", "ppb"): 0.70,
    ("Benzene", "ug/m3"): 1.0, ("Benzene", "mg/m3"): 1000.0,
}
COL_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<unit>[^)]*)\)(?:\.\d+)?$")


def parse_columns(cols):
    """Map raw columns -> list of (raw_col, canonical_var, factor, unit_note)."""
    out = []
    for c in cols:
        m = COL_RE.match(c)
        if not m:
            continue
        name, unit = m["name"].strip(), m["unit"].strip()
        var = NAME_MAP.get(name)
        if var is None:
            continue
        if (var, unit) in UNIT_FACTOR:
            out.append((c, var, UNIT_FACTOR[(var, unit)], "converted" if UNIT_FACTOR[(var, unit)] != 1 else "ok"))
        else:
            # Unit blank or a mislabelled header from the portal (e.g. "RH (W/mt2)").
            # Assume the canonical unit; the physical-range check catches real mismatches.
            out.append((c, var, 1.0, "assumed" if unit not in ("ug/m3", "%", "m/s", "degree C",
                                                               "degree", "deg", "W/mt2", "mmHg", "mm")
                        else "ok"))
    return out


# ------------------------------------------------------------------ QA rules
def stuck_mask(s: pd.Series, day: pd.Series) -> pd.Series:
    """True where value belongs to a run of >= STUCK_RUN_HOURS identical readings (within day)."""
    grp = ((s != s.shift()) | (day != day.shift())).cumsum()
    run = s.groupby(grp).transform("size")
    return s.notna() & (run >= STUCK_RUN_HOURS)


def spike_mask(s: pd.Series, day: pd.Series, floor: float) -> pd.Series:
    med = s.groupby(day).transform("median")
    prev_ = s.groupby(day).shift(1)
    next_ = s.groupby(day).shift(-1)
    isolated = (prev_.isna() | (prev_ < s / 2)) & (next_.isna() | (next_ < s / 2))
    return (s > floor) & (s > 4 * med) & isolated


def clean_station(path):
    st = path.stem
    df = pd.read_csv(path, low_memory=False)
    t = pd.to_datetime(df["From Date"], errors="coerce")
    colmap = parse_columns(df.columns)
    data, log = {}, []
    for var in dict.fromkeys(v for _, v, _, _ in colmap):
        parts = []
        for raw, v, fac, note in colmap:
            if v == var:
                parts.append(pd.to_numeric(df[raw], errors="coerce") * fac)
                log.append(dict(station=st, var=var, raw_col=raw, unit_handling=note))
        # duplicate columns (e.g. "RF (mm).1".."7", "AT"+"Temp"): take first non-null
        s = parts[0]
        for p in parts[1:]:
            s = s.combine_first(p)
        data[var] = s
    h = pd.DataFrame(data)
    h.insert(0, "time", t)
    h = h.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    # complete hourly grid so that shift()/rolling() mean "previous hour", not "previous row"
    full = pd.date_range(h["time"].min(), h["time"].max(), freq="h")
    h = h.set_index("time").reindex(full).rename_axis("time").reset_index()
    for v in POLLUTANTS + MET_ONSITE:
        if v not in h:
            h[v] = np.nan
    # BP sometimes reported in hPa/mbar -> convert to mmHg
    if h["BP"].median() > 850:
        h["BP"] = h["BP"] * 0.750062

    day = h["time"].dt.floor("D")
    qa = {}

    def drop(var, mask, rule):
        mask = mask.fillna(False) & h[var].notna()
        n = int(mask.sum())
        if n:
            h.loc[mask, var] = np.nan
        qa[(var, rule)] = qa.get((var, rule), 0) + n

    for v, (lo, hi) in VALID_RANGE.items():
        drop(v, (h[v] < lo) | (h[v] > hi), "out_of_range")
    drop("PM25", (h["PM10"] > 0) & (h["PM25"] > 1.5 * h["PM10"]), "pm25_gt_pm10")
    for v in AQI_POLLUTANTS + ["NO", "NOx"]:
        drop(v, stuck_mask(h[v], day), "stuck")
    for v, floor in SPIKE_FLOOR.items():
        drop(v, spike_mask(h[v], day, floor), "spike")

    qa_rows = [dict(station=st, var=v, rule=r, n_removed=n,
                    n_valid_raw=int(df.shape[0])) for (v, r), n in qa.items() if n]

    daily = aggregate_daily(h, day)
    daily.insert(0, "station", st)
    hourly = h[["time", "PM25", "PM10", "NO2", "O3", "CO"]].copy()
    hourly.insert(0, "station", st)
    hourly[["PM25", "PM10", "NO2", "O3", "CO"]] = hourly[["PM25", "PM10", "NO2", "O3", "CO"]].astype("float32")
    return daily, hourly, qa_rows, log


# ------------------------------------------------------------------ daily aggregation
def aggregate_daily(h: pd.DataFrame, day: pd.Series) -> pd.DataFrame:
    g = h.groupby(day)
    hour = h["time"].dt.hour
    out = pd.DataFrame(index=g.size().index)
    out.index.name = "date"
    for v in POLLUTANTS:
        n = g[v].count()
        m = g[v].mean().where(n >= MIN_HOURS_PER_DAY)
        out[f"{v}_mean"] = m
        out[f"{v}_nh"] = n                        # valid hours: data-quality / missingness signal
    for v in ["PM25", "PM10", "NO2", "CO"]:
        out[f"{v}_max"] = g[v].max().where(out[f"{v}_nh"] >= MIN_HOURS_PER_DAY)
        ev = h[v].where(hour >= 18).groupby(day).mean()   # evening build-up (18-23h)
        out[f"{v}_eve"] = ev
    # 8-hour rolling max within the day (CPCB rule for CO and O3)
    for v in ["CO", "O3"]:
        # windows ending at 07..23h lie fully inside the day (grid is complete hourly)
        r8 = h[v].rolling(8, min_periods=6).mean().where(hour >= 7)
        out[f"{v}_8h"] = r8.groupby(day).max().where(out[f"{v}_nh"] >= MIN_HOURS_PER_DAY)
    # on-site meteorology
    for v in ["WS", "RH", "AT", "SR", "BP"]:
        out[f"site_{v}"] = g[v].mean().where(g[v].count() >= 12)
    out["site_RF"] = g["RF"].sum(min_count=12)
    rad = np.deg2rad(h["WD"])
    u = (-np.sin(rad) * h["WS"].fillna(1)).groupby(day).mean()
    vv = (-np.cos(rad) * h["WS"].fillna(1)).groupby(day).mean()
    out["site_WD"] = (np.rad2deg(np.arctan2(-u, -vv)) % 360).where(g["WD"].count() >= 12)

    # ---- CPCB AQI
    conc = {"PM25": out["PM25_mean"], "PM10": out["PM10_mean"], "NO2": out["NO2_mean"],
            "SO2": out["SO2_mean"], "NH3": out["NH3_mean"], "CO": out["CO_8h"], "O3": out["O3_8h"]}
    subs = pd.DataFrame({f"SI_{p}": sub_index(conc[p], p) for p in AQI_POLLUTANTS})
    n_si = subs.notna().sum(axis=1)
    has_pm = subs[["SI_PM25", "SI_PM10"]].notna().any(axis=1)
    aqi = subs.max(axis=1).where((n_si >= 3) & has_pm)
    out = pd.concat([out, subs], axis=1)
    out["AQI"] = aqi
    out["AQI_n_sub"] = n_si
    out["AQI_driver"] = subs.fillna(-1).idxmax(axis=1).str.replace("SI_", "").where(aqi.notna())
    out["AQI_cat"] = aqi_category(aqi)
    return out.reset_index()


def sub_index(c: pd.Series, p: str) -> pd.Series:
    bp = AQI_BREAKS[p]
    vals = np.interp(c.clip(lower=0).to_numpy(dtype=float), bp, AQI_INDEX)  # caps at 500
    return pd.Series(vals, index=c.index).where(c.notna())


def aqi_category(aqi: pd.Series) -> pd.Series:
    return pd.cut(aqi, bins=AQI_CAT_EDGES, labels=AQI_CATEGORIES, include_lowest=True)


# ------------------------------------------------------------------ main
def main():
    files = sorted(RAW.glob("[A-Z][A-Z][0-9][0-9][0-9].csv"))
    res = Parallel(n_jobs=10, verbose=5)(delayed(clean_station)(f) for f in files)
    daily = pd.concat([r[0] for r in res], ignore_index=True)
    hourly = pd.concat([r[1] for r in res], ignore_index=True)
    qa = pd.DataFrame([row for r in res for row in r[2]])
    units = pd.DataFrame([row for r in res for row in r[3]])

    info = pd.read_csv(RAW / "stations_info.csv").rename(columns={"file_name": "station"})
    daily = daily.merge(info[["station", "state", "city"]], on="station", how="left")
    float_cols = daily.select_dtypes("float64").columns
    daily[float_cols] = daily[float_cols].astype("float32")
    daily["AQI_cat"] = daily["AQI_cat"].astype(str).replace("nan", None)

    daily.to_parquet(PROC / "daily.parquet", index=False)
    hourly.to_parquet(PROC / "hourly_clean.parquet", index=False)
    qa.to_csv(PROC / "qa_log.csv", index=False)
    units.to_csv(PROC / "unit_handling.csv", index=False)

    print("station-days:", len(daily), " with valid AQI:", int(daily["AQI"].notna().sum()))
    print("AQI category distribution:\n", daily["AQI_cat"].value_counts(normalize=True).round(3))
    print("QA removals by rule/var:\n", qa.groupby(["rule", "var"]).n_removed.sum().unstack(0).fillna(0).astype(int))
    print("unit handling:\n", units.unit_handling.value_counts())


if __name__ == "__main__":
    main()
