"""Step 1 - clean raw CPCB hourly files and aggregate to station-day.

For every station file:
  0. validate timestamps: hourly cadence on the hour; anything sub-hourly / off the hour / duplicated
     is averaged into its clock hour (logged), never silently dropped
  1. harmonise column names and units to canonical CPCB units. Pollutant columns with a unit that
     has no verified conversion are REJECTED (logged), not assumed; a blank unit is assumed to be
     the canonical unit (logged). Met columns are often mislabelled on the portal (e.g. "RH (W/mt2)"),
     so their unit text is ignored and the physical-range check applies.
  2. sensor-fault QA. Every rule looks only at data of the SAME calendar day (or at one reading),
     so cleaning never uses later data:
       - impossible values  (negative, zero PM, outside physical range)
       - PM2.5 > 1.5 x PM10 (physically inconsistent; the PM2.5 BAM reading is removed - humidity
         artefacts of PM2.5 monitors are the usual cause)
       - stuck sensor       (>= 8 identical consecutive hourly readings ABOVE a low level)
       - isolated spike     (> 4x day median AND > floor AND both OBSERVED neighbours < half)
     Raw values are kept in a no-QA AQI (AQI_noqa) and per-day removal counts (qa_*_n) so the
     effect of QA on extreme days and Diwali can be measured.
  3. daily aggregation following CPCB rules (>= 16 valid hours, 8-h max for CO/O3)
  4. CPCB AQI sub-indices, AQI and category for every station-day (src/aqi.py)

Outputs (data/processed/):
  hourly_clean.parquet  - key hourly variables (for EDA: diurnal cycles etc.)
  daily.parquet         - one row per station-day
  qa_log.csv            - number of values removed by each rule, per station/variable
  unit_handling.csv     - unit decision for every raw column of every station
"""
import re
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from aqi import category_name, sub_index  # noqa: E402
from config import (AQI_POLLUTANTS, BP_HPA_MIN, MET_ONSITE, MIN_EVENING_HOURS, MIN_HOURS_PER_DAY,  # noqa: E402
                    MIN_MET_HOURS, POLLUTANTS, PROC, RAW, SPIKE_FLOOR, STATION_FILTER, STUCK_MIN_VALUE,
                    STUCK_RUN_HOURS, VALID_RANGE)

# ------------------------------------------------------------------ name / unit maps
NAME_MAP = {"PM2.5": "PM25", "PM10": "PM10", "NO2": "NO2", "SO2": "SO2", "CO": "CO",
            "Ozone": "O3", "NH3": "NH3", "NO": "NO", "NOx": "NOx", "Benzene": "Benzene",
            "Toluene": "Toluene", "WS": "WS", "WD": "WD", "RH": "RH", "AT": "AT",
            "Temp": "AT", "SR": "SR", "BP": "BP", "RF": "RF"}

# Canonical units: ug/m3 for all pollutants except CO (mg/m3) and NOx (ppb, as reported by CPCB).
# ppb -> ug/m3 factor = molar mass / 24.45 (25 degC, 1 atm); ppm = 1000 x ppb.
_PPB = {"NO2": 1.88, "SO2": 2.62, "O3": 1.96, "NH3": 0.70, "NO": 1.23, "Benzene": 3.19, "Toluene": 3.77}
UNIT_FACTOR = {
    ("CO", "mg/m3"): 1.0, ("CO", "mg/Nm3"): 1.0, ("CO", "ug/m3"): 1e-3, ("CO", "ng/m3"): 1e-6,
    ("CO", "ppm"): 1.145, ("CO", "ppb"): 1.145e-3,
    ("NOx", "ppb"): 1.0, ("NOx", "ppm"): 1000.0, ("NOx", "ug/m3"): 1 / 1.88,
}
for _v, _f in _PPB.items():
    UNIT_FACTOR.update({(_v, "ug/m3"): 1.0, (_v, "mg/m3"): 1000.0, (_v, "ng/m3"): 1e-3,
                        (_v, "ppb"): _f, (_v, "ppm"): 1000 * _f})
UNIT_FACTOR.update({("PM25", "ug/m3"): 1.0, ("PM10", "ug/m3"): 1.0})
MET_VARS = set(MET_ONSITE)
COL_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<unit>[^)]*)\)(?:\.\d+)?$")


def parse_columns(cols):
    """Map raw columns -> list of (raw_col, canonical_var, factor, unit, decision).

    decision: ok | converted | assumed (blank pollutant unit, or met column) | rejected (factor None)."""
    out = []
    for c in cols:
        m = COL_RE.match(c)
        if not m:
            continue
        name, unit = m["name"].strip(), m["unit"].strip()
        var = NAME_MAP.get(name)
        if var is None:
            continue
        if var in MET_VARS:
            # portal headers of met channels are frequently mislabelled; rely on the range checks
            out.append((c, var, 1.0, unit, "assumed"))
        elif (var, unit) in UNIT_FACTOR:
            f = UNIT_FACTOR[(var, unit)]
            out.append((c, var, f, unit, "ok" if f == 1 else "converted"))
        elif unit == "":
            out.append((c, var, 1.0, unit, "assumed"))
        else:
            out.append((c, var, None, unit, "rejected"))
    return out


# ------------------------------------------------------------------ QA rules
def stuck_mask(s: pd.Series, day: pd.Series, min_value: float = 0.0) -> pd.Series:
    """True where value belongs to a run of >= STUCK_RUN_HOURS identical readings within one day,
    and the stuck value is above `min_value` (flat runs at very low levels are plausible)."""
    grp = ((s != s.shift()) | (day != day.shift())).cumsum()
    run = s.groupby(grp).transform("size")
    return s.notna() & (run >= STUCK_RUN_HOURS) & (s > min_value)


def spike_mask(s: pd.Series, day: pd.Series, floor: float) -> pd.Series:
    """Isolated one-hour spike: both neighbouring hours OBSERVED (same day) and below half the value,
    value > 4x the day median and above an absolute floor. A missing neighbour is not evidence."""
    med = s.groupby(day).transform("median")
    prev_ = s.groupby(day).shift(1)
    next_ = s.groupby(day).shift(-1)
    isolated = prev_.notna() & next_.notna() & (prev_ < s / 2) & (next_ < s / 2)
    return (s > floor) & (s > 4 * med) & isolated


def to_hourly(t: pd.Series, h: pd.DataFrame):
    """Put readings on a complete hourly grid. Hourly-on-the-hour data passes through unchanged;
    sub-hourly, off-the-hour or duplicated timestamps are averaged into their clock hour."""
    h = h.assign(time=t).dropna(subset=["time"])
    if h.empty:
        return h, dict(n_rows=0, n_merged=0)
    n_off_hour = int((h["time"] != h["time"].dt.floor("h")).sum())
    h["time"] = h["time"].dt.floor("h")
    n_rows = len(h)
    h = h.groupby("time").mean(numeric_only=True)
    n_hours = len(h)
    full = pd.date_range(h.index.min(), h.index.max(), freq="h")
    h = h.reindex(full).rename_axis("time").reset_index()
    return h, dict(n_rows=n_rows, n_merged=n_rows - n_hours, n_off_hour=n_off_hour)


def wind_direction(h: pd.DataFrame, day: pd.Series) -> pd.Series:
    """Daily vector-mean wind direction (degrees, 'from'), speed-weighted, using only hours where
    BOTH speed and direction are valid; needs >= MIN_MET_HOURS such hours and non-zero mean wind."""
    ok = h["WS"].notna() & h["WD"].notna()
    rad = np.deg2rad(h["WD"].where(ok))
    ws = h["WS"].where(ok)
    u = (-np.sin(rad) * ws).groupby(day).mean()
    v = (-np.cos(rad) * ws).groupby(day).mean()
    n = ok.groupby(day).sum()
    wd = np.rad2deg(np.arctan2(-u, -v)) % 360
    return wd.where((n >= MIN_MET_HOURS) & (np.hypot(u, v) > 1e-6))


def clean_station(path):
    st = path.stem
    df = pd.read_csv(path, low_memory=False)
    t = pd.to_datetime(df["From Date"], errors="coerce")
    colmap = parse_columns(df.columns)
    data, log = {}, []
    for var in dict.fromkeys(v for _, v, _, _, _ in colmap):
        parts = []
        for raw, v, fac, unit, note in colmap:
            if v != var:
                continue
            x = pd.to_numeric(df[raw], errors="coerce")
            log.append(dict(station=st, var=var, raw_col=raw, unit=unit, factor=fac, unit_handling=note,
                            n_values=int(x.notna().sum())))
            if fac is not None:
                parts.append(x * fac)
        if parts:
            # duplicate columns (e.g. "RF (mm).1".."7", "AT"+"Temp"): take first non-null
            s = parts[0]
            for p in parts[1:]:
                s = s.combine_first(p)
            data[var] = s
    h, cad = to_hourly(t, pd.DataFrame(data, index=df.index))
    for v in POLLUTANTS + MET_ONSITE:
        if v not in h:
            h[v] = np.nan
    if h.empty:
        return None, None, [], log, dict(station=st, **cad)
    # BP: decided PER READING (hPa readings are > 850, mmHg readings < 800) -> no use of other days
    hpa = h["BP"] > BP_HPA_MIN
    h.loc[hpa, "BP"] = h.loc[hpa, "BP"] * 0.750062
    raw = h.copy()

    day = h["time"].dt.floor("D")
    qa, qa_day = {}, {}

    def drop(var, mask, rule):
        mask = mask.fillna(False) & h[var].notna()
        n = int(mask.sum())
        if n:
            h.loc[mask, var] = np.nan
            qa_day[rule] = qa_day.get(rule, 0) + mask.astype(int).groupby(day).sum()
        qa[(var, rule)] = qa.get((var, rule), 0) + n

    for v, (lo, hi) in VALID_RANGE.items():
        drop(v, (h[v] < lo) | (h[v] > hi), "out_of_range")
    drop("PM25", (h["PM10"] > 0) & (h["PM25"] > 1.5 * h["PM10"]), "pm25_gt_pm10")
    for v in AQI_POLLUTANTS + ["NO", "NOx"]:
        drop(v, stuck_mask(h[v], day, STUCK_MIN_VALUE.get(v, 0.0)), "stuck")
    for v, floor in SPIKE_FLOOR.items():
        drop(v, spike_mask(h[v], day, floor), "spike")

    qa_rows = [dict(station=st, var=v, rule=r, n_removed=n, n_raw_rows=int(df.shape[0]))
               for (v, r), n in qa.items() if n]

    daily = aggregate_daily(h, day)
    # no-QA AQI (only range-impossible values removed, so units stay sane) for the QA-impact check
    rq = raw.copy()
    for v, (lo, hi) in VALID_RANGE.items():
        rq.loc[(rq[v] < lo) | (rq[v] > hi), v] = np.nan
    daily["AQI_noqa"] = aggregate_daily(rq, day, aqi_only=True)["AQI"].values
    for rule in ["out_of_range", "pm25_gt_pm10", "stuck", "spike"]:
        cnt = qa_day.get(rule)
        daily[f"qa_{rule}_n"] = (cnt.reindex(daily["date"]).fillna(0).astype(int).values
                                 if cnt is not None else 0)
    daily.insert(0, "station", st)
    hourly = h[["time", "PM25", "PM10", "NO2", "O3", "CO"]].copy()
    hourly.insert(0, "station", st)
    hourly[["PM25", "PM10", "NO2", "O3", "CO"]] = hourly[["PM25", "PM10", "NO2", "O3", "CO"]].astype("float32")
    return daily, hourly, qa_rows, log, dict(station=st, **cad)


# ------------------------------------------------------------------ daily aggregation
def aggregate_daily(h: pd.DataFrame, day: pd.Series, aqi_only: bool = False) -> pd.DataFrame:
    g = h.groupby(day)
    hour = h["time"].dt.hour
    out = pd.DataFrame(index=g.size().index)
    out.index.name = "date"
    for v in (AQI_POLLUTANTS if aqi_only else POLLUTANTS):
        n = g[v].count()
        out[f"{v}_mean"] = g[v].mean().where(n >= MIN_HOURS_PER_DAY)
        out[f"{v}_nh"] = n                        # valid hours: data-quality / missingness signal
    # 8-hour rolling max within the day (CPCB rule for CO and O3). Windows ending at 07..23 h lie
    # inside the day (grid is complete hourly); each window needs >= 6 valid hours and the day needs
    # the CPCB 16 valid hours, so a daily MAXIMUM is not taken over a few hours (biased low).
    for v in ["CO", "O3"]:
        r8 = h[v].rolling(8, min_periods=6).mean().where(hour >= 7)
        out[f"{v}_8h"] = r8.groupby(day).max().where(out[f"{v}_nh"] >= MIN_HOURS_PER_DAY)
    if not aqi_only:
        for v in ["PM25", "PM10", "NO2", "CO"]:
            out[f"{v}_max"] = g[v].max().where(out[f"{v}_nh"] >= MIN_HOURS_PER_DAY)
            ev = h[v].where(hour >= 18)                       # evening build-up (18-23 h)
            out[f"{v}_eve"] = ev.groupby(day).mean().where(ev.groupby(day).count() >= MIN_EVENING_HOURS)
        for v in ["WS", "RH", "AT", "SR", "BP"]:
            out[f"site_{v}"] = g[v].mean().where(g[v].count() >= MIN_MET_HOURS)
        out["site_RF"] = g["RF"].sum(min_count=MIN_MET_HOURS)
        out["site_WD"] = wind_direction(h, day)

    # ---- CPCB AQI
    conc = {"PM25": out["PM25_mean"], "PM10": out["PM10_mean"], "NO2": out["NO2_mean"],
            "SO2": out["SO2_mean"], "NH3": out["NH3_mean"], "CO": out["CO_8h"], "O3": out["O3_8h"]}
    subs = pd.DataFrame({f"SI_{p}": sub_index(conc[p], p) for p in AQI_POLLUTANTS})
    n_si = subs.notna().sum(axis=1)
    has_pm = subs[["SI_PM25", "SI_PM10"]].notna().any(axis=1)
    aqi = subs.max(axis=1).where((n_si >= 3) & has_pm)
    if aqi_only:
        return pd.DataFrame({"AQI": aqi.values}, index=out.index).reset_index()
    out = pd.concat([out, subs], axis=1)
    out["AQI"] = aqi
    out["AQI_n_sub"] = n_si
    out["AQI_driver"] = subs.fillna(-1).idxmax(axis=1).str.replace("SI_", "").where(aqi.notna())
    out["AQI_cat"] = pd.Series(category_name(aqi.values), index=out.index).where(aqi.notna())
    return out.reset_index()


# ------------------------------------------------------------------ main
def station_files():
    files = sorted(RAW.glob("[A-Z][A-Z][0-9][0-9][0-9].csv"))
    if STATION_FILTER:
        files = [f for f in files if any(f.stem.startswith(p) for p in STATION_FILTER)]
    if not files:
        raise SystemExit(f"No station files found in {RAW} (filter: {STATION_FILTER or 'none'}). "
                         "Download the Kaggle dataset first (see README).")
    return files


def main():
    files = station_files()
    res = Parallel(n_jobs=min(10, len(files)), verbose=5)(delayed(clean_station)(f) for f in files)
    empty = [r[4]["station"] for r in res if r[0] is None]
    res_ok = [r for r in res if r[0] is not None]
    daily = pd.concat([r[0] for r in res_ok], ignore_index=True)
    hourly = pd.concat([r[1] for r in res_ok], ignore_index=True)
    qa = pd.DataFrame([row for r in res for row in r[2]],
                      columns=["station", "var", "rule", "n_removed", "n_raw_rows"])
    units = pd.DataFrame([row for r in res for row in r[3]])
    cadence = pd.DataFrame([r[4] for r in res])

    info = pd.read_csv(RAW / "stations_info.csv").rename(columns={"file_name": "station"})
    for c in ["state", "city"]:
        info[c] = info[c].astype(str).str.strip()
    daily = daily.merge(info[["station", "state", "city"]], on="station", how="left")
    float_cols = daily.select_dtypes("float64").columns
    daily[float_cols] = daily[float_cols].astype("float32")

    daily.to_parquet(PROC / "daily.parquet", index=False)
    hourly.to_parquet(PROC / "hourly_clean.parquet", index=False)
    qa.to_csv(PROC / "qa_log.csv", index=False)
    units.to_csv(PROC / "unit_handling.csv", index=False)
    cadence.to_csv(PROC / "cadence_log.csv", index=False)

    print("station-days:", len(daily), " with valid AQI:", int(daily["AQI"].notna().sum()))
    if empty:
        print("stations with no valid timestamps (skipped):", empty)
    print("readings averaged into a shared clock hour (sub-hourly/duplicate/off-hour):",
          int(cadence.n_merged.sum()))
    print("AQI category distribution:\n", daily["AQI_cat"].value_counts(normalize=True).round(3))
    if len(qa):
        print("QA removals by rule/var:\n",
              qa.groupby(["rule", "var"]).n_removed.sum().unstack(0).fillna(0).astype(int))
    print("unit handling:\n", units.unit_handling.value_counts())
    rej = units[units.unit_handling == "rejected"]
    if len(rej):
        print("REJECTED pollutant columns (unknown unit, values dropped):\n",
              rej.groupby(["var", "unit"]).agg(stations=("station", "nunique"), values=("n_values", "sum")))


if __name__ == "__main__":
    main()
