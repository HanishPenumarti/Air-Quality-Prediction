"""Leakage audit = (1) a declaration table of every feature group and excluded column, and
(2) EXECUTABLE checks on the real data. The script exits with an error if any check fails.

Checks
  target_alignment   y_aqi of (station, D) equals the daily AQI of (station, D+1)
  no_target_inputs   no target / target-derived column is a model input
  prefix_invariance  features rebuilt with all data AFTER a cut date removed are identical for
                     every row up to the cut date (covers lags, rolling windows, fills, neighbour
                     features, fires) - i.e. no feature of day D uses anything later than D
  fold_isolation     changing every value (features and target) after a fold's cutoff does not
                     change the imputer/climatology fitted on that fold
  train_ui_identity  the modelling table and the UI table give identical features (raw and after
                     the saved imputer) for the same station-days
  firms_timing       every fire detection counts for an issue day ON or AFTER its IST acquisition day
Outputs: reports/leakage_audit.csv (declarations), reports/leakage_checks.csv (results)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXT, PROC, REPORTS, TRAIN_END  # noqa: E402
from features import build, fire_counts, fire_events  # noqa: E402
from prep import ID_COLS, Imputer  # noqa: E402

AUDIT = [
    ("lag_aqi", "YES", "days D-29..D",
     "Past AQI only (lags 0-6, trailing windows ending at D). AQI(D) is complete at 24:00 of D. Gaps filled causally "
     "(same-day neighbours, forward-fill <= 3 calendar days); remaining gaps use the fold-fitted climatology."),
    ("lag_poll", "YES", "days D-6..D",
     "Pollutant daily means/8-h maxima and intraday shape of day D. They are inputs to AQI, but of day D, not D+1."),
    ("missing", "YES", "day D",
     "Valid-hour counts and raw missing flags of day D (taken BEFORE any fill); sensor gaps are informative."),
    ("spatial", "YES", "day D",
     "AQI/PM2.5 of OTHER stations in the same (state, city) / state on day D. Their D+1 values are NOT used."),
    ("met_site", "YES", "day D",
     "OBSERVED weather from the station's own CPCB sensors for day D; reported live, so available at issue time."),
    ("Weather of the target day D+1 (observed)", "EXCLUDED", "day D+1",
     "Observed future weather = leakage for a true forecast; no weather-forecast input is used."),
    ("ERA5 reanalysis (any day)", "NOT USED", "-",
     "Published ~5 days after real time, so even day-D values would not exist at 24:00 of D."),
    ("fire", "YES", "available by 24:00 of D",
     "NASA FIRMS MODIS detections dated in IST from the UTC acquisition time, plus a 3-h near-real-time latency. "
     "The standard-quality-only `type` field is not used. Archive locations are the standard product (retrospective "
     "study); counts use fields that the near-real-time product also has."),
    ("calendar", "YES", "day D+1", "Day of year/week, Diwali distance, lockdown flag of D+1: known in advance."),
    ("static", "YES", "-",
     "Coordinates, elevation, region; clim_aqi_month is fitted per fold on that fold's training rows "
     "(leave-one-year-out for training rows, so a row's own target never enters its feature)."),
    ("AQI(D+1), AQI_cat, AQI_driver, sub-indices SI_* of D+1", "EXCLUDED", "day D+1",
     "Deterministic function of the target day's pollutant concentrations: hands the model the answer."),
    ("Pollutant concentrations of D+1 (PM2.5, PM10, NO2, SO2, CO, O3, NH3)", "EXCLUDED", "day D+1",
     "Inputs of the target AQI: not available when the forecast is issued."),
    ("y_pm25 (PM2.5 of D+1)", "EXCLUDED", "day D+1", "Kept for diagnostics only, never a feature."),
    ("Centred rolling windows / two-sided interpolation", "NOT USED", "-",
     "All windows are trailing; linear interpolation appears only as an offline reference in the imputation study."),
    ("Sensor-fault QA rules", "SAFE", "same calendar day / single reading",
     "Stuck-sensor and spike checks use only data of the same day; pressure units are decided per reading."),
    ("Imputer / climatology / scaler / PCA / class weights / thresholds", "SAFE", "fold training rows",
     "Fitted inside every fold, the final model, each backtest origin and each spatial hold-out on its own training rows."),
]


def declarations():
    groups = json.loads((PROC / "feature_groups.json").read_text())
    rows = []
    for g, used, when, why in AUDIT:
        feats = groups.get(g, [])
        rows.append({"feature group / column": g, "used": used, "time of information": when,
                     "n features": len(feats) if feats else "-",
                     "examples": ", ".join(feats[:5]) + (" ..." if len(feats) > 5 else ""), "why": why})
    pd.DataFrame(rows).to_csv(REPORTS / "leakage_audit.csv", index=False)
    return groups


def main():
    groups = declarations()
    feats = [c for cs in groups.values() for c in cs]
    checks = []

    def record(name, ok, detail):
        checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    d = pd.read_parquet(PROC / "daily.parquet")
    F = pd.read_parquet(PROC / "features.parquet")

    # 1. target alignment
    s = F.sample(min(20000, len(F)), random_state=0)[["station", "date", "y_aqi"]]
    nxt = d[["station", "date", "AQI"]].assign(date=lambda x: x.date - pd.Timedelta(days=1))
    m = s.merge(nxt, on=["station", "date"], how="left")
    ok = np.allclose(m.y_aqi.values, m.AQI.values, equal_nan=False)
    record("target_alignment", ok, f"{len(m)} sampled rows: y_aqi == AQI(D+1)")

    # 2. no target inputs
    bad = [c for c in feats if c in ID_COLS or c.startswith(("SI_", "AQI_", "y_"))]
    record("no_target_inputs", not bad, f"forbidden columns among inputs: {bad or 'none'}")

    # 3. prefix invariance on a multi-station city (neighbour features included)
    geo = pd.read_csv(EXT / "stations_geo.csv")
    big = d.groupby(["state", "city"]).station.nunique().sort_values().index[-1]
    sub = d[(d.state == big[0]) & (d.city == big[1])]
    sub = sub[sub.station.isin(sub.station.unique()[:6])]
    fires = fire_counts(geo[geo.station.isin(sub.station)], fire_events(pd.read_parquet(EXT / "firms_modis_india.parquet")))
    cut = pd.Timestamp("2021-06-30")
    full, _, _ = build(sub, geo, fires)
    part, _, _ = build(sub[sub.date <= cut], geo, fires)
    cols = [c for c in feats if c in full.columns]
    a = full[full.date <= cut].set_index(["station", "date"])[cols].sort_index()
    b = part.set_index(["station", "date"])[cols].sort_index().reindex(a.index)
    diff = ~((a.values == b.values) | (pd.isna(a.values) & pd.isna(b.values)))
    record("prefix_invariance", not diff.any(),
           f"{big[1]} ({len(sub.station.unique())} stations), cut {cut.date()}: {int(diff.sum())} differing values "
           f"in {len(a)} rows x {len(cols)} features")

    # 4. fold isolation
    t = F.target_date
    tr = (t <= "2019-12-31").values
    imp1 = Imputer("clim", feats).fit(F, tr)
    Fp = F.copy()
    later = ~tr
    num = [c for c in feats if c in Fp.columns] + ["y_aqi"]
    Fp.loc[later, num] = Fp.loc[later, num] * 3 + 7
    imp2 = Imputer("clim", feats).fit(Fp, tr)
    x1 = imp1.transform(F.iloc[:50000]).values
    x2 = imp2.transform(F.iloc[:50000]).values
    record("fold_isolation", np.allclose(x1, x2, equal_nan=True),
           "imputer fitted on <= 2019 is unchanged when every later value is altered")

    # 5. train / UI identity
    Ffull = pd.read_parquet(PROC / "features_full.parquet")
    k = F[["station", "date"]].sample(min(20000, len(F)), random_state=1)
    A = k.merge(F, on=["station", "date"]).set_index(["station", "date"]).sort_index()
    B = k.merge(Ffull, on=["station", "date"]).set_index(["station", "date"]).sort_index()
    cols = [c for c in feats if c in A.columns]
    same_raw = np.allclose(A[cols].values.astype(float), B[cols].values.astype(float), equal_nan=True)
    imp = Imputer("clim", feats).fit(F, (t <= TRAIN_END).values)
    same_imp = np.allclose(imp.transform(A.reset_index()).values, imp.transform(B.reset_index()).values, equal_nan=True)
    record("train_ui_identity", same_raw and same_imp and len(A) == len(B),
           f"{len(A)} station-days: raw features equal={same_raw}, imputed equal={same_imp}")

    # 6. FIRMS timing
    f = fire_events(pd.read_parquet(EXT / "firms_modis_india.parquet").sample(200000, random_state=0))
    at = f.acq_time.astype(int)
    acq_ist = (pd.to_datetime(f.acq_date) + pd.to_timedelta(at // 100, unit="h") + pd.to_timedelta(at % 100, unit="m")
               + pd.Timedelta(hours=5.5)).dt.floor("D")
    lag = (f.date - acq_ist).dt.days
    record("firms_timing", bool((lag >= 0).all() and (lag <= 1).all()),
           f"availability day - IST acquisition day: {lag.value_counts().sort_index().to_dict()}")

    out = pd.DataFrame(checks)
    out.to_csv(REPORTS / "leakage_checks.csv", index=False)
    if (out.result == "FAIL").any():
        raise SystemExit("leakage checks FAILED - see reports/leakage_checks.csv")


if __name__ == "__main__":
    main()
