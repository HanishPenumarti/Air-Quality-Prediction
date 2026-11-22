"""Leakage audit: every feature group (and every excluded column) with the reason it is or is
not safe for a next-day (D+1) forecast issued at 24:00 of day D.
Output: reports/leakage_audit.csv"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC, REPORTS  # noqa: E402

AUDIT = [
    # group, used?, time reference, reasoning
    ("lag_aqi", "YES", "days D-29..D",
     "Past AQI only (lags 0-6, rolling windows ending at D). AQI(D) is complete at 24:00 of D. Gaps filled causally "
     "(same-day neighbours, ffill, train-period climatology)."),
    ("lag_poll", "YES", "days D-6..D",
     "Pollutant daily means/8-h maxima and intraday shape of day D. These are inputs to AQI, but of day D, not D+1, "
     "so they are legitimate lags."),
    ("missing", "YES", "day D",
     "Valid-hour counts and missing flags of day D; sensor gaps are informative (more gaps on polluted days)."),
    ("spatial", "YES", "day D",
     "Mean AQI/PM2.5 of OTHER stations in the same city/state on day D. Same-time readings of OTHER stations at "
     "D+1 are NOT used."),
    ("met_site", "YES", "day D",
     "OBSERVED weather from the station's own CPCB sensors (WS, RH, AT, SR, BP, RF, WD) for day D. Reported live, "
     "so available at issue time."),
    ("Weather of the target day D+1 (observed)", "EXCLUDED", "day D+1",
     "Observed future weather = leakage for a true forecast. We use observed weather up to D only (no weather "
     "forecast input)."),
    ("ERA5 reanalysis (any day)", "NOT USED", "-",
     "Published ~5 days after real time (ERA5T), so even day-D values would not exist at 24:00 of D."),
    ("fire", "YES", "days D-6..D", "NASA FIRMS MODIS detections up to day D (near real-time product exists)."),
    ("calendar", "YES", "day D+1", "Day of year/week, Diwali distance, lockdown flag of D+1: known in advance."),
    ("static", "YES", "-",
     "Coordinates, elevation, region; station-month AQI climatology computed from the TRAINING period only."),
    ("AQI(D+1), AQI_cat, AQI_driver, sub-indices SI_* of D+1", "EXCLUDED", "day D+1",
     "Deterministic function of the target day's pollutant concentrations: hands the model the answer."),
    ("Pollutant concentrations of D+1 (PM2.5, PM10, NO2, SO2, CO, O3, NH3)", "EXCLUDED", "day D+1",
     "Inputs of the target AQI: not available when the forecast is issued."),
    ("y_pm25 (PM2.5 of D+1)", "EXCLUDED", "day D+1", "Kept in the table for diagnostics only, never a feature."),
    ("Centred rolling windows / two-sided interpolation", "NOT USED", "-",
     "Would mix D+1.. values into day-D features. All windows are trailing; linear interpolation was evaluated only "
     "in the offline reconstruction test."),
    ("Sensor-fault QA rules", "SAFE", "same calendar day",
     "Stuck-sensor and spike checks use only data of the same day, so cleaning of day D never looks at D+1."),
    ("Imputer / scaler / PCA statistics", "SAFE", "training period",
     "Medians, climatology, StandardScaler and PCA are fitted on training rows only, then applied to val/test."),
]


def main():
    groups = json.loads((PROC / "feature_groups.json").read_text())
    rows = []
    for g, used, when, why in AUDIT:
        feats = groups.get(g, [])
        rows.append({"feature group / column": g, "used": used, "time of information": when,
                     "n features": len(feats) if feats else "-",
                     "examples": ", ".join(feats[:5]) + (" ..." if len(feats) > 5 else ""), "why": why})
    pd.DataFrame(rows).to_csv(REPORTS / "leakage_audit.csv", index=False)
    print(pd.DataFrame(rows)[["feature group / column", "used", "n features"]].to_string())


if __name__ == "__main__":
    main()
