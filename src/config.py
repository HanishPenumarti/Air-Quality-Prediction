"""Shared paths and constants for the air-quality forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cpcb"
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
for _p in (EXT, PROC, MODELS, FIG):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- variables
# Pollutants kept (canonical CPCB units): ug/m3 except CO (mg/m3) and NOx (ppb)
POLLUTANTS = ["PM25", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "NO", "NOx", "Benzene", "Toluene"]
AQI_POLLUTANTS = ["PM25", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
MET_ONSITE = ["WS", "WD", "RH", "AT", "SR", "BP", "RF"]

# Physically plausible hourly ranges (after unit harmonisation). Outside -> NaN.
VALID_RANGE = {
    "PM25": (0.01, 1500), "PM10": (0.01, 2000), "NO2": (0, 1000), "SO2": (0, 1000),
    "CO": (0, 50), "O3": (0, 1000), "NH3": (0, 2000), "NO": (0, 1500), "NOx": (0, 2000),
    "Benzene": (0, 500), "Toluene": (0, 1000),
    "WS": (0, 40), "WD": (0, 360), "RH": (0, 100), "AT": (-30, 55), "SR": (0, 1500),
    "BP": (500, 800), "RF": (0, 200),
}
# Absolute floor for the isolated-spike rule (value must exceed this AND 4x day median)
SPIKE_FLOOR = {"PM25": 150, "PM10": 250, "NO2": 100, "SO2": 100, "CO": 5, "O3": 100}
STUCK_RUN_HOURS = 8        # identical consecutive readings => stuck sensor
MIN_HOURS_PER_DAY = 16     # CPCB rule for a valid 24-h average

# ---------------------------------------------------------------- CPCB AQI
# Breakpoints (concentration) mapped to index [0,50,100,200,300,400,500].
# 24-h averages for PM2.5, PM10, NO2, SO2, NH3; 8-h max for CO (mg/m3) and O3.
AQI_INDEX = [0, 50, 100, 200, 300, 400, 500]
AQI_BREAKS = {
    "PM25": [0, 30, 60, 90, 120, 250, 380],
    "PM10": [0, 50, 100, 250, 350, 430, 510],
    "NO2": [0, 40, 80, 180, 280, 400, 520],
    "SO2": [0, 40, 80, 380, 800, 1600, 2400],
    "CO": [0, 1.0, 2.0, 10, 17, 34, 51],
    "O3": [0, 50, 100, 168, 208, 748, 1000],
    "NH3": [0, 200, 400, 800, 1200, 1800, 2400],
}
AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_CAT_EDGES = [0, 50, 100, 200, 300, 400, 10_000]   # (lo, hi] per category
POOR_THRESHOLD = 200       # target class: AQI > 200  ->  "Poor or worse"

# ---------------------------------------------------------------- splits
TRAIN_END = "2020-12-31"
VAL_END = "2021-12-31"     # test = 2022-01-01 .. 2023-03-31
SEED = 42
