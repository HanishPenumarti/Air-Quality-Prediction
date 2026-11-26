"""Shared paths and constants for the air-quality forecasting project.

Environment overrides (optional):
  AQ_OUT       write all generated artefacts (processed data, models, reports, figures) under this
               folder instead of the project folders. Used for smoke tests / demo bundles so the
               main results are never overwritten.
  AQ_STATIONS  comma-separated station IDs or ID prefixes (e.g. "DL001,DL002,MH") to clean only a
               subset of stations (smoke test / small demo).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cpcb"
EXT = ROOT / "data" / "external"
_OUT = Path(os.environ["AQ_OUT"]).resolve() if os.environ.get("AQ_OUT") else None
PROC = (_OUT / "processed") if _OUT else ROOT / "data" / "processed"
MODELS = (_OUT / "models") if _OUT else ROOT / "models"
REPORTS = (_OUT / "reports") if _OUT else ROOT / "reports"
FIG = REPORTS / "figures"
for _p in (EXT, PROC, MODELS, FIG):
    _p.mkdir(parents=True, exist_ok=True)
STATION_FILTER = [s.strip() for s in os.environ.get("AQ_STATIONS", "").split(",") if s.strip()]

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
# Isolated-spike rule: value must exceed this floor AND 4x the day median AND BOTH observed
# neighbouring hours must be below half of it.
SPIKE_FLOOR = {"PM25": 150, "PM10": 250, "NO2": 100, "SO2": 100, "CO": 5, "O3": 100}
STUCK_RUN_HOURS = 8        # identical consecutive readings => stuck sensor ...
# ... but only above these low levels: long flat runs at/near the detection limit are plausible
# for clean air (or values rounded to an integer), so they are NOT removed.
STUCK_MIN_VALUE = {"PM25": 10, "PM10": 20, "NO2": 5, "SO2": 5, "CO": 0.2, "O3": 5, "NH3": 5,
                   "NO": 2, "NOx": 5}
MIN_HOURS_PER_DAY = 16     # CPCB rule for a valid 24-h average
MIN_EVENING_HOURS = 4      # of the 6 evening hours 18-23 h, for the evening-mean features
MIN_MET_HOURS = 12         # on-site meteorology daily values
BP_HPA_MIN = 850           # a pressure reading above this is in hPa (mmHg readings are < 800)

# ---------------------------------------------------------------- CPCB AQI
# National AQI bands (CPCB, "About National Air Quality Index"): for each pollutant the
# concentration band [c_lo, c_hi] maps linearly to the index band [i_lo, i_hi].
# Concentrations are rounded first (integer ug/m3; CO to 0.1 mg/m3) so the published integer
# bands are contiguous. The Severe band is open-ended in the CPCB table; its upper
# concentration (last c_hi) is a PROJECT CONVENTION, and values above it are capped at 500.
AQI_INDEX_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
AQI_CONC_BANDS = {
    "PM25": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 380)],
    "PM10": [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 510)],
    "NO2": [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 520)],
    "SO2": [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 2400)],
    "CO": [(0, 1.0), (1.1, 2.0), (2.1, 10), (10.1, 17), (17.1, 34), (34.1, 51)],
    "O3": [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 1000)],
    "NH3": [(0, 200), (201, 400), (401, 800), (801, 1200), (1201, 1800), (1801, 2400)],
}
AQI_ROUND = {"CO": 1}      # decimals; every other pollutant is rounded to an integer
AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_CAT_EDGES = [0, 50, 100, 200, 300, 400, 10_000]   # (lo, hi] per category
POOR_THRESHOLD = 200       # target class: AQI > 200  ->  "Poor or worse"
SEVERE_THRESHOLD = 400     # AQI > 400 -> Severe

# ---------------------------------------------------------------- regions
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


def city_key(state, city):
    """Cities are identified by (state, city): e.g. 'Aurangabad' exists in Bihar AND Maharashtra."""
    return f"{str(state).strip()}|{str(city).strip()}"


# ---------------------------------------------------------------- splits
TRAIN_END = "2020-12-31"
VAL_END = "2021-12-31"     # test = 2022-01-01 .. 2023-03-31
TUNE_FOLDS = [("2019-12-31", "2020-12-31"), (TRAIN_END, VAL_END)]   # walk-forward tuning
BACKTEST_YEARS = [2019, 2020, 2021, 2022]   # rolling origin: train <= Y-1, test year Y (2022 -> Mar 2023)
SEED = 42
PERSIST_MAX_AGE = 7        # persistence baseline: last MEASURED AQI at most this many days old

# ---------------------------------------------------------------- external data timing
FIRMS_LATENCY_H = 3        # NRT fire detections become available ~3 h after the overpass
