"""Step 2 - fetch external context data.

  * station coordinates  : Open-Meteo geocoding API (GeoNames), city-level, matched on state
  * NASA FIRMS fires     : MODIS (Collection 6.1) yearly India archive CSVs, 2010-2023
  * festival calendar    : Diwali dates (firecracker spikes) - fixed lunar-calendar dates

Outputs in data/external/: stations_geo.csv, firms_modis_india.parquet
"""
import io
import json
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import EXT, RAW

# GeoNames spellings / nearest town for CPCB city names that don't geocode directly
CITY_ALIASES = {
    "Rajamahendravaram": "Rajahmundry", "Bihar Sharif": "Bihar", "Gurugram": "Gurgaon",
    "Kalaburagi": "Gulbarga", "Belgaum": "Belagavi", "Mysuru": "Mysore", "Vijayapura": "Bijapur",
    "Shivamogga": "Shimoga", "Tumakuru": "Tumkur", "Chikkamagaluru": "Chikmagalur",
    "Hubballi": "Hubli", "Prayagraj": "Allahabad", "Manguraha": "Bettiah",
    "Kunjemura": "Raigarh", "Tumidih": "Raigarh", "Milupara": "Raigarh", "Chhal": "Raigarh",
    "Suakati": "Keonjhar", "Tensa": "Barbil", "Bileipada": "Barbil", "Palkalaiperur": "Perambalur",
    "Byrnihat": "Nongpoh", "Mandikhera": "Nuh", "Kunjemura ": "Raigarh", "Jorapokhar": "Jharia",
    "Nandesari": "Vadodara", "Vatva": "Ahmedabad", "Eloor": "Kochi", "Chengalpattu": "Chengalpattu",
    "Ooty": "Ooty", "Sri Ganganagar": "Ganganagar", "Mandideep": "Bhopal", "Pithampur": "Indore",
}
# Verified coordinates for towns the geocoder cannot resolve unambiguously (lat, lon)
MANUAL_COORDS = {
    "Sivasagar": (26.98, 94.64), "Byrnihat": (26.05, 91.87), "Chhapra": (25.78, 84.73),
    "Gurugram": (28.46, 77.03), "Chikkaballapur": (13.43, 77.73), "Kalaburagi": (17.33, 76.83),
    "Ramanagara": (12.72, 77.28), "Shivamogga": (13.93, 75.57), "Davanagere": (14.46, 75.92),
    "Mandi Gobindgarh": (30.67, 76.30), "Sri Ganganagar": (29.90, 73.88),
    "Gummidipoondi": (13.41, 80.11), "Prayagraj": (25.44, 81.85), "Delhi": (28.63, 77.22),
}
STATE_ALIASES = {"Delhi": "delhi","Jammu and Kashmir": "Kashmir", "Odisha": "Odisha",
                 "Puducherry": "Puducherry"}

DIWALI = ["2010-11-05", "2011-10-26", "2012-11-13", "2013-11-03", "2014-10-23", "2015-11-11",
          "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14", "2021-11-04",
          "2022-10-24", "2023-11-12"]


_CACHE_F = EXT / "geocode_cache.json"
_CACHE = json.loads(_CACHE_F.read_text()) if _CACHE_F.exists() else {}


def _search(name):
    if name in _CACHE:
        return _CACHE[name]
    for attempt in range(6):
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params=dict(name=name, count=10, countryCode="IN", language="en"), timeout=30)
            r.raise_for_status()
            res = r.json().get("results", [])
            _CACHE[name] = res
            _CACHE_F.write_text(json.dumps(_CACHE))
            return res
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"geocoding failed for {name}")


def geocode(city, state):
    name = CITY_ALIASES.get(city, city)
    res = [] if city in MANUAL_COORDS else _search(name)
    st_key = STATE_ALIASES.get(state, state).lower()
    same_state = [x for x in res if st_key in str(x.get("admin1", "")).lower()]
    pick = (same_state or res or [None])[0]
    if pick is None:
        return dict(city=city, state=state, lat=None, lon=None, elevation=None, geo_name=None, geo_ok=False)
    return dict(city=city, state=state, lat=pick["latitude"], lon=pick["longitude"],
                elevation=pick.get("elevation"), geo_name=pick["name"],
                geo_admin1=pick.get("admin1"), geo_ok=bool(same_state))


def apply_manual(geo):
    for c, (la, lo) in MANUAL_COORDS.items():
        m = geo.city.str.strip() == c
        geo.loc[m, ["lat", "lon", "geo_name", "geo_ok"]] = [la, lo, c + " (manual)", True]
    return geo


def stations_geo():
    info = pd.read_csv(RAW / "stations_info.csv")
    cities = info[["city", "state"]].drop_duplicates()
    rows = []
    for c, s in cities.itertuples(index=False):
        rows.append(geocode(c.strip(), s.strip()))
        time.sleep(0.5)
    geo = pd.DataFrame(rows)
    geo = apply_manual(geo)
    out = info.rename(columns={"file_name": "station"}).merge(geo, on=["city", "state"], how="left")
    out.to_csv(EXT / "stations_geo.csv", index=False)
    bad = geo[~geo.geo_ok]
    print(f"geocoded {len(geo)} cities; not matched within state: {len(bad)}")
    if len(bad):
        print(bad[["city", "state", "geo_name", "geo_admin1"]].to_string())
    return out


def firms():
    frames = []
    for y in range(2010, 2024):
        url = f"https://firms.modaps.eosdis.nasa.gov/data/country/modis/{y}/modis_{y}_India.csv"
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        frames.append(df)
        print(y, len(df))
    f = pd.concat(frames, ignore_index=True)
    f = f[["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "daynight"]]
    f.to_parquet(EXT / "firms_modis_india.parquet", index=False)
    print("FIRMS detections:", len(f))


if __name__ == "__main__":
    what = sys.argv[1:] or ["geo", "firms"]
    if "geo" in what:
        stations_geo()
    if "firms" in what:
        firms()
