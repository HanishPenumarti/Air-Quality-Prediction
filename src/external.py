"""Step 2 - fetch external context data.

  * station coordinates
      1. city centre for every (state, city): Open-Meteo geocoding API (GeoNames), matched on state
      2. STATION-level coordinates from OpenAQ (v3 API, which republishes CPCB stations): each CPCB
         station is matched by name within 60 km of its city centre. Stations that do not match
         cleanly keep the city centre and are flagged (coord_source = "city_centroid").
         Needs a free OpenAQ API key: env var OPENAQ_API_KEY or a file `openaq_key.txt` in the
         project root (git-ignored). Without a key, step 2 is skipped (city-level coordinates).
  * NASA FIRMS fires     : MODIS (Collection 6.1) yearly India archive CSVs, 2010-2023, cached per
                           year with retries; checksums + retrieval time in a manifest
  * festival calendar    : Diwali dates (firecracker spikes) - fixed lunar-calendar dates

Usage: python src/external.py [geo] [openaq] [firms]      (default: all three)
Outputs in data/external/: stations_geo.csv, openaq_locations.csv, firms_modis_india.parquet,
                           firms/<year>.csv, download_manifest.json
"""
import difflib
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import EXT, RAW, ROOT  # noqa: E402

# GeoNames spellings / nearest town for CPCB city names that don't geocode directly
CITY_ALIASES = {
    "Rajamahendravaram": "Rajahmundry", "Bihar Sharif": "Bihar", "Gurugram": "Gurgaon",
    "Kalaburagi": "Gulbarga", "Belgaum": "Belagavi", "Mysuru": "Mysore", "Vijayapura": "Bijapur",
    "Shivamogga": "Shimoga", "Tumakuru": "Tumkur", "Chikkamagaluru": "Chikmagalur",
    "Hubballi": "Hubli", "Prayagraj": "Allahabad", "Manguraha": "Bettiah",
    "Kunjemura": "Raigarh", "Tumidih": "Raigarh", "Milupara": "Raigarh", "Chhal": "Raigarh",
    "Suakati": "Keonjhar", "Tensa": "Barbil", "Bileipada": "Barbil", "Palkalaiperur": "Perambalur",
    "Byrnihat": "Nongpoh", "Mandikhera": "Nuh", "Jorapokhar": "Jharia",
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
STATE_ALIASES = {"Delhi": "delhi", "Jammu and Kashmir": "Kashmir", "Odisha": "Odisha",
                 "Puducherry": "Puducherry"}

DIWALI = ["2010-11-05", "2011-10-26", "2012-11-13", "2013-11-03", "2014-10-23", "2015-11-11",
          "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14", "2021-11-04",
          "2022-10-24", "2023-11-12"]

_CACHE_F = EXT / "geocode_cache.json"
_CACHE = json.loads(_CACHE_F.read_text()) if _CACHE_F.exists() else {}
MANIFEST_F = EXT / "download_manifest.json"
UA = {"User-Agent": "air-quality-coursework/1.0"}


def _manifest_update(key, entry):
    m = json.loads(MANIFEST_F.read_text()) if MANIFEST_F.exists() else {}
    m[key] = entry
    MANIFEST_F.write_text(json.dumps(m, indent=1))


def _get(url, retries=6, timeout=120, headers=None, **kw):
    hdr = {**UA, **(headers or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=hdr, **kw)
            if r.status_code == 429:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"download failed: {url}: {e}") from e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed (rate limited): {url}")


# ------------------------------------------------------------------ 1. city centres
def _search(name):
    if name in _CACHE:
        return _CACHE[name]
    r = _get("https://geocoding-api.open-meteo.com/v1/search",
             params=dict(name=name, count=10, countryCode="IN", language="en"), timeout=30)
    res = r.json().get("results", [])
    _CACHE[name] = res
    _CACHE_F.write_text(json.dumps(_CACHE))
    return res


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
        m = geo.city == c
        geo.loc[m, ["lat", "lon", "geo_name", "geo_ok"]] = [la, lo, c + " (manual)", True]
    return geo


def station_info():
    info = pd.read_csv(RAW / "stations_info.csv").rename(columns={"file_name": "station"})
    for c in ["state", "city", "station_location"]:
        info[c] = info[c].astype(str).str.strip()
    return info


def stations_geo():
    info = station_info()
    cities = info[["city", "state"]].drop_duplicates()
    rows = []
    for c, s in cities.itertuples(index=False):
        rows.append(geocode(c, s))
        time.sleep(0.5)
    geo = apply_manual(pd.DataFrame(rows))
    bad = geo[~geo.geo_ok.astype(bool)]
    if len(bad):
        print("cities NOT matched within their state (check these):\n",
              bad[["city", "state", "geo_name", "geo_admin1"]].to_string())
    out = info.merge(geo, on=["city", "state"], how="left", validate="many_to_one")
    assert len(out) == len(info), "geocode merge changed the number of stations"
    out = out.rename(columns={"lat": "city_lat", "lon": "city_lon"})
    out["lat"], out["lon"] = out.city_lat, out.city_lon
    out["coord_source"] = np.where(out.lat.notna(), "city_centroid", "unknown")
    _validate(out)
    out.to_csv(EXT / "stations_geo.csv", index=False)
    print(f"geocoded {len(geo)} cities ((state, city) pairs) for {len(out)} stations; "
          f"not matched within state: {len(bad)}")
    return out


def _validate(geo):
    ok = geo.lat.between(6, 37.5) & geo.lon.between(68, 97.5)
    if (~ok & geo.lat.notna()).any():
        raise ValueError("coordinates outside India:\n" + geo[~ok & geo.lat.notna()].to_string())


# ------------------------------------------------------------------ 2. station coordinates (OpenAQ)
def _openaq_key():
    k = os.environ.get("OPENAQ_API_KEY")
    f = ROOT / "openaq_key.txt"
    if not k and f.exists():
        k = f.read_text().strip()
    return k


def _norm(s):
    s = str(s).lower()
    s = re.split(r"\s+-\s+", s)[0]                 # drop agency suffix: "Anand Vihar, Delhi - DPCC"
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def _haversine(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def fetch_openaq(key):
    """All OpenAQ v3 locations in India -> data/external/openaq_locations.csv"""
    h = {"X-API-Key": key}
    base = "https://api.openaq.org/v3"
    params = dict(limit=1000, iso="IN")
    try:
        _get(f"{base}/locations", params={**params, "page": 1, "limit": 1}, headers=h, timeout=60)
    except RuntimeError:      # older deployments: filter by numeric country id instead of iso
        cs = _get(f"{base}/countries", params=dict(limit=300), headers=h, timeout=60).json()["results"]
        params = dict(limit=1000, countries_id=[c["id"] for c in cs if c.get("code") == "IN"][0])
    rows, page = [], 1
    while True:
        js = _get(f"{base}/locations", params={**params, "page": page}, headers=h, timeout=120).json()
        res = js.get("results", [])
        for r in res:
            c = r.get("coordinates") or {}
            if c.get("latitude") is None:
                continue
            rows.append(dict(openaq_id=r.get("id"), name=r.get("name"), locality=r.get("locality"),
                             provider=(r.get("provider") or {}).get("name"),
                             lat=c["latitude"], lon=c["longitude"], is_mobile=r.get("isMobile")))
        if len(res) < params["limit"]:
            break
        page += 1
        time.sleep(1.1)       # free tier: 60 requests / minute
    loc = pd.DataFrame(rows)
    loc.to_csv(EXT / "openaq_locations.csv", index=False)
    _manifest_update("openaq_locations", dict(url=f"{base}/locations", params=str(params), n=len(loc),
                                              retrieved=datetime.now(timezone.utc).isoformat()))
    print(f"OpenAQ: {len(loc)} locations in India")
    return loc


def match_openaq(geo, loc, max_km=60, min_score=0.72):
    """Match each CPCB station to an OpenAQ location by name, among candidates within `max_km`
    of its city centre. Unmatched stations keep the city centre (flagged)."""
    loc = loc[~loc.is_mobile.fillna(False).astype(bool)].reset_index(drop=True)
    names = loc.name.map(_norm).values
    out = geo.copy()
    out["lat"], out["lon"] = out.city_lat, out.city_lon          # start from the city centre
    out["coord_source"] = np.where(out.lat.notna(), "city_centroid", "unknown")
    for c in ["openaq_name", "match_score", "match_km"]:
        out[c] = np.nan if c != "openaq_name" else None
    for i, r in out.iterrows():
        if pd.isna(r.city_lat):
            continue
        dist = _haversine(r.city_lat, r.city_lon, loc.lat.values, loc.lon.values)
        cand = np.flatnonzero(dist <= max_km)
        if not len(cand):
            continue
        target = _norm(r.station_location)
        scores = np.array([difflib.SequenceMatcher(None, target, names[j]).ratio() for j in cand])
        j = cand[int(np.argmax(scores))]
        best = float(scores.max())
        if best >= min_score:
            out.loc[i, ["lat", "lon"]] = [loc.lat[j], loc.lon[j]]
            out.loc[i, ["openaq_name", "match_score", "match_km"]] = [loc.name[j], best, float(dist[j])]
            out.loc[i, "coord_source"] = "openaq_station"
    # two CPCB stations matched to one OpenAQ location -> ambiguous, revert both to city centre
    dup = out.openaq_name.notna() & out.duplicated(subset=["openaq_name", "lat", "lon"], keep=False)
    out.loc[dup, ["lat", "lon"]] = out.loc[dup, ["city_lat", "city_lon"]].values
    out.loc[dup, "coord_source"] = "city_centroid (ambiguous match)"
    return out


def station_coords():
    key = _openaq_key()
    geo = pd.read_csv(EXT / "stations_geo.csv")
    if not key:
        print("No OpenAQ API key (env OPENAQ_API_KEY or openaq_key.txt) - keeping city-level coordinates.")
        return geo
    if "city_lat" not in geo:            # file written by an older version of this script
        geo["city_lat"], geo["city_lon"] = geo.lat, geo.lon
    loc = fetch_openaq(key)
    geo = match_openaq(geo, loc)
    _validate(geo)
    geo.to_csv(EXT / "stations_geo.csv", index=False)
    print("coordinate source:\n", geo.coord_source.value_counts().to_string())
    print("check low-score matches if any:\n",
          geo[geo.coord_source == "openaq_station"].nsmallest(10, "match_score")
          [["station", "station_location", "openaq_name", "match_score", "match_km"]].to_string())
    return geo


# ------------------------------------------------------------------ 3. NASA FIRMS
FIRMS_COLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence",
              "version", "frp", "daynight", "type"]


def firms():
    folder = EXT / "firms"
    folder.mkdir(exist_ok=True)
    frames = []
    for y in range(2010, 2024):
        f = folder / f"modis_{y}_India.csv"
        url = f"https://firms.modaps.eosdis.nasa.gov/data/country/modis/{y}/modis_{y}_India.csv"
        if not f.exists():
            r = _get(url, timeout=300)
            f.write_bytes(r.content)
            _manifest_update(f"firms_{y}", dict(url=url, sha256=hashlib.sha256(r.content).hexdigest(),
                                                bytes=len(r.content),
                                                retrieved=datetime.now(timezone.utc).isoformat()))
        df = pd.read_csv(f)
        miss = set(["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp"]) - set(df.columns)
        if miss:
            raise ValueError(f"{f.name}: missing columns {miss}")
        frames.append(df[[c for c in FIRMS_COLS if c in df.columns]])
        print(y, len(df))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(EXT / "firms_modis_india.parquet", index=False)
    print("FIRMS detections:", len(out))


if __name__ == "__main__":
    what = sys.argv[1:] or ["geo", "openaq", "firms"]
    if "geo" in what:
        stations_geo()
    if "openaq" in what:
        station_coords()
    if "firms" in what:
        firms()
