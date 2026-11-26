## Data sources

| Source | What | Provider / origin | Access | Licence & caveats |
|---|---|---|---|---|
| **CPCB CCR hourly monitoring** (primary) | 453 CAAQMS stations, 242 cities ((state, city) pairs; 241 distinct names because "Aurangabad" is in both Bihar and Maharashtra), hourly, Jan 2010 to Mar 2023. Pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, benzene, toluene. On-site met: WS, WD, RH, AT, SR, BP, RF | Central Pollution Control Board, Govt. of India (Central Control Room for AQ Management), compiled on Kaggle: `abhisheksjha/time-series-air-quality-data-of-india-2010-2023` | Kaggle API | CC BY-NC-SA 4.0. Scraped from the CPCB portal: units differ between stations, some column headers are mislabelled, values are provisional (not all validated). Unit decisions per column are logged in `unit_handling.csv`. |
| **Station coordinates (station level)** | lat/lon of CPCB stations | OpenAQ v3 API, which republishes CPCB stations; matched to CPCB stations by name within 60 km of the city centre | REST API, free key | CC BY 4.0. Unmatched or ambiguous stations fall back to the city centre (`coord_source` column). Match name, score and distance are kept for checking. |
| **City centres** | lat/lon/elevation of each (state, city) | GeoNames via the Open-Meteo geocoding API, plus 14 manually verified towns | REST API | CC BY 4.0. The fallback when no station-level match exists; elevation is city level. |
| **NASA FIRMS** | MODIS Collection 6.1 active-fire detections over India, 2010-2023 | NASA LANCE / FIRMS yearly country archive | yearly CSV, cached per year with SHA-256 in `download_manifest.json` | Free use with attribution. 1 km pixels, about 4 overpasses per day; clouds and haze hide fires, so counts are a lower bound. The archive is the standard-quality product, which later replaces the near-real-time records, so this is a retrospective study. Only fields the near-real-time product also has are used, with a 3-h availability delay. MODIS was chosen over VIIRS for a consistent sensor over the whole period. |
| **Festival calendar** | Diwali dates 2010-2023 | Public Hindu lunar calendar | hard-coded | Known years in advance, so safe as a forecast feature. |

### Alignment

* **Time:** all sources are aggregated to **IST calendar days**.
  * CPCB timestamps are IST.
  * FIRMS `acq_date` and `acq_time` are UTC. They are converted to IST, and 3 h of near-real-time latency is added. Each detection counts for the issue day on which it would first have been available, so a ~22:30 IST night overpass counts for the next day.
* **Space:** each station uses its own coordinates (OpenAQ) or, failing that, its city centre.
  * Fires are great-circle counts within **100 km / 300 km**, plus a regional Punjab + Haryana total (the stubble-burning signal).
  * Stations without coordinates get missing (not zero) fire features.
* **Daily rules (CPCB):**
  * a 24-h mean needs **≥ 16 valid hours**;
  * CO and O3 use the **maximum 8-h running mean**;
  * the AQI needs **≥ 3 sub-indices, including PM2.5 or PM10**.

### Weather: what we use and why
The brief asks us to state whether we use observed or forecast weather. **We use observed weather up to the issue time only**: the station's own CPCB sensors (wind speed and direction, relative humidity, air temperature, solar radiation, pressure, rainfall) for day D. These are reported live, so they are available at 24:00 of D. Weather of the target day D+1 is never used, because observed future weather would be leakage. There is no weather-forecast input.

ERA5 reanalysis, which would add boundary-layer height, was evaluated and **deliberately not used**. Copernicus publishes it about 5 days after real time, so even the day-D values would not exist when the forecast is issued.
