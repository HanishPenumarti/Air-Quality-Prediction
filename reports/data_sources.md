## Data sources

| Source | What | Provider / origin | Access | Licence & caveats |
|---|---|---|---|---|
| **CPCB CCR hourly monitoring** (primary) | 453 CAAQMS stations, 241 cities, hourly, Jan 2010 to Mar 2023: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, benzene, toluene + on-site met (WS, WD, RH, AT, SR, BP, RF) | Central Pollution Control Board, Govt. of India (Central Control Room for AQ Management), compiled on Kaggle: `abhisheksjha/time-series-air-quality-data-of-india-2010-2023` | Kaggle API | CC BY-NC-SA 4.0. Scraped from the CPCB portal: units differ between stations, some column headers are mislabelled, values are provisional (not all validated). |
| **NASA FIRMS** | MODIS Collection 6.1 active-fire detections over India, 2010-2023 (1.14 M detections) | NASA LANCE / FIRMS | yearly country CSV archive | Free use with attribution. 1 km pixels, 2 overpasses/day; clouds and haze hide fires, so counts are a lower bound. MODIS chosen over VIIRS for a consistent sensor over the whole period. |
| **Station coordinates** | lat/lon/elevation of each station's city | GeoNames via Open-Meteo geocoding API (+14 manually verified towns) | REST API | CC BY 4.0. **City-level**, not exact station location (the Kaggle file has no coordinates); adequate for 100/300 km fire radii and region labels, too coarse for sub-10 km distances. |
| **Festival calendar** | Diwali dates 2010-2023 | Public Hindu lunar calendar | hard-coded | Known years in advance, so safe as a forecast feature. |

### Alignment
* **Time:** all sources aggregated to **IST calendar days** (CPCB timestamps are IST; FIRMS dates are UTC, but MODIS overpasses at ~10:30/13:30 local fall on the same IST day).
* **Space:** each station → its city coordinates. Fires: great-circle counts within **100 km / 300 km** of the station, plus a regional Punjab + Haryana total (stubble-burning signal).
* **Daily rules (CPCB):** a 24-h mean needs **≥ 16 valid hours**; CO and O3 use the **maximum 8-h running mean**; AQI needs **≥ 3 sub-indices including PM2.5 or PM10**.

### Weather: what we use and why
The brief asks us to state whether we use observed or forecast weather. **We use observed weather up to the issue time only**: the station's own CPCB sensors (wind speed and direction, relative humidity, air temperature, solar radiation, pressure, rainfall) for day D. These are reported live, so they are available at 24:00 of D. Weather of the target day D+1 is never used, because observed future weather would be leakage. There is no weather-forecast input.

ERA5 reanalysis (which would add boundary-layer height) was evaluated and **deliberately not used**. Copernicus publishes it about 5 days after real time, so even the day-D values would not exist when the forecast is issued. The ablation also shows weather adds only about 0.1 AQI points of MAE on top of the pollution history.

