# Predicting Air Quality in India: next-day AQI forecasting for 453 stations

## 1. Problem and task definition

| | |
|---|---|
| **Unit of prediction** | station × day |
| **Target (regression)** | Official CPCB **National AQI of day D+1** at that station (0–500) |
| **Target (classification)** | **"Poor or worse"** day: AQI(D+1) > 200. This is the CPCB threshold at which health advisories start. |
| **Forecast issue time** | 24:00 IST of day D. Only information available by then is used (sections 4 and 5). |
| **Horizon** | 1 day ahead (D → D+1) |
| **Two ways to classify** | (a) *derived*: threshold the regression forecast at 200; (b) *direct*: a class-weighted classifier with its decision threshold tuned on the 2021 validation year |

**Why this target.** The AQI is what the public and regulators act on. It is defined by CPCB rules: a sub-index per pollutant, with the maximum taken as the AQI. Within 24 hours of the forecast an authority can still issue an advisory or a GRAP restriction. "Poor or worse" covers 24.2 % of station-days, which is an imbalance large enough to need imbalance-aware metrics.

## 2. Data

### 2.1 Sources

The full provenance and licence table is in [data_sources.md](data_sources.md). The sources are:

* **CPCB CCR hourly monitoring** (primary)
  * 453 continuous stations in 242 cities, Jan 2010 to Mar 2023, about 14.3 M hourly rows.
  * 11 pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, benzene, toluene).
  * 7 on-site meteorological channels.
  * Taken from the Kaggle compilation `abhisheksjha/time-series-air-quality-data-of-india-2010-2023` (CC BY-NC-SA 4.0).
* **NASA FIRMS MODIS C6.1** active fires over India, 2010–2023: 1.14 M detections.
* **Station geolocation:** GeoNames via the Open-Meteo geocoder, plus 14 manually verified towns.
* **Diwali calendar**, 2010–2023.

### 2.2 Size

* **Daily data:** 595,889 station-days after aggregation; **444,322** of them have a valid AQI.
* **Modelling table:** **444,277 rows** with a valid AQI on D+1 (≥ 100 k required).
* **Features:** **78 engineered features** in 8 groups, 83 with missing-value indicators (≥ 50 required).

### 2.3 Alignment

* **Time:** all sources are aggregated to **IST calendar days**.
* **Space:** each station is mapped to its city's coordinates. Fire counts are great-circle counts within 100 km and 300 km, plus a Punjab + Haryana regional total (the stubble-burning signal).
* **Limitation:** coordinates are at city level, because the Kaggle files contain no station coordinates.

### 2.4 Weather: observed up to the issue time only

The brief asks teams to state whether they use observed or forecast weather. **We use observed weather up to the end of day D only.** It comes from the station's own CPCB sensors: wind speed and direction, humidity, temperature, solar radiation, pressure and rainfall. These report live, so they are available when the forecast is issued. Weather of the target day D+1 is never used, because that would be observed future weather (leakage), and no weather-forecast input is used.

**ERA5 reanalysis was considered and rejected.** It would add boundary-layer height, but Copernicus publishes it about 5 days after real time. Even its day-D values therefore would not exist at 24:00 of D, so using them would break the rule that every feature must be available at forecast time.

## 3. Cleaning, sensor-fault detection and EDA

### 3.1 Harmonisation

The raw files mix units and header spellings (`src/scan_raw.py` inventories them).

* **Conversions:** ppb → µg/m³ at 25 °C, CO ppm/µg → mg/m³, pressure hPa → mmHg.
* **Logging:** every conversion is recorded in `data/processed/unit_handling.csv`.

### 3.2 Sensor-fault QA

QA runs per station on the hourly series, and **each rule uses only data from the same calendar day**, so cleaning cannot leak future values.

| Rule | Hourly values removed |
|---|---:|
| **Stuck sensor:** identical value for ≥ 8 consecutive hours | 873,930 |
| **Out of physical range:** e.g. PM2.5 < 0 or > 1,000 | 266,332 |
| **Isolated spike:** a value far above both neighbours, above a floor | 42,142 |
| **PM2.5 > 1.5 × PM10:** physically inconsistent | 12,704 |

Daily values follow the CPCB rules:
* A daily mean needs **≥ 16 valid hours**.
* CO and O3 use the **maximum 8-h running mean**.
* The AQI needs **≥ 3 sub-indices, including PM2.5 or PM10**.

### 3.3 Key EDA findings

Figures `figures/eda_01` to `eda_12`.

* **Network growth.** The network grows from about 10 stations in 2010 to more than 400 by 2022 (`eda_01`), so later years dominate the row count.
* **Class balance.** Poor-or-worse covers **24.2 %** of station-days overall but varies by region (`eda_07`):

  | Region | Poor-or-worse share |
  |---|---:|
  | North (IGP) | 38.9 % |
  | East | 19.9 % |
  | North-East & Hills | 14.2 % |
  | West | 13.3 % |
  | Central | 12.2 % |
  | South | 2.1 % |

* **Which pollutant drives the AQI.** On Poor-or-worse days the driver is **PM2.5 80.5 %** and PM10 17.8 % (`eda_08`).
* **Missingness is not random.** PM2.5 is missing on **28–30 %** of days when the city is Poor or worse, against **22–25 %** on clean days (`eda_04`). Gaps therefore carry information, which is why missing-value flags became features.
* **Spatial dependence.** The correlation of daily AQI between station pairs falls with distance: **0.90** within a city, **0.75** at 50–100 km, **0.40** beyond 1,000 km (`eda_09`). Random K-fold splits would leak through same-day neighbours, so the validation uses time splits plus a leave-one-city-out check (section 5).
* **Seasonality.** There is a strong winter peak in the IGP, plus a post-monsoon stubble-burning bump in Oct–Nov (`eda_05`).
* **Diwali.** The median AQI the day after Diwali is **1.52 ×** the level a week before (`eda_11`).

## 4. Missing data and imputation

### 4.1 Rules

* **The target is never imputed.** Rows without a valid AQI on D+1 are dropped.
* The *history* used for lag features is imputed at the **daily level** with a **hybrid causal fill**, in this order:
  1. the mean of the other stations in the same city on the same day;
  2. otherwise, forward-fill of up to 3 days;
  3. otherwise, the station × month climatology from the training period only.
* Missing-value indicators are kept as features: hours present, number of sub-indices, `days_since_valid` and `isna_*`.
* Any NaNs left in the feature matrix are handled by `prep.Imputer`. It is fitted on training rows only and offers the strategies `native`, `median`, `ffill`, `clim` and `ffill_clim`.

### 4.2 Strategies compared

Two experiments, in `src/imputation.py`:

**(a) Reconstruction test.** 10 % of known daily PM2.5 values were masked, then reconstructed (`imp_01`):

| Method | MAE (µg/m³) | Coverage |
|---|---:|---:|
| ffill (causal) | 21.0 | 99.9 % |
| linear interpolation (uses the future, so offline only) | 16.6 | 99.5 % |
| station-month climatology | 22.3 | 100 % |
| same-city neighbours | 16.5 | 56 % |
| **hybrid: neighbours → ffill → climatology** | **16.3** | **100 %** |

**(b) Downstream test.** XGBoost validation MAE on 2021 for each feature-level strategy (`imp_02`):
* All five strategies land within **24.3–24.5**.
* F1 is within **0.844–0.846**.
* Applying the hybrid fill to the lag history first changes both by less than 0.2 MAE.

**Decision.** The hybrid is used because it is the best causal reconstruction and it covers 100 % of days. Downstream, tree models are insensitive to the choice, which is itself a finding: missing-value flags plus trees cope with the gaps.

## 5. Leakage audit and validation design

### 5.1 Leakage audit

The full table is generated by `src/leakage_audit.py` into [leakage_audit.csv](leakage_audit.csv). Summary:

| Feature group | n | Time of information | Verdict |
|---|---:|---|---|
| AQI history: lags 0–6, trailing 3/7/14/30-day means, SD, max, change, poor-day count | 16 | D-29 to D | **used** |
| Pollutant history: daily means, 8-h maxima, evening (18–23 h) means, daily max, PM ratio | 24 | D-6 to D | **used**. These are AQI inputs, but for day D, not D+1. |
| Missing-value flags: hours present, sub-indices, days since valid | 6 | D | **used** |
| Neighbours: other stations in the same city or state on day D | 5 | D | **used**. Same-day D+1 values of other stations are **not** used. |
| On-site weather, observed | 8 | D | **used**. Reported live, so available at issue time. |
| Observed weather of D+1 | n/a | D+1 | **excluded**: future weather is leakage for a true forecast |
| ERA5 reanalysis | n/a | n/a | **not used**: published about 5 days late, so not available at issue time |
| Fires: FIRMS counts within 100/300 km, FRP, Punjab+Haryana total | 6 | D-6 to D | **used** |
| Calendar for D+1: day of year, day of week, Diwali distance, lockdown | 8 | D+1, known in advance | **used** |
| Static: coordinates, elevation, region, train-period climatology | 5 | n/a | **used** |
| AQI, category, driver and sub-indices of D+1; pollutant concentrations of D+1; `y_pm25` | n/a | D+1 | **excluded** |
| Centred windows, two-sided interpolation | n/a | n/a | **not used**. All windows are trailing. |
| Imputer, scaler and PCA statistics; QA rules | n/a | train only, or same day | **safe** |

**Programmatic checks.**
* For random rows, the target equals the AQI of the next calendar day.
* `aqi_l0` equals the AQI of the same day.
* The `rm3` window covers D-2 to D.
* corr(aqi_l0, y) is 0.90, which is plausible persistence rather than a leak.

### 5.2 Splits

* **Train:** ≤ 2020.
* **Validation:** 2021.
* **Test:** Jan 2022 to Mar 2023, 141,610 rows, used **once**.
* **Tuning:** walk-forward with 2 origins. Each origin trains on all years before its validation year:

  | Origin | Train | Validate |
  |---|---|---|
  | 1 | ≤ 2019 | 2020 |
  | 2 | ≤ 2020 | 2021 |

  Each training window is subsampled to 40 % for speed.
* **Final models** are refitted on ≤ 2021 with the chosen parameters.
* **Classification threshold:** for the direct classifiers it is chosen on 2021, using a model fitted on ≤ 2020, to maximise F1.

### 5.3 Spatial dependence

* Time splits keep all stations in every period, so they measure the *known-station* setting.
* **Leave-one-city-out (LOCO)** (`src/spatial_cv.py`) holds out all stations of a city during training. It does this for 12 large cities across all regions, and then tests on that city.

### 5.4 Baselines

* **Persistence:** AQI(D+1) = AQI(D). If AQI(D) is missing, it falls back to the last valid AQI, then to climatology.
* **Seasonal climatology:** the station × month mean AQI from the training period.

## 6. Models and benchmark

### 6.1 Models

| Model | Implementation | Tuning grid |
|---|---|---|
| Decision Tree | sklearn | depth {8, 12, 16} × min leaf {50, 200} |
| Random Forest | sklearn, 150 trees | depth {16, 24}, min leaf 10, max_features 0.33 |
| **AdaBoost** | sklearn, decision-tree base learners, 100 rounds | base depth {6, 9} × learning rate {0.05, 0.3} |
| **Gradient Boosting** | sklearn, Huber loss, 250 rounds, early stopping | depth {4, 6}, learning rate 0.1 |
| **XGBoost** | `hist`, up to 800 rounds with early stopping | depth {6, 8} × min_child_weight {5, 20}, learning rate 0.05 |

* **Shared setup.** Every model is trained twice:
  * on the **raw** features;
  * after **StandardScaler + PCA** keeping 95 % of the variance (46 components). PCA is fitted on ≤ 2021 only.
* **Imbalance handling for the direct classifiers:**
  * `class_weight="balanced"` for the Decision Tree;
  * `"balanced_subsample"` for the Random Forest;
  * `scale_pos_weight` for XGBoost;
  * balanced sample weights for AdaBoost and Gradient Boosting.
* **Imbalance-aware metrics:** precision, recall, F1, **PR-AUC** and balanced accuracy, not plain accuracy.

### 6.2 Results: regression, test Jan 2022 to Mar 2023

| Model | Features | MAE | RMSE | R² | MAE on top-5 % days | Skill vs persistence | Fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Persistence | – | 28.49 | 46.51 | 0.775 | 50.2 | 0 % | – |
| Seasonal climatology | – | 57.20 | 79.83 | 0.337 | 108.4 | −101 % | – |
| Decision Tree | raw | 26.57 | 40.53 | 0.829 | 51.5 | +6.8 % | 17 |
| Random Forest | raw | 25.05 | 38.46 | 0.846 | **49.7** | +12.1 % | 187 |
| AdaBoost | raw | 26.86 | 39.49 | 0.838 | 50.9 | +5.7 % | 1,331 |
| Gradient Boosting | raw | 24.92 | 38.62 | 0.845 | 49.8 | +12.5 % | 651 |
| **XGBoost** | raw | **24.84** | **38.21** | **0.848** | 50.6 | **+12.8 %** | **10** |

The PCA rows are in [tables.md](tables.md).

### 6.3 Results: Poor-or-worse classification

| Model | Derived F1 | Direct F1 | Direct PR-AUC | Direct recall | Direct balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.791 | – | (0.862)\* | – | – |
| Seasonal climatology | 0.590 | – | – | – | – |
| Decision Tree | 0.813 | 0.815 | 0.891 | 0.845 | 0.889 |
| Random Forest | 0.822 | **0.824** | 0.905 | 0.855 | 0.895 |
| AdaBoost | 0.821 | 0.821 | 0.904 | **0.868** | **0.898** |
| Gradient Boosting | 0.823 | 0.819 | 0.901 | 0.852 | 0.892 |
| XGBoost | **0.824** | 0.823 | **0.907** | 0.836 | 0.889 |

\*PR-AUC of persistence uses AQI(D) as the score.

* **Derived and direct give the same F1** to within ±0.01.
* The direct classifiers' tuned thresholds trade a little precision for **higher recall**, which is the more useful side for a health advisory.
* Every model beats persistence: F1 rises by 2.4–3.3 points, and PR-AUC by 2.9–4.5 points.

### 6.4 PCA with and without

| Model | MAE raw → PCA | Direct F1 raw → PCA |
|---|---|---|
| Decision Tree | 26.57 → 29.79 (+3.2) | 0.815 → 0.788 |
| Random Forest | 25.05 → 28.35 (+3.3) | 0.824 → 0.807 |
| AdaBoost | 26.86 → 30.48 (+3.6) | 0.821 → 0.807 |
| Gradient Boosting | 24.92 → 27.46 (+2.5) | 0.819 → 0.808 |
| XGBoost | 24.84 → 27.57 (+2.7) | 0.823 → 0.808 |

**PCA hurts every model.** Three reasons, consistent with the ablation in section 7.1:
* **Variance is not predictive relevance.** The first component (33 % of variance) is a general pollution-level axis, while the predictive signal lives in a few specific columns (`aqi_l0`, the evening PM means).
* **Rotation destroys axis-aligned splits,** which tree models rely on.
* **It saves little time.** XGBoost trains in 9 s instead of 10 s.

### 6.5 Choice of model

**XGBoost (raw features)** is used in the UI. It has the best MAE and PR-AUC, and trains **65× faster** than sklearn's Gradient Boosting and **19× faster** than the Random Forest. Its file is 1 MB, compared with 56 MB for the Random Forest. The differences among RF, GBM and XGB are small, about 0.2 MAE; the separation from the Decision Tree, AdaBoost and the baselines is clear.

## 7. Additional analyses: A, C and E

### 7.1 Analysis A: feature ablation and selection

Script `src/analysis_ablation.py`, figures `abl_01` to `abl_03`.

**Leave one group out.** The table shows the increase in test MAE when the group is removed:

| Group removed | Decision Tree | Random Forest | XGBoost |
|---|---:|---:|---:|
| pollutant history | +2.07 | +2.46 | **+2.78** |
| AQI history | +1.38 | +0.86 | +0.93 |
| calendar / festivals | −0.06 | +0.03 | +0.14 |
| fires (FIRMS) | −0.06 | +0.01 | +0.11 |
| on-site weather | −0.02 | +0.04 | +0.10 |
| missing-value flags | −0.01 | +0.05 | +0.05 |
| neighbouring stations | −0.06 | 0.00 | +0.04 |
| station context | −0.22 | −0.09 | +0.01 |

* **Pollutant history matters most,** above all the *evening* PM2.5 and PM10 means. The concentration at the end of day D is the best predictor of D+1, and a daily mean hides that.
* **Incremental addition:**
  * AQI history alone gives XGBoost MAE 27.58, which already beats persistence's 28.49.
  * Adding pollutant history gets it to 24.97.
  * Everything after that adds ≤ 0.1.
* **The secondary groups only help XGBoost.** They are worth 0.05–0.14 MAE each to XGBoost but nothing to a single Decision Tree. The Tree even improves slightly without them, a sign of overfitting.

**Selection vs PCA** at an equal number of inputs (`abl_02`), XGBoost MAE:

| Inputs | Top-k by XGBoost gain | PCA with k components |
|---|---:|---:|
| 5 | 25.69 | 31.32 |
| 10 | 25.92 | 28.97 |
| 20 | 25.41 | 28.16 |
| 40 | 25.18 | 27.75 |
| all 83 | 24.84 | n/a |

**Five selected features already get within 0.9 MAE of the full model:** `aqi_l0`, `aqi_last_valid`, `PM25_eve_l0`, `aqi_rm14` and `PM10_eve_l0`. PCA needs more than 40 components to approach that.

**Weather.** Removing the on-site weather costs XGBoost only 0.10 MAE. For a next-day forecast, today's pollution already carries most of what today's weather says about tomorrow.

### 7.2 Analysis C: computational efficiency

Script `src/analysis_efficiency.py`, figures `eff_01` and `eff_02`. Measured on 14 cores, 302,667 training rows.

| Model | Fit (s) | Predict (ms / 1k rows) | Latency for 1 row (ms) | Size (MB) | MAE |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 17 | 0.15 | 0.04 | 0.05 | 26.57 |
| Random Forest | 187 | 4.39 | 14.1 | 56.3 | 25.05 |
| AdaBoost | 1,331 | 13.18 | 3.2 | 2.1 | 26.86 |
| Gradient Boosting | 651 | 6.15 | 0.11 | 0.23 | 24.92 |
| XGBoost | **10** | 0.82 | 0.75 | 1.0 | **24.84** |

**Scaling with training size** (`eff_02`), using 10 %, 25 %, 50 % and 100 % of the training rows:
* **Fit time** grows roughly linearly for every model. AdaBoost and sklearn's Gradient Boosting are 10–100× slower than XGBoost at every size.
* **Accuracy saturates early.**

  | Model | MAE at 10 % of rows (30 k) | MAE at 100 % | Change |
  |---|---:|---:|---:|
  | XGBoost | 25.21 | 24.84 | −0.37 |
  | Random Forest | 25.35 | 25.05 | −0.30 |
  | AdaBoost | 26.29 | 26.86 | **+0.57** |

  AdaBoost gets *worse* with more data: its linear-loss reweighting overweights the extreme and noisy days.
* **Practical conclusion.** Each daily run (about 450 stations) costs XGBoost **< 1 ms**, and a full retrain takes **10 s**, so the system can be retrained daily on a laptop.

### 7.3 Analysis E: error analysis

Script `src/analysis_errors.py`, figures `err_01` to `err_07`, plus per-station errors in `errors_by_station.csv`.

**By region (XGBoost MAE).** Errors scale with pollution level, but XGBoost's gain over persistence is largest exactly where pollution is worst:

| Region | XGBoost MAE | Persistence MAE | Gain over persistence |
|---|---:|---:|---:|
| North (IGP) | 30.8 | 36.5 | −16 % |
| West | 24.9 | 27.2 | |
| East | 21.8 | 24.5 | |
| Central | 20.8 | 22.3 | |
| North-East & Hills | 20.1 | 20.4 | |
| South | 13.2 | 14.3 | −8 % |

**By month.** Error is highest in Jan (32.5), Nov–Dec (about 30) and May (27, dust), and lowest in Jul–Sep (about 13, monsoon washout). The *gain* over persistence is largest in Jan–Feb, by about 5.6 points.

**By true category: the main weakness.**

| True category | XGBoost bias |
|---|---:|
| Good | +10.8 |
| Moderate | +7.5 |
| Poor | −16.3 |
| Very Poor | −26.2 |
| Severe | **−66.4** (MAE 71) |

* **The models shrink towards the mean.** They over-forecast clean days and under-forecast extreme ones.
* **On Severe days they are worse than persistence:** MAE 71 against 63.5.
* **Cause:** squared-error and Huber losses on a target capped at 500.
* **This matters for health warnings,** but the direct classifier compensates partly through its lowered threshold, which gives high recall.

**By size of the day-to-day change |AQI(D+1) − AQI(D)|.**

| Change size | XGBoost MAE | Persistence MAE |
|---|---:|---:|
| < 25 (calm days) | 13.5 | **9.6** |
| > 150 (sudden jumps) | **121** | 187 |

* On calm days persistence wins.
* On sudden jumps XGBoost cuts the error by a third, but jumps are still where most of the error is. The Delhi episode in `err_07` shows forecasts lagging the observed jumps by one day.

**Special days:**
* **Diwali ±2 days:** MAE 37.2, against 24.7 on other days. The calendar features help only partly, since the effect depends on weather and on firecracker bans.
* **Missing AQI on day D:** MAE 28.5, against 24.7 when it was measured. This is far smaller than persistence's jump to 39.5, so the neighbour and climatology fills work.

**Stations.** The worst station is **Byrnihat (AS007)**, an industrial town on the Assam–Meghalaya border with mean AQI 418 in the test period. XGBoost's MAE there is 93, against 56 for persistence. Its pollution level is unlike anything in the training data, which is a clear case of distribution shift.

**Model agreement.** Errors of the five ML models correlate at **0.93–0.98**, and only 0.79–0.82 with persistence (`err_06`). They fail on the same days, so an ensemble of them would add little; the remaining error needs new information (weather forecasts), not a different learner.

### 7.4 Leave-one-city-out

Script `src/spatial_cv.py`, figure `loco.png`.

* **Unseen cities lose very little.** Holding a city out of training raises XGBoost's MAE by only **+0.38 AQI points on average** across the 12 cities. Two examples:
  * Delhi: 36.5 held out, against 35.7 when the city is seen in training.
  * Ahmedabad: 29.1, against 27.9.
* **Persistence is still beaten** in 11 of 12 held-out cities. The exception is Ahmedabad, at 29.1 against 28.6.
* **Why:** the model mostly learns *transferable dynamics* (persistence plus the evening concentration) rather than city identity. A new station can therefore be forecast from its first few days of data. **Spatial leakage is therefore not inflating the benchmark** to any meaningful degree.

## 8. User interface

`streamlit run app/app.py` opens six pages built on the same pipeline artefacts. They load the saved models, the saved imputer and the feature table.

1. **Forecast**
   * Pick a station and an issue date.
   * See the next-day AQI from XGBoost, the CPCB category, and the direct-classifier probability against its tuned threshold, with an alert when it is crossed.
   * Compare all models side by side, see a 60-day history, and a per-feature explanation from XGBoost's SHAP-style `pred_contribs`.
2. **Risk map:** the forecast CPCB category for every station on a chosen date, for any of the 5 models, with the day's count of Poor-or-worse stations and the MAE against observations.
3. **What-if:** change today's AQI, this evening's PM2.5, on-site wind, humidity, temperature and rain, and fire counts, and see every model's forecast respond, with response curves.
4. **Benchmark:** regression and classification tables, and PCA with and without.
5. **Analyses:** A, C, E and LOCO, with figures.
6. **Data & method:** EDA, missingness and QA, the leakage audit, and sources.

All pages were smoke-tested headlessly (Streamlit `AppTest`) with no exceptions.

## 9. Limitations

* **Location.** Station coordinates are city-level; the Kaggle data has no exact station locations.
* **Data status.** CPCB data are provisional, and calibration drift is not corrected.
* **Fires.** FIRMS fire counts are a lower bound because clouds and haze hide fires, and they are city-to-fire distances, not modelled smoke transport.
* **The AQI cap.** The AQI is capped at 500, so Severe days are censored, which contributes to the negative bias on extremes. A quantile loss or a separate extreme-day model is the natural next step.
* **Test period.** The test window (2022 to Mar 2023) contains one winter. LOCO and the walk-forward tuning reduce but do not remove the risk that a single season dominates the result.

## 10. Reproducibility

The steps, timings and outputs are listed in [README](../README.md).

* **Seed:** 42 throughout.
* **Configuration:** all paths, AQI breakpoints and split dates are in `src/config.py`.
* **Full re-run:** `python src/run_all.py`.
* **Tables:** every number in this report comes from files in `reports/`, and [tables.md](tables.md) is generated from them.
