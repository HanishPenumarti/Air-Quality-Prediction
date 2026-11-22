# Air Quality Prediction

**Next-day AQI forecasting for 453 monitoring stations in India (CPCB, 2010–2023).**

This project forecasts the next day's official **CPCB AQI** for **453 monitoring stations in 242 Indian cities**, using data from 2010 to 2023. The forecast is issued at the end of day D and covers day D+1. It is produced in two forms:

* **regression:** the AQI value itself;
* **classification:** whether the day will be "Poor or worse" (AQI > 200).

The full write-up is in [reports/report.md](reports/report.md), and the data provenance is in [reports/data_sources.md](reports/data_sources.md).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/app.py          # UI: needs the trained models in models/ and data/processed/
```

## Rebuilding everything from scratch

1. **Primary data.** Download the CPCB data (Kaggle CLI, about 1.6 GB unzipped):
   ```bash
   kaggle datasets download -d abhisheksjha/time-series-air-quality-data-of-india-2010-2023 -p data/raw/cpcb --unzip
   ```
2. **External data.** Station geocoding and NASA FIRMS fires (minutes):
   ```bash
   python src/external.py
   ```
3. **Pipeline:**
   ```bash
   python src/run_all.py                    # all steps
   python src/run_all.py --from features    # rebuild from feature engineering on
   python src/run_all.py --only errors loco # selected steps
   ```

| Step | Script | Output |
|---|---|---|
| clean | `src/clean.py` | Unit harmonisation, sensor-fault QA, CPCB daily aggregation and AQI → `data/processed/daily.parquet`, `qa_log.csv` |
| eda | `src/eda.py` | `reports/figures/eda_*.png`, `reports/eda_summary.json` |
| features | `src/features.py` | `data/processed/features.parquet` (83 features in 8 groups) |
| audit | `src/leakage_audit.py` | `reports/leakage_audit.csv` |
| impute | `src/imputation.py` | Comparison of imputation strategies → `reports/imputation_results.json` |
| models | `src/models.py` | Benchmark of all models, with and without PCA → `reports/benchmark*.csv`, `models/*.joblib` |
| ablation | `src/analysis_ablation.py` | Analysis A: feature-group ablation and feature selection |
| efficiency | `src/analysis_efficiency.py` | Analysis C: fit time, latency, size and scaling |
| errors | `src/analysis_errors.py` | Analysis E: error slices by season, region, category and episode |
| loco | `src/spatial_cv.py` | Leave-one-city-out spatial check |
| tables | `src/report_tables.py` | Regenerates `reports/tables.md`, the result tables used in the report |

The full run needs about 16 GB of RAM. The benchmark takes about 2.5 h on 14 cores, mostly for AdaBoost and sklearn's Gradient Boosting; XGBoost takes seconds.

## Layout

```
src/        pipeline scripts (config.py holds the paths, AQI breakpoints and split dates)
app/        Streamlit UI (Forecast, Risk map, What-if, Benchmark, Analyses, Data & method)
data/       raw/ (CPCB), external/ (FIRMS, geocodes), processed/ (parquet; not committed)
models/     fitted models + imputer (joblib)
reports/    report.md, data_sources.md, CSV/JSON results, figures/
```

## Licences

The CPCB data compilation is CC BY-NC-SA 4.0, and the geocodes (GeoNames) are CC BY 4.0. NASA FIRMS is free to use with attribution. The project is for non-commercial academic use only.
