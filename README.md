# Air Quality Prediction

**Next-day AQI forecasting for 453 monitoring stations in 242 Indian cities (CPCB, 2010 – Mar 2023).**

The forecast is issued at the end of day D for day D+1, using only information available at that time. It is produced in two forms:

* **regression:** the AQI of D+1, computed with the CPCB breakpoints;
* **classification:** whether D+1 will be "Poor or worse" (AQI > 200), both from a dedicated classifier and derived from the AQI forecast.

Models: Decision Tree, Random Forest, AdaBoost, Gradient Boosting (sklearn) and XGBoost, each on raw features and on PCA components, all compared against persistence and seasonal-climatology baselines.

The write-up is in [reports/report.md](reports/report.md), and the data provenance is in [reports/data_sources.md](reports/data_sources.md).

## How to run

### 1. Set up (once)

```bash
pip install -r requirements.txt          # pinned, tested versions (Python 3.12)
kaggle datasets download -d abhisheksjha/time-series-air-quality-data-of-india-2010-2023 -p data/raw/cpcb --unzip
```

**Station coordinates (optional but recommended).** Create a free OpenAQ account at <https://explore.openaq.org>, copy your API key, and save it in a file named `openaq_key.txt` in the project folder. The file is git-ignored, so it is never committed; setting the environment variable `OPENAQ_API_KEY` also works. Without a key, stations use their city centre.

```bash
python src/external.py                   # city geocodes, OpenAQ station coordinates, NASA FIRMS fires
```

### 2. Check the code (about 20 s, no data needed)

```bash
python -m pytest tests -q
```

### 3. Run the pipeline

```bash
python src/run_all.py                    # everything, roughly 15-18 h on 14 cores / 16 GB RAM
```

It can also run in stages, for example over two nights:

```bash
python src/run_all.py --only clean eda features audit impute models pairwise severe   # ~9-11 h
python src/run_all.py --only backtest ablation efficiency errors loco tables       # ~6-7 h
```

| Step | Script | What it does | Rough time (full data) |
|---|---|---|---|
| clean | `src/clean.py` | units, sensor-fault QA, CPCB daily aggregation, AQI | 10 min |
| eda | `src/eda.py` | EDA figures, summary, QA-impact table | 3 min |
| features | `src/features.py` | issue-time features (no fitted statistics) | 5 min |
| audit | `src/leakage_audit.py` | leakage declarations plus executable checks; **stops the run if a check fails** | 3 min |
| impute | `src/imputation.py` | imputation comparison (validation year only) | 15 min |
| models | `src/models.py` | benchmark: 5 models × raw/PCA × regression/classification | 8-10 h |
| pairwise | `src/pairwise_ci.py` | paired model-vs-model differences with block-bootstrap intervals | 5 min |
| severe | `src/severe.py` | extreme-day variants, prediction intervals | 15 min |
| backtest | `src/backtest.py` | rolling-origin test years 2019, 2020, 2021, 2022-23 | 1-1.5 h |
| ablation | `src/analysis_ablation.py` | Analysis A, all five models, full data | 2.5-3 h |
| efficiency | `src/analysis_efficiency.py` | Analysis C (timings run sequentially) | 2 h |
| errors | `src/analysis_errors.py` | Analysis E | 2 min |
| loco | `src/spatial_cv.py` | buffered leave-one-city-out and leave-one-region-out | 1.5-2 h |
| tables | `src/report_tables.py` | writes every result table into `reports/tables.md` **and into `reports/report.md`** | seconds |

Useful options:
* `--jobs N` sets the parallel jobs for the single-threaded models (Decision Tree, AdaBoost, Gradient Boosting); the default is 6. Lower it if memory is tight.
* `python src/models.py --only XGBoost` runs a **partial** benchmark. It is written to `reports/partial/` and `models/partial/`, so it can never mix with the full results.
* `python src/analysis_ablation.py --frac 0.3` runs a faster ablation on a stratified 30% sample.

### 4. Open the app

```bash
streamlit run app/app.py
```

If something is missing, the app lists the missing files and the command that creates them. The app is a **historical forecast demonstration**: it replays the archive with models trained up to 2021, and has no live data feed.

### Quick demo on a few stations (about 30-60 min)

Writes everything to a separate folder, so full results are never overwritten:

```bash
# PowerShell
$env:AQ_OUT="demo"; $env:AQ_STATIONS="DL00,MH00,KA00,BR00,WB00"
python src/run_all.py --only clean eda features audit models errors tables
streamlit run app/app.py
```

## Layout

```
src/        pipeline (config.py: paths, AQI bands, QA thresholds, split dates; zoo.py: model definitions)
app/        Streamlit UI (Forecast, Risk map, What-if, Benchmark, Analyses, Data & method)
tests/      offline tests on synthetic data
data/       raw/ (CPCB), external/ (FIRMS, geocodes, OpenAQ), processed/ (parquet; not committed)
models/     fitted models, imputer, production.json, thresholds.json, final_params.json, intervals.json
reports/    report.md, data_sources.md, CSV/JSON results, run_manifest.json, figures/
```

## Licences

The CPCB data compilation is CC BY-NC-SA 4.0, the geocodes (GeoNames) are CC BY 4.0, and OpenAQ data are CC BY 4.0. NASA FIRMS is free to use with attribution. The project is for non-commercial academic use only.
