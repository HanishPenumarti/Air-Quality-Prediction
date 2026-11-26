"""Run the whole pipeline end-to-end (or from a given step).

    python src/run_all.py                      # everything (assumes raw data + external downloads exist)
    python src/run_all.py --from features      # rebuild from feature engineering onwards
    python src/run_all.py --only errors tables # selected steps only
    python src/run_all.py --jobs 6             # parallel jobs for the single-threaded models

Steps
  clean      src/clean.py               raw CPCB hourly -> QA -> daily + AQI
  eda        src/eda.py                 EDA figures + summary + QA impact
  features   src/features.py            issue-time forecasting features (no fitted statistics)
  audit      src/leakage_audit.py       leakage declarations + executable checks (stops on failure)
  impute     src/imputation.py          imputation comparison (validation year only)
  models     src/models.py              benchmark of all models, raw and PCA
  severe     src/severe.py              extreme-day variants + prediction intervals
  backtest   src/backtest.py            rolling-origin backtest 2019..2022-23
  ablation   src/analysis_ablation.py   Analysis A (all five models, validation year)
  efficiency src/analysis_efficiency.py Analysis C
  errors     src/analysis_errors.py     Analysis E
  loco       src/spatial_cv.py          buffered leave-one-city-out + leave-one-region-out
  tables     src/report_tables.py       regenerate reports/tables.md and the tables inside report.md
Downloads (run once, separately): kaggle dataset, src/external.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent
STEPS = [
    ("clean", ["clean.py"]), ("eda", ["eda.py"]), ("features", ["features.py"]), ("audit", ["leakage_audit.py"]),
    ("impute", ["imputation.py"]), ("models", ["models.py"]), ("severe", ["severe.py"]),
    ("backtest", ["backtest.py"]), ("ablation", ["analysis_ablation.py"]), ("efficiency", ["analysis_efficiency.py"]),
    ("errors", ["analysis_errors.py"]), ("loco", ["spatial_cv.py"]), ("tables", ["report_tables.py"]),
]
TAKES_JOBS = {"models", "backtest", "ablation"}


def main():
    names = [s for s, _ in STEPS]
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="clean", choices=names)
    ap.add_argument("--only", nargs="+", choices=names)
    ap.add_argument("--jobs", type=int, default=6, help="parallel jobs for single-threaded models")
    a = ap.parse_args()
    run = a.only if a.only else names[names.index(a.start):]
    t_all = time.time()
    for name, cmd in STEPS:
        if name not in run:
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        extra = ["--jobs", str(a.jobs)] if name in TAKES_JOBS else []
        subprocess.run([sys.executable, str(SRC / cmd[0]), *cmd[1:], *extra], check=True)
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print(f"\nall requested steps done in {(time.time() - t_all) / 3600:.2f} h")


if __name__ == "__main__":
    main()
