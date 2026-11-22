"""Run the whole pipeline end-to-end (or from a given step).

    python src/run_all.py                 # everything (assumes raw data + external downloads exist)
    python src/run_all.py --from features # rebuild from feature engineering onwards

Steps
  clean     src/clean.py               raw CPCB hourly -> QA -> daily + AQI
  eda       src/eda.py                 EDA figures + summary
  features  src/features.py            forecasting matrix (+ leakage audit table)
  impute    src/imputation.py          imputation comparison
  models    src/models.py              benchmark of all models, raw and PCA
  ablation  src/analysis_ablation.py   Analysis A
  efficiency src/analysis_efficiency.py Analysis C
  errors    src/analysis_errors.py     Analysis E
  loco      src/spatial_cv.py          leave-one-city-out
  tables    src/report_tables.py       regenerate reports/tables.md
Downloads (run once, separately): kaggle dataset, src/external.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent
STEPS = [
    ("clean", ["clean.py"]), ("eda", ["eda.py"]),
    ("features", ["features.py"]), ("audit", ["leakage_audit.py"]), ("impute", ["imputation.py"]),
    ("models", ["models.py"]), ("ablation", ["analysis_ablation.py"]),
    ("efficiency", ["analysis_efficiency.py"]), ("errors", ["analysis_errors.py"]), ("loco", ["spatial_cv.py"]), ("tables", ["report_tables.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="clean", choices=[s for s, _ in STEPS])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    names = [s for s, _ in STEPS]
    run = names[names.index(a.start):] if not a.only else a.only
    for name, cmd in STEPS:
        if name not in run:
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, str(SRC / cmd[0]), *cmd[1:]], check=True)
        print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
