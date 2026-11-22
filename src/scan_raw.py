"""Scan every raw CPCB station file: rows, time span, sampling step, columns & units.

Output: data/processed/raw_scan.csv (one row per station) and a printed summary.
"""
from collections import Counter
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "cpcb"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    rows = []
    col_counter = Counter()
    for f in sorted(RAW.glob("[A-Z][A-Z][0-9][0-9][0-9].csv")):
        df = pd.read_csv(f, usecols=lambda c: True, low_memory=False)
        t = pd.to_datetime(df["From Date"], errors="coerce")
        step = t.diff().mode()
        cols = [c for c in df.columns if c not in ("From Date", "To Date")]
        col_counter.update(cols)
        pm = next((c for c in cols if c.startswith("PM2.5")), None)
        rows.append(dict(
            station=f.stem, n_rows=len(df), start=t.min(), end=t.max(),
            step=step.iloc[0] if len(step) else None,
            n_cols=len(cols),
            pm25_frac=df[pm].notna().mean() if pm else 0.0,
            cols="|".join(cols),
        ))
    scan = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    scan.to_csv(OUT / "raw_scan.csv", index=False)

    print("stations:", len(scan), " total rows:", f"{scan.n_rows.sum():,}")
    print("sampling step counts:\n", scan.step.value_counts())
    print("start year counts:\n", scan.start.dt.year.value_counts().sort_index())
    print("PM2.5 non-missing fraction (describe):\n", scan.pm25_frac.describe())
    print("\nAll column names (with units) and #stations having them:")
    for c, n in sorted(col_counter.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {c}")


if __name__ == "__main__":
    main()
