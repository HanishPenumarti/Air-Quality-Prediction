"""Shared data preparation: load features, chronological splits, fold-fitted imputer.

Everything that is LEARNED from data lives in `Imputer.fit(F, train_mask)`, so each walk-forward
fold, the final model, the backtest origins and the spatial hold-outs fit it on their own training
rows only. features.py contains no training-period statistics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC, TRAIN_END, VAL_END, region_of  # noqa: E402

TARGET_REG, TARGET_CLS = "y_aqi", "y_poor"
ID_COLS = ["station", "city", "state", "city_key", "date", "target_date", "y_aqi", "y_poor", "y_pm25"]


def load(drop_groups=(), path=None):
    F = pd.read_parquet(path or PROC / "features.parquet")
    groups = json.loads((PROC / "feature_groups.json").read_text())
    feats = [c for g, cols in groups.items() for c in cols if g not in drop_groups]
    return F, feats, groups


def split_masks(F):
    t = F["target_date"]
    return {"train": (t <= TRAIN_END).values,
            "val": ((t > TRAIN_END) & (t <= VAL_END)).values,
            "test": (t > VAL_END).values}


def _keys(F):
    region = F["state"].map(region_of)
    return {"station": F["station"], "city": F["city_key"], "region": region}


class Imputer:
    """Fold-fitted preprocessing.

    fit(F, train_mask) learns, from the training rows only:
      * clim_aqi_month : mean AQI of the TARGET month, by station -> (state, city) -> region -> all.
                         For a training row whose own year is in the fit, that year is left out
                         (leave-one-year-out), so a row's target never contributes to its own feature.
      * fill values for remaining NaNs in the features: median by station x month(D) -> city x month
        -> region x month -> global median -> 0.
    strategy: 'clim' (the hierarchy above), 'median' (global median only) or 'native' (leave NaN,
    for XGBoost). clim_aqi_month is computed for every strategy.
    """
    LEVELS = ["station", "city", "region"]

    def __init__(self, strategy="clim", feats=None):
        self.strategy, self.feats = strategy, list(feats)

    # ---------------------------------------------------------------- fit
    def fit(self, F, train_mask):
        m = np.asarray(train_mask, bool)
        Ft = F.loc[m]
        k = {n: s[m] for n, s in _keys(F).items()}
        self.cols_ = [c for c in self.feats if c in F.columns]
        X = Ft[self.cols_]
        self.median_ = X.median()
        if self.strategy == "clim":
            mon = Ft["date"].dt.month
            self.fill_ = {lv: X.groupby([k[lv], mon]).median() for lv in self.LEVELS}
        # target climatology (by target month), with per-year sums for leave-one-year-out
        y, tm, ty = Ft["y_aqi"], Ft["target_date"].dt.month, Ft["target_date"].dt.year
        self.clim_ = {}
        for lv in self.LEVELS:
            g = y.groupby([k[lv], tm, ty]).agg(["sum", "count"])
            self.clim_[lv] = (g.groupby(level=[0, 1]).sum(), g)
        self.clim_global_ = float(y.mean())
        self.cutoff_ = str(Ft["target_date"].max().date()) if len(Ft) else None
        self.n_train_ = int(m.sum())
        return self

    # ---------------------------------------------------------------- transform
    def target_climatology(self, F):
        k = _keys(F)
        tm, ty = F["target_date"].dt.month.values, F["target_date"].dt.year.values
        out = pd.Series(np.nan, index=F.index)
        for lv in self.LEVELS:
            tot, per_year = self.clim_[lv]
            idx2 = pd.MultiIndex.from_arrays([k[lv].values, tm])
            idx3 = pd.MultiIndex.from_arrays([k[lv].values, tm, ty])
            s_all = tot["sum"].reindex(idx2).values
            n_all = tot["count"].reindex(idx2).values
            s_y = np.nan_to_num(per_year["sum"].reindex(idx3).values)
            n_y = np.nan_to_num(per_year["count"].reindex(idx3).values)
            n = n_all - n_y
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(n > 0, (s_all - s_y) / n, np.nan)
            out = out.fillna(pd.Series(v, index=F.index))
        return out.fillna(self.clim_global_).astype("float32")

    def transform(self, F):
        X = F[self.cols_].copy()
        if "clim_aqi_month" in self.feats:
            X["clim_aqi_month"] = self.target_climatology(F).values
        if self.strategy == "clim":
            k = _keys(F)
            mon = F["date"].dt.month.values
            for lv in self.LEVELS:
                if not X.isna().any().any():
                    break
                C = self.fill_[lv].reindex(pd.MultiIndex.from_arrays([k[lv].values, mon]))
                C.index = X.index
                X = X.fillna(C[[c for c in C.columns if c in X.columns]])
        if self.strategy != "native":
            X = X.fillna(self.median_).fillna(0)
        return X[[c for c in self.feats if c in X.columns]]
