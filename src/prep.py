"""Shared data preparation: load features, chronological splits, imputers, PCA."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC, TRAIN_END, VAL_END  # noqa: E402

TARGET_REG, TARGET_CLS = "y_aqi", "y_poor"
ID_COLS = ["station", "city", "state", "date", "target_date", "y_aqi", "y_poor", "y_pm25"]
# columns that get an explicit missing-indicator before imputation
INDICATOR_COLS = ["PM25_mean_l0", "PM10_mean_l0", "city_aqi_l0", "site_WS_l0", "aqi_l1"]


def load(drop_groups=()):
    F = pd.read_parquet(PROC / "features.parquet")
    groups = json.loads((PROC / "feature_groups.json").read_text())
    feats = [c for g, cols in groups.items() for c in cols if g not in drop_groups]
    return F, feats, groups


def split_masks(F):
    t = F["target_date"]
    return {"train": (t <= TRAIN_END).values,
            "val": ((t > TRAIN_END) & (t <= VAL_END)).values,
            "test": (t > VAL_END).values}


class Imputer:
    """Causal imputation fitted on the training period only.

    strategy:
      native      : leave NaN (only for models with native NaN support) + indicators
      median      : global training median
      ffill       : forward-fill within station (<= 7 days, past only), then training median
      clim        : station x month training median, then global training median
      ffill_clim  : forward-fill (<= 3 days), then station x month climatology, then median
    """

    def __init__(self, strategy="ffill_clim", feats=None):
        self.strategy, self.feats = strategy, list(feats)

    def fit(self, F, train_mask):
        tr = F.loc[train_mask, self.feats]
        self.median_ = tr.median()
        if "clim" in self.strategy:
            key = [F.loc[train_mask, "station"], F.loc[train_mask, "target_date"].dt.month]
            self.clim_ = tr.groupby(key).median()
        return self

    def transform(self, F):
        X = F[self.feats].copy()
        ind = {f"isna_{c}": X[c].isna().astype("int8") for c in INDICATOR_COLS if c in X}
        if self.strategy == "native":
            return pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)
        order = np.lexsort((F["date"].values, F["station"].values))
        if self.strategy.startswith("ffill"):
            limit = 7 if self.strategy == "ffill" else 3
            Xs = X.iloc[order]
            Xs = Xs.groupby(F["station"].values[order]).ffill(limit=limit)
            X = Xs.reindex(X.index)
        if "clim" in self.strategy:
            idx = pd.MultiIndex.from_arrays([F["station"], F["target_date"].dt.month])
            C = self.clim_.reindex(idx)
            C.index = X.index
            X = X.fillna(C)
        X = X.fillna(self.median_).fillna(0)
        return pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)
