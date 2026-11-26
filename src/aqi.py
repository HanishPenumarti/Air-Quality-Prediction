"""CPCB National AQI: sub-indices, AQI, categories and the 'Poor or worse' rule.

Single source of truth for every AQI decision in the project (cleaning, metrics, app).
"""
import numpy as np
import pandas as pd

from config import (AQI_CAT_EDGES, AQI_CATEGORIES, AQI_CONC_BANDS, AQI_INDEX_BANDS, AQI_ROUND,
                    POOR_THRESHOLD, SEVERE_THRESHOLD)

UNKNOWN = "Unknown"


def _round_half_up(x, decimals):
    f = 10.0 ** decimals
    return np.floor(np.asarray(x, dtype=float) * f + 0.5) / f


def sub_index(conc, pollutant):
    """CPCB sub-index: I = i_lo + (i_hi - i_lo) / (c_hi - c_lo) * (C - c_lo) within the band of C.

    C is first rounded (integer ug/m3, CO to 0.1 mg/m3). Above the last band -> 500 (cap).
    NaN in -> NaN out. Returns a numpy array (or a Series if a Series was passed)."""
    idx = conc.index if isinstance(conc, pd.Series) else None
    c = _round_half_up(np.clip(np.asarray(conc, dtype=float), 0, None), AQI_ROUND.get(pollutant, 0))
    out = np.full(c.shape, np.nan)
    bands = AQI_CONC_BANDS[pollutant]
    for (c_lo, c_hi), (i_lo, i_hi) in zip(bands, AQI_INDEX_BANDS):
        m = (c >= c_lo) & (c <= c_hi)
        out[m] = i_lo + (i_hi - i_lo) / (c_hi - c_lo) * (c[m] - c_lo)
    out[c > bands[-1][1]] = 500.0
    return pd.Series(out, index=idx) if idx is not None else out


def category_index(aqi):
    """0..5 for Good..Severe; -1 for missing / non-finite values (never silently 'Severe')."""
    a = np.asarray(aqi, dtype=float)
    out = np.digitize(np.nan_to_num(a, nan=0.0), AQI_CAT_EDGES[1:-1], right=True)
    return np.where(np.isfinite(a), out, -1)


def category_name(aqi):
    """Category label(s); 'Unknown' for missing values."""
    names = np.array(AQI_CATEGORIES + [UNKNOWN])
    return names[category_index(aqi)]          # index -1 -> last element = Unknown


def is_poor(aqi):
    """'Poor or worse' decision on an AQI VALUE: strictly AQI > 200 (target definition)."""
    a = np.asarray(aqi, dtype=float)
    return np.where(np.isfinite(a), a > POOR_THRESHOLD, False)


def is_severe(aqi):
    a = np.asarray(aqi, dtype=float)
    return np.where(np.isfinite(a), a > SEVERE_THRESHOLD, False)
