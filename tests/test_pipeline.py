"""Offline tests with small synthetic data (no dataset needed):   python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqi import category_name, is_poor, sub_index  # noqa: E402
from metrics import best_f1_threshold, cls_metrics_aqi  # noqa: E402


# ------------------------------------------------------------------ AQI
@pytest.mark.parametrize("c,expected", [(0, 0), (30, 50), (31, 51), (60, 100), (61, 101), (90, 200), (91, 201),
                                        (120, 300), (121, 301), (250, 400), (251, 401), (380, 500), (999, 500),
                                        (30.4, 50), (30.5, 51)])
def test_pm25_bands(c, expected):
    assert sub_index(np.array([c]), "PM25")[0] == pytest.approx(expected)


@pytest.mark.parametrize("c,expected", [(1.0, 50), (1.1, 51), (2.0, 100), (2.1, 101), (10, 200), (10.1, 201)])
def test_co_bands(c, expected):
    assert sub_index(np.array([c]), "CO")[0] == pytest.approx(expected)


def test_missing_is_unknown_not_severe():
    assert list(category_name([np.nan, 450, 200, 200.5])) == ["Unknown", "Severe", "Moderate", "Poor"]
    assert list(is_poor([200, 200.01, np.nan])) == [False, True, False]


def test_derived_classification_uses_strict_rule():
    assert cls_metrics_aqi([0, 1], [200, 201])["F1"] == pytest.approx(1.0)


def test_best_threshold_searches_all_scores():
    y = np.array([1] * 9 + [0])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.37, 0.4, 0.5, 0.6, 0.7, 0.36])
    from sklearn.metrics import f1_score
    t = best_f1_threshold(y, s)
    assert f1_score(y, s >= t) == pytest.approx(0.947, abs=1e-3)


# ------------------------------------------------------------------ cleaning
def test_unit_parsing_rejects_unknown_units():
    from clean import parse_columns
    out = {r[0]: r for r in parse_columns(["NO2 (ppb)", "SO2 ()", "CO (ppm)", "NH3 (furlongs)", "RH (W/mt2)"])}
    assert out["NO2 (ppb)"][2] == pytest.approx(1.88) and out["NO2 (ppb)"][4] == "converted"
    assert out["SO2 ()"][4] == "assumed"
    assert out["CO (ppm)"][2] == pytest.approx(1.145)
    assert out["NH3 (furlongs)"][2] is None and out["NH3 (furlongs)"][4] == "rejected"
    assert out["RH (W/mt2)"][4] == "assumed"


def test_spike_needs_two_observed_neighbours():
    from clean import spike_mask
    day = pd.Series(pd.to_datetime(["2022-01-01"] * 4))
    s = pd.Series([600.0, 50, 50, 50])            # first hour: no previous neighbour
    assert not spike_mask(s, day, 150).any()
    s = pd.Series([50.0, 600, 50, 50])
    assert list(spike_mask(s, day, 150)) == [False, True, False, False]


def test_stuck_rule_keeps_low_flat_values():
    from clean import stuck_mask
    day = pd.Series(pd.to_datetime(["2022-01-01"] * 24))
    assert not stuck_mask(pd.Series([0.0] * 24), day, 5).any()
    assert stuck_mask(pd.Series([80.0] * 24), day, 5).all()


def test_wind_direction_and_hourly_aggregation():
    from clean import to_hourly, wind_direction
    t = pd.Series(pd.date_range("2022-01-01", periods=24, freq="h"))
    h = pd.DataFrame({"time": t, "WS": [2.0] * 12 + [np.nan] * 12, "WD": [90.0] * 24})
    wd = wind_direction(h, t.dt.floor("D"))
    assert wd.iloc[0] == pytest.approx(90.0)
    t2 = pd.Series(pd.date_range("2022-01-01", periods=8, freq="15min"))
    hh, info = to_hourly(t2, pd.DataFrame({"PM25": np.arange(8.0)}))
    assert len(hh) == 2 and hh.PM25.tolist() == [1.5, 5.5] and info["n_merged"] == 6
    empty, info = to_hourly(pd.Series(pd.to_datetime([None, None])), pd.DataFrame({"PM25": [1.0, 2.0]}))
    assert empty.empty


# ------------------------------------------------------------------ features
def synthetic_daily(n_days=120, seed=0):
    from features import INTRADAY, POLL_L0, SITE
    rng = np.random.default_rng(seed)
    rows = []
    stations = [("XX001", "Maharashtra", "Aurangabad"), ("XX002", "Maharashtra", "Aurangabad"),
                ("XX003", "Bihar", "Aurangabad")]
    for st, state, city in stations:
        dates = pd.date_range("2020-10-01", periods=n_days, freq="D")
        base = 150 + 60 * np.sin(np.arange(n_days) / 9) + rng.normal(0, 20, n_days)
        for i, dte in enumerate(dates):
            r = {"station": st, "state": state, "city": city, "date": dte, "AQI": max(5, base[i]),
                 "PM25_nh": 24, "PM10_nh": 24, "AQI_n_sub": 5, "site_WD": 180.0}
            for c in POLL_L0 + INTRADAY + SITE:
                r[c] = max(0.1, base[i] / 3 + rng.normal(0, 5))
            rows.append(r)
    d = pd.DataFrame(rows)
    # gaps: station 1 loses 10 days, station 3 (other Aurangabad) is the only one in its city
    gap = (d.station == "XX001") & d.date.between("2020-11-10", "2020-11-19")
    d.loc[gap, ["AQI", "PM25_mean"]] = np.nan
    return d.drop(index=d.index[(d.station == "XX002") & d.date.between("2020-11-08", "2020-11-25")])


def geo_for(d):
    return pd.DataFrame({"station": ["XX001", "XX002", "XX003"], "lat": [19.9, 19.9, 24.75],
                         "lon": [75.3, 75.3, 84.37], "elevation": [500, 500, 100]})


def empty_fires():
    from features import fire_counts, fire_events
    f = pd.DataFrame({"latitude": [np.nan, 20.0], "longitude": [75.0, np.nan], "acq_date": ["2020-10-05"] * 2,
                      "acq_time": [500, 500], "confidence": [80, 80], "frp": [5.0, 5.0]})
    return fire_counts(geo_for(None), fire_events(f))


def test_fire_edge_cases_do_not_crash():
    fc, reg, located = empty_fires()
    assert len(fc) == 0 and len(located) == 3


def test_firms_utc_to_ist_and_latency():
    from features import fire_events
    f = pd.DataFrame({"latitude": [20, 20], "longitude": [75, 75], "acq_date": ["2022-11-01", "2022-11-01"],
                      "acq_time": [2000, 1000], "confidence": [80, 80], "frp": [1, 1]})
    out = fire_events(f)
    # 20:00 UTC = 01:30 IST next day (+3 h latency) -> issue day 2 Nov; 10:00 UTC = 15:30 IST -> 1 Nov
    assert list(out.date.dt.strftime("%Y-%m-%d")) == ["2022-11-02", "2022-11-01"]


def test_features_are_causal_and_aligned():
    from features import build
    d = synthetic_daily()
    fires = empty_fires()
    full, groups, _ = build(d, geo_for(d), fires)
    cut = pd.Timestamp("2020-12-01")
    part, _, _ = build(d[d.date <= cut], geo_for(d), fires)
    feats = [c for cs in groups.values() for c in cs if c in full.columns]
    a = full[full.date <= cut].set_index(["station", "date"])[feats].sort_index()
    b = part.set_index(["station", "date"])[feats].sort_index().reindex(a.index)
    assert np.allclose(a.values.astype(float), b.values.astype(float), equal_nan=True), "future data changed a feature"
    # target = AQI of the next calendar day
    x = full[(full.station == "XX003")].set_index("date")
    assert x.y_aqi.iloc[5] == pytest.approx(d[(d.station == "XX003")].set_index("date").AQI.iloc[6])
    # Aurangabad (Bihar) has no same-city neighbour: the two Aurangabads are separate cities
    assert full[full.station == "XX003"].city_aqi_l0.isna().all()
    # raw missing flag set on the gap even though the lag value is filled
    g = full[(full.station == "XX001") & (full.date == "2020-11-12")].iloc[0]
    assert g.isna_PM25_mean_l0 == 1 and g.miss_aqi_l0 == 1


def test_forward_fill_limit_is_calendar_days():
    from features import causal_fill, complete_grid
    d = pd.DataFrame({"station": "S1", "state": "A", "city": "B",
                      "date": pd.to_datetime(["2022-01-01", "2022-02-01"]), "AQI": [10.0, np.nan]})
    g = complete_grid(d)
    v = causal_fill(g, ["AQI"], how="ffill")["AQI"]
    assert v.iloc[1:4].tolist() == [10.0] * 3 and v.iloc[4:].isna().all()


def test_imputer_fold_isolation_and_leave_one_year_out():
    from features import build
    from prep import Imputer
    d = synthetic_daily(n_days=900)                     # Oct 2020 .. Mar 2023
    F, groups, _ = build(d, geo_for(d), empty_fires())
    F = F[F.y_aqi.notna()].reset_index(drop=True)
    feats = [c for cs in groups.values() for c in cs]
    tr = (F.target_date <= "2022-06-30").values         # Marches of 2021 and 2022 are in training
    imp1 = Imputer("clim", feats).fit(F, tr)
    F2 = F.copy()
    F2.loc[~tr, ["y_aqi", "aqi_l0", "PM25_mean_l0"]] = 9999
    imp2 = Imputer("clim", feats).fit(F2, tr)
    assert np.allclose(imp1.transform(F).values, imp2.transform(F).values, equal_nan=True)
    # leave-one-year-out: a training row's climatology does not contain its own year
    c = imp1.target_climatology(F)
    r = F[tr & (F.station == "XX003") & (F.target_date.dt.year == 2021) & (F.target_date.dt.month == 3)]
    other = F[tr & (F.station == "XX003") & (F.target_date.dt.month == 3) & (F.target_date.dt.year != 2021)]
    assert len(r) and len(other)
    assert c[r.index[0]] == pytest.approx(other.y_aqi.mean(), rel=1e-5)
    # a validation row (not in the fit) uses all training years
    v = F[~tr & (F.station == "XX003") & (F.target_date.dt.month == 3)]
    allm = F[tr & (F.station == "XX003") & (F.target_date.dt.month == 3)]
    assert c[v.index[0]] == pytest.approx(allm.y_aqi.mean(), rel=1e-5)


def test_every_model_family_runs():
    import zoo
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)).astype("float32")
    y = 100 + 30 * X[:, 0] + rng.normal(0, 5, 600)
    yc = (y > 110).astype(int)
    for name in zoo.MODEL_NAMES:
        for task, yy in [("reg", y), ("cls", yc)]:
            for kind in ["raw", "pca"]:
                p = zoo.grid(name, task)[0]
                m, stages, n_best, pv = zoo.fit_eval(name, task, kind, p, X[:400], yy[:400], X[400:], yy[400:])
                assert len(pv) == 200 and np.isfinite(stages).all()
                if name in zoo.BOOSTERS:
                    assert 1 <= n_best <= zoo.MAX_ROUNDS[name]
                    final = zoo.fit_final(name, task, kind, p, X, yy, n_rounds=n_best)
                    assert len(zoo.predict(final, task, X[:5])) == 5


def test_openaq_matching():
    from external import match_openaq
    geo = pd.DataFrame({"station": ["DL001", "DL002", "AP002"], "station_location": [
        "Anand Vihar, Delhi", "Some Unknown Place, Delhi", "PWD Grounds, Vijayawada"],
        "city_lat": [28.63, 28.63, 16.5], "city_lon": [77.22, 77.22, 80.64]})
    loc = pd.DataFrame({"name": ["Anand Vihar, Delhi - DPCC", "PWD Grounds, Vijayawada - APPCB", "Far Away, Mumbai"],
                        "lat": [28.647, 16.507, 19.07], "lon": [77.316, 80.627, 72.87], "is_mobile": [False] * 3})
    out = match_openaq(geo, loc).set_index("station")
    assert out.loc["DL001", "coord_source"] == "openaq_station" and out.loc["DL001", "lat"] == pytest.approx(28.647)
    assert out.loc["DL002", "coord_source"] == "city_centroid" and out.loc["DL002", "lat"] == pytest.approx(28.63)
    assert out.loc["AP002", "coord_source"] == "openaq_station"


def test_fast_adaboost_staging_matches_sklearn():
    import zoo
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    y = 50 + 20 * X[:, 0] + rng.normal(0, 3, 500)
    m = zoo.make("AdaBoost", "reg", dict(depth=3, loss="linear"), 30).fit(X[:300], y[:300])
    P = np.column_stack([e.predict(X[300:]) for e in m.estimators_])
    staged = list(m.staged_predict(X[300:]))
    for k in [1, 10, 20, len(m.estimators_)]:
        assert np.allclose(zoo._weighted_median(P[:, :k], np.asarray(m.estimator_weights_[:k])), staged[k - 1])
