"""Compare imputation strategies (brief: 'compare at least two ... and justify').

Part 1  Reconstruction (causal design, uses the DEPLOYED fill function features.causal_fill):
        * complete daily calendar grid per station; reference statistics from dates <= 2020 only
        * gaps hidden ONLY in 2021 (the validation year; the 2022-23 test period is never touched):
          disjoint, calendar-contiguous runs of 1-7 days placed on fully observed stretches,
          about 10 % of the observed 2021 values (the achieved fraction is reported)
        * methods: ffill <= 3 days | same-city neighbours | station-month climatology (<= 2020,
          hierarchy station -> city -> region -> global, as prep.Imputer) | DEPLOYED hybrid
          (neighbours -> ffill <= 3 days -> climatology) | linear interpolation (uses FUTURE values,
          offline reference only)
        * MAE on each method's own coverage AND on the common support, by true gap length;
          3 random seeds (mean and spread); variables: daily AQI and daily PM2.5
Part 2  Downstream (fixed XGBoost settings, fit <= 2020, validation 2021):
        (a) history-level: the whole feature table is rebuilt from raw daily data with each history
            fill (none | ffill | neighbour | hybrid); residual gaps then use the same 'clim' imputer
        (b) residual-level on the deployed hybrid history: native NaN (XGBoost) | global median | clim
Outputs: reports/imputation_results.json, figures imp_01/imp_02
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aqi import is_poor  # noqa: E402
from config import EXT, FIG, PROC, REPORTS, SEED, TRAIN_END, VAL_END, region_of  # noqa: E402
from features import build, causal_fill, complete_grid, load_fires  # noqa: E402
from prep import Imputer  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402

MASK_YEAR = 2021
TARGET_FRAC = 0.10


def place_gaps(d, var, rng):
    """Disjoint contiguous gaps (1-7 days) on fully observed stretches in MASK_YEAR."""
    obs = d[var].notna().values
    in_year = (d.date.dt.year == MASK_YEAR).values
    st = d.station.values
    cand = np.flatnonzero(obs & in_year)
    target = int(TARGET_FRAC * len(cand))
    mask = np.zeros(len(d), bool)
    gap_len = np.zeros(len(d), int)
    blocked = np.zeros(len(d), bool)
    n_masked = 0
    for s in rng.permutation(cand):
        if n_masked >= target:
            break
        L = int(rng.integers(1, 8))
        e = s + L
        if e > len(d) or (st[s:e] != st[s]).any() or not obs[s:e].all() or not in_year[s:e].all() \
                or blocked[max(0, s - 1):e + 1].any():
            continue
        mask[s:e] = True
        gap_len[s:e] = L
        blocked[s:e] = True
        n_masked += L
    return mask, gap_len, target / max(1, len(cand)), mask.sum() / max(1, len(cand))


def climatology(d, filled):
    """Station x month median of the (filled) history over dates <= 2020, then city, region, global."""
    ref = (d.date <= TRAIN_END).values
    mon = d.date.dt.month
    keys = {"station": d.station, "city": d.city_key, "region": d.state.map(region_of)}
    out = pd.Series(np.nan, index=d.index)
    for lv in ["station", "city", "region"]:
        med = filled[ref].groupby([keys[lv][ref], mon[ref]]).median()
        out = out.fillna(pd.Series(med.reindex(pd.MultiIndex.from_arrays([keys[lv], mon])).values, index=d.index))
    return out.fillna(filled[ref].median())


def reconstruction(d):
    rows, by_gap = [], []
    for var in ["AQI", "PM25_mean"]:
        for seed in [SEED, SEED + 1, SEED + 2]:
            rng = np.random.default_rng(seed)
            mask, gap_len, f_req, f_got = place_gaps(d, var, rng)
            truth = d[var].values
            dm = d.copy()
            dm.loc[mask, var] = np.nan
            est = {}
            est["ffill <= 3 days"] = causal_fill(dm, [var], how="ffill")[var]
            est["same-city neighbours"] = causal_fill(dm, [var], how="neighbour")[var]
            hyb = causal_fill(dm, [var], how="hybrid")[var]
            clim = climatology(dm, hyb)
            est["station-month climatology (<= 2020)"] = clim
            est["DEPLOYED hybrid: neighbour > ffill<=3 > climatology"] = hyb.fillna(clim)
            est["linear interpolation (uses future, offline)"] = dm[var].groupby(dm.station.values).transform(
                lambda s: s.interpolate(limit_area="inside"))
            common = mask.copy()
            for e in est.values():
                common &= e.notna().values
            for name, e in est.items():
                err = np.abs(e.values - truth)
                own = mask & np.isfinite(err)
                rows.append(dict(variable=var, seed=seed, method=name, masked_fraction=f_got,
                                 coverage=float(own.sum() / mask.sum()), mae_own=float(err[own].mean()),
                                 mae_common=float(err[common].mean()), n_common=int(common.sum())))
                for L in range(1, 8):
                    m = own & (gap_len == L)
                    if m.any():
                        by_gap.append(dict(variable=var, seed=seed, method=name, gap=L, mae=float(err[m].mean()),
                                           n=int(m.sum())))
            print(f"{var} seed {seed}: masked {f_got:.3f} of observed {MASK_YEAR} values", flush=True)
    r = pd.DataFrame(rows)
    g = pd.DataFrame(by_gap)
    summ = r.groupby(["variable", "method"]).agg(masked_fraction=("masked_fraction", "mean"),
                                                 coverage=("coverage", "mean"), mae_own=("mae_own", "mean"),
                                                 mae_own_sd=("mae_own", "std"), mae_common=("mae_common", "mean"),
                                                 mae_common_sd=("mae_common", "std")).reset_index()
    gg = g.groupby(["variable", "method", "gap"]).mae.mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, var in zip(axes, ["AQI", "PM25_mean"]):
        for k, name in enumerate(summ.method.unique()):
            s = gg[(gg.variable == var) & (gg.method == name)]
            ax.plot(s.gap, s.mae, marker="o", ms=4, label=name, color=SERIES[k],
                    ls=":" if "future" in name else "-")
        ax.set_xlabel("true gap length (calendar days)")
        ax.set_ylabel(f"MAE of reconstructed daily {'AQI' if var == 'AQI' else 'PM2.5 (ug/m3)'}")
        ax.set_title(f"{var}: hidden in {MASK_YEAR}, statistics from <= 2020 (3 seeds)")
    axes[0].legend(fontsize=7)
    save(fig, FIG / "imp_01_reconstruction.png")
    return summ, gg


def xgb_val(F, feats, strategy):
    import xgboost as xgb
    from sklearn.metrics import f1_score
    t = F.target_date
    tr = (t <= TRAIN_END).values
    va = ((t > TRAIN_END) & (t <= VAL_END)).values
    X = Imputer(strategy, feats).fit(F, tr).transform(F)
    m = xgb.XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05, min_child_weight=20, subsample=0.8,
                         colsample_bytree=0.7, tree_method="hist", n_jobs=12, random_state=SEED)
    m.fit(X[tr], F.y_aqi.values[tr])
    p = m.predict(X[va])
    y = F.y_aqi.values[va]
    return dict(val_mae=float(np.mean(np.abs(p - y))), val_f1_poor=float(f1_score(is_poor(y), is_poor(p))))


def downstream(d0, geo, fires):
    hist, resid = [], []
    for fill in ["none", "ffill", "neighbour", "hybrid"]:
        F, groups, _ = build(d0, geo, fires, fill=fill)
        F = F[F.y_aqi.notna()].reset_index(drop=True)
        feats = [c for cs in groups.values() for c in cs]
        hist.append(dict(history_fill=fill, residual="clim", **xgb_val(F, feats, "clim")))
        print(hist[-1], flush=True)
        if fill == "hybrid":
            for strat in ["native", "median", "clim"]:
                resid.append(dict(history_fill="hybrid", residual=strat, **xgb_val(F, feats, strat)))
                print(resid[-1], flush=True)
    h, r = pd.DataFrame(hist), pd.DataFrame(resid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    for ax, df, col, title in [(axes[0], h, "history_fill", "(a) history fill, residual = clim"),
                               (axes[1], r, "residual", "(b) residual fill, history = hybrid")]:
        ax.barh(df[col], df.val_mae, color=SERIES[0], height=0.55)
        for i, v in enumerate(df.val_mae):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=TEXT2)
        ax.set_xlim(df.val_mae.min() * 0.95, df.val_mae.max() * 1.04)
        ax.set_xlabel("validation MAE 2021 (XGBoost, fixed settings)")
        ax.set_title(title)
    save(fig, FIG / "imp_02_downstream.png")
    return h, r


def main():
    d0 = pd.read_parquet(PROC / "daily.parquet")
    geo = pd.read_csv(EXT / "stations_geo.csv")
    d = complete_grid(d0)
    summ, gg = reconstruction(d)
    print(summ.round(3).to_string())
    fires = load_fires(geo)
    h, r = downstream(d0, geo, fires)
    res = {"design": dict(mask_year=MASK_YEAR, reference_period=f"<= {TRAIN_END}", target_masked_fraction=TARGET_FRAC,
                          gaps="disjoint contiguous 1-7 days on observed stretches", seeds=3),
           "reconstruction": summ.to_dict("records"), "reconstruction_by_gap": gg.to_dict("records"),
           "downstream_history": h.to_dict("records"), "downstream_residual": r.to_dict("records")}
    (REPORTS / "imputation_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
