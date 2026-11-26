"""Spatial generalisation: buffered leave-one-city-out and leave-one-region-out.

Scenario (stated explicitly): a NEW city/region whose monitors already report live data, but
none of whose history was available for training. Concretely, for each hold-out:
  * none of its rows are used for fitting the model, the imputer or the climatology
    (fallbacks for it come from other cities/regions only)
  * its stations are excluded as DONORS of the neighbour features of the training rows
    (e.g. other Haryana stations' state average is recomputed without the held-out city)
  * buffer: training also drops every station within BUFFER_KM of a held-out station, because
    nearby stations are highly correlated (r ~ 0.75 at 50-100 km) - e.g. Noida/Gurugram for Delhi
  * at test time the held-out stations use their own live data (own lags, own same-city
    neighbours), which a deployment would have.
Matched comparison: the "seen" model uses IDENTICAL settings (benchmark params + rounds) and the
standard training set (<= 2021, everything included); both are evaluated on the same test rows.
Models: XGBoost and Random Forest (raw features, regression). 95 % block-bootstrap intervals
(blocks = ISO weeks) for the penalty.
Outputs: reports/loco.csv, figures/loco.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, MODELS, PROC, REPORTS, SEED, VAL_END, city_key, region_of  # noqa: E402
from features import SPATIAL_COLS, spatial_features  # noqa: E402
from metrics import block_bootstrap  # noqa: E402
from models import persistence  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
import zoo  # noqa: E402

CITIES = [("Delhi", "Delhi"), ("Maharashtra", "Mumbai"), ("Karnataka", "Bengaluru"), ("West Bengal", "Kolkata"),
          ("Telangana", "Hyderabad"), ("Tamil Nadu", "Chennai"), ("Uttar Pradesh", "Lucknow"), ("Bihar", "Patna"),
          ("Rajasthan", "Jaipur"), ("Gujarat", "Ahmedabad"), ("Maharashtra", "Pune"), ("Assam", "Guwahati")]
BUFFER_KM = 100
MODELS_USED = ["XGBoost", "RandomForest"]


def hav(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1)[:, None], np.radians(la2)[None, :]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2[None, :] - lo1[:, None]) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    F, feats, _ = load()
    fp = json.loads((MODELS / "final_params.json").read_text())
    grid = pd.read_parquet(PROC / "grid.parquet")
    t = F.target_date
    trv, te = (t <= VAL_END).values, (t > VAL_END).values
    y = F.y_aqi.values.astype(float)
    region = F.state.map(region_of).values
    st_xy = F.groupby("station")[["lat", "lon"]].first()

    # "seen" models: standard training set, identical settings (fitted once)
    imp_seen = Imputer("clim", feats).fit(F, trv)
    X_seen = imp_seen.transform(F).astype("float32").values
    seen_models = {nm: zoo.fit_final(nm, "reg", "raw", fp[nm]["reg"]["raw"]["params"], X_seen[trv], y[trv],
                                     fp[nm]["reg"]["raw"]["n_rounds"]) for nm in MODELS_USED}
    clim_seen = imp_seen.target_climatology(F).values.astype(float)
    pers = persistence(F, clim_seen)

    holdouts = [("city", f"{c} ({s})", (F.city_key == city_key(s, c)).values) for s, c in CITIES]
    holdouts += [("region", r, region == r) for r in sorted(set(region))]
    rows = []
    for kind, label, held in holdouts:
        tm = te & held
        if tm.sum() < 200:
            continue
        held_st = F.station[held].unique()
        excl = held.copy()
        if kind == "city":
            xy = st_xy.dropna()
            hx = xy.loc[[s for s in held_st if s in xy.index]]
            if len(hx):
                dist = hav(xy.lat.values, xy.lon.values, hx.lat.values, hx.lon.values).min(axis=1)
                near = set(xy.index[dist <= BUFFER_KM])
                excl |= F.station.isin(near).values
        trm = trv & ~excl
        # neighbour features of the training rows recomputed without the held-out stations as donors
        donor = ~grid.station.isin(held_st).values
        sp = spatial_features(grid, donor=donor)
        sp = pd.concat([grid[["station", "date"]], sp[SPATIAL_COLS]], axis=1)
        Fh = F.copy()
        alt = Fh[["station", "date"]].merge(sp, on=["station", "date"], how="left")
        for c in SPATIAL_COLS:
            Fh.loc[~held, c] = alt.loc[~held, c].values
        X = Imputer("clim", feats).fit(Fh, trm).transform(Fh).astype("float32").values
        wk = pd.to_datetime(t[tm]).dt.strftime("%G-%V").values
        yt = y[tm]
        for nm in MODELS_USED:
            q = fp[nm]["reg"]["raw"]
            mdl = zoo.fit_final(nm, "reg", "raw", q["params"], X[trm], y[trm], q["n_rounds"])
            ph = mdl.predict(X[tm]).astype(float)
            ps = seen_models[nm].predict(X_seen[tm]).astype(float)
            pp = pers[tm]
            ci = block_bootstrap(wk, {"pen": lambda i: np.mean(np.abs(ph[i] - yt[i]) - np.abs(ps[i] - yt[i]))},
                                 B=500, seed=SEED)["pen"]
            rows.append(dict(holdout=kind, unit=label, model=nm, n_test=int(tm.sum()), n_train=int(trm.sum()),
                             n_buffer_stations=int(len(set(F.station[excl & ~held]))),
                             MAE_held_out=float(np.mean(np.abs(ph - yt))), MAE_seen=float(np.mean(np.abs(ps - yt))),
                             MAE_persistence=float(np.mean(np.abs(pp - yt))),
                             penalty=float(np.mean(np.abs(ph - yt)) - np.mean(np.abs(ps - yt))),
                             penalty_lo95=ci[0], penalty_hi95=ci[1]))
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(REPORTS / "loco.csv", index=False)
    plot(r)


def plot(r):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for ax, kind, title in [(axes[0], "city", f"Leave-one-city-out (+{BUFFER_KM} km buffer)"),
                            (axes[1], "region", "Leave-one-region-out")]:
        x = r[(r.model == "XGBoost") & (r.holdout == kind)].sort_values("MAE_seen")
        if x.empty:
            continue
        xx = np.arange(len(x))
        ax.bar(xx - 0.27, x.MAE_persistence, 0.26, color="#c3c2b7", label="persistence")
        ax.bar(xx, x.MAE_seen, 0.26, color=SERIES[0], label="XGBoost, unit seen in training")
        ax.bar(xx + 0.27, x.MAE_held_out, 0.26, color=SERIES[1], label="XGBoost, unit held out")
        ax.set_xticks(xx, x.unit.str.replace(r" \(.*\)", "", regex=True), rotation=25, ha="right")
        ax.set_title(title)
        ax.text(len(x) - 0.5, ax.get_ylim()[1] * 0.95, f"mean penalty: {x.penalty.mean():+.2f} AQI pts",
                ha="right", fontsize=8, color=TEXT2)
    axes[0].set_ylabel("test MAE (2022-23)")
    axes[0].legend(fontsize=8)
    save(fig, FIG / "loco.png")


if __name__ == "__main__":
    main()
