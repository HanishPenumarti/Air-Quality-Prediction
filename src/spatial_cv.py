"""Spatial generalisation check: leave-one-city-out (LOCO).

Main benchmark = "known stations" (chronological split). Here we ask: how well does the
model forecast for a city it has NEVER seen in training?
For each held-out city: train on all OTHER cities (<= 2021), test on the held-out city
(2022-01 .. 2023-03). Compared with the known-station model on exactly the same rows.
Note: neighbour features of the held-out city come from its own other stations on day D
(they would exist in a deployment), but no row of that city is used for fitting.
Outputs: reports/loco.csv, figures/loco.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from config import FIG, PROC, REPORTS, SEED, VAL_END  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Hyderabad", "Chennai", "Lucknow", "Patna",
          "Jaipur", "Ahmedabad", "Pune", "Guwahati"]


def main():
    F, feats, _ = load()
    t = F.target_date
    trv, te = (t <= VAL_END).values, (t > VAL_END).values
    b = pd.read_csv(REPORTS / "benchmark.csv")
    xgb_params = json.loads(b[(b.model == "XGBoost") & (b.task == "regression") & (b.features == "raw")].params.iloc[0])
    preds = pd.read_parquet(PROC / "preds_test.parquet")
    y = F.y_aqi.values
    rows = []
    for city in CITIES:
        in_city = (F.city == city).values
        if (in_city & te).sum() < 200:
            continue
        trm = trv & ~in_city
        # imputer climatology from training cities only; held-out city falls back to medians
        X = Imputer("ffill_clim", feats).fit(F, trm).transform(F).astype("float32").values
        tm = te & in_city
        known = preds[preds.city == city]
        for name, mdl in [("XGBoost", XGBRegressor(n_estimators=600, subsample=0.8, colsample_bytree=0.7,
                                                   tree_method="hist", n_jobs=12, random_state=SEED, **xgb_params)),
                          ("RandomForest", RandomForestRegressor(n_estimators=80, max_depth=20, min_samples_leaf=10,
                                                                 max_features=0.33, n_jobs=12, random_state=SEED))]:
            mdl.fit(X[trm], y[trm])
            mae = float(np.mean(np.abs(mdl.predict(X[tm]) - y[tm])))
            known_mae = float(np.mean(np.abs(known[f"{name}|reg|raw"] - known.y)))
            pers = float(np.mean(np.abs(known["Persistence|reg"] - known.y)))
            rows.append(dict(city=city, model=name, n_test=int(tm.sum()), MAE_unseen_city=mae,
                             MAE_known_station=known_mae, MAE_persistence=pers))
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(REPORTS / "loco.csv", index=False)

    x = r[r.model == "XGBoost"].sort_values("MAE_known_station")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xx = np.arange(len(x))
    ax.bar(xx - 0.27, x.MAE_persistence, 0.26, color="#c3c2b7", label="persistence")
    ax.bar(xx, x.MAE_known_station, 0.26, color=SERIES[0], label="XGBoost, city seen in training")
    ax.bar(xx + 0.27, x.MAE_unseen_city, 0.26, color=SERIES[1], label="XGBoost, city held out (LOCO)")
    ax.set_xticks(xx, x.city, rotation=25, ha="right")
    ax.set_ylabel("test MAE (2022-23)")
    ax.set_title("Generalisation to unseen cities (leave-one-city-out)")
    ax.legend(fontsize=8)
    ax.text(len(x) - 0.5, ax.get_ylim()[1] * 0.95,
            f"mean penalty: +{(x.MAE_unseen_city - x.MAE_known_station).mean():.2f} AQI pts",
            ha="right", fontsize=8, color=TEXT2)
    save(fig, FIG / "loco.png")


if __name__ == "__main__":
    main()
