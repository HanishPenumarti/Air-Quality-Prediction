"""Extreme-day experiment (part of Analysis E) + prediction intervals.

Every learned model under-forecasts Severe days (AQI > 400). Here we test remedies on the
2021 VALIDATION year only (models fitted <= 2020), then report the test result once:
  * reference      : XGBoost with the benchmark's tuned settings
  * weighted w     : sample weight 1 + w for AQI(D+1) > 300 (w = 2, 5)
  * quantile a     : quantile loss at alpha = 0.6, 0.7 (forecast an upper-ish quantile)
  * delta          : learn the CHANGE from today's AQI (AQI(D+1) - aqi_l0), add it back
  * blend a        : a * model + (1 - a) * persistence, a chosen on 2020 (model fitted <= 2019)
Selection rule (stated before looking at test): a variant replaces the production regressor only if
its validation MAE is at most 1 % worse than the production model's AND its validation MAE on
Severe days is lower AND its Severe recall is not lower. Otherwise production is unchanged.

Prediction intervals (80 %): residual quantiles of the <= 2020 production-type model on 2021,
per forecast category, added to the final model's test forecasts; empirical coverage is checked
by season and by true category (no guarantee under drift - it is measured, not assumed).

Outputs: reports/severe.csv, reports/intervals_coverage.csv, models/intervals.json,
         models/production.json (updated only if a variant is selected), figure err_08_severe.png
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aqi import category_index, is_poor  # noqa: E402
from config import AQI_CATEGORIES, FIG, MODELS, REPORTS, TRAIN_END, VAL_END  # noqa: E402
from metrics import onset_metrics, reg_metrics  # noqa: E402
from models import persistence  # noqa: E402
from prep import Imputer, load  # noqa: E402
from viz import SERIES, TEXT2, plt, save  # noqa: E402
import zoo  # noqa: E402


def xgb(p, n, **kw):
    m = zoo.make("XGBoost", "reg", p, n)
    m.set_params(**kw)
    return m


def evaluate(F, idx, y, pred, today):
    r = reg_metrics(y[idx], pred)
    r.update(poor_recall=float(np.mean(is_poor(pred)[is_poor(y[idx])])),
             poor_false_alert=float(np.mean(is_poor(pred)[~is_poor(y[idx])])))
    r.update(onset_metrics(today[idx], y[idx], is_poor(pred)))
    return {k: r[k] for k in ["MAE", "RMSE", "MAE_top5pct", "MAE_severe", "bias_severe", "severe_recall",
                              "poor_recall", "poor_false_alert", "onset_recall", "false_alert_on_clean_days"]}


def main():
    F, feats, _ = load()
    fp = json.loads((MODELS / "final_params.json").read_text())
    prod = json.loads((MODELS / "production.json").read_text())
    px = fp["XGBoost"]["reg"]["raw"]
    p, n = px["params"], px["n_rounds"]
    t = F.target_date
    y = F.y_aqi.values.astype(float)
    tr19, tr20 = (t <= "2019-12-31").values, (t <= TRAIN_END).values
    va20 = ((t > "2019-12-31") & (t <= TRAIN_END)).values
    va21 = ((t > TRAIN_END) & (t <= VAL_END)).values
    trv, te = (t <= VAL_END).values, (t > VAL_END).values
    today = F.aqi_l0.where(F.miss_aqi_l0 == 0).values.astype(float)        # measured AQI(D) only

    def design(mask_fit):
        imp = Imputer("clim", feats).fit(F, mask_fit)
        X = imp.transform(F).astype("float32")
        return X.values, imp.target_climatology(F).values.astype(float), list(X.columns)

    X20, clim20, cols = design(tr20)
    X21, clim21, _ = design(trv)
    a0 = cols.index("aqi_l0")
    pers20, pers21 = persistence(F, clim20), persistence(F, clim21)

    variants = {
        "reference": lambda X, m: xgb(p, n).fit(X[m], y[m]),
        "weighted w=2": lambda X, m: xgb(p, n).fit(X[m], y[m], sample_weight=1 + 2 * (y[m] > 300)),
        "weighted w=5": lambda X, m: xgb(p, n).fit(X[m], y[m], sample_weight=1 + 5 * (y[m] > 300)),
        "quantile 0.6": lambda X, m: xgb(p, n, objective="reg:quantileerror", quantile_alpha=0.6).fit(X[m], y[m]),
        "quantile 0.7": lambda X, m: xgb(p, n, objective="reg:quantileerror", quantile_alpha=0.7).fit(X[m], y[m]),
        "delta": lambda X, m: xgb(p, n).fit(X[m], y[m] - X[m][:, a0]),
    }

    def predict(vname, model, X, idx):
        out = model.predict(X[idx]).astype(float)
        return np.clip(out + X[idx][:, a0] if vname == "delta" else out, 0, 500)

    rows, val_pred, test_pred = [], {}, {}
    for vname, fit in variants.items():
        m20 = fit(X20, tr20)
        val_pred[vname] = predict(vname, m20, X20, va21)
        rows.append(dict(variant=vname, split="validation 2021", **evaluate(F, va21, y, val_pred[vname], today)))
        m21 = fit(X21, trv)
        test_pred[vname] = predict(vname, m21, X21, te)
        rows.append(dict(variant=vname, split="test 2022-23", **evaluate(F, te, y, test_pred[vname], today)))
        joblib.dump(m21, MODELS / f"severe_{vname.replace(' ', '_').replace('=', '')}.joblib", compress=3)
        print(rows[-2], flush=True)

    # blend weight chosen on 2020 (model <= 2019), so the 2021 validation score is honest
    X19, clim19, _ = design(tr19)
    m19 = variants["reference"](X19, tr19)
    pv20 = m19.predict(X19[va20]).astype(float)
    p19 = persistence(F, clim19)[va20]
    alphas = np.linspace(0, 1, 21)
    a_best = float(alphas[np.argmin([np.mean(np.abs(a * pv20 + (1 - a) * p19 - y[va20])) for a in alphas])])
    vname = f"blend a={a_best:.2f}"
    val_pred[vname] = a_best * val_pred["reference"] + (1 - a_best) * pers20[va21]
    test_pred[vname] = a_best * test_pred["reference"] + (1 - a_best) * pers21[te]
    rows.append(dict(variant=vname, split="validation 2021", **evaluate(F, va21, y, val_pred[vname], today)))
    rows.append(dict(variant=vname, split="test 2022-23", **evaluate(F, te, y, test_pred[vname], today)))
    rows.append(dict(variant="persistence", split="validation 2021", **evaluate(F, va21, y, pers20[va21], today)))
    rows.append(dict(variant="persistence", split="test 2022-23", **evaluate(F, te, y, pers21[te], today)))
    r = pd.DataFrame(rows)

    # ---- selection on validation (rule fixed above) -----------------------------------
    pv = prod["validation_scores"][prod["regressor"]]                   # production val MAE (<= 2020 fit)
    ref_sev = r[(r.variant == "reference") & (r.split == "validation 2021")].iloc[0]
    cand = r[(r.split == "validation 2021") & ~r.variant.isin(["reference", "persistence"])]
    ok = cand[(cand.MAE <= 1.01 * pv) & (cand.MAE_severe < ref_sev.MAE_severe) &
              (cand.severe_recall >= ref_sev.severe_recall)]
    r["selected"] = False
    if len(ok):
        win = ok.sort_values("MAE_severe").iloc[0].variant
        r.loc[r.variant == win, "selected"] = True
        if not win.startswith("blend"):
            prod["regressor_override"] = dict(file=f"severe_{win.replace(' ', '_').replace('=', '')}.joblib",
                                              variant=win, delta=win == "delta",
                                              reason="severe-day rule on 2021 validation")
            (MODELS / "production.json").write_text(json.dumps(prod, indent=1))
        print("selected variant:", win)
    else:
        print("no variant met the selection rule - production regressor unchanged")
    r.to_csv(REPORTS / "severe.csv", index=False)

    # ---- prediction intervals for the production model -----------------------------------
    name, task, kind = prod["regressor"].split("|")
    pp = fp[name][task][kind]
    m20 = zoo.fit_final(name, task, kind, pp["params"], X20[tr20], y[tr20], pp["n_rounds"])
    resid = y[va21] - zoo.predict(m20, "reg", X20[va21])
    cat_v = category_index(zoo.predict(m20, "reg", X20[va21]))
    q = {}
    for c in range(6):
        e = resid[cat_v == c] if (cat_v == c).sum() >= 200 else resid
        q[AQI_CATEGORIES[c]] = [float(np.quantile(e, 0.1)), float(np.quantile(e, 0.9))]
    (MODELS / "intervals.json").write_text(json.dumps(dict(model=prod["regressor"], level=0.8, by_forecast_category=q,
                                                           source="residuals on 2021 of the <= 2020 fit"), indent=1))
    final = joblib.load(MODELS / f"{name}_{task}_{kind}.joblib")
    pt = zoo.predict(final, "reg", X21[te])
    lo = pt + np.array([q[AQI_CATEGORIES[max(c, 0)]][0] for c in category_index(pt)])
    hi = pt + np.array([q[AQI_CATEGORIES[max(c, 0)]][1] for c in category_index(pt)])
    inside = (y[te] >= lo) & (y[te] <= hi)
    season = pd.Series(t[te].dt.month.values).map(lambda m: "Winter (Nov-Feb)" if m in (11, 12, 1, 2) else
                                                  "Pre-monsoon (Mar-May)" if m in (3, 4, 5) else
                                                  "Monsoon (Jun-Sep)" if m in (6, 7, 8, 9) else "Post-monsoon (Oct)")
    cov = pd.concat([
        pd.DataFrame({"slice": "all", "group": ["all"], "coverage": [inside.mean()], "n": [len(inside)]}),
        pd.DataFrame({"slice": "season", "coverage": pd.Series(inside).groupby(season).mean(),
                      "n": pd.Series(inside).groupby(season).size()}).rename_axis("group").reset_index(),
        pd.DataFrame({"slice": "true category",
                      "coverage": pd.Series(inside).groupby(np.array(AQI_CATEGORIES)[category_index(y[te])]).mean(),
                      "n": pd.Series(inside).groupby(np.array(AQI_CATEGORIES)[category_index(y[te])]).size()})
        .rename_axis("group").reset_index()])
    cov.to_csv(REPORTS / "intervals_coverage.csv", index=False)
    print(cov.to_string())
    plot(r)


def plot(r):
    v = r[r.split == "validation 2021"].set_index("variant")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, lab in zip(axes, ["MAE", "MAE_severe", "severe_recall"],
                            ["MAE, all days", "MAE on Severe days (AQI > 400)", "Severe recall"]):
        ax.barh(v.index, v[col], color=[SERIES[1] if s else SERIES[0] for s in v.selected], height=0.6)
        ax.set_title(lab)
        ax.axvline(v.loc["reference", col], color=TEXT2, ls=":", lw=1)
    axes[0].set_xlim(v.MAE.min() * 0.95, v.MAE.max() * 1.02)
    fig.suptitle("Extreme-day variants on 2021 validation (orange = selected by the stated rule)", fontweight="bold")
    save(fig, FIG / "err_08_severe.png")


if __name__ == "__main__":
    main()
