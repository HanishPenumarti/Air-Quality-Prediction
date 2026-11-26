"""AirCast India - Streamlit UI for the next-day AQI forecasting pipeline.

Run:  streamlit run app/app.py
This is a HISTORICAL FORECAST DEMONSTRATION: it replays the archive (2010 - Mar 2023) with the
models trained on <= 2021. It has no live data ingestion, so it cannot forecast today's date.
Uses the artefacts written by the pipeline (src/run_all.py): data/processed/features_full.parquet,
models/*.joblib + production.json + thresholds.json + intervals.json, reports/*.csv|json, figures.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aqi import UNKNOWN, category_name, is_poor  # noqa: E402
from config import AQI_CATEGORIES, EXT, FIG, MODELS, PROC, REPORTS, VAL_END  # noqa: E402

st.set_page_config(page_title="AirCast India", page_icon="🌫️", layout="wide")

AQI_COLORS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffd700",
              "Poor": "#ff9900", "Very Poor": "#ff0000", "Severe": "#a52a2a", UNKNOWN: "#9e9e9e"}
MODEL_NAMES = ["DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]
NICE = {"DecisionTree": "Decision Tree", "RandomForest": "Random Forest", "AdaBoost": "AdaBoost",
        "GradBoost": "Gradient Boosting", "XGBoost": "XGBoost"}
HEALTH = {   # CPCB health statements for the forecast CATEGORY (information, not an automated action)
    "Good": "Minimal impact.",
    "Satisfactory": "Minor breathing discomfort to sensitive people.",
    "Moderate": "Breathing discomfort to people with lung/heart disease, children and older adults.",
    "Poor": "Breathing discomfort to most people on prolonged exposure.",
    "Very Poor": "Respiratory illness on prolonged exposure.",
    "Severe": "Affects healthy people and seriously impacts those with existing disease.",
    UNKNOWN: "No forecast available (missing inputs).",
}
REQUIRED = {
    "data/processed/features_full.parquet": "python src/run_all.py --from features",
    "data/processed/feature_groups.json": "python src/run_all.py --from features",
    "models/imputer.joblib": "python src/run_all.py --from models",
    "models/production.json": "python src/run_all.py --from models",
    "models/thresholds.json": "python src/run_all.py --from models",
    "data/external/stations_geo.csv": "python src/external.py geo",
}


# ---------------------------------------------------------------- preflight
def preflight():
    paths = {"data/processed": PROC, "models": MODELS, "data/external": EXT}
    missing = []
    for rel, cmd in REQUIRED.items():
        top = "/".join(rel.split("/")[:-1])
        if not (paths[top] / rel.split("/")[-1]).exists():
            missing.append((rel, cmd))
    if not any((MODELS / f"{m}_reg_raw.joblib").exists() for m in MODEL_NAMES):
        missing.append(("models/*_reg_raw.joblib", "python src/run_all.py --from models"))
    if missing:
        st.error("The app needs the pipeline's outputs, and some are missing:")
        st.table(pd.DataFrame(missing, columns=["missing file", "create it with"]))
        st.info("See README.md -> 'How to run'. A small demo can be built with "
                "AQ_STATIONS / AQ_OUT (README -> 'Quick demo').")
        st.stop()


# ---------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    import zoo
    reg, cls = {}, {}
    for m in MODEL_NAMES:
        for kind, store in (("reg", reg), ("cls", cls)):
            f = MODELS / f"{m}_{kind}_raw.joblib"
            if f.exists():
                store[m] = zoo.single_thread(joblib.load(f))
    prod = json.loads((MODELS / "production.json").read_text())
    override = None
    ov = prod.get("regressor_override")
    if ov and (MODELS / ov["file"]).exists():
        override = (zoo.single_thread(joblib.load(MODELS / ov["file"])), ov)
    return reg, cls, joblib.load(MODELS / "imputer.joblib"), prod, override


@st.cache_resource(show_spinner="Preparing feature matrix (one-off) ...")
def load_data():
    F = pd.read_parquet(PROC / "features_full.parquet")
    F = F.sort_values(["station", "date"]).reset_index(drop=True)
    _, _, imp, _, _ = load_models()
    X = imp.transform(F).astype("float32")
    return F, X.values, list(X.columns)


@st.cache_data
def load_reports():
    out = {}
    for name in ["benchmark", "benchmark_ci", "ablation", "permutation_importance", "efficiency", "loco", "severe",
                 "backtest", "errors_by_station", "errors_classification", "leakage_audit", "leakage_checks",
                 "qa_impact", "coverage", "intervals_coverage"]:
        f = REPORTS / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f)
    for name in ["benchmark_meta", "eda_summary", "imputation_results"]:
        f = REPORTS / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


@st.cache_data
def thresholds():
    th = json.loads((MODELS / "thresholds.json").read_text())
    return {m: th.get(f"{m}|cls|raw", {}).get("direct") for m in MODEL_NAMES}


@st.cache_data
def intervals():
    f = MODELS / "intervals.json"
    return json.loads(f.read_text()) if f.exists() else None


def cat_of(a):
    return str(category_name(np.array([a], dtype=float))[0])


def badge(a, label="Forecast AQI", sub=None):
    c = cat_of(a)
    fg = "#000" if c in ("Moderate", "Satisfactory", "Good", "Poor", UNKNOWN) else "#fff"
    val = "-" if not np.isfinite(a) else f"{a:.0f}"
    st.markdown(
        f"""<div style="background:{AQI_COLORS[c]};color:{fg};padding:18px 22px;border-radius:12px">
        <div style="font-size:14px;opacity:.85">{label}</div>
        <div style="font-size:44px;font-weight:700;line-height:1.1">{val}</div>
        <div style="font-size:20px;font-weight:600">{c}</div>
        {f'<div style="font-size:13px;opacity:.85">{sub}</div>' if sub else ''}</div>""", unsafe_allow_html=True)


def headline_regression(X, idx):
    """Production regressor (chosen on 2021 validation; may be a severe-day variant)."""
    reg, _, _, prod, override = load_models()
    _, _, cols = load_data()
    if override:
        mdl, ov = override
        p = mdl.predict(X[idx]).astype(float)
        if ov.get("delta"):
            p = p + X[idx][:, cols.index("aqi_l0")]
        return np.clip(p, 0, 500), f"XGBoost ({ov['variant']})"
    name = prod["regressor"].split("|")[0]
    return np.clip(reg[name].predict(X[idx]).astype(float), 0, 500), NICE[name]


def predict_rows(idx):
    reg, cls, _, _, _ = load_models()
    _, X, _ = load_data()
    thr = thresholds()
    out = {m: np.clip(mdl.predict(X[idx]).astype(float), 0, 500) for m, mdl in reg.items()}
    for m, mdl in cls.items():
        out[f"{m}_score"] = mdl.predict_proba(X[idx])[:, 1]
        out[f"{m}_thr"] = thr.get(m)
    return out


def contributions(i, top=12):
    """XGBoost TreeSHAP attributions; the 'all other features' term makes them sum to the forecast."""
    import xgboost as xgb
    reg, _, _, _, _ = load_models()
    _, X, cols = load_data()
    c = reg["XGBoost"].get_booster().predict(xgb.DMatrix(X[[i]]), pred_contribs=True)[0]
    s = pd.Series(c[:-1], index=cols)
    order = s.abs().sort_values(ascending=False).index
    shown = s[order[:top]]
    shown["all other features"] = s[order[top:]].sum()
    return shown, float(c[-1]), float(c.sum())


def station_picker(F, key):
    info = F.groupby("station")[["state", "city"]].first()
    states = sorted(info.state.dropna().unique())
    c1, c2, c3 = st.columns(3)
    state = c1.selectbox("State", states, index=states.index("Delhi") if "Delhi" in states else 0, key=key + "s")
    cities = sorted(info[info.state == state].city.unique())
    city = c2.selectbox("City", cities, key=key + "c")
    stations = info[(info.state == state) & (info.city == city)].index.tolist()
    geo = pd.read_csv(EXT / "stations_geo.csv").set_index("station")
    loc = geo["station_location"] if "station_location" in geo else pd.Series(dtype=str)
    return c3.selectbox("Station", stations, key=key + "t", format_func=lambda s: f"{s} - {str(loc.get(s, ''))[:40]}")


def period_note(date):
    if date <= pd.Timestamp(VAL_END):
        st.warning("This issue date lies in the TRAINING period (<= 2021): the models have seen these days, so this "
                   "is an in-sample replay, not a forecast skill demonstration. Pick a date from 2022 onwards for "
                   "held-out behaviour.")


# ---------------------------------------------------------------- pages
def page_forecast():
    st.title("🌫️ Next-day AQI forecast (historical demonstration)")
    st.caption("Replays the archive: the forecast is issued at midnight after day D for AQI(D+1), using only "
               "information available at that time. Models were trained on data up to 2021.")
    F, X, cols = load_data()
    station = station_picker(F, "fc")
    rows = F.index[F.station == station]
    dates = F.loc[rows, "date"]
    d = st.slider("Issue day D (forecast is for D+1)", min_value=dates.min().date(), max_value=dates.max().date(),
                  value=dates.max().date(), format="YYYY-MM-DD")
    pos = rows[np.searchsorted(dates.values, np.datetime64(d), side="right") - 1]
    row = F.loc[pos]
    period_note(row.date)
    p = predict_rows([pos])
    head, head_name = headline_regression(X, [pos])
    head = float(head[0])
    iv = intervals()
    sub = None
    if iv and np.isfinite(head):
        lo, hi = iv["by_forecast_category"].get(cat_of(head), [np.nan, np.nan])
        sub = f"80% range {max(0, head + lo):.0f} - {min(500, head + hi):.0f} (validation residuals)"
    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    with c1:
        badge(head, f"{head_name} forecast for {row.target_date.date()}", sub)
        st.caption(f"Category information: {HEALTH[cat_of(head)]}")
    with c2:
        if pd.notna(row.y_aqi):
            badge(float(row.y_aqi), "Observed AQI (archive)")
        else:
            st.markdown("#### Observed\nNo observed AQI for this day in the archive (sensor gap or end of data).")
    with c3:
        _, cls, _, prod, _ = load_models()
        cname = prod["classifier"].split("|")[0] if prod.get("classifier") else "XGBoost"
        if f"{cname}_score" in p and p[f"{cname}_thr"] is not None:
            score, thr = float(p[f"{cname}_score"][0]), float(p[f"{cname}_thr"])
            alert = score >= thr
            st.metric(f"Poor-or-worse risk score ({NICE[cname]} classifier)", f"{score:.2f}",
                      help=f"Model score, not a calibrated probability (see Analyses -> calibration). Alert rule: "
                           f"score >= {thr:.2f}, chosen for F1 on 2021 validation.")
            (st.error if alert else st.success)(
                ("⚠️ Poor-or-worse ALERT threshold crossed" if alert else "Poor-or-worse alert threshold not crossed")
                + f" (score {score:.2f} vs threshold {thr:.2f}).")
            if alert != bool(is_poor(head)):
                st.info(f"The alert classifier and the AQI forecast disagree: the forecast AQI is {head:.0f} "
                        f"({'above' if is_poor(head) else 'at or below'} 200). They are separate models; the "
                        "classifier is tuned to catch more Poor-or-worse days.")
        miss = bool(row.miss_aqi_l0) if pd.notna(row.miss_aqi_l0) else True
        st.caption(f"Today's (D) AQI: {row.aqi_l0:.0f}" + (" (filled - sensor gap)" if miss else "")
                   if pd.notna(row.aqi_l0) else "Today's (D) AQI: unavailable")

    st.subheader("All models")
    reg, _, imp, _, _ = load_models()
    clim = float(X[pos, cols.index("clim_aqi_month")]) if "clim_aqi_month" in cols else np.nan
    pers = row.aqi_l0 if not row.miss_aqi_l0 else (row.aqi_last_valid if row.days_since_valid <= 7 else clim)
    names = [m for m in MODEL_NAMES if m in p]
    tab = pd.DataFrame({"Model": [NICE[m] for m in names] + ["Persistence (baseline)", "Seasonal climatology"],
                        "Forecast AQI": [float(p[m][0]) for m in names] + [float(pers), clim]})
    tab["Category"] = tab["Forecast AQI"].map(cat_of)
    tab["Poor+ alert (classifier)"] = [("yes" if p[f"{m}_score"][0] >= p[f"{m}_thr"] else "no")
                                       if f"{m}_score" in p and p[f"{m}_thr"] is not None else "-" for m in names] + ["-", "-"]
    if pd.notna(row.y_aqi):
        tab["Error"] = tab["Forecast AQI"] - row.y_aqi
    st.dataframe(tab.style.format({"Forecast AQI": "{:.0f}", "Error": "{:+.0f}"}, na_rep="-")
                 .map(lambda c: f"background-color:{AQI_COLORS.get(c, '')}" if c in AQI_COLORS else "",
                      subset=["Category"]), hide_index=True, width="stretch")

    lo_ = max(rows.min(), pos - 60)
    hist = F.loc[lo_:pos]
    fig = go.Figure()
    for c, (a, b) in zip(AQI_CATEGORIES, [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]):
        fig.add_hrect(y0=a, y1=b, fillcolor=AQI_COLORS[c], opacity=0.10, line_width=0)
    fig.add_scatter(x=hist.date, y=hist.aqi_l0.where(hist.miss_aqi_l0 == 0), mode="lines+markers",
                    name="observed AQI", line=dict(color="#222"))
    hp, _ = headline_regression(X, list(range(lo_, pos + 1)))
    fig.add_scatter(x=hist.target_date, y=hp, mode="lines", name=f"{head_name} forecast (for D+1)",
                    line=dict(color="#2a78d6", dash="dot"))
    fig.add_scatter(x=[row.target_date], y=[head], mode="markers", name="this forecast",
                    marker=dict(size=14, color="#eb6834", symbol="star"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="AQI",
                      title="Last 60 days: observed vs forecast")
    st.plotly_chart(fig, width="stretch")

    if "XGBoost" in reg:
        st.subheader("Why? XGBoost model attributions (TreeSHAP)")
        s, base, total = contributions(pos)
        xgb_pred = float(reg["XGBoost"].predict(X[[pos]])[0])
        fig = go.Figure(go.Waterfall(orientation="h", y=["base value"] + list(s.index) + ["forecast"],
                                     x=[base] + list(s.values) + [0], measure=["absolute"] + ["relative"] * len(s) + ["total"],
                                     increasing=dict(marker=dict(color="#e34948")),
                                     decreasing=dict(marker=dict(color="#2a78d6"))))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                          title=f"Base value {base:.0f} + attributions = {total:.0f} (model output {xgb_pred:.0f})",
                          xaxis_title="AQI points")
        st.plotly_chart(fig, width="stretch")
        st.caption("Attributions explain the model's arithmetic, not physical causes. 'Base value' is the model's "
                   "explanation baseline. The sum matches the XGBoost output (before clipping to 0-500).")


def page_map():
    st.title("🗺️ Next-day risk map (historical demonstration)")
    F, X, cols = load_data()
    dmin, dmax = F.date.min().date(), F.date.max().date()
    default = min(pd.Timestamp("2022-11-03").date(), dmax)
    d = st.date_input("Issue day D", value=default, min_value=dmin, max_value=dmax)
    period_note(pd.Timestamp(d))
    idx = F.index[F.date == pd.Timestamp(d)]
    if len(idx) == 0:
        st.warning("No stations reporting on this day.")
        return
    reg, cls, _, _, _ = load_models()
    model = st.radio("Model", [m for m in MODEL_NAMES if m in reg], horizontal=True, format_func=NICE.get,
                     index=[m for m in MODEL_NAMES if m in reg].index("XGBoost") if "XGBoost" in reg else 0)
    color_by = st.radio("Colour by", ["Forecast AQI category", "Classifier alert (Poor or worse)"], horizontal=True)
    m = F.loc[idx, ["station", "city", "state", "lat", "lon", "y_aqi"]].copy()
    m["forecast"] = np.clip(reg[model].predict(X[idx]), 0, 500)
    m["category"] = m.forecast.map(cat_of)
    thr = thresholds().get(model)
    if model in cls and thr is not None:
        m["alert"] = np.where(cls[model].predict_proba(X[idx])[:, 1] >= thr, "alert", "no alert")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations", len(m))
    c2.metric("Forecast AQI > 200", f"{int(is_poor(m.forecast).sum())} ({is_poor(m.forecast).mean() * 100:.0f}%)")
    if m.y_aqi.notna().any():
        mm = m.dropna(subset=["y_aqi"])
        c3.metric("MAE vs observed (this day)", f"{np.mean(np.abs(mm.forecast - mm.y_aqi)):.1f}")
    import plotly.express as px
    if color_by.startswith("Classifier") and "alert" in m:
        fig = px.scatter_map(m, lat="lat", lon="lon", color="alert", hover_name="station",
                             color_discrete_map={"alert": "#e34948", "no alert": "#2a78d6"},
                             hover_data={"city": True, "forecast": ":.0f", "y_aqi": ":.0f", "lat": False, "lon": False},
                             zoom=3.6, center=dict(lat=22.5, lon=80), height=620, map_style="carto-positron")
        rule = f"Colour = {NICE[model]} classifier alert (score >= {thr:.2f})."
    else:
        fig = px.scatter_map(m, lat="lat", lon="lon", color="category", hover_name="station",
                             hover_data={"city": True, "forecast": ":.0f", "y_aqi": ":.0f", "lat": False, "lon": False},
                             color_discrete_map=AQI_COLORS, category_orders={"category": AQI_CATEGORIES + [UNKNOWN]},
                             zoom=3.6, center=dict(lat=22.5, lon=80), height=620, map_style="carto-positron")
        rule = "Colour = CPCB category of the forecast AQI ('Poor or worse' = forecast AQI > 200)."
    fig.update_traces(marker=dict(size=11))
    st.plotly_chart(fig, width="stretch")
    geo = pd.read_csv(EXT / "stations_geo.csv")
    n_exact = int((geo.get("coord_source", pd.Series(dtype=str)) == "openaq_station").sum())
    st.caption(rule + f" {n_exact} of {len(geo)} stations are at station coordinates; the rest at their city "
               "centre (they overlap).")
    st.dataframe(m.sort_values("forecast", ascending=False).head(25), hide_index=True, width="stretch")


def page_whatif():
    st.title("🎛️ What-if: local sensitivity of one forecast")
    st.caption("Change one input at a time and re-run the models for THIS station-day (an ICE-style local view, "
               "not a population average and not a causal effect). Dependent features such as rolling means do not "
               "move with a slider. With no slider moved, the forecast equals the original.")
    F, X, cols = load_data()
    station = station_picker(F, "wi")
    rows = F.index[F.station == station]
    d = st.select_slider("Issue day D", options=list(F.loc[rows, "date"].dt.date), value=F.loc[rows[-1], "date"].date())
    pos = rows[np.flatnonzero(F.loc[rows, "date"].dt.date.values == d)[0]]
    x0 = X[pos].copy()
    ci = {c: i for i, c in enumerate(cols)}
    knobs = [("aqi_l0", "Today's AQI", 0.0, 500.0), ("PM25_eve_l0", "PM2.5 this evening, 18-23 h (µg/m³)", 0.0, 500.0),
             ("site_WS_l0", "Wind speed today, on-site (m/s)", 0.0, 10.0),
             ("site_RH_l0", "Relative humidity today, on-site (%)", 0.0, 100.0),
             ("site_AT_l0", "Air temperature today, on-site (°C)", 0.0, 45.0),
             ("site_RF_l0", "Rainfall today, on-site (mm)", 0.0, 30.0),
             ("fire_PB_HR_l0", "Fires in Punjab+Haryana today", 0.0, 3000.0),
             ("fire_300km_l0", "Fires within 300 km today", 0.0, 2000.0)]
    knobs = [k for k in knobs if k[0] in ci]
    key = f"wi_{station}_{d}"
    if st.button("Reset all inputs"):
        for c, *_ in knobs:
            st.session_state.pop(f"{key}_{c}", None)
    x = x0.copy()
    changed = []
    cols_ui = st.columns(2)
    for j, (c, lab, lo, hi) in enumerate(knobs):
        v0 = float(x0[ci[c]])
        lo2, hi2 = min(lo, np.floor(v0)), max(hi, np.ceil(v0))       # the range always contains the original
        v = cols_ui[j % 2].slider(lab + (" ✱" if f"{key}_{c}" in st.session_state and
                                         st.session_state[f"{key}_{c}"] != v0 else ""),
                                  float(lo2), float(hi2), v0, key=f"{key}_{c}")
        if v != v0:
            x[ci[c]] = v
            changed.append(lab)
    reg, _, _, _, _ = load_models()
    base = {m: float(reg[m].predict(x0[None])[0]) for m in reg}
    new = {m: float(reg[m].predict(x[None])[0]) if changed else base[m] for m in reg}
    tab = pd.DataFrame({"Model": [NICE[m] for m in reg], "Original": base.values(), "What-if": new.values()})
    tab["Change"] = tab["What-if"] - tab["Original"]
    tab["Category"] = tab["What-if"].map(cat_of)
    st.caption("Changed inputs: " + (", ".join(changed) if changed else "none"))
    c1, c2 = st.columns([1, 2])
    with c1:
        if "XGBoost" in new:
            badge(new["XGBoost"], "XGBoost what-if forecast")
    with c2:
        st.dataframe(tab.style.format({"Original": "{:.0f}", "What-if": "{:.0f}", "Change": "{:+.0f}"}),
                     hide_index=True, width="stretch")
    if knobs:
        c, lab, lo, hi = knobs[st.selectbox("Response curve for", range(len(knobs)), format_func=lambda i: knobs[i][1])]
        grid = np.linspace(min(lo, x0[ci[c]]), max(hi, x0[ci[c]]), 40)
        XX = np.repeat(x[None], len(grid), axis=0)
        XX[:, ci[c]] = grid
        fig = go.Figure()
        for m in reg:
            fig.add_scatter(x=grid, y=reg[m].predict(XX), name=NICE[m], mode="lines")
        fig.add_hline(y=200, line_dash="dot", annotation_text="Poor-or-worse threshold (> 200)")
        fig.update_layout(height=360, xaxis_title=lab, yaxis_title="forecast AQI (D+1)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")


def show_fig(name, caption=None):
    f = FIG / name
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")


def table(R, key, **fmt):
    if key in R:
        st.dataframe(R[key].style.format(precision=fmt.get("precision", 3), na_rep="-"), hide_index=True, width="stretch")


def page_benchmark():
    st.title("📊 Model benchmark (test: Jan 2022 - Mar 2023)")
    R = load_reports()
    if "benchmark" not in R:
        st.info("Run the benchmark first (python src/run_all.py --from models).")
        return
    b = R["benchmark"]
    meta = R.get("benchmark_meta", {})
    prod = meta.get("production", {})
    st.caption(f"{meta.get('n_trainval', 0):,} training station-days (<= 2021), {meta.get('n_test', 0):,} test "
               f"station-days, {meta.get('n_features', '?')} features. Same split, target and fold-fitted features for "
               f"every model. Production model (chosen on 2021 validation, not on test): {prod.get('regressor')}.")
    t1, t2, t3, t4, t5 = st.tabs(["Regression", "Classification: Poor or worse", "Confidence intervals", "PCA",
                                  "Backtest 2019-2023"])
    with t1:
        r = b[b.task == "regression"][[c for c in ["model", "features", "MAE", "RMSE", "R2", "MAE_top5pct",
                                                   "MAE_severe", "skill_vs_persistence_pct", "val_score", "fit_s",
                                                   "fit_cpu_s"] if c in b]]
        st.dataframe(r.sort_values("MAE").style.format(precision=2, na_rep="-"), hide_index=True, width="stretch")
    with t2:
        c = b[b.task.isin(["direct_cls", "derived_cls", "derived_cls_tuned"])]
        c = c[[x for x in ["model", "features", "task", "precision", "recall", "F1", "AP", "ROC_AUC", "brier",
                           "false_alert_rate", "threshold"] if x in c]]
        st.markdown("**direct_cls** = classifier score >= threshold (tuned on 2021); **derived_cls** = forecast AQI > 200; "
                    "**derived_cls_tuned** = forecast AQI > a 2021-tuned AQI threshold. AP = average precision.")
        st.dataframe(c.sort_values("AP", ascending=False).style.format(precision=3, na_rep="-"), hide_index=True,
                     width="stretch")
    with t3:
        st.caption("95% paired block-bootstrap intervals (blocks = ISO weeks). Overlapping intervals mean the "
                   "difference is not established.")
        table(R, "benchmark_ci")
    with t4:
        show_fig("pca_explained_variance.png")
    with t5:
        show_fig("bt_01_backtest.png")
        table(R, "backtest", precision=2)


def page_analyses():
    st.title("🔍 Additional analyses")
    R = load_reports()
    a, c, e, s, l = st.tabs(["A. Ablation", "C. Efficiency", "E. Error analysis", "E. Severe days", "Spatial hold-outs"])
    with a:
        st.caption("All five models, fitted <= 2020 and evaluated on 2021 validation (column-group ablations).")
        for f in ["abl_01_leave_one_group_out.png", "abl_03_incremental.png", "abl_04_permutation.png",
                  "abl_02_pca_vs_selection.png"]:
            show_fig(f)
        table(R, "ablation", precision=2)
    with c:
        st.caption("Estimator timings (not end-to-end system cost, except the 'end_to_end' column).")
        show_fig("eff_01_accuracy_vs_cost.png")
        show_fig("eff_02_scaling.png")
        table(R, "efficiency")
    with e:
        for f in ["err_01_month_weekday.png", "err_02_region.png", "err_03_by_category.png", "err_04_change_size.png",
                  "err_05_stations.png", "err_06_error_correlation.png", "err_07_pred_vs_actual_episode.png",
                  "err_09_calibration.png"]:
            show_fig(f)
        if "errors_classification" in R:
            w = R["errors_classification"]
            st.markdown("**Early-warning performance**")
            st.dataframe(w[w.slice == "warning"].dropna(axis=1, how="all"), hide_index=True, width="stretch")
        if "errors_by_station" in R:
            st.markdown("**Hardest stations**")
            st.dataframe(R["errors_by_station"].head(20), hide_index=True, width="stretch")
    with s:
        show_fig("err_08_severe.png")
        table(R, "severe")
        table(R, "intervals_coverage")
    with l:
        show_fig("loco.png")
        table(R, "loco", precision=2)


def page_data():
    st.title("🧾 Data, preprocessing & leakage audit")
    R = load_reports()
    s = R.get("eda_summary", {})
    c = st.columns(4)
    c[0].metric("Stations", s.get("stations"))
    c[1].metric("Cities (state, city)", s.get("cities"))
    c[2].metric("Station-days", f"{s.get('station_days', 0):,}")
    c[3].metric("Poor-or-worse days", f"{s.get('poor_or_worse_pct', 0)}%")
    t1, t2, t3, t4 = st.tabs(["EDA", "Missing data & QA", "Leakage audit", "Sources"])
    with t1:
        for f in ["eda_10_station_map.png", "eda_05_seasonal_region.png", "eda_06_diurnal.png",
                  "eda_07_class_balance.png", "eda_08_aqi_dist_driver.png", "eda_09_spatial_corr.png",
                  "eda_11_diwali.png", "eda_01_network_growth.png"]:
            show_fig(f)
    with t2:
        for f in ["eda_02_missing_pollutant_year.png", "eda_03_missing_month_station.png",
                  "eda_04_missing_vs_pollution.png", "eda_12_qa_removed.png", "imp_01_reconstruction.png",
                  "imp_02_downstream.png"]:
            show_fig(f)
        table(R, "qa_impact")
        table(R, "coverage", precision=0)
    with t3:
        table(R, "leakage_checks")
        table(R, "leakage_audit")
    with t4:
        f = REPORTS / "data_sources.md"
        if not f.exists():
            f = ROOT / "reports" / "data_sources.md"
        st.markdown(f.read_text(encoding="utf-8") if f.exists() else "See report.")


preflight()
pg = st.navigation([
    st.Page(page_forecast, title="Forecast", icon="🌫️", url_path="forecast", default=True),
    st.Page(page_map, title="Risk map", icon="🗺️", url_path="map"),
    st.Page(page_whatif, title="What-if", icon="🎛️", url_path="whatif"),
    st.Page(page_benchmark, title="Benchmark", icon="📊", url_path="benchmark"),
    st.Page(page_analyses, title="Analyses", icon="🔍", url_path="analyses"),
    st.Page(page_data, title="Data & method", icon="🧾", url_path="data"),
])
pg.run()
