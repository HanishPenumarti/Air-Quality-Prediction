"""AirCast India - Streamlit UI for the next-day AQI forecasting pipeline.

Run:  streamlit run app/app.py
Uses the artefacts written by the pipeline (src/run_all.py): features_full.parquet,
models/*.joblib, reports/*.csv|json, reports/figures/*.png
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
from config import AQI_CATEGORIES, FIG, MODELS, PROC, REPORTS  # noqa: E402
from metrics import category  # noqa: E402

st.set_page_config(page_title="AirCast India", page_icon="🌫️", layout="wide")

AQI_COLORS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffd700",
              "Poor": "#ff9900", "Very Poor": "#ff0000", "Severe": "#a52a2a"}
MODEL_NAMES = ["DecisionTree", "RandomForest", "AdaBoost", "GradBoost", "XGBoost"]
NICE = {"DecisionTree": "Decision Tree", "RandomForest": "Random Forest", "AdaBoost": "AdaBoost",
        "GradBoost": "Gradient Boosting", "XGBoost": "XGBoost"}
HEALTH = {
    "Good": "Minimal impact.",
    "Satisfactory": "Minor breathing discomfort to sensitive people.",
    "Moderate": "Breathing discomfort to people with lung/heart disease, children and older adults.",
    "Poor": "Breathing discomfort to most people on prolonged exposure. Issue advisory.",
    "Very Poor": "Respiratory illness on prolonged exposure. Consider GRAP restrictions.",
    "Severe": "Affects healthy people, serious impact on those with existing disease. Emergency measures.",
}


# ---------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    reg, cls = {}, {}
    for m in MODEL_NAMES:
        for kind, store in (("reg", reg), ("cls", cls)):
            f = MODELS / f"{m}_{kind}_raw.joblib"
            if f.exists():
                store[m] = joblib.load(f)
    return reg, cls, joblib.load(MODELS / "imputer.joblib")


@st.cache_resource(show_spinner="Preparing feature matrix (one-off) ...")
def load_data():
    F = pd.read_parquet(PROC / "features_full.parquet")
    F = F.sort_values(["station", "date"]).reset_index(drop=True)
    _, _, imp = load_models()
    X = imp.transform(F).astype("float32")
    return F, X.values, list(X.columns)


@st.cache_data
def load_reports():
    out = {}
    for name in ["benchmark", "ablation", "efficiency", "efficiency_scaling", "loco", "errors_by_station",
                 "leakage_audit"]:
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
    b = load_reports().get("benchmark")
    t = b[(b.task == "direct_cls") & (b.features == "raw")].set_index("model").threshold
    return t.to_dict()


def cat_of(a):
    return AQI_CATEGORIES[int(category(np.array([a]))[0])]


def badge(a, label="Forecast AQI"):
    c = cat_of(a)
    fg = "#000" if c in ("Moderate", "Satisfactory", "Good", "Poor") else "#fff"
    st.markdown(
        f"""<div style="background:{AQI_COLORS[c]};color:{fg};padding:18px 22px;border-radius:12px">
        <div style="font-size:14px;opacity:.85">{label}</div>
        <div style="font-size:44px;font-weight:700;line-height:1.1">{a:.0f}</div>
        <div style="font-size:20px;font-weight:600">{c}</div></div>""", unsafe_allow_html=True)


def predict_rows(idx):
    reg, cls, _ = load_models()
    F, X, cols = load_data()
    thr = thresholds()
    out = {}
    for m, mdl in reg.items():
        out[f"{m}"] = mdl.predict(X[idx])
    for m, mdl in cls.items():
        out[f"{m}_prob"] = mdl.predict_proba(X[idx])[:, 1]
        out[f"{m}_thr"] = thr.get(m, 0.5)
    return out


def contributions(i, top=12):
    """Per-feature contributions (TreeSHAP) of the XGBoost regressor for one forecast."""
    import xgboost as xgb
    reg, _, _ = load_models()
    F, X, cols = load_data()
    booster = reg["XGBoost"].get_booster()
    c = booster.predict(xgb.DMatrix(X[[i]]), pred_contribs=True)[0]
    s = pd.Series(c[:-1], index=cols)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:top]
    return s, float(c[-1])


def station_picker(F, key):
    info = F.groupby("station")[["state", "city"]].first()
    c1, c2, c3 = st.columns(3)
    state = c1.selectbox("State", sorted(info.state.dropna().unique()),
                         index=sorted(info.state.dropna().unique()).index("Delhi"), key=key + "s")
    cities = sorted(info[info.state == state].city.unique())
    city = c2.selectbox("City", cities, key=key + "c")
    stations = info[(info.state == state) & (info.city == city)].index.tolist()
    geo = pd.read_csv(ROOT / "data" / "external" / "stations_geo.csv").set_index("station")
    station = c3.selectbox("Station", stations, key=key + "t",
                           format_func=lambda s: f"{s} - {str(geo.station_location.get(s, ''))[:40]}")
    return station


# ---------------------------------------------------------------- pages
def page_forecast():
    st.title("🌫️ Next-day AQI forecast")
    st.caption("Forecast issued at midnight after day D for the CPCB AQI of day D+1, using only "
               "information available at that time.")
    F, X, cols = load_data()
    station = station_picker(F, "fc")
    rows = F.index[F.station == station]
    dates = F.loc[rows, "date"]
    d = st.slider("Issue day D (forecast is for D+1)", min_value=dates.min().date(), max_value=dates.max().date(),
                  value=dates.max().date(), format="YYYY-MM-DD")
    pos = rows[np.searchsorted(dates.values, np.datetime64(d), side="right") - 1]
    row = F.loc[pos]
    if row.date.date() != d:
        st.info(f"No data for {d}; showing the closest earlier day {row.date.date()}.")
    p = predict_rows([pos])
    xgb_val = float(p["XGBoost"][0])
    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    with c1:
        badge(xgb_val, f"XGBoost forecast for {row.target_date.date()}")
    with c2:
        if pd.notna(row.y_aqi):
            badge(float(row.y_aqi), "Observed AQI (actual)")
        else:
            st.markdown("#### Observed\nNot yet available, so this is a real forecast.")
    with c3:
        prob = float(p["XGBoost_prob"][0])
        st.metric("P(Poor or worse) - direct XGBoost classifier", f"{prob * 100:.0f}%",
                  help=f"Alert raised when probability >= {p['XGBoost_thr']:.2f} (threshold tuned for F1 on 2021)")
        alert = prob >= p["XGBoost_thr"]
        (st.error if alert else st.success)(
            ("⚠️ EARLY WARNING: Poor-or-worse air expected. " if alert else "No unhealthy-air alert. ")
            + HEALTH[cat_of(xgb_val)])
        st.caption(f"Today's (D) AQI: {row.aqi_l0:.0f}" + (" (imputed - sensor gap)" if row.miss_aqi_l0 else ""))

    st.subheader("All models")
    tab = pd.DataFrame({
        "Model": [NICE[m] for m in MODEL_NAMES if m in p] + ["Persistence (baseline)", "Seasonal climatology"],
        "Forecast AQI": [float(p[m][0]) for m in MODEL_NAMES if m in p] +
                        [float(row.aqi_last_valid), float(row.clim_aqi_month)],
    })
    tab["Category"] = tab["Forecast AQI"].map(cat_of)
    tab["P(Poor+) direct"] = [f"{p[m + '_prob'][0] * 100:.0f}%" if m + "_prob" in p else "-" for m in MODEL_NAMES
                              if m in p] + ["-", "-"]
    if pd.notna(row.y_aqi):
        tab["Error"] = tab["Forecast AQI"] - row.y_aqi
    st.dataframe(tab.style.format({"Forecast AQI": "{:.0f}", "Error": "{:+.0f}"})
                 .map(lambda c: f"background-color:{AQI_COLORS.get(c, '')}" if c in AQI_COLORS else "",
                      subset=["Category"]), hide_index=True, width="stretch")

    # history plot
    lo = max(rows.min(), pos - 60)
    hist = F.loc[lo:pos]
    fig = go.Figure()
    for c, (a, b) in zip(AQI_CATEGORIES, [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]):
        fig.add_hrect(y0=a, y1=b, fillcolor=AQI_COLORS[c], opacity=0.10, line_width=0)
    raw = hist.aqi_l0.where(hist.miss_aqi_l0 == 0)
    fig.add_scatter(x=hist.date, y=raw, mode="lines+markers", name="observed AQI", line=dict(color="#222"))
    hp = predict_rows(list(range(lo, pos + 1)))
    fig.add_scatter(x=hist.target_date, y=hp["XGBoost"], mode="lines", name="XGBoost forecast (for D+1)",
                    line=dict(color="#2a78d6", dash="dot"))
    fig.add_scatter(x=[row.target_date], y=[xgb_val], mode="markers", name="this forecast",
                    marker=dict(size=14, color="#eb6834", symbol="star"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="AQI",
                      title="Last 60 days: observed vs forecast")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Why this forecast? (XGBoost feature contributions)")
    s, base = contributions(pos)
    fig = go.Figure(go.Bar(x=s.values[::-1], y=s.index[::-1], orientation="h",
                           marker_color=["#e34948" if v > 0 else "#2a78d6" for v in s.values[::-1]]))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Start from the average forecast ({base:.0f}) and add contributions; red pushes AQI up",
                      xaxis_title="contribution to forecast (AQI points)")
    st.plotly_chart(fig, width="stretch")


def page_map():
    st.title("🗺️ Tomorrow's risk map")
    F, X, cols = load_data()
    dmin, dmax = F.date.min().date(), F.date.max().date()
    d = st.date_input("Issue day D", value=pd.Timestamp("2022-11-03").date(), min_value=dmin, max_value=dmax)
    idx = F.index[F.date == pd.Timestamp(d)]
    if len(idx) == 0:
        st.warning("No stations reporting on this day.")
        return
    model = st.radio("Model", MODEL_NAMES, index=4, horizontal=True, format_func=NICE.get)
    reg, _, _ = load_models()
    pred = reg[model].predict(X[idx])
    m = F.loc[idx, ["station", "city", "state", "lat", "lon", "y_aqi"]].copy()
    m["forecast"] = pred
    m["category"] = m.forecast.map(cat_of)
    c1, c2, c3 = st.columns(3)
    c1.metric("Stations", len(m))
    c2.metric("Forecast Poor or worse", f"{(m.forecast > 200).sum()} ({(m.forecast > 200).mean() * 100:.0f}%)")
    if m.y_aqi.notna().any():
        mm = m.dropna(subset=["y_aqi"])
        c3.metric("MAE vs observed (this day)", f"{np.mean(np.abs(mm.forecast - mm.y_aqi)):.1f}")
    import plotly.express as px
    fig = px.scatter_map(m, lat="lat", lon="lon", color="category", hover_name="station",
                         hover_data={"city": True, "forecast": ":.0f", "y_aqi": ":.0f", "lat": False, "lon": False},
                         color_discrete_map=AQI_COLORS, category_orders={"category": AQI_CATEGORIES},
                         zoom=3.6, center=dict(lat=22.5, lon=80), height=620, map_style="carto-positron")
    fig.update_traces(marker=dict(size=11))
    st.plotly_chart(fig, width="stretch")
    st.caption("Station coordinates are city-level (geocoded); stations in the same city overlap.")
    st.dataframe(m.sort_values("forecast", ascending=False).head(25), hide_index=True, width="stretch")


def page_whatif():
    st.title("🎛️ What-if: sensitivity of the forecast")
    st.caption("Change one driver at a time and re-run the trained models. Other features stay fixed, so this is a "
               "partial-dependence style view: correlated features (e.g. rolling means) do not move with the slider.")
    F, X, cols = load_data()
    station = station_picker(F, "wi")
    rows = F.index[F.station == station]
    d = st.select_slider("Issue day D", options=list(F.loc[rows, "date"].dt.date), value=F.loc[rows[-1], "date"].date())
    pos = rows[np.flatnonzero(F.loc[rows, "date"].dt.date.values == d)[0]]
    x0 = X[pos].copy()
    ci = {c: i for i, c in enumerate(cols)}
    knobs = [("aqi_l0", "Today's AQI", 0.0, 500.0), ("PM25_eve_l0", "PM2.5 this evening, 18-23 h (µg/m³)", 0.0, 500.0),
             ("site_WS_l0", "Wind speed today, on-site (m/s)", 0.0, 10.0),
             ("site_RH_l0", "Relative humidity today, on-site (%)", 5.0, 100.0),
             ("site_AT_l0", "Air temperature today, on-site (°C)", 0.0, 45.0),
             ("site_RF_l0", "Rainfall today, on-site (mm)", 0.0, 30.0),
             ("fire_PB_HR_l0", "Fires in Punjab+Haryana today", 0.0, 3000.0),
             ("fire_300km_l0", "Fires within 300 km today", 0.0, 2000.0)]
    knobs = [k for k in knobs if k[0] in ci]
    x = x0.copy()
    cols_ui = st.columns(2)
    for j, (c, lab, lo, hi) in enumerate(knobs):
        v = cols_ui[j % 2].slider(lab, lo, hi, float(np.clip(x0[ci[c]], lo, hi)), key="k" + c)
        x[ci[c]] = v
    reg, cls, _ = load_models()
    base = {m: float(reg[m].predict(x0[None])[0]) for m in reg}
    new = {m: float(reg[m].predict(x[None])[0]) for m in reg}
    tab = pd.DataFrame({"Model": [NICE[m] for m in reg], "Original": base.values(), "What-if": new.values()})
    tab["Change"] = tab["What-if"] - tab["Original"]
    tab["Category"] = tab["What-if"].map(cat_of)
    c1, c2 = st.columns([1, 2])
    with c1:
        badge(new["XGBoost"], "XGBoost what-if forecast")
    with c2:
        st.dataframe(tab.style.format({"Original": "{:.0f}", "What-if": "{:.0f}", "Change": "{:+.0f}"}),
                     hide_index=True, width="stretch")
    if knobs:
        c, lab, lo, hi = knobs[st.selectbox("Response curve for", range(len(knobs)), format_func=lambda i: knobs[i][1])]
        grid = np.linspace(lo, hi, 40)
        XX = np.repeat(x[None], len(grid), axis=0)
        XX[:, ci[c]] = grid
        fig = go.Figure()
        for k, m in enumerate(reg):
            fig.add_scatter(x=grid, y=reg[m].predict(XX), name=NICE[m], mode="lines")
        fig.add_hline(y=200, line_dash="dot", annotation_text="Poor threshold")
        fig.update_layout(height=360, xaxis_title=lab, yaxis_title="forecast AQI (D+1)",
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")


def show_fig(name, caption=None):
    f = FIG / name
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")


def page_benchmark():
    st.title("📊 Model benchmark (test: Jan 2022 - Mar 2023)")
    R = load_reports()
    b = R["benchmark"]
    meta = R.get("benchmark_meta", {})
    st.caption(f"{meta.get('n_trainval', 0):,} training station-days (<= 2021), {meta.get('n_test', 0):,} test "
               f"station-days, {meta.get('n_features', '?')} features. All models share the same split, target and "
               "imputed features.")
    t1, t2, t3 = st.tabs(["Regression", "Classification: Poor or worse", "PCA"])
    with t1:
        r = b[b.task == "regression"][["model", "features", "MAE", "RMSE", "R2", "MAE_top5pct", "MAE_top1pct",
                                       "skill_vs_persistence_pct", "cat_acc", "fit_s"]]
        st.dataframe(r.sort_values("MAE").style.format(precision=2).highlight_min(subset=["MAE", "RMSE"], color="#cde2fb"),
                     hide_index=True, width="stretch")
        fig = go.Figure()
        rr = r.sort_values("MAE")
        fig.add_bar(x=rr.model + " (" + rr.features + ")", y=rr.MAE, marker_color="#2a78d6")
        fig.update_layout(height=320, yaxis_title="test MAE (AQI points)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    with t2:
        c = b[b.task.isin(["direct_cls", "derived_cls"])][["model", "features", "task", "precision", "recall", "F1",
                                                           "PR_AUC", "ROC_AUC", "threshold"]]
        st.markdown("**direct_cls** = classifier trained on the label; **derived_cls** = regression forecast > 200.")
        st.dataframe(c.sort_values("F1", ascending=False).style.format(precision=3), hide_index=True,
                     width="stretch")
    with t3:
        show_fig("pca_explained_variance.png")
        if meta:
            st.write(f"95% of variance needs **{meta['pca_n95']}** of {meta['n_features']} components.")


def page_analyses():
    st.title("🔍 Additional analyses")
    R = load_reports()
    a, c, e, l = st.tabs(["A. Ablation", "C. Efficiency", "E. Error analysis", "Spatial (LOCO)"])
    with a:
        show_fig("abl_01_leave_one_group_out.png")
        show_fig("abl_03_incremental.png")
        show_fig("abl_02_pca_vs_selection.png")
        if "ablation" in R:
            st.dataframe(R["ablation"].style.format(precision=2), hide_index=True, width="stretch")
    with c:
        show_fig("eff_01_accuracy_vs_cost.png")
        show_fig("eff_02_scaling.png")
        if "efficiency" in R:
            st.dataframe(R["efficiency"].style.format(precision=3), hide_index=True, width="stretch")
    with e:
        for f in ["err_01_month_weekday.png", "err_02_region.png", "err_03_by_category.png", "err_04_change_size.png",
                  "err_05_stations.png", "err_06_error_correlation.png", "err_07_pred_vs_actual_episode.png"]:
            show_fig(f)
        if "errors_by_station" in R:
            st.markdown("**Hardest stations (XGBoost)**")
            st.dataframe(R["errors_by_station"].head(20), hide_index=True, width="stretch")
    with l:
        show_fig("loco.png")
        if "loco" in R:
            st.dataframe(R["loco"].style.format(precision=2), hide_index=True, width="stretch")


def page_data():
    st.title("🧾 Data, preprocessing & leakage audit")
    R = load_reports()
    s = R.get("eda_summary", {})
    c = st.columns(4)
    c[0].metric("Stations", s.get("stations"))
    c[1].metric("Cities", s.get("cities"))
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
    with t3:
        if "leakage_audit" in R:
            st.dataframe(R["leakage_audit"], hide_index=True, width="stretch")
    with t4:
        st.markdown((ROOT / "reports" / "data_sources.md").read_text(encoding="utf-8")
                    if (ROOT / "reports" / "data_sources.md").exists() else "See report.")


pg = st.navigation([
    st.Page(page_forecast, title="Forecast", icon="🌫️", url_path="forecast", default=True),
    st.Page(page_map, title="Risk map", icon="🗺️", url_path="map"),
    st.Page(page_whatif, title="What-if", icon="🎛️", url_path="whatif"),
    st.Page(page_benchmark, title="Benchmark", icon="📊", url_path="benchmark"),
    st.Page(page_analyses, title="Analyses", icon="🔍", url_path="analyses"),
    st.Page(page_data, title="Data & method", icon="🧾", url_path="data"),
])
pg.run()
