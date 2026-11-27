"""Regenerate every result table from the saved results:
  * reports/tables.md          - all tables
  * reports/report.md          - the blocks between <!-- TABLE:name --> and <!-- /TABLE:name -->
                                 are replaced in place, so the report's tables always match the run
Run after any re-run of the pipeline:   python src/report_tables.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import REPORTS  # noqa: E402

NOT_RUN = "_not run yet - run the pipeline (see README)_"


def md(df, floatfmt=".3g"):
    """Markdown table; '|' inside text cells (e.g. 'XGBoost|reg|raw') is escaped so columns don't split."""
    if not len(df):
        return "_empty_"
    df = df.copy()
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].map(lambda v: v.replace("|", "\\|") if isinstance(v, str) else v)
    return df.to_markdown(index=False, floatfmt=floatfmt)


def read(name):
    p = REPORTS / name
    return pd.read_csv(p).rename(columns={"Unnamed: 0": ""}) if p.exists() else None


def pick(df, cols):
    return df[[c for c in cols if c in df.columns]]


def tables():
    T = {}
    b = read("benchmark.csv")
    if b is not None:
        T["bench_reg"] = md(pick(b[b.task == "regression"], ["model", "features", "MAE", "RMSE", "R2", "MAE_top5pct",
                                                             "MAE_severe", "bias_severe", "skill_vs_persistence_pct",
                                                             "val_score", "n_rounds", "pca_components"]), ".2f")
        T["bench_cls"] = md(pick(b[b.task.isin(["derived_cls", "derived_cls_tuned", "direct_cls"])],
                                 ["model", "features", "task", "precision", "recall", "F1", "AP", "ROC_AUC",
                                  "false_alert_rate", "brier", "threshold"]), ".3f")
        raw = b[b.features == "raw"].set_index(["model", "task"])
        pca = b[b.features == "pca"].set_index(["model", "task"])
        j = raw.join(pca, lsuffix="_raw", rsuffix="_pca", how="inner").reset_index()
        j = j[j.task.isin(["regression", "direct_cls"])]
        j["dMAE"] = j.MAE_pca - j.MAE_raw
        j["dF1"] = j.F1_pca - j.F1_raw
        T["bench_pca"] = md(pick(j, ["model", "task", "MAE_raw", "MAE_pca", "dMAE", "F1_raw", "F1_pca", "dF1",
                                     "params_raw", "params_pca", "pca_components_pca"]), ".3f")
        T["bench_cost"] = md(pick(b[b.task.isin(["regression", "direct_cls"]) & b.fit_s.notna()],
                                  ["model", "features", "task", "fit_s", "fit_cpu_s", "threads", "n_rounds",
                                   "predict_ms_per_1k", "size_mb"]), ".3g")
    ci = read("benchmark_ci.csv")
    if ci is not None:
        T["bench_ci"] = md(ci[ci.metric.isin(["MAE", "dMAE_vs_persistence", "AP", "F1"])], ".2f")
    pw = read("benchmark_pairwise.csv")
    if pw is not None:
        T["bench_pairwise"] = md(pw, ".3f")
    for key, f, fmt in [("severe", "severe.csv", ".3f"), ("intervals", "intervals_coverage.csv", ".3f"),
                        ("backtest", "backtest.csv", ".2f"), ("leakage_checks", "leakage_checks.csv", None),
                        ("leakage_audit", "leakage_audit.csv", None), ("permutation", "permutation_importance.csv", ".3f"),
                        ("efficiency", "efficiency.csv", ".4g"), ("loco", "loco.csv", ".2f"),
                        ("qa_impact", "qa_impact.csv", ".4g"), ("coverage", "coverage.csv", ".0f"),
                        ("errors_region", "errors_region.csv", ".2f"), ("errors_true_cat_bias", "errors_true_cat_bias.csv", ".2f"),
                        ("errors_pred_cat_bias", "errors_pred_cat_bias.csv", ".2f"),
                        ("errors_today_cat", "errors_today_cat_mae.csv", ".2f"), ("errors_change", "errors_change.csv", ".2f"),
                        ("errors_station_type", "errors_station_type.csv", ".2f"),
                        ("errors_averaging", "errors_averaging.csv", ".2f"), ("errors_diwali", "errors_diwali.csv", ".2f"),
                        ("errors_gap", "errors_gap.csv", ".2f"), ("calibration", "errors_calibration.csv", ".3f")]:
        d = read(f)
        if d is not None:
            T[key] = md(d, fmt) if fmt else md(d)
    w = read("errors_classification.csv")
    if w is not None:
        T["errors_warning"] = md(pick(w[w.slice == "warning"], ["model", "group", "recall_poor", "recall_on_severe_days",
                                                                 "onsets", "onset_recall", "false_alert_rate",
                                                                 "false_alert_on_clean_days"]), ".3f")
    a = read("ablation.csv")
    if a is not None:
        base = a[a.experiment == "ALL features"].set_index("model").MAE
        a = a.assign(dMAE=a.MAE - a.model.map(base))
        for key, cond in [("ablation_logo", a.experiment.str.startswith("- without")),
                          ("ablation_single", a.experiment.str.startswith("only ")),
                          ("ablation_inc", a.experiment.str.startswith("inc:") | (a.experiment == "ALL features")),
                          ("ablation_select", a.experiment.str.startswith(("select", "PCA")))]:
            col = "dMAE" if key == "ablation_logo" else "MAE"
            piv = a[cond].pivot(index="experiment", columns="model", values=col)
            piv = piv.reindex(list(dict.fromkeys(a[cond].experiment)))
            T[key] = md(piv.reset_index(), ".2f")
    sc = read("efficiency_scaling.csv")
    if sc is not None:
        g = sc.groupby(["model", "frac"]).agg(n_rows=("n_rows", "mean"), fit_s=("fit_s", "mean"),
                                              fit_cpu_s=("fit_cpu_s", "mean"), val_MAE=("val_MAE", "mean"),
                                              val_MAE_min=("val_MAE", "min"), val_MAE_max=("val_MAE", "max"))
        T["efficiency_scaling"] = md(g.reset_index(), ".4g")
    p = REPORTS / "imputation_results.json"
    if p.exists():
        imp = json.loads(p.read_text())
        for key in ["reconstruction", "downstream_history", "downstream_residual"]:
            if key in imp:
                T[f"imputation_{key}"] = md(pd.DataFrame(imp[key]), ".3f")
    return T


def fill_report(T, path=REPORTS / "report.md"):
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    n = 0

    def rep(m):
        nonlocal n
        n += 1
        return f"<!-- TABLE:{m.group(1)} -->\n{T.get(m.group(1), NOT_RUN)}\n<!-- /TABLE:{m.group(1)} -->"
    text = re.sub(r"<!-- TABLE:(\w+) -->.*?<!-- /TABLE:\1 -->", rep, text, flags=re.S)
    path.write_text(text, encoding="utf-8")
    return n


def main():
    T = tables()
    (REPORTS / "tables.md").write_text("# Result tables (auto-generated by src/report_tables.py, do not edit)\n\n" +
                                       "\n".join(f"### {k}\n\n{v}\n" for k, v in T.items()), encoding="utf-8")
    n = fill_report(T)
    print(f"wrote {REPORTS / 'tables.md'} ({len(T)} tables); filled {n} table blocks in report.md")


if __name__ == "__main__":
    main()
