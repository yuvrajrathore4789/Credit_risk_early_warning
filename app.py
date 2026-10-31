"""Streamlit dashboard: ``streamlit run app.py`` (run ``python train.py`` first)."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import data_processing as dp
from src import database as db
from src import early_warning as ew
from src import explainability as xai
from src import monitoring
from src.inference import InputValidationError, validate_input
from src.model_training import load_bundle

monitoring.setup_logging()
logger = logging.getLogger("app")

st.set_page_config(page_title="Credit Risk Early Warning", page_icon="📉", layout="wide")

OUT = config.OUTPUTS_DIR
PAGES = ["Executive Overview", "Exploratory Data Analysis", "Model Comparison", "Individual Risk Analysis",
         "Early Warning", "Monitoring & Audit"]
BLUE, RED, GREY = "#3b6ea5", "#d1495b", "#8d99ae"


# ------------------------------------------------------------------ cached loaders (keyed by file mtime)
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading model...")
def get_bundle(_mtime_key: float):
    return load_bundle()


@st.cache_resource
def get_engine():
    engine = db.get_engine()
    db.init_db(engine)
    return engine


@st.cache_data
def read_csv(path: str, _mtime_key: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def read_json(path: str, _mtime_key: float) -> dict:
    return json.loads(Path(path).read_text())


def csv(path: Path) -> pd.DataFrame:
    return read_csv(str(path), _mtime(path))


def jsn(path: Path) -> dict:
    return read_json(str(path), _mtime(path))


def require_artifacts() -> None:
    needed = [config.MODEL_PATH, OUT / "results.json", OUT / "test_predictions.csv", config.EDA_DIR / "eda_summary.json"]
    if any(not p.exists() for p in needed):
        st.title("Credit Risk Prediction & Early Warning System")
        st.error("Trained artifacts not found. Train the models first, then reload this page.")
        st.code("python train.py", language="bash")
        st.caption("See data/README.md for how to obtain the dataset.")
        st.stop()


require_artifacts()
bundle = get_bundle(_mtime(config.MODEL_PATH))
results = jsn(OUT / "results.json")
engine = get_engine()
db.ensure_model_run(engine, bundle)

# ------------------------------------------------------------------ sidebar: page + shared policy settings
st.sidebar.title("Credit Risk Early Warning")
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.subheader("Illustrative policy settings")
thr = st.sidebar.slider("Review threshold", 0.05, 0.95, float(bundle.threshold), 0.01,
                        help="Accounts with estimated probability >= threshold are flagged for analyst review.")
st.sidebar.caption(f"Validation-selected default: {bundle.threshold:.2f} ({bundle.threshold_method})")
with st.sidebar.expander("Risk-band cut-offs"):
    e1 = st.number_input("Medium starts at", 0.01, 0.98, float(config.RISK_BAND_EDGES[0]), 0.01)
    e2 = st.number_input("High starts at", 0.01, 0.98, float(config.RISK_BAND_EDGES[1]), 0.01)
    e3 = st.number_input("Very High starts at", 0.01, 0.99, float(config.RISK_BAND_EDGES[2]), 0.01)
edges = [e1, e2, e3]
try:
    ew.validate_bands(edges, config.RISK_BAND_LABELS)
except ValueError as exc:
    st.sidebar.error(f"{exc} Using defaults.")
    edges = list(config.RISK_BAND_EDGES)
LABELS = config.RISK_BAND_LABELS
BAND_COLORS = dict(zip(LABELS, ["#2a9d8f", "#e9c46a", "#f4a261", "#d1495b"]))
st.sidebar.info(config.DISCLAIMER)


def band_bar(alerts: pd.DataFrame, title: str):
    s = ew.band_summary(alerts, LABELS)
    fig = px.bar(s, x="risk_band", y="count", color="risk_band", color_discrete_map=BAND_COLORS,
                 text=s["share"].map("{:.1%}".format), title=title, category_orders={"risk_band": LABELS})
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Accounts")
    return fig


def metric_row(items: list[tuple[str, str]]) -> None:
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)


test_pred = csv(OUT / "test_predictions.csv")
sel_test = results["metrics"]["at_selected"]

# ================================================================== A. Executive overview
if page == PAGES[0]:
    st.title("Executive Overview")
    st.caption(f"Model **{bundle.model_name}** · version `{bundle.version}` · trained {bundle.trained_at}")
    eda_sum = jsn(config.EDA_DIR / "eda_summary.json")
    metric_row([("Records analysed", f"{eda_sum['n_records']:,}"),
                ("Observed historical default rate", f"{eda_sum['default_rate']:.1%}"),
                ("Test ROC-AUC", f"{sel_test['roc_auc']:.3f}"), ("Test PR-AUC", f"{sel_test['pr_auc']:.3f}")])
    metric_row([("Recall @ selected threshold", f"{sel_test['recall']:.1%}"),
                ("Precision @ selected threshold", f"{sel_test['precision']:.1%}"),
                ("F1", f"{sel_test['f1']:.3f}"), ("Brier score", f"{sel_test['brier']:.3f}")])
    st.info(eda_sum["dataset_note"])
    alerts = ew.build_alerts(test_pred["probability"], thr, edges=edges)
    c1, c2 = st.columns(2)
    fig = px.histogram(test_pred, x="probability", color=test_pred["default"].map({0: "No default", 1: "Default"}),
                       nbins=40, barmode="overlay", opacity=0.6, title="Predicted default probability (held-out test records)",
                       color_discrete_sequence=[BLUE, RED], labels={"color": "Observed outcome"})
    fig.add_vline(x=thr, line_dash="dash", annotation_text=f"threshold {thr:.2f}")
    c1.plotly_chart(fig, width="stretch")
    c2.plotly_chart(band_bar(alerts, "Risk-band distribution (held-out test records)"), width="stretch")
    st.caption("Predictions are for HELD-OUT HISTORICAL records. Predicted risk is not an observed default.")

# ================================================================== B. EDA
elif page == PAGES[1]:
    st.title("Exploratory Data Analysis")
    st.caption("Descriptive only: associations in a 2005 Taiwanese card portfolio, not causal effects.")
    eda_sum = jsn(config.EDA_DIR / "eda_summary.json")
    sample = csv(config.EDA_DIR / "eda_sample.csv")
    t1, t2, t3, t4, t5 = st.tabs(["Class distribution", "Feature distributions", "Correlation", "Segment default rates", "Data cleaning"])
    with t1:
        cc = eda_sum["class_counts"]
        d = pd.DataFrame({"class": ["No default", "Default"], "customers": [cc["0"], cc["1"]]})
        st.plotly_chart(px.bar(d, x="class", y="customers", color="class", text="customers",
                               color_discrete_sequence=[BLUE, RED]), width="stretch")
        st.write(f"Default rate **{eda_sum['default_rate']:.1%}** — imbalanced, so accuracy alone is misleading "
                 "(predicting 'no default' for everyone would already score ~78%). PR-AUC and recall are tracked too.")
    with t2:
        feat = st.selectbox("Feature", [c for c in sample.columns if c != config.TARGET_COLUMN], index=0)
        st.plotly_chart(px.histogram(sample, x=feat, color=sample[config.TARGET_COLUMN].map({0: "No default", 1: "Default"}),
                                     nbins=40, barmode="overlay", opacity=0.6, color_discrete_sequence=[BLUE, RED],
                                     labels={"color": "Outcome"}), width="stretch")
        st.caption("Stratified sample of ~6,000 records.")
        st.dataframe(csv(config.EDA_DIR / "descriptive_stats.csv").rename(columns={"Unnamed: 0": "feature"}), width="stretch")
    with t3:
        corr = csv(config.EDA_DIR / "correlation.csv").set_index("Unnamed: 0")
        top = corr[config.TARGET_COLUMN].drop(config.TARGET_COLUMN).abs().sort_values(ascending=False).index[:12].tolist()
        picks = st.multiselect("Features", list(corr.columns), default=top + [config.TARGET_COLUMN])
        if len(picks) >= 2:
            st.plotly_chart(px.imshow(corr.loc[picks, picks], color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                      aspect="auto", text_auto=".2f"), width="stretch")
        st.caption("Correlation shows linear association only; bill/payment columns are strongly inter-correlated.")
    with t4:
        seg = csv(config.EDA_DIR / "segment_default_rates.csv")
        name = st.selectbox("Segment", seg["segment"].unique())
        g = seg[seg["segment"] == name]
        fig = px.bar(g, x="level", y="default_rate", text=g["default_rate"].map("{:.1%}".format), hover_data=["n"],
                     color_discrete_sequence=[RED])
        fig.add_hline(y=eda_sum["default_rate"], line_dash="dash", annotation_text="overall")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(g, width="stretch", hide_index=True)
    with t5:
        st.json(eda_sum["cleaning_report"])

# ================================================================== C. Model comparison
elif page == PAGES[2]:
    st.title("Model Comparison")
    comp = csv(OUT / "model_comparison.csv")
    st.write("Model **selected on validation PR-AUC** (test set untouched until final evaluation). "
             "Test metrics use each model's own validation-selected threshold.")
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if comp[c].dtype == float}).apply(
        lambda r: ["background-color: rgba(59,110,165,.18)" if r["selected"] else "" for _ in r], axis=1),
        width="stretch", hide_index=True)
    curves = jsn(OUT / "curves.json")
    c1, c2 = st.columns(2)
    f1, f2 = go.Figure(), go.Figure()
    for m, c in curves.items():
        f1.add_scatter(x=c["fpr"], y=c["tpr"], name=m, mode="lines")
        f2.add_scatter(x=c["recall"], y=c["precision"], name=m, mode="lines")
    f1.add_scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(dash="dash", color=GREY))
    f2.add_hline(y=float(test_pred["default"].mean()), line_dash="dash", annotation_text="base rate")
    f1.update_layout(title="ROC curve (test)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    f2.update_layout(title="Precision-recall curve (test)", xaxis_title="Recall", yaxis_title="Precision")
    c1.plotly_chart(f1, width="stretch"); c2.plotly_chart(f2, width="stretch")
    st.caption("When defaults are the minority class, PR-AUC and recall focus on how well the rare class is found; "
               "ROC-AUC can look flattering because the many non-defaults inflate the true-negative side.")

    model = st.selectbox("Model detail", list(results["all_models"].keys()), index=list(results["all_models"]).index(results["selected_model"]))
    m = results["all_models"][model]
    which = st.radio("Evaluate at", ["Validation-selected threshold", "Default threshold 0.50"], horizontal=True)
    key = "at_selected" if which.startswith("Validation") else "at_0.5"
    tm = m["test"][key]
    cm = pd.DataFrame([[tm["tn"], tm["fp"]], [tm["fn"], tm["tp"]]], index=["Actual: no default", "Actual: default"],
                      columns=["Predicted: no default", "Predicted: default"])
    d1, d2 = st.columns([1, 1])
    d1.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=f"Confusion matrix (test, threshold {tm['threshold']:.2f})"),
                    width="stretch")
    d2.dataframe(pd.DataFrame({"Validation": m["validation"][key], "Test": tm}).drop(["tn", "fp", "fn", "tp"]).style.format("{:.3f}"),
                 width="stretch")
    st.markdown(f"**False positives ({tm['fp']:,})** are non-defaulters flagged for review — wasted analyst effort. "
                f"**False negatives ({tm['fn']:,})** are eventual defaulters the model missed — usually the costlier error, "
                "which is why the threshold is picked with a recall-leaning illustrative cost ratio "
                f"({config.ILLUSTRATIVE_COST_FALSE_NEGATIVE:g}:{config.ILLUSTRATIVE_COST_FALSE_POSITIVE:g}).")

    thr_tab = csv(OUT / "threshold_analysis.csv")
    thr_tab = thr_tab[thr_tab["model"] == model]
    ft = px.line(thr_tab, x="threshold", y=["precision", "recall", "f1"], title="Threshold analysis (validation set)")
    ft.add_vline(x=m["selected_threshold"], line_dash="dash", annotation_text="selected")
    st.plotly_chart(ft, width="stretch")
    fc = px.line(thr_tab, x="threshold", y="avg_cost_per_account", title="Illustrative average cost per account (validation)")
    fc.add_vline(x=m["selected_threshold"], line_dash="dash")
    st.plotly_chart(fc, width="stretch")

    cal = csv(OUT / "calibration.csv")
    cal = cal[cal["model"] == model]
    fcal = px.line(cal, x="mean_predicted", y="observed_rate", color="kind", markers=True, title="Calibration (test)")
    fcal.add_scatter(x=[0, 1], y=[0, 1], name="perfect", line=dict(dash="dash", color=GREY))
    st.plotly_chart(fcal, width="stretch")
    st.caption(f"'raw' = model output with class weighting; 'calibrated' = after Platt scaling fitted on validation data. "
               f"Test Brier score raw {m['test_raw_brier']:.3f} → calibrated {tm['brier']:.3f}.")

# ================================================================== D. Individual risk
elif page == PAGES[3]:
    st.title("Individual Risk Analysis")
    st.warning("Illustrative model output for exploration. It is not a credit decision and must not be used to approve or reject anyone.")
    cols = bundle.input_columns
    preset = st.selectbox("Pre-fill the form with", ["Training-set medians"] + [f"Real held-out record (ID {i})" for i in test_pred[config.ID_COLUMN].head(5)])
    if preset.startswith("Training"):
        base = {c: bundle.train_medians[c] for c in cols}
    else:
        rid = int(preset.split("ID ")[1].rstrip(")"))
        base = test_pred.loc[test_pred[config.ID_COLUMN] == rid, cols].iloc[0].to_dict()
    st.caption("Every field below is required and shown explicitly; nothing is imputed after you submit. "
               "Amounts are in NT$ (the dataset's currency).")
    values: dict[str, float] = {}
    with st.form("borrower"):
        a, b, c = st.columns(3)
        values["limit_bal"] = a.number_input("Credit limit (limit_bal)", 1.0, 1e8, float(base["limit_bal"]), 1000.0, key=f"{preset}limit")
        values["age"] = b.number_input("Age", 0, 120, int(base["age"]), key=f"{preset}age")
        values["education"] = c.selectbox("Education", [1, 2, 3, 4], index=int(base["education"]) - 1 if int(base["education"]) in (1, 2, 3, 4) else 3,
                                          format_func={1: "1 Graduate school", 2: "2 University", 3: "3 High school", 4: "4 Other"}.get, key=f"{preset}edu")
        values["marriage"] = a.selectbox("Marital status", [1, 2, 3], index=min(int(base["marriage"]), 3) - 1,
                                         format_func={1: "1 Married", 2: "2 Single", 3: "3 Other"}.get, key=f"{preset}mar")
        st.markdown("**Repayment status** (-2 no consumption, -1 paid in full, 0 revolving, 1-9 months delayed). `pay_1` is the latest month.")
        pc = st.columns(6)
        for i, col in enumerate(config.PAY_STATUS_COLS):
            values[col] = pc[i].number_input(col, -2, 9, int(base[col]), key=f"{preset}{col}")
        st.markdown("**Bill amounts** (`bill_amt1` latest)")
        bc = st.columns(6)
        for i, col in enumerate(config.BILL_COLS):
            values[col] = bc[i].number_input(col, -1e7, 1e8, float(base[col]), 100.0, key=f"{preset}{col}")
        st.markdown("**Payment amounts** (`pay_amt1` latest)")
        qc = st.columns(6)
        for i, col in enumerate(config.PAY_AMT_COLS):
            values[col] = qc[i].number_input(col, 0.0, 1e8, float(base[col]), 100.0, key=f"{preset}{col}")
        log_it = st.checkbox("Log this prediction to the audit table", value=True)
        submitted = st.form_submit_button("Estimate risk")
    if submitted:
        try:
            frame = validate_input(pd.DataFrame([values]))
            p = float(bundle.predict_proba(frame)[0])
            alert = ew.build_alerts([p], thr, edges=edges, labels=LABELS)
            metric_row([("Estimated default probability", f"{p:.1%}"), ("Illustrative risk band", alert.loc[0, "risk_band"]),
                        ("Review flag", "FLAG for review" if alert.loc[0, "flagged"] else "Not flagged")])
            st.caption(f"Threshold used: {thr:.2f}. The population base rate is {test_pred['default'].mean():.1%}. "
                       "A flag means 'worth a second look', not 'will default'.")
            ex = xai.explain_row(bundle, frame, top_k=10)
            ex = ex.iloc[::-1]
            fig = go.Figure(go.Bar(x=ex["contribution"], y=[f"{f} = {v:,.2f}" for f, v in zip(ex["feature"], ex["value"])], orientation="h",
                                   marker_color=[RED if x > 0 else BLUE for x in ex["contribution"]]))
            units = "log-odds" if bundle.model_name != "Random Forest" else "probability"
            fig.update_layout(title=f"Top feature contributions to the model score (SHAP, {units}; red raises, blue lowers)",
                              xaxis_title="SHAP contribution", height=420)
            st.plotly_chart(fig, width="stretch")
            up = ex[ex["contribution"] > 0].iloc[::-1].head(3)["feature"].tolist()
            down = ex[ex["contribution"] < 0].iloc[::-1].head(3)["feature"].tolist()
            st.caption("SHAP explains the model's raw (class-weighted) score. The probability shown above is that score after a "
                       "monotone Platt recalibration, so the direction and ranking of contributions carry over, but the units differ.")
            st.markdown(f"Features pushing the estimate **up**: {', '.join(up) or 'none'}. Features pushing it **down**: {', '.join(down) or 'none'}. "
                        "These describe how the model uses its inputs, not causes of default.")
            if log_it:
                db.insert_predictions(engine, alert, bundle.version, "individual_form", thr, inputs=[values])
        except InputValidationError as exc:
            st.error("Please fix the following:")
            for problem in exc.problems:
                st.write(f"- {problem}")
        except Exception:  # noqa: BLE001 - show a friendly message, keep details in the log
            logger.exception("Individual prediction failed")
            st.error("Prediction failed unexpectedly; see outputs/app.log.")

# ================================================================== E. Early warning
elif page == PAGES[4]:
    st.title("Early Warning")
    st.info("**Held-out HISTORICAL records** scored by the model (the outcome column is the actual historical label). "
            "Flags are illustrative review triggers, not decisions.")
    df = test_pred.drop(columns=["probability", "flagged", "risk_band"]).copy()
    al = ew.build_alerts(test_pred["probability"], thr, ids=test_pred[config.ID_COLUMN], edges=edges, labels=LABELS)
    df["predicted_probability"], df["flagged_for_review"], df["risk_band"] = al["probability"], al["flagged"], al["risk_band"]
    f1, f2, f3 = st.columns(3)
    bands = f1.multiselect("Risk band", LABELS, default=LABELS)
    only_flag = f2.checkbox("Flagged only", value=False)
    pay_rng = f3.slider("Latest repayment status (pay_1)", int(df["pay_1"].min()), int(df["pay_1"].max()),
                        (int(df["pay_1"].min()), int(df["pay_1"].max())))
    view = df[df["risk_band"].isin(bands) & df["pay_1"].between(*pay_rng)]
    if only_flag:
        view = view[view["flagged_for_review"]]
    metric_row([("Accounts shown", f"{len(view):,}"), ("Flagged for review", f"{int(view['flagged_for_review'].sum()):,}"),
                ("Flag rate", f"{view['flagged_for_review'].mean():.1%}" if len(view) else "–"),
                ("Observed default rate (shown)", f"{view['default'].mean():.1%}" if len(view) else "–")])
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(band_bar(view.assign(probability=view["predicted_probability"]), "Risk bands (current filter)"), width="stretch")
    c2.dataframe(ew.band_summary(view, LABELS).style.format({"share": "{:.1%}"}), hide_index=True, width="stretch")
    show = view.sort_values("predicted_probability", ascending=False).rename(columns={"default": "observed_outcome"})
    st.dataframe(show.style.format({"predicted_probability": "{:.3f}"}), width="stretch", hide_index=True, height=420)
    st.download_button("Download CSV (held-out demonstration predictions)", show.to_csv(index=False).encode(),
                       "early_warning_heldout_predictions.csv", "text/csv")

# ================================================================== F. Monitoring & audit
else:
    st.title("Monitoring & Audit")
    t1, t2, t3 = st.tabs(["Model & audit log", "Data quality, drift & performance", "Database analytics"])
    with t1:
        st.subheader("Model metadata")
        st.json({"model_version": bundle.version, "model": bundle.model_name, "trained_at": bundle.trained_at,
                 "random_seed": bundle.seed, "dataset_sha256": bundle.dataset_sha256, "review_threshold": bundle.threshold,
                 "threshold_method": bundle.threshold_method, "best_params": bundle.best_params,
                 "input_columns": bundle.input_columns, "sensitive_features_used": config.USE_SENSITIVE_FEATURES,
                 "split_sizes": {k: results[k] for k in ("n_train", "n_val", "n_test")}, "training_config": results["config"]})
        st.subheader("Prediction log (non-held-out predictions)")
        logs = db.run_query(engine, "recent_prediction_log")
        if logs.empty:
            st.caption("No logged predictions yet. Submit the Individual Risk form or run `python predict.py --log-to-db`.")
        else:
            st.dataframe(logs, width="stretch", hide_index=True)
    with t2:
        src = st.radio("Data to check", ["Held-out test records (historical)", "SYNTHETIC shifted scenario", "Upload CSV"], horizontal=True)
        new, labelled = None, False
        if src.startswith("Held-out"):
            new = test_pred.copy()
        elif src.startswith("SYNTHETIC"):
            st.warning("Synthetic stress test: bills ×1.35, limits ×0.8, 25% of accounts get +1 month delay. Not real customers.")
            new = monitoring.simulate_shift(test_pred)
        else:
            up = st.file_uploader("CSV with the model's input columns (optionally the default label)", type=["csv"])
            if up is not None:
                try:
                    new = pd.read_csv(io.BytesIO(up.getvalue()))
                    if len(new) > 100_000:
                        st.error("Please upload at most 100,000 rows."); new = None
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read CSV: {exc}")
        if new is not None:
            std = dp.standardize_columns(new)
            tgt = next((c for c in std.columns if c in config.TARGET_ALIASES or c.startswith("default_payment")), None)
            st.subheader("Data-quality checks")
            dq = monitoring.data_quality_report(std)
            st.dataframe(dq, hide_index=True, width="stretch")
            if (dq["status"] == "FAIL").any():
                st.error("Data-quality failures: fix the input before drift and performance checks.")
            else:
                psi = monitoring.compute_psi(bundle.reference, std)
                st.subheader("Input drift (PSI vs training data)")
                fig = px.bar(psi, x="feature", y="psi", color="status",
                             color_discrete_map={"OK": "#2a9d8f", "WARN": "#e9c46a", "ALERT": RED})
                fig.add_hline(y=config.PSI_WARN, line_dash="dot"); fig.add_hline(y=config.PSI_ALERT, line_dash="dash")
                st.plotly_chart(fig, width="stretch")
                st.caption(f"PSI < {config.PSI_WARN} stable · {config.PSI_WARN}-{config.PSI_ALERT} moderate · > {config.PSI_ALERT} significant. Indicative only for small batches.")
                frame = validate_input(std)
                proba = bundle.predict_proba(frame)
                train_mean = float(test_pred["probability"].mean())
                st.write(f"Mean predicted probability **{proba.mean():.3f}** (held-out baseline {train_mean:.3f}).")
                perf = None
                if tgt is not None:
                    st.subheader("Performance (labels available)")
                    try:
                        perf = monitoring.performance_report(std[tgt].astype(int), proba, thr, baseline=sel_test)
                        st.dataframe(pd.Series(perf, name="value").to_frame(), width="stretch")
                    except ValueError as exc:
                        st.warning(str(exc))
                else:
                    st.caption("No label column found, so performance cannot be measured yet.")
                if st.button("Save these results to the monitoring log"):
                    db.log_monitoring(engine, bundle.version, "drift_psi", [{"name": r.feature, "value": r.psi, "status": r.status} for r in psi.itertuples()], src)
                    db.log_monitoring(engine, bundle.version, "data_quality", [{"name": r.check, "status": r.status, "details": {"detail": r.detail}} for r in dq.itertuples()], src)
                    if perf:
                        db.log_monitoring(engine, bundle.version, "performance", [{"name": k, "value": v} for k, v in perf.items()], src)
                    st.success("Saved.")
    with t3:
        st.caption("Reusable SQL in sql/analytics.sql, run through SQLAlchemy.")
        queries = db.load_queries()
        qn = st.selectbox("Query", [q for q in queries if q not in ("recent_prediction_log",)])
        params = {"model_version": bundle.version} if ":model_version" in queries[qn] else None
        st.code(queries[qn], language="sql")
        st.dataframe(db.run_query(engine, qn, params), width="stretch", hide_index=True)
