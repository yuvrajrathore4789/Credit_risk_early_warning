import numpy as np
import pandas as pd
import pytest

import config
from src import early_warning as ew
from src import evaluation as ev
from src import explainability as xai
from src import monitoring
from src.inference import InputValidationError, predict_dataframe, validate_input


def _inputs(df, n=50):
    return df[[config.ID_COLUMN, *config.model_input_columns()]].head(n).copy()


@pytest.mark.parametrize("fx", ["xgb_bundle", "rf_bundle", "lr_bundle"])
def test_probabilities_within_bounds(fx, request, clean_df):
    b = request.getfixturevalue(fx)
    p = b.predict_proba(_inputs(clean_df, 300))
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1)) and np.isfinite(p).all()


def test_training_and_inference_use_identical_features(xgb_bundle, clean_df):
    X = _inputs(clean_df)
    assert list(xgb_bundle.pipeline[:-1].transform(X).columns) == xgb_bundle.transformed_feature_names
    batch = xgb_bundle.predict_proba(X)
    single = np.array([xgb_bundle.predict_proba(X.iloc[[i]])[0] for i in range(5)])
    np.testing.assert_allclose(batch[:5], single, rtol=1e-6)
    # extra / reordered columns do not change predictions
    shuffled = X[list(reversed(X.columns))].assign(junk=1)
    np.testing.assert_allclose(xgb_bundle.predict_proba(shuffled), batch, rtol=1e-6)


def test_predict_accepts_raw_uci_spellings(xgb_bundle, raw_df):
    out = predict_dataframe(xgb_bundle, raw_df.head(20).drop(columns=["default.payment.next.month"]))
    assert len(out) == 20 and {"probability", "flagged", "risk_band", "customer_id"} <= set(out.columns)
    assert (out["flagged"] == (out["probability"] >= xgb_bundle.threshold)).all()


def test_invalid_inputs_are_rejected_with_clear_messages(xgb_bundle, clean_df):
    X = _inputs(clean_df, 5)
    with pytest.raises(InputValidationError, match="missing required columns"):
        validate_input(X.drop(columns=["age"]))
    with pytest.raises(InputValidationError, match="limit_bal"):
        validate_input(X.assign(limit_bal=-5))
    with pytest.raises(InputValidationError, match="non-numeric"):
        validate_input(X.assign(age="old"))
    with pytest.raises(InputValidationError, match="pay_1"):
        validate_input(X.assign(pay_1=42))
    with pytest.raises(InputValidationError, match="no rows"):
        validate_input(X.iloc[0:0])


def test_missing_values_not_silently_filled(xgb_bundle, clean_df):
    X = _inputs(clean_df, 5)
    X.loc[X.index[0], "bill_amt1"] = np.nan
    with pytest.raises(InputValidationError, match="missing values"):
        validate_input(X)
    out = predict_dataframe(xgb_bundle, X, allow_missing=True)  # explicit opt-in
    assert np.isfinite(out["probability"]).all()


@pytest.mark.parametrize("fx", ["xgb_bundle", "rf_bundle", "lr_bundle"])
def test_shap_matches_the_model_being_explained(fx, request, clean_df):
    b = request.getfixturevalue(fx)
    X = _inputs(clean_df, 40)
    res = xai.compute_shap(b, X)
    assert res.values.shape == (40, len(b.transformed_feature_names))
    assert xai.additivity_gap(b, X, res) < 1e-3     # base + sum(SHAP) == model output
    ex = xai.explain_row(b, X.iloc[[0]], top_k=5)
    assert len(ex) == 5 and set(ex["direction"]) <= {"raises estimated risk", "lowers estimated risk"}


def test_risk_bands_and_alert_validation():
    s = ew.assign_risk_band([0.0, 0.149, 0.15, 0.3, 0.5, 1.0])
    assert s.astype(str).tolist() == ["Low", "Low", "Medium", "High", "Very High", "Very High"]
    with pytest.raises(ValueError):
        ew.assign_risk_band([0.1], edges=[0.5, 0.2], labels=list("abc"))
    with pytest.raises(ValueError):
        ew.assign_risk_band([1.2])
    with pytest.raises(ValueError):
        ew.build_alerts([0.2], threshold=1.5)
    a = ew.build_alerts([0.1, 0.4, 0.9], 0.3)
    assert a["flagged"].tolist() == [False, True, True]
    summ = ew.band_summary(a)
    assert summ["count"].sum() == 3 and summ["share"].sum() == pytest.approx(1.0)


def test_metrics_and_threshold_selection_are_computed_not_hardcoded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    m = ev.compute_metrics(y, p, 0.5)
    assert m["tp"] + m["fn"] == y.sum() and 0.5 < m["roc_auc"] <= 1
    table = ev.threshold_table(y, p)
    t_cost, t_f1 = ev.select_threshold(table, "min_cost"), ev.select_threshold(table, "max_f1")
    assert t_cost <= t_f1 + 0.3 and 0.05 <= t_cost <= 0.95  # cost-based picks a recall-leaning cut
    with pytest.raises(ValueError):
        ev.select_threshold(table, "magic")


def test_psi_detects_synthetic_shift(xgb_bundle, clean_df):
    same = monitoring.compute_psi(xgb_bundle.reference, clean_df.iloc[500:])
    shifted = monitoring.compute_psi(xgb_bundle.reference, monitoring.simulate_shift(clean_df.iloc[500:]))
    assert same["psi"].max() < config.PSI_ALERT
    assert (shifted["status"] == "ALERT").any()


def test_data_quality_and_performance_reports(xgb_bundle, clean_df):
    dq = monitoring.data_quality_report(_inputs(clean_df))
    assert (dq["status"] == "PASS").all()
    assert (monitoring.data_quality_report(_inputs(clean_df).drop(columns=["age"]))["status"] == "FAIL").any()
    X = _inputs(clean_df, 400)
    perf = monitoring.performance_report(clean_df["default"].head(400), xgb_bundle.predict_proba(X), 0.3, {"roc_auc": 0.7})
    assert "roc_auc_delta_vs_baseline" in perf
    with pytest.raises(ValueError):
        monitoring.performance_report([0, 0, 0], [0.1, 0.2, 0.3], 0.3)
