import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

import config
from src import data_processing as dp
from src import database as db
from src import early_warning as ew


@pytest.fixture()
def engine():
    e = db.get_engine(":memory:")
    db.init_db(e)
    return e


@pytest.fixture()
def labelled(clean_df):
    train, val, test = dp.split_data(clean_df, seed=1)
    return pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")], ignore_index=True)


def test_schema_created_and_validated(engine):
    db.validate_schema(engine)
    db.init_db(engine)  # idempotent


def test_schema_validation_detects_problems(engine):
    with engine.begin() as c:
        c.execute(text("DROP TABLE monitoring_log"))
    with pytest.raises(db.SchemaError, match="monitoring_log"):
        db.validate_schema(engine)


def test_insert_and_retrieve_clean_records(engine, labelled):
    stats = db.load_clean_records(engine, labelled)
    assert stats["inserted"] == len(labelled)
    back = db.read_table(engine, "clean_credit_clients", limit=10_000).sort_values("customer_id")
    assert len(back) == len(labelled)
    src = labelled.sort_values("id")
    assert back["default_next_month"].tolist() == src["default"].tolist()
    assert set(back["split"]) == {"train", "val", "test"}


def test_duplicate_loads_are_idempotent(engine, labelled, raw_df):
    db.load_clean_records(engine, labelled)
    again = db.load_clean_records(engine, labelled)
    assert again["inserted"] == 0 and again["skipped_existing"] == len(labelled)
    raw = dp.prepare_raw(raw_df)
    assert db.load_raw_records(engine, raw)["inserted"] == len(raw)
    assert db.load_raw_records(engine, raw)["inserted"] == 0
    assert db._count(engine, "raw_credit_clients") == len(raw)


def test_replace_mode_updates_splits(engine, labelled):
    db.load_clean_records(engine, labelled)
    flipped = labelled.assign(split="train")
    db.load_clean_records(engine, flipped, replace=True)
    assert db.run_query(engine, "split_summary")["split"].tolist() == ["train"]


def test_load_validates_inputs(engine, clean_df, labelled):
    with pytest.raises(db.SchemaError, match="split"):
        db.load_clean_records(engine, clean_df)
    with pytest.raises(db.SchemaError):
        db.load_clean_records(engine, labelled.assign(split="oops"))
    with pytest.raises(db.SchemaError, match="missing columns"):
        db.load_clean_records(engine, labelled.drop(columns=["age"]))


def test_nan_features_stored_as_null(engine, labelled):
    df = labelled.copy()
    df.loc[df.index[0], "pay_amt1"] = float("nan")
    db.load_clean_records(engine, df)
    with engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM clean_credit_clients WHERE pay_amt1 IS NULL")).scalar_one() == 1


def test_analytics_queries_run_and_match_pandas(engine, labelled):
    db.load_clean_records(engine, labelled)
    summary = db.run_query(engine, "portfolio_summary").iloc[0]
    assert summary["n_customers"] == len(labelled)
    assert summary["default_rate"] == pytest.approx(labelled["default"].mean(), abs=1e-4)
    by_pay = db.run_query(engine, "default_rate_by_latest_repayment_status").set_index("latest_repayment_status")
    expected = labelled.groupby("pay_1")["default"].mean()
    for k, v in expected.items():
        assert by_pay.loc[k, "default_rate"] == pytest.approx(v, abs=1e-4)
    for name in ["default_rate_by_education", "default_rate_by_marriage", "default_rate_by_age_band",
                 "default_rate_by_limit_band", "default_rate_by_months_delayed", "split_summary"]:
        assert not db.run_query(engine, name).empty
    with pytest.raises(KeyError):
        db.run_query(engine, "drop_everything")


def _model_run(version="v1"):
    return {"model_version": version, "trained_at": "2026-01-01T00:00:00", "selected_model": "XGBoost",
            "random_seed": 42, "threshold": 0.3, "metrics": {"x": 1}, "config": {}, "feature_names": ["a"]}


def test_predictions_logging_and_audit_queries(engine, labelled):
    db.load_clean_records(engine, labelled)
    db.insert_model_run(engine, _model_run())
    test_ids = labelled.loc[labelled["split"] == "test", "id"].head(30).tolist()
    alerts = ew.build_alerts([0.05 * (i % 20) for i in range(30)], 0.3, ids=test_ids)
    assert db.insert_predictions(engine, alerts, "v1", "heldout_test", 0.3) == 30
    summary = db.run_query(engine, "risk_band_summary", {"model_version": "v1"})
    assert summary["n"].sum() == 30
    assert not db.run_query(engine, "flagged_accounts", {"model_version": "v1"}).empty
    db.insert_predictions(engine, alerts.head(1).drop(columns="customer_id"), "v1", "individual_form", 0.3, inputs=[{"age": 30}])
    log = db.run_query(engine, "recent_prediction_log")
    assert len(log) == 1 and log.loc[0, "source"] == "individual_form"
    db.replace_heldout_predictions(engine, "v1")
    assert db._count(engine, "predictions") == 1


def test_prediction_logging_rejects_bad_source_and_unknown_model(engine):
    alerts = ew.build_alerts([0.2], 0.3)
    db.insert_model_run(engine, _model_run())
    with pytest.raises(ValueError):
        db.insert_predictions(engine, alerts, "v1", "made_up", 0.3)
    with pytest.raises(IntegrityError):  # foreign key to model_runs enforced
        db.insert_predictions(engine, alerts, "ghost", "individual_form", 0.3)


def test_model_run_upsert_and_monitoring_log(engine):
    db.insert_model_run(engine, _model_run())
    db.insert_model_run(engine, {**_model_run(), "threshold": 0.4})
    runs = db.read_table(engine, "model_runs")
    assert len(runs) == 1 and runs.loc[0, "threshold"] == 0.4
    db.log_monitoring(engine, "v1", "drift_psi", [{"name": "age", "value": 0.02, "status": "OK"}], "test")
    assert db._count(engine, "monitoring_log") == 1
