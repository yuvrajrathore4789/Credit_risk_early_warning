import numpy as np
import pandas as pd
import pytest

import config
from src import data_processing as dp
from src.feature_engineering import add_features, build_pipeline, feature_groups
from sklearn.linear_model import LogisticRegression


def test_standardize_columns_maps_aliases(raw_df):
    out = dp.standardize_columns(raw_df)
    assert "pay_1" in out.columns and "pay_0" not in out.columns
    assert "default_payment_next_month" in out.columns
    assert all(c == c.lower() and " " not in c and "." not in c for c in out.columns)


def test_find_target_validates_binary(raw_df):
    std = dp.standardize_columns(raw_df)
    assert dp.find_target_column(std) == "default_payment_next_month"
    bad = std.copy()
    bad["default_payment_next_month"] = 3
    with pytest.raises(dp.DataValidationError, match="0 and 1"):
        dp.find_target_column(bad)
    with pytest.raises(dp.DataValidationError, match="exactly one target"):
        dp.find_target_column(std.drop(columns="default_payment_next_month"))


def test_target_must_have_both_classes(raw_df):
    one = raw_df.copy()
    one["default.payment.next.month"] = 0
    with pytest.raises(dp.DataValidationError):
        dp.validate_and_clean(one)


def test_missing_required_column_gives_clear_error(raw_df):
    with pytest.raises(dp.DataValidationError, match="missing required columns"):
        dp.validate_and_clean(raw_df.drop(columns=["AGE"]))


def test_cleaning_handles_duplicates_missing_and_invalid(raw_df):
    df = raw_df.copy()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)       # duplicate ID
    df = df.astype({"default.payment.next.month": float, "BILL_AMT1": float, "LIMIT_BAL": object})
    df.loc[1, "default.payment.next.month"] = np.nan            # missing target -> dropped
    df.loc[2, "BILL_AMT1"] = np.nan                             # missing feature -> kept as NaN
    df.loc[3, "AGE"] = 5                                        # rule violation -> dropped
    df.loc[4, "LIMIT_BAL"] = "abc"                              # non-numeric -> NaN
    clean, rep = dp.validate_and_clean(df)
    assert rep["duplicate_id_rows_removed"] == 1
    assert rep["rows_dropped_missing_id_or_target"] == 1
    assert rep["rows_removed_rule_violations"] == 1
    assert rep["non_numeric_values_coerced_to_nan"] == {"limit_bal": 1}
    assert clean["bill_amt1"].isna().sum() == 1                  # NOT silently filled
    assert clean["id"].is_unique and len(clean) == len(raw_df) - 2


def test_recode_categoricals_is_idempotent(raw_df):
    clean, rep = dp.validate_and_clean(raw_df)
    assert set(clean["education"]) <= {1, 2, 3, 4} and set(clean["marriage"]) <= {1, 2, 3}
    assert dp.recode_categoricals(clean).equals(clean)
    assert rep["education_codes_recoded"] > 0


def test_identifier_never_reaches_model(clean_df):
    numeric, categorical = feature_groups()
    assert config.ID_COLUMN not in numeric + categorical
    assert config.TARGET_COLUMN not in numeric + categorical
    assert "sex" not in numeric + categorical  # protected attribute excluded by default
    pipe = build_pipeline(LogisticRegression(max_iter=300), True)
    X = clean_df[[config.ID_COLUMN, *config.model_input_columns()]]
    pipe.fit(X, clean_df[config.TARGET_COLUMN])
    names = list(pipe[:-1].transform(X).columns)
    assert not any(n in names for n in ("id", "default"))


def test_engineered_features_do_not_use_target(clean_df):
    a = add_features(clean_df)
    b = add_features(clean_df.assign(default=1 - clean_df["default"]))
    pd.testing.assert_frame_equal(a.drop(columns="default"), b.drop(columns="default"))


def test_split_is_stratified_disjoint_and_reproducible(clean_df):
    t1, v1, s1 = dp.split_data(clean_df, seed=7)
    t2, v2, s2 = dp.split_data(clean_df, seed=7)
    assert t1.equals(t2) and v1.equals(v2) and s1.equals(s2)
    ids = [set(d["id"]) for d in (t1, v1, s1)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(t1) + len(v1) + len(s1) == len(clean_df)
    rates = [d["default"].mean() for d in (t1, v1, s1)]
    assert max(rates) - min(rates) < 0.02


def test_preprocessing_is_reproducible(clean_df):
    X = clean_df[[config.ID_COLUMN, *config.model_input_columns()]]
    y = clean_df["default"]
    out = [build_pipeline(LogisticRegression(max_iter=300), True).fit(X, y)[:-1].transform(X) for _ in range(2)]
    pd.testing.assert_frame_equal(out[0], out[1])


def test_imputer_is_fit_on_training_data_only(clean_df):
    X = clean_df[[config.ID_COLUMN, *config.model_input_columns()]].copy()
    y = clean_df["default"]
    pipe = build_pipeline(LogisticRegression(max_iter=300), True).fit(X.iloc[:400], y.iloc[:400])
    test_row = X.iloc[[500]].copy()
    test_row["limit_bal"] = np.nan
    filled = pipe[:-1].transform(test_row)["limit_bal"].iloc[0]
    train_median = X.iloc[:400]["limit_bal"].median()
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    idx = feature_groups()[0].index("limit_bal")
    assert filled == pytest.approx((train_median - scaler.mean_[idx]) / scaler.scale_[idx])


def test_find_dataset_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No dataset found"):
        dp.find_dataset(tmp_path)
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.csv").write_text("x\n1\n")
    with pytest.raises(FileNotFoundError, match="Several candidate"):
        dp.find_dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        dp.find_dataset(tmp_path / "nope")


def test_load_raw_roundtrip_and_bad_type(tmp_path, raw_df):
    p = tmp_path / "UCI_Credit_Card.csv"
    raw_df.to_csv(p, index=False)
    assert len(dp.load_raw(p)) == len(raw_df)
    assert dp.find_dataset(tmp_path) == p
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(dp.DataValidationError, match="Unsupported"):
        dp.load_raw(tmp_path / "x.txt")
