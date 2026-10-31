"""Stateless feature engineering + the sklearn preprocessing pipeline.

Engineered features use only information available at prediction time (past bills, payments and
repayment status) and never the target. They are deterministic row-wise functions, so training
and inference apply exactly the same transformation. Learned steps (imputation, scaling,
one-hot) live in the ColumnTransformer and are fit on the training split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.data_processing import recode_categoricals

ENGINEERED_COLS = [
    "avg_bill_amt", "avg_pay_amt", "utilization_latest", "utilization_avg", "payment_ratio",
    "max_delay", "mean_delay", "n_delayed_months", "bill_trend",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered columns appended (NaN-safe, no target use)."""
    out = recode_categoricals(df)
    bills = out[config.BILL_COLS]
    pays = out[config.PAY_AMT_COLS]
    delays = out[config.PAY_STATUS_COLS].clip(lower=0)  # -2/-1/0 mean no delay; >=1 = months late
    limit = out["limit_bal"].where(out["limit_bal"] > 0)

    out["avg_bill_amt"] = bills.mean(axis=1)
    out["avg_pay_amt"] = pays.mean(axis=1)
    out["utilization_latest"] = out["bill_amt1"] / limit
    out["utilization_avg"] = out["avg_bill_amt"] / limit
    # PAY_AMT(k) settles BILL_AMT(k+1): compare payments in months 1-5 with bills of months 2-6
    owed = out[config.BILL_COLS[1:]].clip(lower=0).sum(axis=1)
    paid = out[config.PAY_AMT_COLS[:-1]].sum(axis=1)
    out["payment_ratio"] = (paid / owed.where(owed > 0)).fillna(1.0).clip(0, 5)
    out["max_delay"] = delays.max(axis=1)
    out["mean_delay"] = delays.mean(axis=1)
    out["n_delayed_months"] = (delays >= 1).sum(axis=1)
    out["bill_trend"] = (out["bill_amt1"] - out["bill_amt6"]) / limit
    out[["utilization_latest", "utilization_avg", "bill_trend"]] = (
        out[["utilization_latest", "utilization_avg", "bill_trend"]].clip(-5, 5)
    )
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper around :func:`add_features` (stateless)."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame.")
        return add_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.input_columns_) + ENGINEERED_COLS)


def feature_groups() -> tuple[list[str], list[str]]:
    """(numeric_columns, categorical_columns) fed to the ColumnTransformer."""
    cat = config.categorical_columns()
    base = [c for c in config.model_input_columns() if c not in cat]
    return base + ENGINEERED_COLS, cat


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric, categorical = feature_groups()
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:  # needed by Logistic Regression; trees are scale-invariant
        num_steps.append(("scale", StandardScaler()))
    ct = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",  # drops id/sex/etc: identifiers can never reach the model
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def build_pipeline(estimator, scale_numeric: bool) -> Pipeline:
    """features -> preprocess -> model. The same object is used for training and inference."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("prep", build_preprocessor(scale_numeric)),
        ("model", estimator),
    ])
