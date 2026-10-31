"""Shared fixtures. Tests use a small SYNTHETIC frame with the UCI schema so they run offline,
fast, and without the real dataset. It is test scaffolding only, never used for reported results."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.feature_engineering import build_pipeline  # noqa: E402
from src.model_training import ModelBundle, PlattCalibrator  # noqa: E402
from src.monitoring import build_reference  # noqa: E402


def make_raw(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame using the ORIGINAL UCI column spellings (PAY_0, default.payment.next.month)."""
    rng = np.random.default_rng(seed)
    pay = rng.choice([-2, -1, 0, 0, 0, 1, 2, 3], size=(n, 6))
    limit = rng.choice([20000, 50000, 100000, 200000, 400000], n)
    bills = rng.integers(0, 150000, (n, 6)).astype(float)
    pays = rng.integers(0, 20000, (n, 6)).astype(float)
    logit = -1.6 + 0.8 * np.clip(pay[:, 0], 0, None) + 0.3 * np.clip(pay[:, 1], 0, None) + rng.normal(0, 0.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1), "LIMIT_BAL": limit, "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[.01, .3, .45, .15, .05, .02, .02]),
        "MARRIAGE": rng.choice([0, 1, 2, 3], n, p=[.01, .45, .5, .04]), "AGE": rng.integers(21, 70, n),
    })
    for i, c in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        df[c] = pay[:, i]
    for i in range(6):
        df[f"BILL_AMT{i + 1}"] = bills[:, i]
    for i in range(6):
        df[f"PAY_AMT{i + 1}"] = pays[:, i]
    df["default.payment.next.month"] = y
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from src.data_processing import validate_and_clean
    df, _ = validate_and_clean(raw_df)
    return df


def _bundle(estimator, scale: bool, df: pd.DataFrame, name: str) -> ModelBundle:
    cols = config.model_input_columns()
    X = df[[config.ID_COLUMN, *cols]]
    y = df[config.TARGET_COLUMN]
    pipe = build_pipeline(estimator, scale).fit(X.iloc[:500], y.iloc[:500])
    cal = PlattCalibrator().fit(pipe.predict_proba(X.iloc[500:])[:, 1], y.iloc[500:])
    prep = pipe[:-1].transform(X.iloc[:500])
    return ModelBundle(
        pipeline=pipe, calibrator=cal, model_name=name, version="test-0", trained_at="2026-01-01T00:00:00",
        threshold=0.3, threshold_method="min_cost", input_columns=cols,
        transformed_feature_names=list(prep.columns), reference=build_reference(df.iloc[:500]),
        shap_background=prep.sample(100, random_state=0),
        train_medians={c: float(df[c].median()) for c in cols},
    )


@pytest.fixture(scope="session")
def xgb_bundle(clean_df):
    return _bundle(XGBClassifier(n_estimators=30, max_depth=3, random_state=0, n_jobs=1), False, clean_df, "XGBoost")


@pytest.fixture(scope="session")
def rf_bundle(clean_df):
    return _bundle(RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0, n_jobs=1), False, clean_df, "Random Forest")


@pytest.fixture(scope="session")
def lr_bundle(clean_df):
    return _bundle(LogisticRegression(max_iter=500), True, clean_df, "Logistic Regression")
