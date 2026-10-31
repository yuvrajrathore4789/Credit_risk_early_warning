"""Central configuration: paths, schema constants, seeds and illustrative policy settings.

Everything an analyst might reasonably want to tweak lives here. The risk bands and
costs below are ILLUSTRATIVE demonstration values, not a validated lending policy.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

DB_PATH = DATA_DIR / "credit_risk.db"
MODEL_PATH = MODELS_DIR / "credit_risk_model.joblib"
LOG_PATH = OUTPUTS_DIR / "app.log"
SCHEMA_SQL = SQL_DIR / "schema.sql"
ANALYTICS_SQL = SQL_DIR / "analytics.sql"

# ---------------------------------------------------------------- reproducibility
RANDOM_SEED = 42
VAL_SIZE = 0.20   # fraction of all rows used for validation (model/threshold selection)
TEST_SIZE = 0.20  # fraction of all rows held out until final evaluation
CV_FOLDS = 3

MODEL_VERSION_PREFIX = "1.0.0"

# ---------------------------------------------------------------- dataset schema
# The UCI file ships as "UCI_Credit_Card.csv" (Kaggle mirror) or an .xls from UCI itself.
DATASET_CANDIDATES = [
    "UCI_Credit_Card.csv",
    "default of credit card clients.csv",
    "default_of_credit_card_clients.csv",
    "default of credit card clients.xls",
    "default of credit card clients.xlsx",
]
ID_COLUMN = "id"
TARGET_COLUMN = "default"  # standardized name; several raw spellings are accepted
TARGET_ALIASES = ["default", "default_payment_next_month", "default_payment_next_month_"]
COLUMN_RENAMES = {"pay_0": "pay_1"}  # UCI skips PAY_1; we make the series contiguous

PAY_STATUS_COLS = [f"pay_{i}" for i in range(1, 7)]       # repayment status, 1 = latest month
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]
RAW_FEATURE_COLS = (
    ["limit_bal", "sex", "education", "marriage", "age"]
    + PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS
)

# Protected attribute handling: sex is kept for descriptive analysis but is excluded from the
# model by default because using it in real credit decisions is generally impermissible.
SENSITIVE_FEATURES = ["sex"]
USE_SENSITIVE_FEATURES = False

CATEGORICAL_BASE = ["education", "marriage"]


def model_input_columns() -> list[str]:
    """Raw columns a caller must supply for inference."""
    cols = list(RAW_FEATURE_COLS)
    if not USE_SENSITIVE_FEATURES:
        cols = [c for c in cols if c not in SENSITIVE_FEATURES]
    return cols


def categorical_columns() -> list[str]:
    return CATEGORICAL_BASE + (SENSITIVE_FEATURES if USE_SENSITIVE_FEATURES else [])


# ---------------------------------------------------------------- illustrative policy
# Costs used ONLY to pick a demonstration threshold on the validation set.
ILLUSTRATIVE_COST_FALSE_NEGATIVE = 3.0  # missed eventual defaulter
ILLUSTRATIVE_COST_FALSE_POSITIVE = 1.0  # unnecessary review

RISK_BAND_EDGES = [0.15, 0.30, 0.50]
RISK_BAND_LABELS = ["Low", "Medium", "High", "Very High"]

# Drift (Population Stability Index) rules of thumb
PSI_WARN = 0.10
PSI_ALERT = 0.25

DISCLAIMER = (
    "Predicted risk is a statistical estimate, not an observed default, and is not a lending "
    "decision. Thresholds and bands here are illustrative; a real deployment needs independent "
    "validation, governance and compliance review."
)
