"""Dataset discovery, loading, validation, cleaning and splitting (Phase 1).

All functions here are stateless with respect to the target: nothing is "fitted" on the full
data, so splitting happens before any learned preprocessing (imputers/scalers live in the
sklearn pipeline and are fit on the training split only).
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

__all__ = [
    "DataValidationError", "find_dataset", "file_sha256", "load_raw", "standardize_columns",
    "find_target_column", "coerce_numeric", "recode_categoricals", "check_rules",
    "validate_and_clean", "split_data", "prepare_raw",
]


class DataValidationError(ValueError):
    """Raised when the dataset (or a caller-supplied frame) fails validation."""


# ------------------------------------------------------------------ discovery / loading
def find_dataset(data_dir: Path | str = config.DATA_DIR) -> Path:
    """Locate the dataset file without assuming a single filename."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for name in config.DATASET_CANDIDATES:
        if (data_dir / name).is_file():
            return data_dir / name
    others = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".csv", ".xls", ".xlsx"})
    if len(others) == 1:
        return others[0]
    if not others:
        raise FileNotFoundError(
            f"No dataset found in {data_dir}. See data/README.md for download instructions, "
            "or pass --data-path to train.py."
        )
    raise FileNotFoundError(
        f"Several candidate files in {data_dir}: {[p.name for p in others]}. "
        "Pass --data-path to choose one."
    )


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read CSV/XLS(X). The UCI Excel version has a banner row above the header."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=0)
        if not any("pay" in str(c).lower() for c in df.columns):  # banner row present
            df = pd.read_excel(path, header=1)
    else:
        raise DataValidationError(f"Unsupported file type '{suffix}'. Use .csv, .xls or .xlsx.")
    if df.empty:
        raise DataValidationError(f"Dataset file {path.name} is empty.")
    return df


# ------------------------------------------------------------------ standardization
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """lower_snake_case names; maps known aliases (PAY_0 -> pay_1, target spellings)."""
    out = df.copy()
    names = [re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_") for c in out.columns]
    out.columns = [config.COLUMN_RENAMES.get(n, n) for n in names]
    if out.columns.duplicated().any():
        dups = out.columns[out.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate column names after standardization: {dups}")
    return out


def find_target_column(df: pd.DataFrame) -> str:
    """Find (never assume) the target column and verify it is binary."""
    hits = [c for c in df.columns if c in config.TARGET_ALIASES or c.startswith("default_payment")]
    if len(hits) != 1:
        raise DataValidationError(
            f"Expected exactly one target column among {config.TARGET_ALIASES}; found {hits}. "
            f"Columns present: {list(df.columns)}"
        )
    col = hits[0]
    values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
    if not values or not values <= {0, 1} or len(values) < 2:
        raise DataValidationError(f"Target '{col}' must contain both 0 and 1 only; found {sorted(values)}")
    return col


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce columns to numeric; returns count of values that were non-numeric (became NaN)."""
    out = df.copy()
    coerced: dict[str, int] = {}
    for c in cols:
        before = out[c].isna().sum()
        out[c] = pd.to_numeric(out[c], errors="coerce")
        n = int(out[c].isna().sum() - before)
        if n:
            coerced[c] = n
    return out, coerced


def recode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented codes into the documented 'other' category.

    UCI documents EDUCATION 1-4 and MARRIAGE 1-3, but the data also holds EDUCATION 0/5/6 and
    MARRIAGE 0. They are folded into "other" (4 / 3). Idempotent and target-free, so it is
    safe to apply identically at training and inference time.
    """
    out = df.copy()
    if "education" in out:
        out["education"] = out["education"].where(~out["education"].isin([0, 5, 6]), 4)
    if "marriage" in out:
        out["marriage"] = out["marriage"].where(out["marriage"] != 0, 3)
    return out


# ------------------------------------------------------------------ validation rules
def _rule_specs() -> list[tuple[str, str, "callable"]]:
    """(column, message, predicate that is True for VALID values)."""
    specs = [
        ("limit_bal", "limit_bal must be between 1 and 1e8", lambda s: s.between(1, 1e8)),
        ("age", "age must be between 18 and 100", lambda s: s.between(18, 100)),
        ("education", "education must be an integer code 0-6", lambda s: s.isin(range(0, 7))),
        ("marriage", "marriage must be an integer code 0-3", lambda s: s.isin(range(0, 4))),
        ("sex", "sex must be 1 or 2", lambda s: s.isin([1, 2])),
    ]
    specs += [(c, f"{c} must be an integer between -2 and 9", lambda s: s.isin(range(-2, 10)))
              for c in config.PAY_STATUS_COLS]
    specs += [(c, f"{c} must be within +/-1e8", lambda s: s.abs().le(1e8)) for c in config.BILL_COLS]
    specs += [(c, f"{c} must be between 0 and 1e8", lambda s: s.between(0, 1e8))
              for c in config.PAY_AMT_COLS]
    return specs


def check_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Hard validity rules -> {message: boolean mask of VIOLATING rows}.

    Missing values are not violations here; they are reported/handled separately.
    """
    out: dict[str, pd.Series] = {}
    for col, message, is_valid in _rule_specs():
        if col in df:
            s = df[col]
            out[message] = (~is_valid(s)) & s.notna()
    return out


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardized but NOT cleaned/recoded copy of the source rows (for the raw DB table)."""
    df = standardize_columns(raw)
    target = find_target_column(df)
    df = df.rename(columns={target: config.TARGET_COLUMN})
    required = [config.ID_COLUMN, *config.RAW_FEATURE_COLS, config.TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")
    df, _ = coerce_numeric(df[required], required)
    df = df.dropna(subset=[config.ID_COLUMN, config.TARGET_COLUMN])
    df = df.drop_duplicates(subset=[config.ID_COLUMN], keep="first")
    return df.astype({config.ID_COLUMN: int, config.TARGET_COLUMN: int}).reset_index(drop=True)


# ------------------------------------------------------------------ main entry point
def validate_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean the raw dataset. Returns (clean_df, report).

    clean_df columns: id, <raw feature columns>, default. Missing feature values are left as NaN
    (imputed later inside the sklearn pipeline, fit on the training split only).
    """
    df = standardize_columns(raw)
    target = find_target_column(df)
    if target != config.TARGET_COLUMN:
        df = df.rename(columns={target: config.TARGET_COLUMN})

    required = [config.ID_COLUMN, *config.RAW_FEATURE_COLS, config.TARGET_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Dataset is missing required columns: {missing_cols}")
    df = df[required].copy()

    report: dict = {"raw_rows": int(len(df)), "target_column_found": target}
    df, coerced = coerce_numeric(df, required)
    report["non_numeric_values_coerced_to_nan"] = coerced

    report["missing_by_column"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    bad_key = df[config.ID_COLUMN].isna() | df[config.TARGET_COLUMN].isna()
    report["rows_dropped_missing_id_or_target"] = int(bad_key.sum())
    df = df.loc[~bad_key]

    dup_id = df.duplicated(subset=[config.ID_COLUMN], keep="first")
    report["duplicate_id_rows_removed"] = int(dup_id.sum())
    df = df.loc[~dup_id]
    # Identical feature rows with *different* IDs are plausibly different customers: reported, kept.
    feat_dup = df.duplicated(subset=[*config.RAW_FEATURE_COLS, config.TARGET_COLUMN], keep=False)
    report["identical_feature_rows_with_distinct_ids_kept"] = int(feat_dup.sum())

    violations = check_rules(df)
    bad_rows = pd.Series(False, index=df.index)
    for mask in violations.values():
        bad_rows |= mask
    report["rule_violations"] = {k: int(v.sum()) for k, v in violations.items() if v.any()}
    report["rows_removed_rule_violations"] = int(bad_rows.sum())
    df = df.loc[~bad_rows]

    report["education_codes_recoded"] = int(df["education"].isin([0, 5, 6]).sum())
    report["marriage_codes_recoded"] = int((df["marriage"] == 0).sum())
    df = recode_categoricals(df)

    for c in [config.ID_COLUMN, config.TARGET_COLUMN]:
        df[c] = df[c].astype(int)
    df = df.reset_index(drop=True)

    if df[config.TARGET_COLUMN].nunique() != 2:
        raise DataValidationError("After cleaning the target has fewer than two classes.")
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    report["final_rows"] = int(len(df))
    report["class_counts"] = {int(k): int(v) for k, v in counts.items()}
    report["default_rate"] = float(df[config.TARGET_COLUMN].mean())
    logger.info("Cleaned dataset: %s", {k: v for k, v in report.items() if k != "missing_by_column"})
    return df, report


# ------------------------------------------------------------------ splitting
def split_data(
    df: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split (done BEFORE any preprocessing is fit)."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1).")
    y = df[config.TARGET_COLUMN]
    train_val, test = train_test_split(df, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[config.TARGET_COLUMN], random_state=seed
    )
    assert not (set(train[config.ID_COLUMN]) & set(test[config.ID_COLUMN]))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
