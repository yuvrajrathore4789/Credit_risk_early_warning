"""Input validation + prediction used identically by predict.py, the dashboard and tests."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

import config
from src import data_processing as dp
from src.early_warning import build_alerts
from src.model_training import ModelBundle


class InputValidationError(ValueError):
    """Raised with a human-readable list of everything wrong with the supplied input."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Invalid input: " + " | ".join(problems))


def validate_input(df: pd.DataFrame, allow_missing: bool = False, max_messages: int = 12) -> pd.DataFrame:
    """Validate borrower rows and return a numeric frame of the model's input columns.

    Nothing is filled in silently: missing values are an error unless ``allow_missing=True``
    is passed explicitly, in which case the pipeline imputes them with TRAINING medians/modes.
    """
    if df is None or len(df) == 0:
        raise InputValidationError(["no rows supplied"])
    frame = dp.standardize_columns(df)
    required = config.model_input_columns()
    missing_cols = [c for c in required if c not in frame.columns]
    if missing_cols:
        raise InputValidationError([f"missing required columns: {missing_cols}"])

    keep = ([config.ID_COLUMN] if config.ID_COLUMN in frame.columns else []) + required
    frame = frame[keep].copy()
    frame[required], coerced = dp.coerce_numeric(frame[required], required)
    problems: list[str] = []
    if coerced:
        problems.append(f"non-numeric values in {coerced}")
    n_missing = frame[required].isna().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing) and not allow_missing:
        problems.append(f"missing values (not imputed unless allow_missing=True): {n_missing.to_dict()}")
    for message, mask in dp.check_rules(frame).items():
        if mask.any():
            rows = (mask[mask].index[:5] + 1).tolist()
            problems.append(f"{message} (rows {rows}{'...' if mask.sum() > 5 else ''})")
    if problems:
        raise InputValidationError(problems[:max_messages])
    return frame.reset_index(drop=True)


def predict_dataframe(
    bundle: ModelBundle, df: pd.DataFrame, threshold: float | None = None,
    edges: Sequence[float] = config.RISK_BAND_EDGES, labels: Sequence[str] = config.RISK_BAND_LABELS,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Validated -> calibrated probabilities, review flag and risk band for every row."""
    frame = validate_input(df, allow_missing=allow_missing)
    thr = bundle.threshold if threshold is None else float(threshold)
    proba = bundle.predict_proba(frame)
    ids = frame[config.ID_COLUMN] if config.ID_COLUMN in frame.columns else None
    return build_alerts(proba, thr, ids=ids, edges=edges, labels=labels)
