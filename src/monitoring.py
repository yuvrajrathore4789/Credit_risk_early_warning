"""Logging setup, data-quality checks, drift (PSI), performance monitoring (Phase 6)."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from src import data_processing as dp

_EPS = 1e-4


def setup_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    """Console + file logging (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_credit_risk_configured", False):
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if to_file:
        try:
            config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(config.LOG_PATH, encoding="utf-8"))
        except OSError:  # read-only filesystems (e.g. some hosts): console only
            pass
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    root._credit_risk_configured = True  # type: ignore[attr-defined]


# ------------------------------------------------------------------ data quality
def data_quality_report(df: pd.DataFrame, required: Sequence[str] | None = None) -> pd.DataFrame:
    """Row-level data-quality checks. Returns check / status (PASS|WARN|FAIL) / detail."""
    df = dp.standardize_columns(df)
    required = list(required or config.model_input_columns())
    checks: list[dict] = []

    def add(check: str, status: str, detail: str) -> None:
        checks.append({"check": check, "status": status, "detail": detail})

    missing_cols = [c for c in required if c not in df.columns]
    add("Required columns present", "FAIL" if missing_cols else "PASS",
        f"missing: {missing_cols}" if missing_cols else f"all {len(required)} present")
    if missing_cols:
        return pd.DataFrame(checks)
    num, coerced = dp.coerce_numeric(df, required)
    add("Numeric types", "FAIL" if coerced else "PASS", f"non-numeric values: {coerced}" if coerced else "all numeric")
    n_missing = int(num[required].isna().any(axis=1).sum())
    add("Missing values", "FAIL" if n_missing else "PASS", f"{n_missing} rows with missing inputs")
    viol = {k: int(v.sum()) for k, v in dp.check_rules(num).items() if v.any()}
    add("Value-range rules", "FAIL" if viol else "PASS", str(viol) if viol else "no violations")
    if config.ID_COLUMN in df:
        d = int(df[config.ID_COLUMN].duplicated().sum())
        add("Duplicate IDs", "WARN" if d else "PASS", f"{d} duplicated IDs")
    add("Row count", "WARN" if len(df) < 30 else "PASS", f"{len(df):,} rows" + (" (too few for stable drift stats)" if len(df) < 30 else ""))
    return pd.DataFrame(checks)


# ------------------------------------------------------------------ drift (PSI)
def drift_columns() -> list[str]:
    return [c for c in config.model_input_columns() if c not in config.categorical_columns()]


def build_reference(train: pd.DataFrame, columns: Sequence[str] | None = None, n_bins: int = 10) -> dict:
    """Quantile bin edges + expected proportions from TRAINING data."""
    ref: dict = {}
    for c in columns or drift_columns():
        x = train[c].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
        ref[c] = {"edges": edges.tolist(), "expected": _proportions(x, edges).tolist()}
    return ref


def _proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def compute_psi(reference: dict, current: pd.DataFrame) -> pd.DataFrame:
    """Population Stability Index per feature. Rule of thumb: <0.10 stable, 0.10-0.25 moderate,
    >0.25 significant. Indicative only - PSI is noisy for small batches."""
    current = dp.standardize_columns(current)
    rows = []
    for c, r in reference.items():
        if c not in current:
            continue
        x = pd.to_numeric(current[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        e = np.clip(np.array(r["expected"]), _EPS, None)
        a = np.clip(_proportions(x, np.array(r["edges"])), _EPS, None)
        psi = float(np.sum((a - e) * np.log(a / e)))
        status = "ALERT" if psi >= config.PSI_ALERT else "WARN" if psi >= config.PSI_WARN else "OK"
        rows.append({"feature": c, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_shift(df: pd.DataFrame, bill_multiplier: float = 1.35, limit_multiplier: float = 0.8,
                   extra_delay: int = 1, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """SYNTHETIC stress scenario for demonstrating drift detection (not real data)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in config.BILL_COLS:
        out[c] = out[c] * bill_multiplier
    out["limit_bal"] = (out["limit_bal"] * limit_multiplier).round()
    hit = rng.random(len(out)) < 0.25  # a quarter of accounts show an extra month of delay
    out.loc[hit, "pay_1"] = (out.loc[hit, "pay_1"] + extra_delay).clip(upper=9)
    return out


# ------------------------------------------------------------------ performance monitoring
def performance_report(y_true, proba, threshold: float, baseline: dict | None = None) -> dict:
    """Live metrics once ground-truth labels are available, vs. the held-out baseline."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes in the labelled batch to compute ROC/PR-AUC.")
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    report = {
        "n": int(len(y)),
        "observed_default_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "recall": float(tp / max(int((y == 1).sum()), 1)),
        "precision": float(tp / max(int(pred.sum()), 1)),
    }
    if baseline:
        for k in ("roc_auc", "pr_auc", "recall", "precision"):
            if k in baseline:
                report[f"{k}_delta_vs_baseline"] = report[k] - float(baseline[k])
    return report
