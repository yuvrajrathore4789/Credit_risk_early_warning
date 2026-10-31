"""Rule-based early-warning layer on top of model probabilities (Phase 6).

Bands and the review threshold are ILLUSTRATIVE and configurable. A flag means "worth an
analyst's second look", never an automatic approve/reject decision.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

import config


def validate_bands(edges: Sequence[float], labels: Sequence[str]) -> None:
    if len(labels) != len(edges) + 1:
        raise ValueError("Need exactly one more band label than band edges.")
    if any(not 0 < e < 1 for e in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("Band edges must be strictly increasing values between 0 and 1 (exclusive).")


def assign_risk_band(
    proba, edges: Sequence[float] = config.RISK_BAND_EDGES, labels: Sequence[str] = config.RISK_BAND_LABELS
) -> pd.Series:
    """Map probabilities to ordered band labels (lower edge inclusive)."""
    validate_bands(edges, labels)
    p = pd.Series(np.asarray(proba, dtype=float))
    if p.isna().any() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    bins = [-np.inf, *edges, np.inf]
    return pd.cut(p, bins=bins, labels=list(labels), right=False, ordered=True)


def build_alerts(
    proba, threshold: float, ids=None, edges: Sequence[float] = config.RISK_BAND_EDGES,
    labels: Sequence[str] = config.RISK_BAND_LABELS,
) -> pd.DataFrame:
    """DataFrame with probability, review flag and band (+ customer_id when supplied)."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1].")
    p = np.asarray(proba, dtype=float)
    out = pd.DataFrame({
        "probability": p,
        "flagged": p >= threshold,
        "risk_band": assign_risk_band(p, edges, labels).astype(str).to_numpy(),
    })
    if ids is not None:
        out.insert(0, "customer_id", np.asarray(ids))
    return out


def band_summary(alerts: pd.DataFrame, labels: Sequence[str] = config.RISK_BAND_LABELS) -> pd.DataFrame:
    """Counts and proportions per band, in band order (empty bands shown as 0)."""
    counts = alerts["risk_band"].value_counts().reindex(list(labels), fill_value=0)
    total = max(int(counts.sum()), 1)
    return pd.DataFrame({"risk_band": counts.index, "count": counts.to_numpy(), "share": counts.to_numpy() / total})
