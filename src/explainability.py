"""SHAP explanations for the model actually being evaluated (Phase 5).

* Tree models (XGBoost, Random Forest) -> ``shap.TreeExplainer`` (exact, fast).
* Logistic Regression -> ``shap.LinearExplainer`` with a training background sample.

Units differ by model and are documented in the return value: XGBoost/LogReg explain the
log-odds (margin) of the RAW model score; Random Forest explains the raw class-1 probability.
The Platt calibration step is a monotone map applied afterwards, so SHAP orderings and signs
carry over. SHAP describes how the MODEL uses features, not causal effects on default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.model_training import ModelBundle  # noqa: E402


@dataclass
class ShapResult:
    values: np.ndarray          # (n_rows, n_features) contributions
    base_value: float
    data: pd.DataFrame          # transformed features the model saw
    feature_names: list[str]
    units: str


def _positive_class_values(raw, n_features: int) -> np.ndarray:
    """Normalise SHAP output formats across versions/models to (n, n_features) for class 1."""
    if isinstance(raw, list):                       # old API: [class0, class1]
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3:                               # new API: (n, features, classes)
        return arr[:, :, 1] if arr.shape[2] == 2 else arr[:, :, -1]
    if arr.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP output shape {arr.shape}")
    return arr


def _base_value(explainer, model) -> float:
    ev = np.asarray(explainer.expected_value).ravel()
    return float(ev[1] if (isinstance(model, RandomForestClassifier) and ev.size == 2) else ev[-1])


def compute_shap(bundle: ModelBundle, X_raw: pd.DataFrame) -> ShapResult:  # noqa: N803
    pipe = bundle.pipeline
    data = pipe[:-1].transform(X_raw)
    model = pipe[-1]
    if isinstance(model, LogisticRegression):
        if bundle.shap_background is None:
            raise ValueError("Bundle has no SHAP background sample for the linear explainer.")
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(bundle.shap_background, max_samples=200))
        units = "log-odds of raw model score"
    else:
        explainer = shap.TreeExplainer(model)
        units = ("probability of raw model score" if isinstance(model, RandomForestClassifier)
                 else "log-odds of raw model score")
    values = _positive_class_values(explainer.shap_values(data), data.shape[1])
    return ShapResult(values, _base_value(explainer, model), data, list(data.columns), units)


def additivity_gap(bundle: ModelBundle, X_raw: pd.DataFrame, result: ShapResult) -> float:  # noqa: N803
    """Max |(base + sum SHAP) - model output| in the explained units (sanity check ~ 0)."""
    model = bundle.pipeline[-1]
    data = result.data
    if isinstance(model, RandomForestClassifier):
        out = model.predict_proba(data)[:, 1]
    else:
        out = model.decision_function(data) if isinstance(model, LogisticRegression) else (
            model.predict(data, output_margin=True))
    return float(np.max(np.abs(result.base_value + result.values.sum(axis=1) - out)))


def global_importance(result: ShapResult) -> pd.DataFrame:
    imp = np.abs(result.values).mean(axis=0)
    return (pd.DataFrame({"feature": result.feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def explain_row(bundle: ModelBundle, X_row: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:  # noqa: N803
    """Top contributions for ONE borrower: feature, value, SHAP contribution, direction."""
    if len(X_row) != 1:
        raise ValueError("explain_row expects exactly one row.")
    res = compute_shap(bundle, X_row)
    engineered = bundle.pipeline.named_steps["features"].transform(X_row).iloc[0]
    rows = []
    for name, shap_v in zip(res.feature_names, res.values[0]):
        shown = engineered[name] if name in engineered.index else res.data.iloc[0][name]
        rows.append({"feature": name, "value": float(shown), "contribution": float(shap_v)})
    df = pd.DataFrame(rows)
    df["direction"] = np.where(df["contribution"] >= 0, "raises estimated risk", "lowers estimated risk")
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


# ------------------------------------------------------------------ figures
def save_global_plots(result: ShapResult, out_dir: Path, top_k: int = 15) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imp = global_importance(result).head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp["feature"], imp["mean_abs_shap"], color="#3b6ea5")
    ax.set(xlabel=f"mean |SHAP| ({result.units})", title="Global feature importance (association, not causation)")
    fig.tight_layout(); fig.savefig(out_dir / "shap_global_importance.png", dpi=130); plt.close(fig)

    shap.summary_plot(result.values, result.data, show=False, max_display=top_k)
    plt.title("SHAP summary (red = high feature value)")
    plt.tight_layout(); plt.savefig(out_dir / "shap_summary.png", dpi=130); plt.close()


def save_individual_plot(result: ShapResult, idx: int, out_path: Path) -> None:
    exp = shap.Explanation(values=result.values[idx], base_values=result.base_value,
                           data=result.data.iloc[idx].to_numpy(), feature_names=result.feature_names)
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
