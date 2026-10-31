"""Metrics, threshold selection, calibration and evaluation plots (Phase 4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

import config  # noqa: E402

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """All metrics are computed from the supplied predictions; nothing is hardcoded."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "flagged_rate": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def classification_report_text(y_true, proba, threshold: float) -> str:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return classification_report(y_true, pred, target_names=["no default", "default"], zero_division=0)


def threshold_table(
    y_true, proba, thresholds=THRESHOLD_GRID,
    cost_fn: float = config.ILLUSTRATIVE_COST_FALSE_NEGATIVE,
    cost_fp: float = config.ILLUSTRATIVE_COST_FALSE_POSITIVE,
) -> pd.DataFrame:
    """Precision/recall/F1/flag-rate and illustrative cost per threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "threshold": float(t), "precision": prec, "recall": rec, "f1": f1,
            "flagged_rate": float(pred.mean()), "false_positives": fp, "false_negatives": fn,
            "avg_cost_per_account": (cost_fn * fn + cost_fp * fp) / len(y),
        })
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame, method: str = "min_cost") -> float:
    """Pick the threshold on VALIDATION data. ``min_cost`` minimises the illustrative
    FN/FP cost (recall-leaning, since a missed defaulter is assumed costlier than an extra
    review); ``max_f1`` is offered as a cost-free alternative."""
    if method == "min_cost":
        return float(table.loc[table["avg_cost_per_account"].idxmin(), "threshold"])
    if method == "max_f1":
        return float(table.loc[table["f1"].idxmax(), "threshold"])
    raise ValueError("method must be 'min_cost' or 'max_f1'")


def curve_points(y_true, proba, max_points: int = 400) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    def thin(*arrays):
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, min(max_points, len(arrays[0]))).astype(int))
        return [np.asarray(a)[idx].round(5).tolist() for a in arrays]

    fpr, tpr = thin(fpr, tpr)
    prec, rec = thin(prec, rec)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


# ------------------------------------------------------------------ plots (PNG, for README/outputs)
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc_pr(curves: dict[str, dict], base_rate: float, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, c in curves.items():
        ax[0].plot(c["fpr"], c["tpr"], label=name)
        ax[1].plot(c["recall"], c["precision"], label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax[1].axhline(base_rate, color="k", ls="--", lw=0.8, label=f"base rate {base_rate:.2f}")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test)")
    for a in ax:
        a.legend(frameon=False); a.grid(alpha=0.25)
    _save(fig, path)


def plot_confusion(metrics: dict, path: Path, title: str) -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
           yticklabels=["No default", "Default"], xlabel="Predicted", ylabel="Actual", title=title)
    _save(fig, path)


def plot_calibration(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for name, t in tables.items():
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Calibration (test)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    _save(fig, path)


def plot_threshold_analysis(table: pd.DataFrame, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for col in ["precision", "recall", "f1"]:
        ax.plot(table["threshold"], table[col], label=col)
    ax2 = ax.twinx()
    ax2.plot(table["threshold"], table["avg_cost_per_account"], color="grey", ls=":", label="illustrative cost")
    ax.axvline(chosen, color="red", ls="--", lw=1, label=f"selected {chosen:.2f}")
    ax.set(xlabel="Threshold", ylabel="Metric", title="Threshold analysis (validation)")
    ax2.set_ylabel("Avg cost / account")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right"); ax.grid(alpha=0.25)
    _save(fig, path)
