"""Exploratory analysis: summary artifacts (for the dashboard) and PNG figures (Phase 1).

Everything is descriptive. Differences in default rates between segments are observed
associations in this Taiwanese 2005 card portfolio and must not be read as causal.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.feature_engineering import add_features  # noqa: E402

T = config.TARGET_COLUMN
EDU = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MAR = {1: "Married", 2: "Single", 3: "Other"}
SEX = {1: "Male", 2: "Female"}


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: segment, level, n, default_rate."""
    d = add_features(df)
    d["education_lbl"] = d["education"].map(EDU)
    d["marriage_lbl"] = d["marriage"].map(MAR)
    d["sex_lbl"] = d["sex"].map(SEX)
    d["age_band"] = pd.cut(d["age"], [17, 24, 34, 44, 54, 120], labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    d["limit_band"] = pd.cut(d["limit_bal"], [0, 5e4, 1e5, 2e5, 4e5, np.inf],
                             labels=["<50k", "50-100k", "100-200k", "200-400k", "400k+"])
    d["latest_repayment_status"] = d["pay_1"].astype(int).astype(str)
    d["months_delayed_of_6"] = d["n_delayed_months"].astype(int).astype(str)
    specs = {"Education": "education_lbl", "Marital status": "marriage_lbl", "Sex (descriptive only)": "sex_lbl",
             "Age band": "age_band", "Credit limit (NT$)": "limit_band",
             "Latest repayment status (pay_1)": "latest_repayment_status",
             "Months delayed (of last 6)": "months_delayed_of_6"}
    parts = []
    for name, col in specs.items():
        g = d.groupby(col, observed=True)[T].agg(n="size", default_rate="mean").reset_index()
        g.columns = ["level", "n", "default_rate"]
        g.insert(0, "segment", name)
        g["level"] = g["level"].astype(str)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    # keep numeric order for the two numeric-coded segments
    out["_k"] = pd.to_numeric(out["level"], errors="coerce")
    return out.sort_values(["segment", "_k"], kind="stable").drop(columns="_k").reset_index(drop=True)


def write_eda_artifacts(df: pd.DataFrame, report: dict, out_dir: Path = config.EDA_DIR, seed: int = config.RANDOM_SEED) -> dict:
    """Write CSV/JSON used by the dashboard + PNG figures. Returns the summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    enriched = add_features(df)
    numeric = enriched.drop(columns=[config.ID_COLUMN]).select_dtypes("number")
    corr = numeric.corr()
    corr.to_csv(out_dir / "correlation.csv")
    corr_target = corr[T].drop(T).sort_values(key=np.abs, ascending=False)
    seg = segment_table(df)
    seg.to_csv(out_dir / "segment_default_rates.csv", index=False)
    desc = numeric.describe().T
    desc.to_csv(out_dir / "descriptive_stats.csv")

    # stratified sample used by the dashboard for histograms (full data not needed there)
    sample = enriched.groupby(T, group_keys=False).sample(frac=min(1.0, 6000 / len(df)), random_state=seed)
    sample.drop(columns=[config.ID_COLUMN]).to_csv(out_dir / "eda_sample.csv", index=False)

    summary = {
        "n_records": int(len(df)), "default_rate": float(df[T].mean()),
        "class_counts": report["class_counts"], "cleaning_report": report,
        "top_target_correlations": {k: float(v) for k, v in corr_target.head(10).items()},
        "dataset_note": ("UCI Default of Credit Card Clients: Taiwanese credit-card holders, April-September 2005. "
                         "Not representative of Indian SME lending; not Protium data."),
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figures
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    counts = df[T].value_counts().sort_index()
    ax.bar(["No default", "Default"], counts.values, color=["#4c78a8", "#e45756"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set(ylabel="Customers", title="Class distribution", ylim=(0, counts.max() * 1.2))
    fig.tight_layout(); fig.savefig(fig_dir / "eda_class_distribution.png", dpi=130); plt.close(fig)

    cols = ["limit_bal", "age", "bill_amt1", "pay_amt1", "utilization_latest", "payment_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        lo, hi = enriched[c].quantile([0.005, 0.995])
        for val, color, lbl in [(0, "#4c78a8", "No default"), (1, "#e45756", "Default")]:
            ax.hist(enriched.loc[enriched[T] == val, c].clip(lo, hi), bins=40, alpha=0.55, color=color, label=lbl, density=True)
        ax.set_title(c)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Feature distributions by outcome (clipped at 0.5/99.5 percentiles)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_feature_distributions.png", dpi=130); plt.close(fig)

    keep = ["limit_bal", "age", "pay_1", "pay_2", "bill_amt1", "pay_amt1", *[c for c in
            ["max_delay", "n_delayed_months", "utilization_latest", "payment_ratio"]], T]
    c = enriched[keep].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)), keep, rotation=60, ha="right"); ax.set_yticks(range(len(keep)), keep)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlation (Pearson)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_correlation.png", dpi=130); plt.close(fig)

    show = seg[seg["segment"].isin(["Education", "Age band", "Latest repayment status (pay_1)", "Credit limit (NT$)"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, g) in zip(axes.ravel(), show.groupby("segment", sort=False)):
        ax.bar(g["level"], g["default_rate"], color="#e45756")
        ax.axhline(df[T].mean(), color="k", ls="--", lw=0.8)
        ax.set_title(name); ax.set_ylabel("Observed default rate")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Observed default rate by segment (dashed = overall; associations, not causes)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_segment_default_rates.png", dpi=130); plt.close(fig)
    return summary
