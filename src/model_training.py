"""Model specs, tuning, probability calibration and the persisted ModelBundle (Phase 3)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config
from src.feature_engineering import build_pipeline

logger = logging.getLogger(__name__)

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


def get_model_specs(y_train: pd.Series, seed: int, fast: bool = False) -> dict[str, dict[str, Any]]:
    """Estimators + small hyperparameter spaces. Class imbalance is handled per model:
    class_weight for LR/RF, scale_pos_weight (neg/pos from TRAIN only) for XGBoost."""
    pos = float((y_train == 1).sum()); neg = float((y_train == 0).sum())
    n_rf = [60] if fast else [200, 300]
    n_xgb = [60] if fast else [150, 300]
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
            "params": {"C": [0.01, 0.1, 1.0, 10.0]},
            "scale": True, "n_iter": 4,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
            "params": {"n_estimators": n_rf, "max_depth": [6, 8, 12], "min_samples_leaf": [10, 20, 50],
                       "max_features": ["sqrt", 0.5]},
            "scale": False, "n_iter": 5,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=-1,
                scale_pos_weight=neg / pos, random_state=seed,
            ),
            "params": {"n_estimators": n_xgb, "max_depth": [3, 4, 5], "learning_rate": [0.03, 0.05, 0.1],
                       "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 5]},
            "scale": False, "n_iter": 8,
        },
    }


def tune_model(name: str, spec: dict[str, Any], X: pd.DataFrame, y: pd.Series, seed: int, fast: bool = False) -> tuple[Pipeline, dict]:  # noqa: N803
    """Randomised CV search scored by PR-AUC on the TRAINING split only; refits on all of train."""
    pipe = build_pipeline(spec["estimator"], spec["scale"])
    grid_size = int(np.prod([len(v) for v in spec["params"].values()]))
    n_iter = min(2 if fast else spec["n_iter"], grid_size)
    search = RandomizedSearchCV(
        pipe, {f"model__{k}": v for k, v in spec["params"].items()}, n_iter=n_iter,
        scoring="average_precision", cv=StratifiedKFold(2 if fast else config.CV_FOLDS, shuffle=True, random_state=seed),
        random_state=seed, refit=True, n_jobs=1, error_score="raise",
    )
    search.fit(X, y)
    best = search.cv_results_
    i = search.best_index_
    info = {
        "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "cv_pr_auc_mean": float(best["mean_test_score"][i]),
        "cv_pr_auc_std": float(best["std_test_score"][i]),
        "n_candidates": n_iter,
    }
    logger.info("%s tuned: %s", name, info)
    return search.best_estimator_, info


class PlattCalibrator:
    """Sigmoid (Platt) recalibration of a model's score, fit on VALIDATION predictions only.

    Class weighting inflates raw probabilities; recalibrating on a held-out split makes the
    displayed probabilities readable as frequencies. It is monotone, so ranking metrics
    (ROC-AUC, PR-AUC) and SHAP orderings are unchanged.
    """

    def __init__(self) -> None:
        self.lr: LogisticRegression | None = None

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_proba, y) -> "PlattCalibrator":
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw_proba), np.asarray(y))
        return self

    def transform(self, raw_proba) -> np.ndarray:
        if self.lr is None:
            raise RuntimeError("Calibrator is not fitted.")
        return self.lr.predict_proba(self._logit(raw_proba))[:, 1]


@dataclass
class ModelBundle:
    """Everything needed for inference + audit, persisted as ONE joblib file."""

    pipeline: Pipeline
    calibrator: PlattCalibrator
    model_name: str
    version: str
    trained_at: str
    threshold: float
    threshold_method: str
    input_columns: list[str]
    transformed_feature_names: list[str]
    reference: dict = field(default_factory=dict)        # train distributions for drift (PSI)
    shap_background: pd.DataFrame | None = None           # small transformed sample (linear SHAP)
    train_medians: dict = field(default_factory=dict)     # shown in the form as explicit defaults
    seed: int = config.RANDOM_SEED
    dataset_sha256: str | None = None
    metrics: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Calibrated default probability in [0, 1]."""
        return np.clip(self.calibrator.transform(self.predict_raw(X)), 0.0, 1.0)

    def save(self, path=config.MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)


def load_bundle(path=config.MODEL_PATH) -> ModelBundle:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}. Run `python train.py` first.")
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle.")
    return bundle
