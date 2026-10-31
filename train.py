"""End-to-end training pipeline.

    python train.py                 # full run (a few minutes on CPU)
    python train.py --fast          # tiny search, for smoke tests
    python train.py --data-path path/to/UCI_Credit_Card.csv

Order of operations (no leakage): load/validate -> clean -> stratified split -> fit everything
on TRAIN -> tune/calibrate/choose threshold on VALIDATION -> report TEST once.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import data_processing as dp
from src import database as db
from src import eda, evaluation as ev, explainability as xai, monitoring
from src.early_warning import build_alerts
from src.model_training import MODEL_NAMES, ModelBundle, PlattCalibrator, get_model_specs, tune_model

logger = logging.getLogger("train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the credit-risk models.")
    p.add_argument("--data-path", type=Path, default=None, help="Dataset file (default: auto-detect in data/).")
    p.add_argument("--db-path", type=Path, default=config.DB_PATH)
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--fast", action="store_true", help="Tiny hyperparameter search (smoke test).")
    p.add_argument("--threshold-method", choices=["min_cost", "max_f1"], default="min_cost")
    p.add_argument("--skip-plots", action="store_true", help="Skip PNG figures (EDA data files still written).")
    return p.parse_args(argv)


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def run(args: argparse.Namespace) -> dict:
    seed = args.seed
    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- Phase 1: data
    data_path = args.data_path or dp.find_dataset()
    logger.info("Using dataset %s", data_path)
    raw = dp.load_raw(data_path)
    clean, report = dp.validate_and_clean(raw)
    train, val, test = dp.split_data(clean, seed)
    logger.info("Split sizes train/val/test = %d/%d/%d", len(train), len(val), len(test))
    summary = eda.write_eda_artifacts(clean, report, seed=seed)

    # ---------------------------------------------------------------- Phase 2: database
    engine = db.get_engine(args.db_path)
    db.init_db(engine)
    raw_stats = db.load_raw_records(engine, dp.prepare_raw(raw))
    labelled = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    clean_stats = db.load_clean_records(engine, labelled, replace=True)  # splits may change with seed

    # ---------------------------------------------------------------- Phase 3: models
    cols = config.model_input_columns()
    X = {k: d[[config.ID_COLUMN, *cols]] for k, d in {"train": train, "val": val, "test": test}.items()}
    y = {k: d[config.TARGET_COLUMN] for k, d in {"train": train, "val": val, "test": test}.items()}

    specs = get_model_specs(y["train"], seed, fast=args.fast)
    fitted, cv_info, cals, val_raw, test_raw = {}, {}, {}, {}, {}
    for name in MODEL_NAMES:
        logger.info("Tuning %s ...", name)
        fitted[name], cv_info[name] = tune_model(name, specs[name], X["train"], y["train"], seed, fast=args.fast)
        val_raw[name] = fitted[name].predict_proba(X["val"])[:, 1]
        cals[name] = PlattCalibrator().fit(val_raw[name], y["val"])      # validation only
        test_raw[name] = fitted[name].predict_proba(X["test"])[:, 1]

    # ---------------------------------------------------------------- Phase 4: evaluation
    results: dict[str, dict] = {}
    curves, cal_rows, thr_tables, thresholds = {}, [], [], {}
    for name in MODEL_NAMES:
        pv, pt = cals[name].transform(val_raw[name]), cals[name].transform(test_raw[name])
        table = ev.threshold_table(y["val"], pv)
        thr = ev.select_threshold(table, args.threshold_method)       # chosen on validation
        thresholds[name] = thr
        thr_tables.append(table.assign(model=name))
        results[name] = {
            "cv": cv_info[name],
            "validation": {"at_0.5": ev.compute_metrics(y["val"], pv, 0.5), "at_selected": ev.compute_metrics(y["val"], pv, thr)},
            "test": {"at_0.5": ev.compute_metrics(y["test"], pt, 0.5), "at_selected": ev.compute_metrics(y["test"], pt, thr)},
            "selected_threshold": thr,
            "test_raw_brier": ev.compute_metrics(y["test"], test_raw[name], 0.5)["brier"],
        }
        curves[name] = ev.curve_points(y["test"], pt)
        cal_rows.append(ev.calibration_table(y["test"], pt).assign(model=name, kind="calibrated"))
        cal_rows.append(ev.calibration_table(y["test"], test_raw[name]).assign(model=name, kind="raw"))

    # model SELECTION uses validation PR-AUC only (test untouched until here)
    selected = max(MODEL_NAMES, key=lambda n: results[n]["validation"]["at_selected"]["pr_auc"])
    thr = thresholds[selected]
    logger.info("Selected model: %s (validation PR-AUC %.4f), threshold %.2f", selected,
                results[selected]["validation"]["at_selected"]["pr_auc"], thr)

    comparison = pd.DataFrame([{
        "model": n, "selected": n == selected,
        "cv_pr_auc": results[n]["cv"]["cv_pr_auc_mean"],
        "val_roc_auc": results[n]["validation"]["at_selected"]["roc_auc"],
        "val_pr_auc": results[n]["validation"]["at_selected"]["pr_auc"],
        "threshold": results[n]["selected_threshold"],
        **{f"test_{k}": results[n]["test"]["at_selected"][k]
           for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]},
    } for n in MODEL_NAMES])

    # ---------------------------------------------------------------- bundle
    version = f"{config.MODEL_VERSION_PREFIX}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    pipe = fitted[selected]
    prep_train = pipe[:-1].transform(X["train"])
    bg = prep_train.sample(min(200, len(prep_train)), random_state=seed)
    bundle = ModelBundle(
        pipeline=pipe, calibrator=cals[selected], model_name=selected, version=version,
        trained_at=db.utc_now(), threshold=thr, threshold_method=args.threshold_method,
        input_columns=cols, transformed_feature_names=list(prep_train.columns),
        reference=monitoring.build_reference(train), shap_background=bg,
        train_medians={c: float(train[c].median()) for c in cols}, seed=seed,
        dataset_sha256=dp.file_sha256(data_path),
        metrics={"validation": results[selected]["validation"], "test": results[selected]["test"]},
        best_params=cv_info[selected]["best_params"],
    )
    bundle.save(args.model_path)

    # ---------------------------------------------------------------- Phase 5: SHAP on the selected model
    shap_sample = X["test"].sample(min(2000, len(X["test"])), random_state=seed)
    sres = xai.compute_shap(bundle, shap_sample)
    gap = xai.additivity_gap(bundle, shap_sample, sres)
    logger.info("SHAP additivity gap (should be ~0): %.2e", gap)
    imp = xai.global_importance(sres)
    imp.to_csv(config.EDA_DIR.parent / "shap_global_importance.csv", index=False)

    # ---------------------------------------------------------------- Phase 6: early warning on held-out test
    p_test = bundle.predict_proba(X["test"])
    alerts = build_alerts(p_test, thr, ids=X["test"][config.ID_COLUMN])
    test_out = test[[config.ID_COLUMN, *config.RAW_FEATURE_COLS, config.TARGET_COLUMN]].copy()
    test_out["probability"], test_out["flagged"], test_out["risk_band"] = alerts["probability"], alerts["flagged"], alerts["risk_band"]
    test_out.to_csv(config.OUTPUTS_DIR / "test_predictions.csv", index=False)
    X["test"].head(20).to_csv(config.OUTPUTS_DIR / "sample_input.csv", index=False)  # real held-out rows

    drift = monitoring.compute_psi(bundle.reference, test)
    drift.to_csv(config.OUTPUTS_DIR / "drift_test_vs_train.csv", index=False)

    # ---------------------------------------------------------------- persist results
    meta = {
        "model_version": version, "trained_at": bundle.trained_at, "selected_model": selected,
        "random_seed": seed, "dataset_sha256": bundle.dataset_sha256,
        "n_train": len(train), "n_val": len(val), "n_test": len(test), "threshold": thr,
        "threshold_method": args.threshold_method,
        "config": {
            "fast_mode": args.fast, "cv_folds": config.CV_FOLDS, "val_size": config.VAL_SIZE, "test_size": config.TEST_SIZE,
            "cost_fn": config.ILLUSTRATIVE_COST_FALSE_NEGATIVE, "cost_fp": config.ILLUSTRATIVE_COST_FALSE_POSITIVE,
            "use_sensitive_features": config.USE_SENSITIVE_FEATURES, "risk_band_edges": config.RISK_BAND_EDGES,
            "risk_band_labels": config.RISK_BAND_LABELS, "best_params": bundle.best_params,
        },
        "metrics": results[selected]["test"], "feature_names": bundle.transformed_feature_names,
    }
    db.insert_model_run(engine, meta)
    db.replace_heldout_predictions(engine, version)
    db.insert_predictions(engine, alerts, version, "heldout_test", thr)
    db.log_monitoring(engine, version, "drift_psi", [
        {"name": r.feature, "value": r.psi, "status": r.status} for r in drift.itertuples()], "heldout_test_vs_train")

    out = config.OUTPUTS_DIR
    _save_json({**meta, "all_models": results, "classification_report_test": ev.classification_report_text(y["test"], p_test, thr),
                "data_cleaning_report": report, "db_load": {"raw": raw_stats, "clean": clean_stats},
                "shap_additivity_gap": gap, "shap_units": sres.units, "eda_dataset_note": summary["dataset_note"]},
               out / "results.json")
    _save_json(curves, out / "curves.json")
    comparison.to_csv(out / "model_comparison.csv", index=False)
    pd.concat(thr_tables).to_csv(out / "threshold_analysis.csv", index=False)
    pd.concat(cal_rows).to_csv(out / "calibration.csv", index=False)

    if not args.skip_plots:
        ev.plot_roc_pr(curves, float(y["test"].mean()), config.FIGURES_DIR / "roc_pr_curves.png")
        ev.plot_confusion(results[selected]["test"]["at_selected"], config.FIGURES_DIR / "confusion_matrix.png",
                          f"{selected} @ {thr:.2f} (test)")
        cal_plot = {n: t[["mean_predicted", "observed_rate"]] for (n, k), t in
                    pd.concat(cal_rows).groupby(["model", "kind"]) if k == "calibrated"}
        ev.plot_calibration(cal_plot, config.FIGURES_DIR / "calibration.png")
        ev.plot_threshold_analysis(thr_tables[MODEL_NAMES.index(selected)], thr, config.FIGURES_DIR / "threshold_analysis.png")
        xai.save_global_plots(sres, config.FIGURES_DIR)
        top = int(np.argmax(bundle.predict_proba(shap_sample)))
        xai.save_individual_plot(sres, top, config.FIGURES_DIR / "shap_individual_example.png")

    logger.info("Done. Model saved to %s (version %s)", args.model_path, version)
    print("\n" + comparison.round(4).to_string(index=False))
    print(f"\nSelected: {selected} | threshold {thr:.2f} ({args.threshold_method}) | version {version}")
    return meta


if __name__ == "__main__":
    monitoring.setup_logging()
    try:
        run(parse_args())
    except (FileNotFoundError, dp.DataValidationError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
