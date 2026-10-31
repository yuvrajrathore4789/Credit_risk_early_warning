"""Batch inference from the command line.

    python predict.py --input outputs/sample_input.csv --output outputs/predictions.csv
    python predict.py --input my_borrowers.csv --threshold 0.30 --log-to-db

Input CSV must contain the model's input columns (see README). Missing values are rejected
unless --allow-missing is given (then training medians/modes are used, and this is reported).
Output probabilities are ESTIMATES of risk, not decisions.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import config
from src import database as db
from src import early_warning as ew
from src import monitoring
from src.data_processing import DataValidationError, load_raw
from src.inference import InputValidationError, predict_dataframe
from src.model_training import load_bundle

logger = logging.getLogger("predict")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score borrower records with the trained model.")
    p.add_argument("--input", type=Path, required=True, help="CSV/XLSX with borrower records.")
    p.add_argument("--output", type=Path, default=config.OUTPUTS_DIR / "predictions.csv")
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    p.add_argument("--threshold", type=float, default=None, help="Review threshold (default: validation-selected).")
    p.add_argument("--allow-missing", action="store_true", help="Impute missing inputs with training medians/modes.")
    p.add_argument("--log-to-db", action="store_true", help="Write predictions to the audit table.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    monitoring.setup_logging()
    args = parse_args(argv)
    try:
        bundle = load_bundle(args.model_path)
        raw = load_raw(args.input)
        alerts = predict_dataframe(bundle, raw, threshold=args.threshold, allow_missing=args.allow_missing)
    except (FileNotFoundError, InputValidationError, DataValidationError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    thr = bundle.threshold if args.threshold is None else args.threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(args.output, index=False)
    print(f"Model {bundle.model_name} {bundle.version} | review threshold {thr:.2f} | {len(alerts)} rows")
    print(ew.band_summary(alerts).to_string(index=False))
    print(f"Flagged for review: {int(alerts['flagged'].sum())}  ->  {args.output}")
    print(config.DISCLAIMER)

    if args.log_to_db:
        engine = db.get_engine()
        db.init_db(engine)
        db.ensure_model_run(engine, bundle)
        n = db.insert_predictions(engine, alerts, bundle.version, "uploaded_batch", thr)
        logger.info("Logged %d predictions to %s", n, config.DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
