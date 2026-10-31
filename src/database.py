"""SQLite/SQLAlchemy data layer (Phase 2): schema, idempotent loads, analytics, audit logging.

All writes use parameterized SQLAlchemy statements (never string-formatted SQL).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, create_engine, event, inspect, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import config

logger = logging.getLogger(__name__)

SOURCES = {"heldout_test", "uploaded_batch", "individual_form", "synthetic_demo"}
EXPECTED_TABLES = {
    "raw_credit_clients": {"customer_id", "limit_bal", "default_next_month", "loaded_at"},
    "clean_credit_clients": {"customer_id", "default_next_month", "split", "cleaned_at"},
    "model_runs": {"model_version", "trained_at", "selected_model", "random_seed"},
    "predictions": {"pred_id", "model_version", "source", "predicted_probability", "predicted_at"},
    "monitoring_log": {"log_id", "model_version", "check_type", "name"},
}
_CHUNK = 2000


class SchemaError(RuntimeError):
    """Raised when the database schema does not match what the code expects."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ engine / schema
def get_engine(db_path: Path | str | None = None) -> Engine:
    """Create an engine; ``":memory:"`` is supported (used by tests)."""
    target = str(db_path if db_path is not None else config.DB_PATH)
    if target == ":memory:":
        from sqlalchemy.pool import StaticPool
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    else:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{target}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):  # noqa: ANN001
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine, schema_path: Path = config.SCHEMA_SQL) -> None:
    """Create tables if they do not exist, then validate the schema."""
    if not Path(schema_path).is_file():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    sql = Path(schema_path).read_text(encoding="utf-8")
    sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    with engine.begin() as conn:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                conn.exec_driver_sql(stmt)
    validate_schema(engine)


def validate_schema(engine: Engine) -> None:
    """Raise SchemaError if a table or required column is missing."""
    insp = inspect(engine)
    problems = []
    for table, required in EXPECTED_TABLES.items():
        if not insp.has_table(table):
            problems.append(f"missing table {table}")
            continue
        missing = required - {c["name"] for c in insp.get_columns(table)}
        if missing:
            problems.append(f"table {table} missing columns {sorted(missing)}")
    if problems:
        raise SchemaError("Database schema invalid: " + "; ".join(problems))


def _table(engine: Engine, name: str) -> Table:
    return Table(name, MetaData(), autoload_with=engine)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list of dicts with native Python types and None for NaN (sqlite-safe)."""
    out = []
    for row in df.to_dict("records"):
        out.append({k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()})
    return out


def _count(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())  # table name is a constant


# ------------------------------------------------------------------ loading
def _to_db_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={config.ID_COLUMN: "customer_id", config.TARGET_COLUMN: "default_next_month"})
    needed = ["customer_id", *config.RAW_FEATURE_COLS, "default_next_month"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise SchemaError(f"DataFrame is missing columns required for loading: {missing}")
    return out[needed + [c for c in ("split",) if c in out.columns]].copy()


def _load(engine: Engine, table: str, df: pd.DataFrame, stamp_col: str, replace: bool) -> dict[str, int]:
    """Idempotent chunked insert. Existing primary keys are skipped (or replaced if ``replace``)."""
    frame = _to_db_frame(df)
    frame[stamp_col] = utc_now()
    tbl = _table(engine, table)
    before = _count(engine, table)
    with engine.begin() as conn:
        for start in range(0, len(frame), _CHUNK):
            chunk = _records(frame.iloc[start:start + _CHUNK])
            stmt = sqlite_insert(tbl)
            if replace:
                update = {c: stmt.excluded[c] for c in frame.columns if c != "customer_id"}
                stmt = stmt.on_conflict_do_update(index_elements=["customer_id"], set_=update)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=["customer_id"])
            conn.execute(stmt, chunk)
    after = _count(engine, table)
    inserted = after - before
    result = {"submitted": len(frame), "inserted": inserted,
              ("updated_existing" if replace else "skipped_existing"): len(frame) - inserted}
    logger.info("Loaded %s: %s", table, result)
    return result


def load_raw_records(engine: Engine, df: pd.DataFrame, replace: bool = False) -> dict[str, int]:
    """Load standardized-but-unrecoded source rows into raw_credit_clients."""
    return _load(engine, "raw_credit_clients", df, "loaded_at", replace)


def load_clean_records(engine: Engine, df: pd.DataFrame, replace: bool = False) -> dict[str, int]:
    """Load cleaned rows (must contain a ``split`` column) into clean_credit_clients."""
    if "split" not in df.columns:
        raise SchemaError("clean records must carry a 'split' column (train/val/test).")
    if not set(df["split"].unique()) <= {"train", "val", "test"}:
        raise SchemaError("split must be one of train/val/test.")
    return _load(engine, "clean_credit_clients", df, "cleaned_at", replace)


# ------------------------------------------------------------------ model runs / predictions / monitoring
def insert_model_run(engine: Engine, meta: dict[str, Any]) -> None:
    row = {
        "model_version": meta["model_version"], "trained_at": meta["trained_at"],
        "selected_model": meta["selected_model"], "random_seed": meta["random_seed"],
        "dataset_sha256": meta.get("dataset_sha256"), "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"), "n_test": meta.get("n_test"), "threshold": meta.get("threshold"),
        "threshold_method": meta.get("threshold_method"),
        "config_json": json.dumps(meta.get("config", {}), default=str),
        "metrics_json": json.dumps(meta.get("metrics", {}), default=str),
        "feature_names_json": json.dumps(meta.get("feature_names", [])),
    }
    stmt = sqlite_insert(_table(engine, "model_runs"))
    stmt = stmt.on_conflict_do_update(index_elements=["model_version"], set_={k: stmt.excluded[k] for k in row if k != "model_version"})
    with engine.begin() as conn:
        conn.execute(stmt, [row])


def ensure_model_run(engine: Engine, bundle: Any) -> None:
    """Make sure the bundle's version exists in model_runs (needed by the predictions FK),
    e.g. when the database was recreated after the model was trained. Never overwrites."""
    stmt = sqlite_insert(_table(engine, "model_runs")).on_conflict_do_nothing(index_elements=["model_version"])
    row = {
        "model_version": bundle.version, "trained_at": bundle.trained_at, "selected_model": bundle.model_name,
        "random_seed": bundle.seed, "dataset_sha256": bundle.dataset_sha256, "threshold": bundle.threshold,
        "threshold_method": bundle.threshold_method, "metrics_json": json.dumps(bundle.metrics, default=str),
        "feature_names_json": json.dumps(bundle.transformed_feature_names),
        "config_json": json.dumps({"best_params": bundle.best_params, "restored_from_bundle": True}, default=str),
    }
    with engine.begin() as conn:
        conn.execute(stmt, [row])


def insert_predictions(
    engine: Engine, alerts: pd.DataFrame, model_version: str, source: str, threshold: float,
    inputs: list[dict] | None = None,
) -> int:
    """Log predictions. ``alerts`` needs: probability, flagged, risk_band (+ optional customer_id)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}")
    if alerts.empty:
        return 0
    now = utc_now()
    rows = []
    for i, r in enumerate(alerts.reset_index(drop=True).itertuples(index=False)):
        cid = getattr(r, "customer_id", None)
        rows.append({
            "model_version": model_version,
            "customer_id": None if cid is None or pd.isna(cid) else int(cid),
            "source": source,
            "predicted_probability": float(r.probability),
            "threshold": float(threshold),
            "flagged_for_review": int(bool(r.flagged)),
            "risk_band": str(r.risk_band),
            "input_json": json.dumps(inputs[i], default=str) if inputs else None,
            "predicted_at": now,
        })
    tbl = _table(engine, "predictions")
    with engine.begin() as conn:
        for start in range(0, len(rows), _CHUNK):
            conn.execute(tbl.insert(), rows[start:start + _CHUNK])
    return len(rows)


def replace_heldout_predictions(engine: Engine, model_version: str) -> None:
    """Make re-running training idempotent: drop earlier held-out rows for this version."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions WHERE model_version = :v AND source = 'heldout_test'"), {"v": model_version})


def log_monitoring(engine: Engine, model_version: str, check_type: str, rows: list[dict], source: str) -> None:
    now = utc_now()
    payload = [{
        "model_version": model_version, "created_at": now, "check_type": check_type,
        "name": r["name"], "value": None if r.get("value") is None else float(r["value"]),
        "status": r.get("status"), "source": source,
        "details_json": json.dumps(r.get("details", {}), default=str),
    } for r in rows]
    if payload:
        with engine.begin() as conn:
            conn.execute(_table(engine, "monitoring_log").insert(), payload)


# ------------------------------------------------------------------ analytics queries
def load_queries(path: Path = config.ANALYTICS_SQL) -> dict[str, str]:
    """Parse analytics.sql into {name: sql} using '-- name: x' markers."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Analytics SQL file not found: {path}")
    queries: dict[str, str] = {}
    name, buf = None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            if name:
                queries[name] = "\n".join(buf).strip().rstrip(";")
            name, buf = m.group(1), []
        elif name and not line.strip().startswith("--"):
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip().rstrip(";")
    return queries


def run_query(engine: Engine, name: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    queries = load_queries()
    if name not in queries:
        raise KeyError(f"Unknown query '{name}'. Available: {sorted(queries)}")
    with engine.connect() as conn:
        return pd.read_sql(text(queries[name]), conn, params=params or {})


def read_table(engine: Engine, table: str, limit: int = 500) -> pd.DataFrame:
    if table not in EXPECTED_TABLES:
        raise KeyError(f"Unknown table {table}")
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table} ORDER BY 1 DESC LIMIT :n"), conn, params={"n": int(limit)})
