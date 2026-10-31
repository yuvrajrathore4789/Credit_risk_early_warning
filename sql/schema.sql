-- Schema for the Credit Risk Early Warning System (SQLite).
-- Statements are separated by semicolons and executed idempotently (IF NOT EXISTS).

-- raw_credit_clients: records exactly as loaded from the source file (column names
-- standardized, values NOT recoded). Kept so the cleaning step is auditable.
CREATE TABLE IF NOT EXISTS raw_credit_clients (
    customer_id        INTEGER PRIMARY KEY,
    limit_bal          REAL,
    sex                INTEGER,
    education          INTEGER,
    marriage           INTEGER,
    age                INTEGER,
    pay_1 INTEGER, pay_2 INTEGER, pay_3 INTEGER, pay_4 INTEGER, pay_5 INTEGER, pay_6 INTEGER,
    bill_amt1 REAL, bill_amt2 REAL, bill_amt3 REAL, bill_amt4 REAL, bill_amt5 REAL, bill_amt6 REAL,
    pay_amt1 REAL, pay_amt2 REAL, pay_amt3 REAL, pay_amt4 REAL, pay_amt5 REAL, pay_amt6 REAL,
    default_next_month INTEGER,
    loaded_at          TEXT NOT NULL
);

-- clean_credit_clients: validated, recoded records plus the train/val/test assignment.
CREATE TABLE IF NOT EXISTS clean_credit_clients (
    customer_id        INTEGER PRIMARY KEY,
    limit_bal          REAL NOT NULL CHECK (limit_bal > 0),
    sex                INTEGER NOT NULL,
    education          INTEGER NOT NULL CHECK (education BETWEEN 1 AND 4),
    marriage           INTEGER NOT NULL CHECK (marriage BETWEEN 1 AND 3),
    age                INTEGER NOT NULL CHECK (age BETWEEN 18 AND 100),
    pay_1 INTEGER, pay_2 INTEGER, pay_3 INTEGER, pay_4 INTEGER, pay_5 INTEGER, pay_6 INTEGER,
    bill_amt1 REAL, bill_amt2 REAL, bill_amt3 REAL, bill_amt4 REAL, bill_amt5 REAL, bill_amt6 REAL,
    pay_amt1 REAL, pay_amt2 REAL, pay_amt3 REAL, pay_amt4 REAL, pay_amt5 REAL, pay_amt6 REAL,
    default_next_month INTEGER NOT NULL CHECK (default_next_month IN (0, 1)),
    split              TEXT CHECK (split IN ('train', 'val', 'test')),
    cleaned_at         TEXT NOT NULL
);

-- model_runs: one row per training run (metadata, configuration and evaluation metrics).
CREATE TABLE IF NOT EXISTS model_runs (
    model_version      TEXT PRIMARY KEY,
    trained_at         TEXT NOT NULL,
    selected_model     TEXT NOT NULL,
    random_seed        INTEGER NOT NULL,
    dataset_sha256     TEXT,
    n_train            INTEGER,
    n_val              INTEGER,
    n_test             INTEGER,
    threshold          REAL,
    threshold_method   TEXT,
    config_json        TEXT,
    metrics_json       TEXT,
    feature_names_json TEXT
);

-- predictions: audit log of every scored record (batch, held-out, form or synthetic demo).
CREATE TABLE IF NOT EXISTS predictions (
    pred_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version         TEXT NOT NULL REFERENCES model_runs (model_version),
    customer_id           INTEGER,
    source                TEXT NOT NULL CHECK (source IN
                              ('heldout_test', 'uploaded_batch', 'individual_form', 'synthetic_demo')),
    predicted_probability REAL NOT NULL CHECK (predicted_probability BETWEEN 0 AND 1),
    threshold             REAL NOT NULL,
    flagged_for_review    INTEGER NOT NULL CHECK (flagged_for_review IN (0, 1)),
    risk_band             TEXT NOT NULL,
    input_json            TEXT,
    predicted_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_version ON predictions (model_version, source);
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON predictions (customer_id);

-- monitoring_log: drift (PSI), data-quality and performance observations over time.
CREATE TABLE IF NOT EXISTS monitoring_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    check_type    TEXT NOT NULL CHECK (check_type IN ('drift_psi', 'performance', 'data_quality')),
    name          TEXT NOT NULL,
    value         REAL,
    status        TEXT,
    source        TEXT,
    details_json  TEXT
);
