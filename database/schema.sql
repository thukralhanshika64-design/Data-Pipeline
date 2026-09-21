-- =====================================================================================
--  ML Interview Lab - database schema (SQLite dialect; tested with sqlite3)
--
--  You do NOT have to run this: common/db.py creates the platform tables automatically
--  (SQLAlchemy) and the data tables are written by pandas. This file is the readable
--  reference + something you can run by hand:
--        sqlite3 ml_lab.db < database/schema.sql
--
--  PostgreSQL porting notes (SQLAlchemy already handles this if you set DATABASE_URL):
--    INTEGER PRIMARY KEY            ->  SERIAL PRIMARY KEY  (or GENERATED ALWAYS AS IDENTITY)
--    DATETIME                       ->  TIMESTAMP
--    REAL                           ->  DOUBLE PRECISION
--    TEXT holding JSON              ->  JSONB   (enables features->>'monthly_charges' queries)
-- =====================================================================================

PRAGMA foreign_keys = ON;

-- ------------------------------ PLATFORM TABLES (experiment tracking) -----------------

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    n_rows      INTEGER,
    n_cols      INTEGER,
    created_at  DATETIME
);

CREATE TABLE IF NOT EXISTS experiments (           -- one row per training run
    id           INTEGER PRIMARY KEY,
    project      VARCHAR(100) NOT NULL,            -- e.g. p04_feature_factory
    run_name     VARCHAR(200),
    model_name   VARCHAR(100),
    dataset_name VARCHAR(100),
    params       TEXT,                             -- JSON string of hyper-parameters
    notes        TEXT,
    created_at   DATETIME
);

CREATE TABLE IF NOT EXISTS metrics (               -- tall/narrow: add new metrics without ALTER TABLE
    id            INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    split         VARCHAR(20) NOT NULL,            -- train | val | test | cv | sim
    metric_name   VARCHAR(50) NOT NULL,            -- roc_auc | rmse | f1 | ...
    metric_value  REAL
);
CREATE INDEX IF NOT EXISTS ix_metrics_experiment ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS ix_metrics_name_split ON metrics(metric_name, split);

CREATE TABLE IF NOT EXISTS model_registry (        -- versioned artifacts + lifecycle stage
    id            INTEGER PRIMARY KEY,
    model_name    VARCHAR(100) NOT NULL,
    version       INTEGER NOT NULL,
    artifact_path VARCHAR(500),
    experiment_id INTEGER REFERENCES experiments(id),
    stage         VARCHAR(20) DEFAULT 'staging',   -- staging | production | archived
    created_at    DATETIME,
    UNIQUE (model_name, version)
);

CREATE TABLE IF NOT EXISTS prediction_log (        -- online predictions, for monitoring
    id            INTEGER PRIMARY KEY,
    model_name    VARCHAR(100),
    model_version INTEGER,
    features      TEXT,                            -- JSON string of the request
    prediction    REAL,
    latency_ms    REAL,
    created_at    DATETIME
);
CREATE INDEX IF NOT EXISTS ix_predlog_model ON prediction_log(model_name, model_version);

-- ------------------------------ DATA TABLES (written by pandas) -----------------------
-- Reference DDL. pandas.to_sql(..., if_exists='replace') recreates these from the DataFrames.

CREATE TABLE IF NOT EXISTS customers (
    customer_id       TEXT PRIMARY KEY,
    signup_date       DATETIME,
    tenure_months     INTEGER,
    contract_type     TEXT,                        -- month-to-month | one-year | two-year
    internet_service  TEXT,                        -- fiber | dsl | none
    payment_method    TEXT,                        -- NULL for ~2% (missing-value practice)
    city              TEXT,                        -- 30 levels (high cardinality practice)
    monthly_charges   REAL,
    total_charges     REAL,                        -- NULL for ~3%
    num_support_calls INTEGER,
    churned           INTEGER                      -- target: 1 = churned
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id                TEXT PRIMARY KEY,
    timestamp             DATETIME,
    amount                REAL,
    hour                  INTEGER,
    merchant_category     TEXT,
    is_foreign            INTEGER,
    distance_from_home_km REAL,
    card_age_days         INTEGER,
    txn_count_last_24h    INTEGER,
    is_fraud              INTEGER                  -- target: ~1% positive
);

CREATE TABLE IF NOT EXISTS houses (
    house_id          INTEGER PRIMARY KEY,
    sqft              REAL,
    bedrooms          INTEGER,
    bathrooms         INTEGER,
    age_years         INTEGER,
    dist_to_center_km REAL,
    has_garage        INTEGER,
    neighborhood      TEXT,
    price             REAL                         -- target (regression)
);

-- Derived tables that the projects create for you:
--   customer_segments(customer_id, segment)          <- project 1  (K-Means result)
--   customer_features(customer_id, <25 features>, churned)   <- project 4  (mini feature store)
--   house_predictions(<features>, actual, predicted) <- project 5
