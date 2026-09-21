"""
Shared database layer for every project.

Default backend : SQLite file  ->  <repo>/ml_lab.db   (zero setup)
Switch backend  : export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/mllab

Tables
------
Data tables (created by seed_all / get_dataset)
    customers, customers_nl, transactions, houses
Platform tables (created by init_db)
    datasets        - catalogue of datasets
    experiments     - one row per training run (project, model, params)
    metrics         - one row per (experiment, split, metric)  [tall format]
    model_registry  - versioned models with stage: staging / production / archived
    prediction_log  - online predictions for monitoring (latency, drift checks)

Run once (optional):   python -m common.db
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, MetaData,
                        String, Table, Text, create_engine, inspect, text)

from functools import partial

from common.data import make_customers, make_houses, make_transactions

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'ml_lab.db'}")


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


metadata = MetaData()

datasets = Table(
    "datasets", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100), unique=True, nullable=False),
    Column("description", Text),
    Column("n_rows", Integer),
    Column("n_cols", Integer),
    Column("created_at", DateTime, default=_now),
)

experiments = Table(
    "experiments", metadata,
    Column("id", Integer, primary_key=True),
    Column("project", String(100), nullable=False),
    Column("run_name", String(200)),
    Column("model_name", String(100)),
    Column("dataset_name", String(100)),
    Column("params", Text),          # JSON string
    Column("notes", Text),
    Column("created_at", DateTime, default=_now),
)

metrics = Table(
    "metrics", metadata,
    Column("id", Integer, primary_key=True),
    Column("experiment_id", Integer, ForeignKey("experiments.id"), nullable=False),
    Column("split", String(20), nullable=False),        # train / val / test / cv
    Column("metric_name", String(50), nullable=False),  # roc_auc, rmse, ...
    Column("metric_value", Float),
)

model_registry = Table(
    "model_registry", metadata,
    Column("id", Integer, primary_key=True),
    Column("model_name", String(100), nullable=False),
    Column("version", Integer, nullable=False),
    Column("artifact_path", String(500)),
    Column("experiment_id", Integer, ForeignKey("experiments.id")),
    Column("stage", String(20), default="staging"),
    Column("created_at", DateTime, default=_now),
)

prediction_log = Table(
    "prediction_log", metadata,
    Column("id", Integer, primary_key=True),
    Column("model_name", String(100)),
    Column("model_version", Integer),
    Column("features", Text),        # JSON string
    Column("prediction", Float),
    Column("latency_ms", Float),
    Column("created_at", DateTime, default=_now),
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, future=True)
    return _engine


def init_db():
    """Create platform tables if they do not exist."""
    metadata.create_all(get_engine())


# --------------------------------------------------------------------------- #
# Data tables
# --------------------------------------------------------------------------- #
_GENERATORS = {
    "customers": (make_customers, "Telco-style customers with churn label", ["signup_date"]),
    "customers_nl": (partial(make_customers, nonlinear=True, seed=43),
                     "Customers with NON-LINEAR churn rules (harder; used to make model choice a real trade-off)",
                     ["signup_date"]),
    "transactions": (make_transactions, "Card transactions, ~1% fraud", ["timestamp"]),
    "houses": (make_houses, "House sales with price target", []),
}


def table_exists(name: str) -> bool:
    return inspect(get_engine()).has_table(name)


def save_table(df: pd.DataFrame, name: str, if_exists: str = "replace"):
    df.to_sql(name, get_engine(), if_exists=if_exists, index=False)


def load_table(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_sql_table(name, get_engine(), parse_dates=parse_dates)
    df.columns = [str(c) for c in df.columns]   # SQLAlchemy returns quoted_name objects; sklearn wants str
    return df


def get_dataset(name: str, refresh: bool = False, **gen_kwargs) -> pd.DataFrame:
    """
    Read a dataset from the DB; generate + store it on first use.
    Pass refresh=True (and e.g. n=20000) to regenerate.
    """
    if name not in _GENERATORS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from {list(_GENERATORS)}")
    init_db()
    gen, desc, date_cols = _GENERATORS[name]
    if refresh or not table_exists(name):
        df = gen(**gen_kwargs)
        save_table(df, name)
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM datasets WHERE name = :n"), {"n": name})
            conn.execute(datasets.insert().values(
                name=name, description=desc, n_rows=len(df), n_cols=df.shape[1], created_at=_now()))
    return load_table(name, parse_dates=date_cols)


def seed_all():
    for name in _GENERATORS:
        df = get_dataset(name, refresh=True)
        print(f"  seeded {name:<13} rows={len(df):>6}  cols={df.shape[1]}")


# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #
def _py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def log_experiment(project: str, run_name: str, model_name: str, metrics_dict: dict,
                   params: dict | None = None, dataset_name: str | None = None,
                   notes: str | None = None, default_split: str = "test") -> int:
    """
    metrics_dict can be flat   {"roc_auc": 0.91}                -> stored under default_split
    or nested                  {"train": {"rmse": 1.0}, "cv": {"rmse": 1.4}}
    Returns the experiment id.
    """
    init_db()
    with get_engine().begin() as conn:
        res = conn.execute(experiments.insert().values(
            project=project, run_name=run_name, model_name=model_name,
            dataset_name=dataset_name, params=json.dumps(params or {}, default=str),
            notes=notes, created_at=_now()))
        exp_id = res.inserted_primary_key[0]
        rows = []
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                rows += [dict(experiment_id=exp_id, split=k, metric_name=mk, metric_value=float(_py(mv)))
                         for mk, mv in v.items()]
            else:
                rows.append(dict(experiment_id=exp_id, split=default_split, metric_name=k,
                                 metric_value=float(_py(v))))
        if rows:
            conn.execute(metrics.insert(), rows)
    return exp_id


def leaderboard(metric: str = "roc_auc", split: str = "test", project: str | None = None,
                top: int = 10, ascending: bool = False) -> pd.DataFrame:
    order = "ASC" if ascending else "DESC"
    where = "AND e.project = :project" if project else ""
    sql = text(f"""
        SELECT e.id, e.project, e.run_name, e.model_name, m.split, m.metric_name, m.metric_value
        FROM experiments e JOIN metrics m ON m.experiment_id = e.id
        WHERE m.metric_name = :metric AND m.split = :split {where}
        ORDER BY m.metric_value {order} LIMIT :top""")
    params = {"metric": metric, "split": split, "top": top}
    if project:
        params["project"] = project
    return pd.read_sql(sql, get_engine(), params=params)


# --------------------------------------------------------------------------- #
# Model registry + prediction logging
# --------------------------------------------------------------------------- #
def next_model_version(model_name: str) -> int:
    """Version number the next register_model() call for this model will receive."""
    init_db()
    with get_engine().connect() as conn:
        last = conn.execute(text("SELECT COALESCE(MAX(version), 0) FROM model_registry WHERE model_name=:n"),
                            {"n": model_name}).scalar()
    return int(last) + 1


def register_model(model_name: str, artifact_path: str, experiment_id: int | None = None,
                   stage: str = "staging") -> int:
    """Insert a new version. Promoting to 'production' archives the previous production version."""
    init_db()
    with get_engine().begin() as conn:
        last = conn.execute(text("SELECT COALESCE(MAX(version), 0) FROM model_registry WHERE model_name=:n"),
                            {"n": model_name}).scalar()
        version = int(last) + 1
        if stage == "production":
            conn.execute(text("UPDATE model_registry SET stage='archived' "
                              "WHERE model_name=:n AND stage='production'"), {"n": model_name})
        conn.execute(model_registry.insert().values(
            model_name=model_name, version=version, artifact_path=str(artifact_path),
            experiment_id=experiment_id, stage=stage, created_at=_now()))
    return version


def get_production_model(model_name: str) -> dict | None:
    init_db()
    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT model_name, version, artifact_path FROM model_registry "
            "WHERE model_name=:n AND stage='production' ORDER BY version DESC LIMIT 1"),
            {"n": model_name}).mappings().first()
    return dict(row) if row else None


def log_prediction(model_name: str, model_version: int, features: dict, prediction: float,
                   latency_ms: float):
    init_db()
    with get_engine().begin() as conn:
        conn.execute(prediction_log.insert().values(
            model_name=model_name, model_version=model_version,
            features=json.dumps(features, default=str), prediction=float(prediction),
            latency_ms=float(latency_ms), created_at=_now()))


if __name__ == "__main__":
    print(f"Database: {DATABASE_URL}")
    init_db()
    seed_all()
    print("Done. Platform tables:", [t for t in metadata.tables])
