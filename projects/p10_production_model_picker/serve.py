"""
FastAPI service for the model that main.py registered as PRODUCTION.

Start (from the repo root):
    uvicorn projects.p10_production_model_picker.serve:app --port 8000
Try it:
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
      "tenure_months": 3, "monthly_charges": 95.5, "total_charges": 280.0, "num_support_calls": 4,
      "contract_type": "month-to-month", "internet_service": "fiber", "payment_method": "e-check"}'
    open http://localhost:8000/docs   (interactive Swagger UI)

Every prediction is written to the `prediction_log` table (features, prediction, latency) -
the raw material for monitoring: latency percentiles, prediction drift, data drift.
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.db import get_production_model, log_prediction

MODEL_NAME = "churn_model"
state: dict = {}


def load_model():
    info = get_production_model(MODEL_NAME)
    if not info:
        raise RuntimeError("No production model in the registry. Run main.py first.")
    path = Path(info["artifact_path"])
    if not path.is_absolute():                 # registry stores paths relative to the repo root
        path = ROOT / path
    state["model"] = joblib.load(path)
    state["version"] = info["version"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Churn model service", lifespan=lifespan)


class Customer(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float | None = None
    num_support_calls: int = Field(ge=0)
    contract_type: str
    internet_service: str
    payment_method: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": state.get("version")}


@app.post("/predict")
def predict(c: Customer):
    payload = c.model_dump()
    row = pd.DataFrame([payload])
    row["total_charges"] = pd.to_numeric(row["total_charges"])       # None -> NaN (imputed by the pipeline)
    t0 = time.perf_counter()
    try:
        proba = float(state["model"].predict_proba(row)[0, 1])
    except Exception as e:                                           # bad category etc.
        raise HTTPException(status_code=422, detail=f"prediction failed: {e}")
    latency_ms = (time.perf_counter() - t0) * 1000
    log_prediction(MODEL_NAME, state["version"], payload, proba, latency_ms)
    return {"churn_probability": round(proba, 4), "model_version": state["version"],
            "latency_ms": round(latency_ms, 2)}


@app.post("/reload")
def reload_model():
    """Hot-swap after promoting a new version in the registry (no restart)."""
    load_model()
    return {"reloaded": True, "version": state["version"]}
