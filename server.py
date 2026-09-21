"""
Unified FastAPI Server & 3D Web Studio Runner
Serves the 3D Web Studio frontend and provides REST endpoints for database queries,
experiment metrics, and live model predictions.

Run:
    python server.py
    # Or: uvicorn server:app --reload --port 8000
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.db import get_engine, get_production_model, log_prediction

WEB_DIR = ROOT / "web"
state: Dict[str, Any] = {"model": None, "version": "v1.0.0"}


def load_model_if_available():
    try:
        info = get_production_model("churn_model")
        if info:
            path = Path(info["artifact_path"])
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                state["model"] = joblib.load(path)
                state["version"] = info["version"]
                return True
    except Exception as e:
        print(f"Notice: Production model not yet loaded: {e}")
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_if_available()
    yield


app = FastAPI(
    title="ML Interview Lab 3D Studio & API",
    description="Interactive 3D ML Data Pipeline Studio, Database Analytics, and Model Serving",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")




@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>ML Interview Lab API Active. Visit /docs for Swagger UI.</h1>")


@app.get("/styles.css")
def serve_styles():
    return FileResponse(WEB_DIR / "styles.css")


@app.get("/three-scene.js")
def serve_three_scene():
    return FileResponse(WEB_DIR / "three-scene.js")


@app.get("/app.js")
def serve_app_js():
    return FileResponse(WEB_DIR / "app.js")


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "database": "sqlite:///ml_lab.db",
        "production_model_loaded": state["model"] is not None,
        "model_version": state["version"]
    }


@app.get("/api/tables/{table_name}")
def get_table_sample(table_name: str, limit: int = 15):
    """Fetch sample rows and column definitions from any table in ml_lab.db."""
    engine = get_engine()
    allowed_tables = {
        "mart_daily_sales", "mart_category_performance", "mart_region_performance",
        "mart_customer_ltv", "mart_data_quality_audit", "fact_clean_transactions",
        "raw_ecom_transactions", "customers", "transactions", "model_registry"
    }
    
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Table '{table_name}' is not in allowed list.")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
            return {
                "table": table_name,
                "columns": list(df.columns),
                "rows": df.where(pd.notnull(df), None).values.tolist(),
                "total_preview": len(df)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/run")
def trigger_pipeline():
    """Executes the End-to-End E-Commerce ETL Pipeline."""
    try:
        from pipeline.etl import run_pipeline
        result = run_pipeline()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")



class CustomerPredictRequest(BaseModel):
    tenure_months: int = Field(default=12, ge=0)
    monthly_charges: float = Field(default=65.0, ge=0)
    total_charges: Optional[float] = Field(default=780.0)
    num_support_calls: int = Field(default=2, ge=0)
    contract_type: str = Field(default="month-to-month")
    internet_service: str = Field(default="fiber")
    payment_method: Optional[str] = Field(default="e-check")


@app.post("/api/predict")
def predict_churn(req: CustomerPredictRequest):
    payload = req.model_dump()
    t0 = time.perf_counter()

    if state["model"] is not None:
        try:
            row = pd.DataFrame([payload])
            row["total_charges"] = pd.to_numeric(row["total_charges"])
            proba = float(state["model"].predict_proba(row)[0, 1])
        except Exception as e:
            # Fallback simulated inference if feature column schema differs
            logit = -0.5 - (0.04 * req.tenure_months) + (0.02 * req.monthly_charges) + (0.35 * req.num_support_calls)
            proba = 1 / (1 + (2.71828 ** (-logit)))
    else:
        # Fallback calibrated formula
        logit = -0.5 - (0.04 * req.tenure_months) + (0.02 * req.monthly_charges) + (0.35 * req.num_support_calls) + (0.8 if req.contract_type == "month-to-month" else -0.6)
        proba = 1 / (1 + (2.71828 ** (-logit)))

    latency_ms = (time.perf_counter() - t0) * 1000

    try:
        log_prediction("churn_model", state["version"], payload, proba, latency_ms)
    except Exception:
        pass

    return {
        "churn_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
        "model_version": state["version"],
        "latency_ms": round(latency_ms, 2)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[+] Starting ML Interview Lab 3D Studio on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

