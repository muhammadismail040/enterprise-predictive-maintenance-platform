from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import DATABASE_URL, MLFLOW_TRACKING_URI
from src.data.synthetic_data import load_or_create_dataset
from src.mlops.mlflow_manager import MLflowManager
from src.models.predictive_maintenance import PredictiveMaintenanceEngine


class SensorSample(BaseModel):
    timestamp: Optional[str] = None
    temperature: float = Field(..., description="Temperature in degrees Celsius")
    pressure: float = Field(..., description="Pressure in psi")
    vibration: float = Field(..., description="Vibration in mm/s")
    rpm: float = Field(..., description="Rotations per minute")
    voltage: float = Field(..., description="Power voltage")
    current: float = Field(..., description="Electrical current")
    humidity: float = Field(..., description="Relative humidity (%)")
    load: float = Field(..., description="Machine load (%)")
    maintenance_history: float = Field(default=0.0)
    failure_log: int = Field(default=0)
    operating_hours: float = Field(default=0.0)


class PredictionRequest(BaseModel):
    machine_id: str = Field(..., description="Machine identifier")
    sensor_data: List[SensorSample] = Field(..., min_length=1)


app = FastAPI(
    title="Enterprise Predictive Maintenance Platform",
    version="1.0.0",
    description="AI-driven industrial equipment failure intelligence and maintenance planning API.",
)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DEFAULT_DATASET = load_or_create_dataset()
ML_ENGINE = PredictiveMaintenanceEngine(DEFAULT_DATASET)
MLOPS_MANAGER = MLflowManager(tracking_uri=MLFLOW_TRACKING_URI)


@app.get("/dashboard-ui")
def dashboard_ui() -> FileResponse:
    dashboard_path = STATIC_DIR / "index.html"
    if not dashboard_path.exists():
        dashboard_path = STATIC_DIR / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(dashboard_path)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "enterprise-predictive-maintenance-platform",
        "version": "1.0.0",
        "environment": "development",
        "mlops": MLOPS_MANAGER.__class__.__name__,
    }


@app.get("/machines")
def list_machines() -> Dict[str, Any]:
    machines = sorted(DEFAULT_DATASET["machine_id"].unique().tolist())
    return {"machines": machines, "count": len(machines)}


@app.post("/predict")
def predict_machine(payload: PredictionRequest) -> Dict[str, Any]:
    try:
        sensor_records = [sample.model_dump() for sample in payload.sensor_data]
        prediction = ML_ENGINE.predict_machine(payload.machine_id, sensor_records)
        return prediction
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dashboard")
def dashboard_summary() -> Dict[str, Any]:
    summary = ML_ENGINE.summarize_dashboard(DEFAULT_DATASET)
    summary["dataset_rows"] = int(len(DEFAULT_DATASET))
    summary["machines_in_scope"] = int(DEFAULT_DATASET["machine_id"].nunique())
    return summary


@app.get("/model/metrics")
def model_metrics() -> Dict[str, Any]:
    return {
        "training_metrics": ML_ENGINE.training_metrics,
        "feature_importance": dict(sorted(ML_ENGINE.feature_importance.items(), key=lambda item: item[1], reverse=True)[:8]),
        "status": "trained",
    }


@app.get("/explainability/{machine_id}")
def explainability(machine_id: str) -> Dict[str, Any]:
    machine_records = DEFAULT_DATASET[DEFAULT_DATASET["machine_id"] == machine_id].to_dict(orient="records")
    if not machine_records:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    prediction = ML_ENGINE.predict_machine(machine_id, machine_records)
    return {
        "machine_id": machine_id,
        "top_risk_factors": prediction["top_risk_factors"],
        "feature_importance": prediction["feature_importance"],
        "recommended_action": prediction["recommended_action"],
    }


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Enterprise Predictive Maintenance Platform API is running."}
