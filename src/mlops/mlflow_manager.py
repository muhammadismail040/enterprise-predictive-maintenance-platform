from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

from src.config import MLFLOW_TRACKING_URI

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


@dataclass
class MLflowManager:
    """Minimal MLflow-backed experiment tracking manager with a graceful offline fallback."""

    experiment_name: str = "enterprise-predictive-maintenance"
    tracking_uri: Optional[str] = MLFLOW_TRACKING_URI
    _active_run: Any = field(default=None, init=False)

    def _tracking_uri_available(self, target_uri: str) -> bool:
        if target_uri.startswith("http://") or target_uri.startswith("https://"):
            try:
                with urlopen(target_uri, timeout=1) as response:
                    return response.status < 500
            except (URLError, TimeoutError, OSError):
                return False
        return True

    def __post_init__(self) -> None:
        if mlflow is not None:
            target_uri = self.tracking_uri or MLFLOW_TRACKING_URI
            if not self._tracking_uri_available(target_uri):
                self.experiment_id = None
                return
            mlflow.set_tracking_uri(target_uri)
            self.experiment_id = mlflow.set_experiment(self.experiment_name)
        else:
            self.experiment_id = None

    def log_run(self, params: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        params = params or {}
        metrics = metrics or {}
        if mlflow is None:
            return {"status": "offline", "experiment": self.experiment_name, "params": params, "metrics": metrics}

        with mlflow.start_run(run_name="maintenance-model") as run:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            self._active_run = run
            return {"status": "logged", "run_id": run.info.run_id, "experiment": self.experiment_name}

    def log_model(self, model_name: str, model: Any) -> Dict[str, Any]:
        if mlflow is None:
            return {"status": "skipped", "model_name": model_name, "message": "MLflow not installed"}
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        return {"status": "logged", "model_name": model_name}

    def drift_report(self, current_metrics: Dict[str, float], baseline_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        baseline_metrics = baseline_metrics or {"temperature": 76.0, "vibration": 5.5, "load": 62.0}
        drift = {}
        for key, baseline in baseline_metrics.items():
            current = float(current_metrics.get(key, baseline))
            drift[key] = round(abs(current - baseline) / max(1.0, abs(baseline)), 4)
        return {"status": "ok", "drift": drift}
