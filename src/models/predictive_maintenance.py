from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    RandomForestClassifier = None
    LinearRegression = None
    accuracy_score = None
    f1_score = None
    train_test_split = None


FEATURE_COLUMNS = [
    "temperature",
    "pressure",
    "vibration",
    "rpm",
    "voltage",
    "current",
    "humidity",
    "load",
    "maintenance_history",
    "failure_log",
    "operating_hours",
    "temperature_rolling_mean",
    "temperature_rolling_std",
    "vibration_rolling_mean",
    "vibration_rolling_std",
    "load_rolling_mean",
    "load_rolling_std",
    "current_trend",
    "temperature_gradient",
    "utilization_rate",
    "failure_frequency",
    "maintenance_interval",
]


class FeatureEngineer:
    """Compute time-series features relevant to machine health and failure risk."""

    @staticmethod
    def _safe_rolling(values: Sequence[float], window: int = 5) -> pd.Series:
        series = pd.Series(values, dtype=float)
        rolling = series.rolling(window=window, min_periods=1)
        return rolling.mean(), rolling.std(ddof=0).fillna(0)

    @classmethod
    def transform(cls, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        enriched = enriched.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        enriched["temperature_rolling_mean"] = 0.0
        enriched["temperature_rolling_std"] = 0.0
        enriched["vibration_rolling_mean"] = 0.0
        enriched["vibration_rolling_std"] = 0.0
        enriched["load_rolling_mean"] = 0.0
        enriched["load_rolling_std"] = 0.0
        enriched["current_trend"] = 0.0
        enriched["temperature_gradient"] = 0.0
        enriched["utilization_rate"] = 0.0
        enriched["failure_frequency"] = 0.0
        enriched["maintenance_interval"] = 0.0

        for machine_id, group in enriched.groupby("machine_id", sort=False):
            temp_mean, temp_std = cls._safe_rolling(group["temperature"].to_numpy(), window=5)
            vib_mean, vib_std = cls._safe_rolling(group["vibration"].to_numpy(), window=5)
            load_mean, load_std = cls._safe_rolling(group["load"].to_numpy(), window=5)
            enriched.loc[group.index, "temperature_rolling_mean"] = temp_mean.to_numpy()
            enriched.loc[group.index, "temperature_rolling_std"] = temp_std.to_numpy()
            enriched.loc[group.index, "vibration_rolling_mean"] = vib_mean.to_numpy()
            enriched.loc[group.index, "vibration_rolling_std"] = vib_std.to_numpy()
            enriched.loc[group.index, "load_rolling_mean"] = load_mean.to_numpy()
            enriched.loc[group.index, "load_rolling_std"] = load_std.to_numpy()

            current_values = group["current"].to_numpy()
            trend = np.concatenate([[0.0], np.diff(current_values)])
            enriched.loc[group.index, "current_trend"] = trend

            temperature_values = group["temperature"].to_numpy()
            gradient = np.concatenate([[0.0], np.diff(temperature_values)])
            enriched.loc[group.index, "temperature_gradient"] = gradient

            utilization = (group["load"].to_numpy() / 100.0).clip(0.0, 1.0)
            enriched.loc[group.index, "utilization_rate"] = utilization

            prior_failures = group["failure_log"].cumsum().to_numpy()
            enriched.loc[group.index, "failure_frequency"] = prior_failures

            maintenance_gap = group["maintenance_history"].to_numpy()
            enriched.loc[group.index, "maintenance_interval"] = maintenance_gap

        return enriched.fillna(0.0)


class PredictiveMaintenanceEngine:
    """Lightweight predictive maintenance engine with heuristic fallback for offline use."""

    def __init__(self, dataset: pd.DataFrame | None = None) -> None:
        self.dataset = dataset.copy() if dataset is not None else None
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.rul_model = None
        self.feature_importance = {}
        self.training_metrics: Dict[str, float] = {}
        if dataset is not None:
            self.fit(dataset)

    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self.feature_engineer.transform(df)
        for column in FEATURE_COLUMNS:
            if column not in prepared.columns:
                prepared[column] = 0.0
        return prepared

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        prepared = self._prepare_training_data(df)
        target = prepared["failure_label"]
        features = prepared[FEATURE_COLUMNS].copy()

        if RandomForestClassifier is not None:
            train_x, test_x, train_y, test_y = train_test_split(
                features,
                target,
                test_size=0.2,
                random_state=42,
                stratify=target,
            )
            self.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
            self.model.fit(train_x, train_y)
            predictions = self.model.predict(test_x)
            self.training_metrics = {
                "accuracy": float(accuracy_score(test_y, predictions)),
                "f1_score": float(f1_score(test_y, predictions, zero_division=0)),
                "samples": int(len(df)),
            }
            self.feature_importance = dict(
                zip(features.columns, self.model.feature_importances_.tolist())
            )
        else:
            self.training_metrics = {"accuracy": 0.0, "f1_score": 0.0, "samples": int(len(df))}
            self.feature_importance = {column: 1.0 / len(FEATURE_COLUMNS) for column in FEATURE_COLUMNS}

        if LinearRegression is not None:
            self.rul_model = LinearRegression()
            self.rul_model.fit(features, prepared["remaining_useful_life"].astype(float))
        return self.training_metrics

    def predict_machine(self, machine_id: str, sensor_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        records = list(sensor_records)
        if not records:
            raise ValueError("Sensor data is required to produce a maintenance prediction.")

        frame = pd.DataFrame(records)
        if "machine_id" not in frame.columns:
            frame["machine_id"] = machine_id
        if "timestamp" not in frame.columns:
            frame["timestamp"] = pd.Timestamp.utcnow().isoformat()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        prepared = self._prepare_training_data(pd.concat([self.dataset, frame], ignore_index=True) if self.dataset is not None else frame)
        machine_frame = prepared[prepared["machine_id"] == machine_id].copy() if "machine_id" in prepared.columns else prepared.copy()
        if machine_frame.empty:
            machine_frame = prepared.iloc[[-1]].copy()

        latest = machine_frame.iloc[-1]
        risk = float(np.clip(latest.get("risk_score", 0.5), 0.0, 1.0))
        health_score = float(np.clip(latest.get("health_score", 100 - (risk * 100)), 0.0, 100.0))
        remaining_useful_life = int(max(0, float(latest.get("remaining_useful_life", 90 - risk * 60))))

        if self.model is not None:
            candidate = machine_frame[FEATURE_COLUMNS].fillna(0.0).iloc[[-1]]
            probability = float(self.model.predict_proba(candidate)[0, 1])
            risk = float(np.clip(probability, 0.0, 1.0))
            health_score = float(np.clip(100 - (risk * 100), 0.0, 100.0))
            if self.rul_model is not None:
                remaining_useful_life = int(max(0, float(self.rul_model.predict(candidate)[0])))

        if risk >= 0.75 or health_score < 35:
            recommendation = "Immediate Repair"
        elif risk >= 0.5 or health_score < 55:
            recommendation = "Scheduled Maintenance"
        else:
            recommendation = "Continue Monitoring"

        explanation = self._build_explanation(machine_frame)

        return {
            "machine_id": machine_id,
            "risk_score": round(risk, 4),
            "health_score": round(health_score, 2),
            "remaining_useful_life_days": max(0, remaining_useful_life),
            "failure_probability": round(risk, 4),
            "recommended_action": recommendation,
            "top_risk_factors": explanation["top_risk_factors"],
            "feature_importance": explanation["feature_importance"],
        }

    def _build_explanation(self, machine_frame: pd.DataFrame) -> Dict[str, Any]:
        risks: List[Dict[str, Any]] = []
        for column, importance in self.feature_importance.items():
            if column not in machine_frame.columns:
                continue
            value = float(machine_frame[column].iloc[-1]) if not machine_frame.empty else 0.0
            score = abs(float(value)) * float(importance)
            if column in {"temperature", "vibration", "current", "load", "maintenance_history"}:
                risks.append({"feature": column, "score": round(score, 4), "value": round(value, 4)})

        ranked = sorted(risks, key=lambda item: item["score"], reverse=True)[:5]
        sorted_importance = dict(
            sorted(self.feature_importance.items(), key=lambda item: item[1], reverse=True)[:8]
        )
        return {"top_risk_factors": ranked, "feature_importance": sorted_importance}

    def summarize_dashboard(
        self,
        dataset: pd.DataFrame,
        saved_predictions: Iterable[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        if dataset.empty:
            return {"machines": 0, "average_health_score": 0.0, "critically_at_risk": 0, "maintenance_recommendations": {}}

        registered_machines = {
            str(machine_id) for machine_id in dataset["machine_id"].dropna().unique()
        }
        latest_predictions = {}
        for item in saved_predictions:
            machine_id = str(item.get("machine_id", "")).strip()
            if machine_id in registered_machines:
                current = latest_predictions.get(machine_id)
                item_key = (str(item.get("created_at", "")), int(item.get("id", 0)))
                current_key = (
                    (str(current.get("created_at", "")), int(current.get("id", 0)))
                    if current
                    else None
                )
                if current_key is None or item_key > current_key:
                    latest_predictions[machine_id] = item

        if not latest_predictions:
            return {
                "machines": 0,
                "average_health_score": 0.0,
                "critically_at_risk": 0,
                "maintenance_recommendations": {},
            }

        health_scores = [
            float(item["health_score"]) for item in latest_predictions.values()
        ]
        statuses = {
            machine_id: str(item.get("predicted_status", "Healthy"))
            for machine_id, item in latest_predictions.items()
        }
        avg_health = float(np.mean(health_scores))
        critical = sum(status == "Critical" for status in statuses.values())
        queue = sum(status in {"Critical", "Warning"} for status in statuses.values())
        recommendations = pd.Series(
            item["recommendation"] for item in latest_predictions.values()
        ).value_counts().to_dict()

        return {
            "machines": len(latest_predictions),
            "average_health_score": round(avg_health, 2),
            "critically_at_risk": critical,
            "maintenance_recommendations": {str(key): int(value) for key, value in recommendations.items()},
            "maintenance_queue": queue,
        }


def build_demo_prediction(machine_id: str = "M-001") -> Dict[str, Any]:
    """Simple default prediction useful for API demos and smoke tests."""
    return {
        "machine_id": machine_id,
        "risk_score": 0.34,
        "health_score": 74.2,
        "remaining_useful_life_days": 46,
        "failure_probability": 0.34,
        "recommended_action": "Continue Monitoring",
        "top_risk_factors": [
            {"feature": "temperature", "score": 0.21, "value": 81.2},
            {"feature": "vibration", "score": 0.15, "value": 6.9},
        ],
        "feature_importance": {"temperature": 0.28, "vibration": 0.2, "current": 0.15},
    }
