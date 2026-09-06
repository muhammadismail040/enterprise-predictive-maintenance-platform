from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "machine_sensor_data.csv"


def generate_synthetic_dataset(
    num_machines: int = 12,
    records_per_machine: int = 200,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate a realistic synthetic machine sensor dataset for predictive maintenance."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    at_risk_count = max(1, int(np.ceil(num_machines * 0.2)))
    at_risk_machines = set(
        rng.choice(num_machines, size=at_risk_count, replace=False).tolist()
    )

    for machine_idx in range(num_machines):
        machine_at_risk = machine_idx in at_risk_machines
        base_temp = 70 + rng.normal(0, 8)
        base_pressure = 100 + rng.normal(0, 7)
        base_vibration = 4 + rng.normal(0, 1.2)
        base_rpm = 1500 + rng.normal(0, 150)
        base_voltage = 230 + rng.normal(0, 10)
        base_current = 65 + rng.normal(0, 12)
        base_humidity = 45 + rng.normal(0, 12)
        base_load = 60 + rng.normal(0, 18)
        maintenance_gap = 30 + rng.integers(0, 90)

        for reading_idx in range(records_per_machine):
            time_delta = timedelta(minutes=reading_idx * 15)
            timestamp = start_time + time_delta
            time_factor = reading_idx / max(1, records_per_machine)

            temperature = base_temp + 18 * np.sin(time_factor * 2 * np.pi) + rng.normal(0, 4)
            pressure = base_pressure + 6 * np.cos(time_factor * 3 * np.pi) + rng.normal(0, 3)
            vibration = max(0.5, base_vibration + 3 * time_factor + rng.normal(0, 1.4))
            rpm = base_rpm + 250 * np.sin(time_factor * 2.5 * np.pi) + rng.normal(0, 120)
            voltage = base_voltage + np.sin(time_factor * np.pi) * 8 + rng.normal(0, 3)
            current = base_current + 18 * time_factor + rng.normal(0, 7)
            humidity = base_humidity + 5 * np.cos(time_factor * np.pi) + rng.normal(0, 6)
            load = min(100, max(10, base_load + 25 * np.sin(time_factor * 3 * np.pi) + rng.normal(0, 10)))

            if machine_at_risk:
                temperature += 22
                vibration += 5
                current += 28
                load = min(100, load + 20)

            if machine_idx % 4 == 0 and reading_idx > records_per_machine * 0.7:
                temperature += 18
                vibration += 4
                current += 20
                load += 15

            maintenance_history = max(0, maintenance_gap - reading_idx % maintenance_gap)
            failure_log = 1 if machine_at_risk or (reading_idx > records_per_machine * 0.88 and machine_idx % 3 == 0) else 0
            operating_hours = reading_idx * 0.25

            risk_score = min(
                1.0,
                (
                    max(0, (temperature - 80) / 25)
                    + max(0, (vibration - 6) / 6)
                    + max(0, (current - 80) / 40)
                    + max(0, (load - 80) / 30)
                    + failure_log * 0.35
                )
                / 4.5,
            )

            failure_label = 1 if machine_at_risk else 0
            if machine_at_risk:
                risk_score = max(risk_score, 0.82)
            health_score = max(0, min(100, 100 - (risk_score * 100) - (failure_log * 15)))
            remaining_useful_life = max(0, int(90 - (risk_score * 80) - (failure_log * 30)))

            if health_score < 45:
                recommended_action = "Immediate Repair"
            elif health_score < 65:
                recommended_action = "Scheduled Maintenance"
            elif risk_score > 0.55:
                recommended_action = "Continue Monitoring"
            else:
                recommended_action = "Continue Monitoring"

            failure_type = "Bearing Failure" if vibration > 8 else "Overheating" if temperature > 95 else "Motor Failure" if current > 90 else "Electrical Fault" if voltage < 200 else "Normal"
            if failure_label == 0:
                failure_type = "Normal"

            rows.append(
                {
                    "machine_id": f"M-{machine_idx + 1:03d}",
                    "timestamp": timestamp.isoformat(),
                    "temperature": round(float(temperature), 3),
                    "pressure": round(float(pressure), 3),
                    "vibration": round(float(vibration), 3),
                    "rpm": round(float(rpm), 3),
                    "voltage": round(float(voltage), 3),
                    "current": round(float(current), 3),
                    "humidity": round(float(humidity), 3),
                    "load": round(float(load), 3),
                    "maintenance_history": round(float(maintenance_history), 3),
                    "failure_log": int(failure_log),
                    "operating_hours": round(float(operating_hours), 3),
                    "risk_score": round(float(risk_score), 4),
                    "failure_label": int(failure_label),
                    "health_score": round(float(health_score), 3),
                    "remaining_useful_life": int(remaining_useful_life),
                    "recommended_action": recommended_action,
                    "failure_type": failure_type,
                }
            )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

    return df


def load_or_create_dataset(path: Optional[str | Path] = None) -> pd.DataFrame:
    target_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not target_path.exists():
        return generate_synthetic_dataset(output_path=target_path)
    return pd.read_csv(target_path)
