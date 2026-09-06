from datetime import datetime, timedelta, timezone


ASSETS = [
    {"id": "A-1001", "name": "Compressor-1", "status": "operational"},
    {"id": "A-1002", "name": "Conveyor-3", "status": "degraded"},
]


PREDICTIONS = [
    {
        "asset_id": "A-1001",
        "failure_risk": 0.12,
        "prediction_window_hours": 72,
    },
    {
        "asset_id": "A-1002",
        "failure_risk": 0.64,
        "prediction_window_hours": 72,
    },
]


SCHEDULES = [
    {
        "asset_id": "A-1002",
        "task": "Bearing inspection",
        "scheduled_for": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
    }
]


ALERTS = [
    {
        "asset_id": "A-1002",
        "severity": "high",
        "message": "Vibration exceeds baseline threshold",
    }
]
