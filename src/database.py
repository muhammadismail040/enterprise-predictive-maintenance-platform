from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from src.config import DATABASE_URL

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None


CREATE_SQLITE_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    input_features TEXT NOT NULL,
    predicted_status TEXT NOT NULL,
    risk_score REAL NOT NULL,
    health_score REAL NOT NULL,
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

CREATE_POSTGRES_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id BIGSERIAL PRIMARY KEY,
    machine_id TEXT NOT NULL,
    input_features JSONB NOT NULL,
    predicted_status TEXT NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    health_score DOUBLE PRECISION NOT NULL,
    recommendation TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
)
"""


def is_postgres() -> bool:
    return DATABASE_URL.startswith(("postgresql://", "postgres://"))


@contextmanager
def connection() -> Iterator[Any]:
    if is_postgres():
        if psycopg is None:
            raise RuntimeError(
                "PostgreSQL is configured but psycopg is not installed."
            )
        with psycopg.connect(DATABASE_URL) as conn:
            yield conn
    else:
        database_path = DATABASE_URL.removeprefix("sqlite:///")
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            yield conn


def initialize_prediction_history() -> None:
    with connection() as conn:
        conn.execute(CREATE_POSTGRES_TABLE if is_postgres() else CREATE_SQLITE_TABLE)
        conn.commit()


def save_prediction(
    machine_id: str,
    input_features: dict[str, Any],
    predicted_status: str,
    risk_score: float,
    health_score: float,
    recommendation: str,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    serialized_features = json.dumps(input_features)
    query = """
    INSERT INTO prediction_history
        (machine_id, input_features, predicted_status, risk_score,
         health_score, recommendation, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    if is_postgres():
        query = query.replace("?", "%s")
        query += " RETURNING id"
        serialized_features = json.dumps(input_features)

    with connection() as conn:
        cursor = conn.execute(
            query,
            (
                machine_id,
                serialized_features,
                predicted_status,
                risk_score,
                health_score,
                recommendation,
                created_at,
            ),
        )
        row_id = cursor.fetchone()[0] if is_postgres() else cursor.lastrowid
        conn.commit()
    return {
        "id": row_id,
        "machine_id": machine_id,
        "predicted_status": predicted_status,
        "risk_score": risk_score,
        "health_score": health_score,
        "recommendation": recommendation,
        "created_at": created_at,
    }


def get_prediction_history(
    machine_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    query = """
    SELECT id, machine_id, input_features, predicted_status, risk_score,
           health_score, recommendation, created_at
    FROM prediction_history
    """
    parameters: tuple[Any, ...] = ()
    if machine_id:
        query += " WHERE machine_id = " + ("%s" if is_postgres() else "?")
        parameters = (machine_id,)
    query += " ORDER BY created_at DESC LIMIT " + (
        "%s" if is_postgres() else "?"
    )
    parameters += (limit,)

    with connection() as conn:
        rows = conn.execute(query, parameters).fetchall()
    history = []
    for row in rows:
        values = dict(row) if isinstance(row, sqlite3.Row) else row
        if not isinstance(values, dict):
            values = {
                "id": values[0],
                "machine_id": values[1],
                "input_features": values[2],
                "predicted_status": values[3],
                "risk_score": values[4],
                "health_score": values[5],
                "recommendation": values[6],
                "created_at": values[7],
            }
        if isinstance(values["input_features"], str):
            values["input_features"] = json.loads(values["input_features"])
        history.append(values)
    return history


def delete_prediction(prediction_id: int) -> bool:
    query = "DELETE FROM prediction_history WHERE id = " + (
        "%s" if is_postgres() else "?"
    )
    with connection() as conn:
        cursor = conn.execute(query, (prediction_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
    return deleted
