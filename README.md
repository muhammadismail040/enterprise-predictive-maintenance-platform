# Enterprise Predictive Maintenance Platform

This project provides a production-minded predictive maintenance platform for industrial equipment. It combines synthetic sensor data, a machine-learning risk model, explainability outputs, and a FastAPI backend that exposes operational monitoring endpoints and a Vite + React dashboard.

## What the project does

- Generates synthetic machine telemetry for industrial assets
- Engineers time-series features for fault and degradation risk analysis
- Predicts the likelihood of failure and recommends maintenance actions
- Exposes machine health, probability, and explainability APIs
- Tracks experiments with MLflow and surfaces fleet-level summaries
- Serves a live operational dashboard for monitoring the fleet

## Architecture diagram (in words)

API -> model -> MLflow -> Postgres

The FastAPI service serves the application and operational endpoints. It calls the predictive maintenance model for machine scoring and explanation. Model metadata and experiment outputs are sent to MLflow for tracking and monitoring. In the Docker deployment, the API and MLflow services are coordinated around a PostgreSQL database for persistence and data access.

## Local run instructions

1. Create and activate a virtual environment if needed:
   `python -m venv .venv`
   `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy the environment template if needed:
   `copy .env.example .env` or `cp .env.example .env`
4. Build the React dashboard:
   `cd frontend && npm install && npm run build`
5. Start the API:
   `python main.py`
6. Open the dashboard:
   `http://localhost:8000/dashboard-ui`
7. Open the interactive API docs:
   `http://localhost:8000/docs`

## Docker run instructions

1. Build and start the services:
   `docker compose up --build`
2. The API image builds the React frontend into `src/static` before starting FastAPI.
3. Wait for the API, Postgres, and MLflow containers to initialize.
4. Open the dashboard:
   `http://localhost:8000/dashboard-ui`
5. Stop the stack when finished:
   `docker compose down`

## Live dashboard route

- Dashboard UI: `http://localhost:8000/dashboard-ui`

## API highlights

- `GET /health` — service health
- `GET /machines` — machine inventory
- `POST /predict` — predictive maintenance score for a machine
- `GET /dashboard` — fleet-level health summary
- `GET /model/metrics` — training metrics and feature importance
- `GET /explainability/{machine_id}` — model explanation for a specific machine

## Notes

The project uses synthetic data to keep local execution simple and reproducible. The API is designed to run with SQLite defaults locally and switch to PostgreSQL in Docker without changing the predictive model logic or the existing endpoint contracts.
