import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DASHBOARD_POLL_INTERVAL_MS = 30_000;
const DASHBOARD_STALE_AFTER_MS = 90_000;

const sensorSample = {
  temperature: 80,
  pressure: 125,
  vibration: 5.9,
  rpm: 1800,
  voltage: 240,
  current: 24,
  humidity: 52,
  load: 68,
  maintenance_history: 3,
  failure_log: 1,
  operating_hours: 5400,
};

function riskClass(risk) {
  if (risk >= 0.75) return "critical";
  if (risk >= 0.5) return "warning";
  return "ok";
}

function riskLabel(risk) {
  if (risk >= 0.75) return "Critical";
  if (risk >= 0.5) return "Warning";
  return "Healthy";
}

function App() {
  const [machines, setMachines] = useState([]);
  const [selectedMachine, setSelectedMachine] = useState("");
  const [dashboard, setDashboard] = useState({
    average_health_score: 0,
    critically_at_risk: 0,
    maintenance_recommendations: {},
  });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [lastUpdatedAt, setLastUpdatedAt] = useState(null);
  const [currentTime, setCurrentTime] = useState(() => Date.now());

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      try {
        const [machinesResponse, dashboardResponse] = await Promise.all([
          fetch("/machines"),
          fetch("/dashboard"),
        ]);
        if (!machinesResponse.ok || !dashboardResponse.ok) {
          throw new Error("Unable to load dashboard data.");
        }
        const machinesPayload = await machinesResponse.json();
        const dashboardPayload = await dashboardResponse.json();
        const machineList = machinesPayload.machines || [];
        if (cancelled) return;
        setMachines(machineList);
        setSelectedMachine(machineList[0] || "");
        setDashboard(dashboardPayload);
        setLastUpdatedAt(Date.now());
        setConnectionStatus("connected");
      } catch (loadError) {
        if (cancelled) return;
        setError(loadError.message);
        setConnectionStatus("offline");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    async function pollDashboard() {
      try {
        const response = await fetch("/dashboard");
        if (!response.ok) throw new Error("Dashboard polling failed.");
        const dashboardPayload = await response.json();
        if (cancelled) return;
        setDashboard(dashboardPayload);
        setLastUpdatedAt(Date.now());
        setConnectionStatus("connected");
        setError("");
      } catch (pollError) {
        if (cancelled) return;
        setConnectionStatus("offline");
        setError(pollError.message);
      }
    }

    loadDashboard();
    const pollTimer = window.setInterval(pollDashboard, DASHBOARD_POLL_INTERVAL_MS);
    const clockTimer = window.setInterval(() => setCurrentTime(Date.now()), 1000);

    return () => {
      cancelled = true;
      window.clearInterval(pollTimer);
      window.clearInterval(clockTimer);
    };
  }, []);

  const predictionRisk = prediction
    ? Number(prediction.failure_probability ?? prediction.risk_score ?? 0)
    : 0;
  const predictionStatus = riskLabel(predictionRisk);
  const recommendation =
    predictionRisk >= 0.75
      ? "Immediate Repair"
      : predictionRisk >= 0.5
        ? "Scheduled Maintenance"
        : "Continue Monitoring";

  async function runPrediction() {
    if (!selectedMachine) return;
    setError("");
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          machine_id: selectedMachine,
          sensor_data: [sensorSample],
        }),
      });
      if (!response.ok) throw new Error("Prediction request failed.");
      setPrediction(await response.json());
    } catch (predictionError) {
      setError(predictionError.message);
    }
  }

  const queueSize = useMemo(
    () => Object.keys(dashboard.maintenance_recommendations || {}).length,
    [dashboard.maintenance_recommendations],
  );
  const isStale =
    !lastUpdatedAt || currentTime - lastUpdatedAt > DASHBOARD_STALE_AFTER_MS;
  const badgeStatus = connectionStatus === "connected" && !isStale
    ? "connected"
    : connectionStatus === "connecting"
      ? "connecting"
      : "stale";
  const badgeLabel = badgeStatus === "connected"
    ? "LIVE MONITORING"
    : badgeStatus === "connecting"
      ? "CONNECTING"
      : "STALE DATA";
  const lastUpdatedLabel = lastUpdatedAt
    ? `Updated ${Math.max(0, Math.floor((currentTime - lastUpdatedAt) / 1000))}s ago`
    : "Waiting for dashboard data";

  if (loading) {
    return <main className="container muted">Loading operations data...</main>;
  }

  return (
    <main className="container">
      <header className="topbar">
        <div>
          <p className="eyebrow">Enterprise Predictive Maintenance</p>
          <h1>Industrial Ops Console</h1>
        </div>
        <div className="status-block">
          <span className={`tag ${badgeStatus}`}>{badgeLabel}</span>
          <span className="status-detail">{lastUpdatedLabel}</span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="grid metrics" aria-label="Fleet metrics">
        <MetricCard
          label="Fleet Health"
          value={`${Number(dashboard.average_health_score || 0).toFixed(1)}%`}
        />
        <MetricCard
          label="Critical Assets"
          value={String(dashboard.critically_at_risk || 0)}
        />
        <MetricCard label="Maintenance Queue" value={String(queueSize)} />
      </section>

      <section className="layout">
        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Asset inventory</p>
              <h2>Machine Overview</h2>
            </div>
            <div className="controls">
              <label htmlFor="machine-selector" className="sr-only">
                Select machine
              </label>
              <select
                id="machine-selector"
                value={selectedMachine}
                onChange={(event) => setSelectedMachine(event.target.value)}
              >
                {machines.map((machine) => (
                  <option key={machine} value={machine}>
                    {machine}
                  </option>
                ))}
              </select>
              <button type="button" onClick={runPrediction}>
                Run Prediction
              </button>
            </div>
          </div>
          <div className="table-wrapper">
            <table className="machine-list">
              <thead>
                <tr>
                  <th>Machine</th>
                  <th>Status</th>
                  <th>Risk</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {machines.map((machine) => {
                  const machineRisk =
                    prediction?.machine_id === machine ? predictionRisk : 0;
                  return (
                    <tr key={machine}>
                      <td>{machine}</td>
                      <td>
                        <span className={`pill ${riskClass(machineRisk)}`}>
                          {riskLabel(machineRisk)}
                        </span>
                      </td>
                      <td>{machineRisk ? machineRisk.toFixed(3) : "Low"}</td>
                      <td>
                        {prediction?.machine_id === machine
                          ? recommendation
                          : "Continue Monitoring"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card prediction-card">
          <p className="eyebrow">Model output</p>
          <h2>Prediction Details</h2>
          {prediction ? (
            <dl className="details">
              <Detail label="Machine" value={prediction.machine_id} />
              <Detail label="Status" value={predictionStatus} />
              <Detail label="Risk" value={predictionRisk.toFixed(3)} />
              <Detail
                label="Health"
                value={`${Number(prediction.health_score ?? 0).toFixed(1)}%`}
              />
              <Detail label="Recommendation" value={recommendation} />
              <Detail
                label="Top risk factors"
                value={
                  (prediction.top_risk_factors || [])
                    .map((factor) => factor.feature)
                    .join(", ") || "N/A"
                }
              />
            </dl>
          ) : (
            <p className="muted">
              Select a machine and run a prediction to inspect model output.
            </p>
          )}
        </section>
      </section>
    </main>
  );
}

function MetricCard({ label, value }) {
  return (
    <article className="card metric-card">
      <p className="eyebrow">{label}</p>
      <p className="metric">{value}</p>
    </article>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
