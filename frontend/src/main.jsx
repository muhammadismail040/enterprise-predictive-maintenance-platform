import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DASHBOARD_POLL_INTERVAL_MS = 30_000;
const DASHBOARD_STALE_AFTER_MS = 90_000;

const manualFields = [
  { name: "temperature", label: "Temperature", unit: "°C", min: 20, max: 150 },
  { name: "pressure", label: "Pressure", unit: "PSI", min: 40, max: 160 },
  { name: "vibration", label: "Vibration", unit: "mm/s", min: 0, max: 25 },
  { name: "rpm", label: "RPM", unit: "rev/min", min: 500, max: 3000 },
  { name: "voltage", label: "Voltage", unit: "V", min: 150, max: 260 },
  { name: "current", label: "Current", unit: "A", min: 0, max: 180 },
  { name: "humidity", label: "Humidity", unit: "%", min: 0, max: 100 },
  { name: "load", label: "Load", unit: "%", min: 0, max: 100 },
  { name: "maintenance_history", label: "Maintenance history", unit: "days", min: 0, max: 365 },
  { name: "failure_log", label: "Failure log", unit: "events", min: 0, max: 1, step: 1 },
  { name: "operating_hours", label: "Operating hours", unit: "hours", min: 0, max: 25000 },
];

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
  const [dashboard, setDashboard] = useState({
    average_health_score: 0,
    critically_at_risk: 0,
    maintenance_recommendations: {},
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [lastUpdatedAt, setLastUpdatedAt] = useState(null);
  const [currentTime, setCurrentTime] = useState(() => Date.now());
  const [manualMachine, setManualMachine] = useState("");
  const [manualForm, setManualForm] = useState({});
  const [manualPrediction, setManualPrediction] = useState(null);
  const [manualError, setManualError] = useState("");
  const [manualLoading, setManualLoading] = useState(false);
  const [manualSaved, setManualSaved] = useState(false);
  const [history, setHistory] = useState([]);

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
        setManualMachine(machineList[0] || "");
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

  useEffect(() => {
    if (!manualMachine) return;
    let cancelled = false;

    async function loadManualDefaults() {
      setManualLoading(true);
      setManualError("");
      try {
        const response = await fetch(
          `/machines/${encodeURIComponent(manualMachine)}/sensor-defaults`,
        );
        if (!response.ok) throw new Error("Unable to load machine sensor defaults.");
        const defaults = await response.json();
        if (!cancelled) setManualForm(defaults);
      } catch (defaultsError) {
        if (!cancelled) setManualError(defaultsError.message);
      } finally {
        if (!cancelled) setManualLoading(false);
      }
    }

    loadManualDefaults();
    return () => {
      cancelled = true;
    };
  }, [manualMachine]);

  useEffect(() => {
    async function loadHistory() {
      const response = await fetch("/predictions/history");
      if (response.ok) {
        const payload = await response.json();
        setHistory(payload.predictions || []);
      }
    }
    loadHistory();
  }, []);

  async function runManualPrediction(event) {
    event.preventDefault();
    setManualError("");
    setManualSaved(false);
    const invalidField = manualFields.find(
      (field) =>
        manualForm[field.name] === undefined ||
        manualForm[field.name] === "" ||
        !Number.isFinite(Number(manualForm[field.name])),
    );
    if (invalidField) {
      setManualError(`${invalidField.label} must be a numeric value.`);
      return;
    }

    try {
      const sensorData = Object.fromEntries(
        manualFields.map((field) => [
          field.name,
          field.name === "failure_log"
            ? Number.parseInt(manualForm[field.name], 10)
            : Number(manualForm[field.name]),
        ]),
      );
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          machine_id: manualMachine,
          sensor_data: [sensorData],
          save_to_history: true,
        }),
      });
      if (!response.ok) throw new Error("Manual prediction request failed.");
      const payload = await response.json();
      setManualPrediction(payload);
      setManualSaved(Boolean(payload.history));
      const historyResponse = await fetch("/predictions/history");
      if (historyResponse.ok) {
        const historyPayload = await historyResponse.json();
        setHistory(historyPayload.predictions || []);
      }
       const dashboardResponse = await fetch("/dashboard");
       if (!dashboardResponse.ok) throw new Error("Dashboard refresh failed.");
       setDashboard(await dashboardResponse.json());
       setLastUpdatedAt(Date.now());
       setConnectionStatus("connected");
    } catch (predictionError) {
      setManualError(predictionError.message);
    }
  }

  async function deleteHistoryRecord(predictionId) {
    if (!window.confirm("Delete this prediction history record?")) return;
    setManualError("");
    try {
      const response = await fetch(`/predictions/history/${predictionId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Unable to delete prediction history record.");
      }
      const historyResponse = await fetch("/predictions/history");
      if (!historyResponse.ok) throw new Error("Unable to refresh prediction history.");
      const historyPayload = await historyResponse.json();
      setHistory(historyPayload.predictions || []);
      const dashboardResponse = await fetch("/dashboard");
      if (!dashboardResponse.ok) throw new Error("Dashboard refresh failed.");
      setDashboard(await dashboardResponse.json());
      setLastUpdatedAt(Date.now());
      setConnectionStatus("connected");
    } catch (deleteError) {
      setManualError(deleteError.message);
    }
  }

  const queueSize = useMemo(
    () =>
      Object.entries(dashboard.maintenance_recommendations || {}).reduce(
        (total, [action, count]) =>
          action === "Continue Monitoring" ? total : total + Number(count),
        0,
      ),
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
          value={`${Number(
            dashboard.fleet_health_percent ?? dashboard.average_health_score ?? 0,
          ).toFixed(1)}%`}
        />
        <MetricCard
          label="Critical Assets"
          value={String(dashboard.critical_assets ?? dashboard.critically_at_risk ?? 0)}
        />
        <MetricCard
          label="Maintenance Queue"
          value={String(dashboard.maintenance_queue ?? queueSize)}
        />
      </section>

      <section className="manual-layout">
        <section className="card">
          <p className="eyebrow">Ad-hoc sensor input</p>
          <h2>Manual Risk Check</h2>
          <form className="manual-form" onSubmit={runManualPrediction} noValidate>
            <label htmlFor="manual-machine">Machine defaults</label>
            <select
              id="manual-machine"
              value={manualMachine}
              onChange={(event) => setManualMachine(event.target.value)}
            >
              {machines.map((machine) => (
                <option key={machine} value={machine}>{machine}</option>
              ))}
            </select>
            {manualLoading && <p className="muted">Loading current sensor values...</p>}
            <div className="manual-fields">
              {manualFields.map((field) => (
                <label key={field.name} className="field">
                  <span>{field.label} ({field.unit})</span>
                  <input
                    type="number"
                    name={field.name}
                    value={manualForm[field.name] ?? ""}
                    min={field.min}
                    max={field.max}
                    step={field.step || "any"}
                    placeholder={`${field.min}-${field.max}`}
                    onChange={(event) =>
                      setManualForm((current) => ({
                        ...current,
                        [field.name]: event.target.value,
                      }))
                    }
                  />
                  <small>Typical range: {field.min}-{field.max}</small>
                </label>
              ))}
            </div>
            {manualError && <p className="inline-error">{manualError}</p>}
            <div className="manual-actions">
              <button type="submit" disabled={manualLoading}>Predict</button>
              {manualSaved && <span className="saved-confirmation">Saved to history</span>}
            </div>
          </form>
        </section>
        <PredictionDetails
          prediction={manualPrediction}
          emptyMessage="Submit the edited sensor values to see a manual risk result."
        />
      </section>
      <section className="card history-panel">
        <p className="eyebrow">Audit trail</p>
        <h2>Prediction History</h2>
        {history.length ? (
          <div className="table-wrapper">
            <table className="machine-list">
              <thead>
                <tr><th>Machine</th><th>Status</th><th>Risk</th><th>Timestamp</th><th>Action</th></tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td>{item.machine_id}</td>
                    <td><span className={`pill ${riskClass(Number(item.risk_score))}`}>{item.predicted_status}</span></td>
                    <td>{Number(item.risk_score).toFixed(3)}</td>
                    <td>{new Date(item.created_at).toLocaleString()}</td>
                    <td>
                      <button
                        type="button"
                        className="delete-button"
                        onClick={() => deleteHistoryRecord(item.id)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="muted">No saved manual predictions yet.</p>}
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

function PredictionDetails({ prediction, emptyMessage }) {
  const risk = prediction
    ? Number(prediction.failure_probability ?? prediction.risk_score ?? 0)
    : 0;
  const recommendation = risk >= 0.75
    ? "Immediate Repair"
    : risk >= 0.5
      ? "Scheduled Maintenance"
      : "Continue Monitoring";

  return (
    <section className="card prediction-card">
      <p className="eyebrow">Model output</p>
      <h2>Prediction Details</h2>
      {prediction ? (
        <dl className="details">
          <Detail label="Machine" value={prediction.machine_id} />
          <Detail label="Status" value={riskLabel(risk)} />
          <Detail label="Risk" value={risk.toFixed(3)} />
          <Detail label="Health" value={`${Number(prediction.health_score ?? 0).toFixed(1)}%`} />
          <Detail label="Recommendation" value={recommendation} />
          <Detail
            label="Top risk factors"
            value={(prediction.top_risk_factors || []).map((factor) => factor.feature).join(", ") || "N/A"}
          />
        </dl>
      ) : (
        <p className="muted">{emptyMessage}</p>
      )}
    </section>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
