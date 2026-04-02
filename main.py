from __future__ import annotations

import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pymysql
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/fraud_model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"))

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "fraud_db")
MYSQL_USER = os.getenv("MYSQL_USER", "fraud_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "fraud_pass")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

app = FastAPI(
    title="Fraud Detection API",
    description="Minimal API and monitoring page for the fraud detection demo.",
    version="3.0.0",
)

MODEL: Any | None = None
PREPROCESSOR: dict[str, Any] | None = None
MODEL_LOAD_ERROR: str | None = None


def load_artifacts() -> None:
    global MODEL, PREPROCESSOR, MODEL_LOAD_ERROR

    try:
        MODEL = joblib.load(MODEL_PATH)
        PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
        MODEL_LOAD_ERROR = None
    except FileNotFoundError:
        MODEL = None
        PREPROCESSOR = None
        MODEL_LOAD_ERROR = (
            "Model artifacts are missing. Run train.py first so the API can load the model."
        )


def preprocess_transaction(transaction: dict[str, Any], preprocessor: dict[str, Any]) -> pd.DataFrame:
    feature_columns = preprocessor["feature_columns"]
    categorical_columns = set(preprocessor["categorical_columns"])
    encoders = preprocessor["encoders"]
    fill_value = preprocessor["fill_value"]

    processed_row: dict[str, float] = {}

    for column in feature_columns:
        value = transaction.get(column, fill_value)
        if value in (None, "") or pd.isna(value):
            value = fill_value

        if column in categorical_columns:
            mapping_info = encoders[column]
            mapping = mapping_info["mapping"]
            default_value = mapping_info["default_value"]
            processed_row[column] = float(mapping.get(str(value), default_value))
        else:
            try:
                processed_row[column] = float(value)
            except (TypeError, ValueError):
                processed_row[column] = float(fill_value)

    return pd.DataFrame([processed_row], columns=feature_columns).astype("float32")


def get_mysql_connection() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def save_prediction(
    connection: pymysql.connections.Connection,
    txn_id: int,
    amount: float,
    is_fraud: int,
    fraud_probability: float,
) -> None:
    query = """
        INSERT INTO fraud_predictions (txn_id, amount, is_fraud, fraud_prob, processed_at)
        VALUES (%s, %s, %s, %s, %s)
    """
    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                txn_id,
                amount,
                is_fraud,
                fraud_probability,
                datetime.utcnow(),
            ),
        )
    connection.commit()


def check_mysql_status() -> bool:
    try:
        connection = get_mysql_connection()
        connection.close()
        return True
    except Exception:
        return False


def check_kafka_status() -> bool:
    try:
        first_server = KAFKA_BOOTSTRAP_SERVERS.split(",")[0].strip()
        host, port = first_server.split(":")
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except Exception:
        return False


def get_monitor_snapshot() -> dict[str, Any]:
    mysql_ok = check_mysql_status()
    kafka_ok = check_kafka_status()

    snapshot: dict[str, Any] = {
        "services": {
            "api": True,
            "model": MODEL is not None,
            "mysql": mysql_ok,
            "kafka": kafka_ok,
        },
        "summary": {
            "total_processed": 0,
            "fraud_alerts": 0,
        },
        "recent_predictions": [],
    }

    if not mysql_ok:
        return snapshot

    connection = get_mysql_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS total_processed, COALESCE(SUM(is_fraud), 0) AS fraud_alerts
                FROM fraud_predictions
                """
            )
            summary = cursor.fetchone() or {}
            snapshot["summary"] = {
                "total_processed": int(summary.get("total_processed", 0) or 0),
                "fraud_alerts": int(summary.get("fraud_alerts", 0) or 0),
            }

            cursor.execute(
                """
                SELECT txn_id, amount, is_fraud, fraud_prob, processed_at
                FROM fraud_predictions
                ORDER BY processed_at DESC
                LIMIT 12
                """
            )
            rows = cursor.fetchall() or []

        snapshot["recent_predictions"] = [
            {
                "txn_id": int(row["txn_id"]),
                "amount": float(row["amount"]),
                "status": "fraud" if int(row["is_fraud"]) == 1 else "normal",
                "fraud_probability": round(float(row["fraud_prob"]), 4),
                "processed_at": row["processed_at"].strftime("%Y-%m-%d %H:%M:%S"),
            }
            for row in rows
        ]
    finally:
        connection.close()

    return snapshot


@app.on_event("startup")
def startup_event() -> None:
    load_artifacts()


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fraud Monitor</title>
  <style>
    :root {
      --bg: #f7f7f2;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --ok: #15803d;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 920px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 32px;
      font-weight: 700;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      flex-wrap: wrap;
    }
    .topbar a {
      color: var(--text);
      text-decoration: none;
      font-weight: 600;
    }
    .section-title {
      margin-top: 28px;
      margin-bottom: 12px;
      font-size: 18px;
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
    }
    .label {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .value {
      font-size: 28px;
      font-weight: 700;
    }
    .status-ok { color: var(--ok); }
    .status-bad { color: var(--bad); }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 14px;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: #fafaf9;
    }
    tr:last-child td {
      border-bottom: none;
    }
    .pill {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .pill-fraud {
      background: #fee2e2;
      color: #b91c1c;
    }
    .pill-normal {
      background: #dcfce7;
      color: #166534;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Fraud Monitor</h1>
        <p>Simple live view of connection status and fraud results.</p>
      </div>
      <a href="/docs">Open API Docs</a>
    </div>

    <div class="section-title">System Status</div>
    <div class="grid">
      <div class="card"><div class="label">API</div><div class="value" id="api-status">...</div></div>
      <div class="card"><div class="label">Model</div><div class="value" id="model-status">...</div></div>
      <div class="card"><div class="label">MySQL</div><div class="value" id="mysql-status">...</div></div>
      <div class="card"><div class="label">Kafka</div><div class="value" id="kafka-status">...</div></div>
    </div>

    <div class="section-title">Fraud Summary</div>
    <div class="grid">
      <div class="card"><div class="label">Total Processed</div><div class="value" id="total-processed">0</div></div>
      <div class="card"><div class="label">Fraud Alerts</div><div class="value" id="fraud-alerts">0</div></div>
    </div>

    <div class="section-title">Recent Transactions</div>
    <table>
      <thead>
        <tr>
          <th>Txn ID</th>
          <th>Amount</th>
          <th>Status</th>
          <th>Probability</th>
          <th>Processed At</th>
        </tr>
      </thead>
      <tbody id="recent-body">
        <tr><td colspan="5">Loading...</td></tr>
      </tbody>
    </table>
  </div>

  <script>
    function setStatus(id, ok) {
      const el = document.getElementById(id);
      el.textContent = ok ? "Connected" : "Not Ready";
      el.className = "value " + (ok ? "status-ok" : "status-bad");
    }

    async function loadData() {
      const res = await fetch("/monitor-data");
      const data = await res.json();

      setStatus("api-status", data.services.api);
      setStatus("model-status", data.services.model);
      setStatus("mysql-status", data.services.mysql);
      setStatus("kafka-status", data.services.kafka);

      document.getElementById("total-processed").textContent = data.summary.total_processed;
      document.getElementById("fraud-alerts").textContent = data.summary.fraud_alerts;

      const tbody = document.getElementById("recent-body");
      if (!data.recent_predictions.length) {
        tbody.innerHTML = '<tr><td colspan="5">No transactions yet.</td></tr>';
        return;
      }

      tbody.innerHTML = data.recent_predictions.map((row) => `
        <tr>
          <td>${row.txn_id}</td>
          <td>${row.amount.toFixed(2)}</td>
          <td><span class="pill ${row.status === "fraud" ? "pill-fraud" : "pill-normal"}">${row.status}</span></td>
          <td>${row.fraud_probability}</td>
          <td>${row.processed_at}</td>
        </tr>
      `).join("");
    }

    loadData();
    setInterval(loadData, 5000);
  </script>
</body>
</html>
        """
    )


@app.get("/monitor-data", include_in_schema=False)
def monitor_data() -> dict[str, Any]:
    return get_monitor_snapshot()


@app.get("/health")
def health() -> dict[str, Any]:
    snapshot = get_monitor_snapshot()
    return {
        "status": "ok" if snapshot["services"]["model"] else "model_not_ready",
        "services": snapshot["services"],
        "details": MODEL_LOAD_ERROR or "Model loaded successfully.",
    }


@app.post("/predict")
def predict(transaction: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)

    feature_frame = preprocess_transaction(transaction, PREPROCESSOR)
    threshold = float(PREPROCESSOR.get("threshold", 0.5))
    id_column = PREPROCESSOR.get("id_column", "TransactionID")

    fraud_probability = float(MODEL.predict_proba(feature_frame)[0][1])
    is_fraud = int(fraud_probability > threshold)
    txn_id = int(transaction.get(id_column) or int(datetime.utcnow().timestamp() * 1_000_000))
    amount = float(transaction.get("TransactionAmt") or 0.0)

    connection = get_mysql_connection()
    try:
        save_prediction(
            connection=connection,
            txn_id=txn_id,
            amount=amount,
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
        )
    finally:
        connection.close()

    return {
        "txn_id": txn_id,
        "prediction": "fraud" if is_fraud else "normal",
        "fraud_probability": round(fraud_probability, 4),
    }
