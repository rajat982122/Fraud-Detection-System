#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
RUN_DIR="$ROOT_DIR/run"
VENV_DIR="$ROOT_DIR/.venv"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is not installed. Install it first from https://brew.sh and then run this script again."
  exit 1
fi

echo "Installing local dependencies with Homebrew..."
brew install python mysql kafka

KAFKA_BIN="$(brew --prefix kafka)/bin"

echo "Starting MySQL and Kafka services..."
brew services start mysql || true
brew services start kafka || true

echo "Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

echo "Preparing artifacts..."
cd "$ROOT_DIR"
python prepare_artifacts.py

echo "Initializing MySQL schema..."
MYSQL_ROOT_USER="${MYSQL_ROOT_USER:-root}"
MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-}"

if [[ -n "$MYSQL_ROOT_PASSWORD" ]]; then
  mysql -u "$MYSQL_ROOT_USER" -p"$MYSQL_ROOT_PASSWORD" < "$ROOT_DIR/mysql/init.sql"
else
  mysql -u "$MYSQL_ROOT_USER" < "$ROOT_DIR/mysql/init.sql"
fi

echo "Creating Kafka topic if needed..."
"$KAFKA_BIN/kafka-topics" \
  --create \
  --if-not-exists \
  --topic transactions \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1 >/dev/null

export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_TOPIC="transactions"
export KAFKA_GROUP_ID="fraud-consumer-group"
export MYSQL_HOST="localhost"
export MYSQL_PORT="3306"
export MYSQL_DATABASE="fraud_db"
export MYSQL_USER="fraud_user"
export MYSQL_PASSWORD="fraud_pass"
export MODEL_PATH="artifacts/fraud_model.pkl"
export PREPROCESSOR_PATH="artifacts/preprocessor.pkl"

python wait_for_services.py --kafka-host localhost --kafka-port 9092 --mysql-host localhost --mysql-port 3306

stop_if_running() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      sleep 1
    fi
    rm -f "$pid_file"
  fi
}

stop_if_running "$RUN_DIR/api.pid"
stop_if_running "$RUN_DIR/consumer.pid"
stop_if_running "$RUN_DIR/producer.pid"

echo "Starting FastAPI..."
nohup "$VENV_DIR/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 > "$LOG_DIR/api.log" 2>&1 &
echo $! > "$RUN_DIR/api.pid"

echo "Starting Kafka consumer..."
nohup "$VENV_DIR/bin/python" consumer.py > "$LOG_DIR/consumer.log" 2>&1 &
echo $! > "$RUN_DIR/consumer.pid"

sleep 5

echo "Starting Kafka producer..."
nohup "$VENV_DIR/bin/python" producer.py --bootstrap-servers localhost:9092 > "$LOG_DIR/producer.log" 2>&1 &
echo $! > "$RUN_DIR/producer.pid"

cat <<EOF

Local fraud detection demo is starting.

Website:  http://localhost:8000
API Docs: http://localhost:8000/docs

Useful logs:
  tail -f "$LOG_DIR/api.log"
  tail -f "$LOG_DIR/consumer.log"
  tail -f "$LOG_DIR/producer.log"

To stop the local background processes later:
  bash "$ROOT_DIR/scripts/stop_local_macos.sh"
EOF
