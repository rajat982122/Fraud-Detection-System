#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/run"

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      echo "Stopped process $pid from $pid_file"
    fi
    rm -f "$pid_file"
  fi
}

stop_pid_file "$RUN_DIR/api.pid"
stop_pid_file "$RUN_DIR/consumer.pid"
stop_pid_file "$RUN_DIR/producer.pid"

echo "Stopped local Python processes for the fraud detection demo."
echo "Homebrew services are still running. If you want to stop them too, run:"
echo "brew services stop kafka"
echo "brew services stop mysql"
