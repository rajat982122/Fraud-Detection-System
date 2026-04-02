from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kafka import KafkaProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream holdout transactions into Kafka.")
    parser.add_argument(
        "--csv",
        default="artifacts/stream_transactions.csv",
        help="CSV file created by train.py that will be streamed to Kafka.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:29092",
        help="Kafka bootstrap servers.",
    )
    parser.add_argument(
        "--topic",
        default="transactions",
        help="Kafka topic name.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.006,
        help="Delay between transactions in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for demo runs.",
    )
    return parser.parse_args()


def preprocess_transaction_for_stream(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column, value in row.items():
        if value == "" or pd.isna(value):
            payload[column] = None
        elif isinstance(value, np.generic):
            payload[column] = value.item()
        else:
            payload[column] = value
    return payload


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run train.py first so the holdout CSV exists."
        )

    transactions = pd.read_csv(csv_path)
    if args.limit is not None:
        transactions = transactions.head(args.limit)

    producer = KafkaProducer(
        bootstrap_servers=[server.strip() for server in args.bootstrap_servers.split(",")],
        value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
        linger_ms=0,
    )

    try:
        for _, row in transactions.iterrows():
            payload = preprocess_transaction_for_stream(row)
            transaction_id = payload.get("TransactionID", "UNKNOWN")
            producer.send(args.topic, value=payload)
            producer.flush()
            print(f"Sent transaction ID: {transaction_id}")
            time.sleep(args.delay)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
