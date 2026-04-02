from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pymysql
from kafka import KafkaConsumer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/fraud_model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-consumer-group")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "fraud_db")
MYSQL_USER = os.getenv("MYSQL_USER", "fraud_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "fraud_pass")


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run train.py first so fraud_model.pkl and preprocessor.pkl exist."
        )
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


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


def main() -> None:
    model, preprocessor = load_artifacts()
    threshold = float(preprocessor.get("threshold", 0.5))
    id_column = preprocessor.get("id_column", "TransactionID")

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[server.strip() for server in KAFKA_BOOTSTRAP_SERVERS.split(",")],
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=KAFKA_GROUP_ID,
        value_deserializer=lambda message: json.loads(message.decode("utf-8")),
    )

    connection = get_mysql_connection()

    print(f"Listening to Kafka topic '{KAFKA_TOPIC}' on {KAFKA_BOOTSTRAP_SERVERS} ...")

    try:
        for message in consumer:
            transaction = message.value
            feature_frame = preprocess_transaction(transaction, preprocessor)

            fraud_probability = float(model.predict_proba(feature_frame)[0][1])
            is_fraud = int(fraud_probability > threshold)
            txn_id = int(transaction.get(id_column, message.offset))
            amount = float(transaction.get("TransactionAmt") or 0.0)

            save_prediction(
                connection=connection,
                txn_id=txn_id,
                amount=amount,
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
            )

            print(
                f"Processed txn_id={txn_id} | amount={amount:.2f} | "
                f"is_fraud={is_fraud} | fraud_prob={fraud_probability:.4f}"
            )

            # this took me forever to figure out - you need to commit the offset manually
            # after the database insert succeeds, otherwise Kafka will replay old messages.
            consumer.commit()
    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")
    finally:
        connection.close()
        consumer.close()


if __name__ == "__main__":
    main()
