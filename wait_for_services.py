from __future__ import annotations

import argparse
import socket
import time

import pymysql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for Kafka and MySQL to become reachable.")
    parser.add_argument("--kafka-host", default=None)
    parser.add_argument("--kafka-port", type=int, default=9092)
    parser.add_argument("--mysql-host", default=None)
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument("--mysql-user", default="fraud_user")
    parser.add_argument("--mysql-password", default="fraud_pass")
    parser.add_argument("--mysql-database", default="fraud_db")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--delay", type=int, default=0)
    return parser.parse_args()


def wait_for_socket(host: str, port: int, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"Connected to {host}:{port}")
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def wait_for_mysql(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    timeout_seconds: int,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            connection = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connect_timeout=3,
            )
            connection.close()
            print(f"MySQL is ready on {host}:{port}")
            return
        except pymysql.MySQLError:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for MySQL at {host}:{port}")


def main() -> None:
    args = parse_args()

    if args.kafka_host:
        wait_for_socket(args.kafka_host, args.kafka_port, args.timeout)

    if args.mysql_host:
        wait_for_mysql(
            host=args.mysql_host,
            port=args.mysql_port,
            user=args.mysql_user,
            password=args.mysql_password,
            database=args.mysql_database,
            timeout_seconds=args.timeout,
        )

    if args.delay > 0:
        print(f"Waiting an extra {args.delay} seconds before starting.")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
