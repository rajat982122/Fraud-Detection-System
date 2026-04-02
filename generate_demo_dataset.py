from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic fraud dataset so the project runs without Kaggle downloads."
    )
    parser.add_argument(
        "--output",
        default="data/demo_train_transaction.csv",
        help="Where to save the generated CSV.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=12000,
        help="Number of synthetic rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the CSV even if it already exists.",
    )
    return parser.parse_args()


def build_dataset(row_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    product_choices = np.array(["W", "C", "H", "S", "R"])
    product_probs = np.array([0.48, 0.18, 0.14, 0.11, 0.09])

    card4_choices = np.array(["visa", "mastercard", "discover", "american express"])
    card4_probs = np.array([0.50, 0.28, 0.15, 0.07])

    card6_choices = np.array(["debit", "credit", "charge card"])
    card6_probs = np.array([0.62, 0.33, 0.05])

    email_choices = np.array(
        [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "icloud.com",
            "anonymous.com",
            "protonmail.com",
        ]
    )
    email_probs = np.array([0.34, 0.20, 0.14, 0.13, 0.10, 0.05, 0.04])

    transaction_ids = np.arange(100000, 100000 + row_count)
    transaction_amt = np.clip(rng.lognormal(mean=4.2, sigma=0.85, size=row_count), 1, 8000)
    product_cd = rng.choice(product_choices, size=row_count, p=product_probs)
    card1 = rng.integers(1000, 20000, size=row_count)
    card2 = rng.integers(100, 600, size=row_count)
    card3 = rng.choice([150, 185, 187, 188, 189, 190], size=row_count)
    card4 = rng.choice(card4_choices, size=row_count, p=card4_probs)
    card5 = rng.integers(100, 300, size=row_count)
    card6 = rng.choice(card6_choices, size=row_count, p=card6_probs)
    p_email = rng.choice(email_choices, size=row_count, p=email_probs)
    r_email = rng.choice(email_choices, size=row_count, p=email_probs)

    data: dict[str, np.ndarray] = {
        "TransactionID": transaction_ids,
        "TransactionAmt": transaction_amt.round(2),
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
    }

    base_signal = (
        0.0025 * transaction_amt
        + 0.95 * np.isin(product_cd, ["C", "W"]).astype(float)
        + 1.30 * np.isin(card4, ["discover", "american express"]).astype(float)
        + 1.10 * np.isin(card6, ["credit", "charge card"]).astype(float)
        + 0.85 * np.isin(p_email, ["anonymous.com", "protonmail.com"]).astype(float)
        + 0.55 * (p_email != r_email).astype(float)
        + 0.00008 * np.abs(card1 - 11000)
        + rng.normal(0, 0.55, size=row_count)
    )

    for feature_index in range(1, 31):
        noise = rng.normal(0, 1, size=row_count)
        direction = 1 if feature_index % 3 != 0 else -1
        v_feature = (
            noise
            + direction * base_signal * rng.uniform(0.35, 0.75)
            + (transaction_amt / 1000.0) * rng.uniform(0.02, 0.08)
        )
        data[f"V{feature_index}"] = v_feature.astype("float32")

    score = (
        base_signal
        + 0.90 * (data["V3"] < -1.0).astype(float)
        + 0.80 * (data["V7"] > 2.0).astype(float)
        + 0.65 * (data["V14"] > 1.5).astype(float)
        + 0.60 * (data["V21"] < -1.2).astype(float)
    )

    fraud_count = max(int(row_count * 0.035), 1)
    fraud_indices = np.argsort(score)[-fraud_count:]
    is_fraud = np.zeros(row_count, dtype=int)
    is_fraud[fraud_indices] = 1

    for feature_index in range(1, 31):
        shift = rng.normal(1.8, 0.35, size=row_count) * is_fraud
        if feature_index % 3 == 0:
            data[f"V{feature_index}"] = data[f"V{feature_index}"] - shift
        else:
            data[f"V{feature_index}"] = data[f"V{feature_index}"] + shift

    data["TransactionAmt"] = (
        data["TransactionAmt"] + is_fraud * rng.normal(150, 45, size=row_count)
    ).clip(1, 9000).round(2)
    data["isFraud"] = is_fraud

    frame = pd.DataFrame(data)

    for column in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]:
        missing_mask = rng.random(row_count) < 0.04
        frame.loc[missing_mask, column] = np.nan

    for column in [f"V{index}" for index in range(1, 31)]:
        missing_mask = rng.random(row_count) < 0.02
        frame.loc[missing_mask, column] = np.nan

    return frame


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        print(f"Demo dataset already exists at {output_path}. Skipping generation.")
        return

    frame = build_dataset(row_count=args.rows, seed=args.seed)
    frame.to_csv(output_path, index=False)

    fraud_rate = frame["isFraud"].mean() * 100
    print(f"Saved demo dataset to {output_path}")
    print(f"Rows: {len(frame):,} | Fraud rate: {fraud_rate:.2f}%")


if __name__ == "__main__":
    main()
