from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "fraud_model.pkl"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
STREAM_DATA_PATH = ARTIFACT_DIR / "stream_transactions.csv"
HOLDOUT_LABELS_PATH = ARTIFACT_DIR / "holdout_labels.csv"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

FILL_VALUE = -999
ID_COLUMN = "TransactionID"
TARGET_COLUMN = "isFraud"

BASE_FEATURES = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost fraud model.")
    parser.add_argument(
        "--train-csv",
        default="data/train_transaction.csv",
        help="Path to the Kaggle IEEE-CIS train_transaction.csv file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to keep for the holdout test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting, SMOTE, and XGBoost.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional stratified subset size for memory-limited laptops.",
    )
    return parser.parse_args()


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    v_columns = sorted(
        [column for column in frame.columns if column.startswith("V")],
        key=lambda name: int(name[1:]) if name[1:].isdigit() else name,
    )
    selected_columns = [column for column in BASE_FEATURES if column in frame.columns]
    selected_columns.extend(v_columns)
    return selected_columns


def maybe_take_sample(
    frame: pd.DataFrame,
    target_column: str,
    sample_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_rows is None or sample_rows >= len(frame):
        return frame

    sampled_frame, _ = train_test_split(
        frame,
        train_size=sample_rows,
        stratify=frame[target_column],
        random_state=random_state,
    )
    return sampled_frame.reset_index(drop=True)


def fit_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working_frame = frame[feature_columns].copy()
    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(working_frame[column])
        or pd.api.types.is_categorical_dtype(working_frame[column])
    ]

    encoders: dict[str, dict[str, Any]] = {}

    for column in feature_columns:
        if column in categorical_columns:
            encoded_series = working_frame.loc[:, column].fillna(str(FILL_VALUE)).astype(str)
            labels = sorted(encoded_series.unique().tolist())
            if "__UNKNOWN__" not in labels:
                labels.append("__UNKNOWN__")
            mapping = {label: index for index, label in enumerate(labels)}
            default_value = mapping["__UNKNOWN__"]
            working_frame.loc[:, column] = encoded_series.map(
                lambda value: mapping.get(value, default_value)
            )
            encoders[column] = {
                "mapping": mapping,
                "default_value": default_value,
            }
        else:
            working_frame.loc[:, column] = (
                pd.to_numeric(working_frame.loc[:, column], errors="coerce")
                .fillna(FILL_VALUE)
                .astype("float32")
            )

    processed_frame = working_frame.astype("float32")
    preprocessor = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "encoders": encoders,
        "fill_value": FILL_VALUE,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "threshold": 0.5,
    }
    return processed_frame, preprocessor


def transform_features(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = preprocessor["feature_columns"]
    categorical_columns = set(preprocessor["categorical_columns"])
    encoders = preprocessor["encoders"]
    fill_value = preprocessor["fill_value"]

    working_frame = frame[feature_columns].copy()

    for column in feature_columns:
        if column in categorical_columns:
            encoded_series = working_frame.loc[:, column].fillna(str(fill_value)).astype(str)
            mapping_info = encoders[column]
            mapping = mapping_info["mapping"]
            default_value = mapping_info["default_value"]
            working_frame.loc[:, column] = encoded_series.map(
                lambda value: mapping.get(value, default_value)
            )
        else:
            working_frame.loc[:, column] = (
                pd.to_numeric(working_frame.loc[:, column], errors="coerce")
                .fillna(fill_value)
                .astype("float32")
            )

    return working_frame.astype("float32")


def print_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities > threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }

    print("\nEvaluation Metrics")
    print("-" * 60)
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1 Score       : {metrics['f1_score']:.4f}")
    print(f"AUC-ROC        : {metrics['auc_roc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    print("-" * 60)

    return metrics


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_csv_path = Path(args.train_csv)
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {train_csv_path}. "
            "Download Kaggle's train_transaction.csv and place it in data/."
        )

    print(f"Loading dataset from {train_csv_path} ...")
    dataset = pd.read_csv(train_csv_path)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            "The dataset is missing the isFraud target column. "
            "Make sure you are using train_transaction.csv, not test_transaction.csv."
        )

    dataset = maybe_take_sample(
        dataset,
        target_column=TARGET_COLUMN,
        sample_rows=args.sample_rows,
        random_state=args.random_state,
    )

    feature_columns = select_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("No usable feature columns were found in the dataset.")

    print(f"Using {len(feature_columns)} features ({len([c for c in feature_columns if c.startswith('V')])} V-features included).")

    train_frame, test_frame = train_test_split(
        dataset,
        test_size=args.test_size,
        stratify=dataset[TARGET_COLUMN],
        random_state=args.random_state,
    )

    y_train = train_frame[TARGET_COLUMN].astype(int)
    y_test = test_frame[TARGET_COLUMN].astype(int)

    X_train, preprocessor = fit_preprocessor(train_frame, feature_columns)
    X_test = transform_features(test_frame, preprocessor)

    negative_samples = int((y_train == 0).sum())
    positive_samples = int((y_train == 1).sum())
    scale_pos_weight = negative_samples / max(positive_samples, 1)

    print(f"Training rows before SMOTE : {len(X_train):,}")
    print(f"Fraud rows before SMOTE    : {positive_samples:,}")
    print("Applying SMOTE only on the training split to avoid data leakage ...")

    smote = SMOTE(random_state=args.random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Training rows after SMOTE  : {len(X_train_resampled):,}")
    print(f"scale_pos_weight           : {scale_pos_weight:.2f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        learning_rate=0.05,
        n_estimators=450,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        gamma=0.1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        n_jobs=max((os.cpu_count() or 2) - 1, 1),
    )

    print("Training XGBoost model ...")
    model.fit(X_train_resampled, y_train_resampled)

    threshold = float(preprocessor["threshold"])
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = print_metrics(y_test, probabilities, threshold)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    stream_frame = test_frame[[ID_COLUMN] + feature_columns].copy()
    stream_frame.to_csv(STREAM_DATA_PATH, index=False)
    test_frame[[ID_COLUMN, TARGET_COLUMN]].to_csv(HOLDOUT_LABELS_PATH, index=False)

    metrics_payload = {
        **metrics,
        "threshold": threshold,
        "scale_pos_weight": float(scale_pos_weight),
        "feature_count": int(len(feature_columns)),
        "categorical_feature_count": int(len(preprocessor["categorical_columns"])),
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2))

    print(f"Saved model        : {MODEL_PATH}")
    print(f"Saved preprocessor : {PREPROCESSOR_PATH}")
    print(f"Saved stream CSV   : {STREAM_DATA_PATH}")
    print(f"Saved metrics JSON : {METRICS_PATH}")


if __name__ == "__main__":
    main()
