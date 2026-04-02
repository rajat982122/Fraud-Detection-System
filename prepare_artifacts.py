from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MODEL_PATH = Path("artifacts/fraud_model.pkl")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.pkl")
KAGGLE_DATASET_PATH = Path("data/train_transaction.csv")
DEMO_DATASET_PATH = Path("data/demo_train_transaction.csv")


def run_command(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        print("Model artifacts already exist. Skipping preparation.")
        return

    if KAGGLE_DATASET_PATH.exists():
        print("Found Kaggle IEEE-CIS dataset. Training with the real dataset.")
        dataset_path = KAGGLE_DATASET_PATH
    else:
        print("Kaggle dataset not found. Generating a demo dataset so the project can run immediately.")
        run_command([sys.executable, "generate_demo_dataset.py"])
        dataset_path = DEMO_DATASET_PATH

    run_command([sys.executable, "train.py", "--train-csv", str(dataset_path)])
    print("Artifacts are ready.")


if __name__ == "__main__":
    main()
