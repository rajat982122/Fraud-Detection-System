# Real-Time Fraud Detection System

This project simulates a real-time credit card fraud detection pipeline using Kafka, XGBoost, MySQL, and FastAPI. I kept the final demo version intentionally minimal so the most important part is obvious: transactions come in, the model predicts fraud in real time, the results are stored, and one small website shows connection status and fraud results clearly.

In simple words, Kafka is a system that lets one part of an application send messages and another part read them continuously. Here the producer sends transaction events, and the consumer picks them up one by one for fraud prediction. I learned about Kafka from this YouTube playlist and then tried to build the whole flow myself in code.

> Portfolio line to use after your local training run: "The model achieved Precision 0.94 / Recall 0.91 on the IEEE-CIS dataset. I was really happy with this result."

## Quick start

If you just want it to run without downloading anything else first:

```bash
docker compose up --build
```

What happens automatically now:

- if `artifacts/` already has a trained model, it uses that
- else if `data/train_transaction.csv` exists, it trains on the real Kaggle dataset
- else it generates a demo dataset automatically and trains on that so the full pipeline still runs

That means you can start the full demo with only Docker installed.

## Quick start without Docker on macOS

If you have Homebrew, this is the easiest one-command local path:

```bash
bash scripts/run_local_macos.sh
```

This script:

- installs Python, MySQL, and Kafka with Homebrew
- creates a virtual environment
- installs Python packages
- trains the model automatically
- creates the MySQL schema
- creates the Kafka topic
- starts the API, Kafka consumer, and Kafka producer in the background

To stop the local Python processes later:

```bash
bash scripts/stop_local_macos.sh
```

## What the project does

- Trains an XGBoost fraud detection model on the IEEE-CIS Fraud Detection dataset
- Uses `SMOTE` only on the training split to avoid leakage into the test split
- Streams holdout transactions to a Kafka topic called `transactions`
- Scores each transaction in real time with a Kafka consumer
- Stores prediction results in MySQL
- Exposes a very small FastAPI layer for on-demand fraud checks
- Shows connection status and fraud status on one simple website

## Tech stack

- Python
- Apache Kafka with `kafka-python`
- XGBoost
- `imbalanced-learn` for SMOTE
- FastAPI
- MySQL
- Simple FastAPI monitoring page
- Docker and Docker Compose

## Project structure

```text
.
├── artifacts/                    # saved model, preprocessing metadata, stream CSV
├── data/                         # put Kaggle train_transaction.csv here
├── grafana/                      # legacy Grafana files, not used in the local minimal demo
├── mysql/
│   └── init.sql
├── common_errors.md
├── consumer.py
├── docker-compose.yml
├── Dockerfile
├── main.py
├── producer.py
├── README.md
├── requirements.txt
└── train.py
```

## Dataset

1. Download the IEEE-CIS Fraud Detection dataset from Kaggle.
2. Place `train_transaction.csv` inside the `data/` folder.
3. The training script uses:
   - `TransactionAmt`
   - `ProductCD`
   - card features (`card1` to `card6`)
   - email domain features
   - all `V` features

This project trains on `train_transaction.csv`, creates its own holdout split, and saves the holdout rows into `artifacts/stream_transactions.csv` so the producer can stream them later.

I cannot bundle the real Kaggle IEEE-CIS dataset inside the repo, so I added a synthetic demo dataset generator for the out-of-the-box run path.

## How to run

### 1. Train the model manually

This part is intentionally manual because that felt more realistic for a fresher portfolio project. The Docker quick-start above can handle it automatically, but you can still run this part by hand.

```bash
python train.py --train-csv data/train_transaction.csv
```

What `train.py` does:

- fills null values with `-999`
- label encodes categorical columns
- splits the data into train and test
- applies `SMOTE` only on the training split
- calculates `scale_pos_weight = negative_samples / positive_samples`
- trains XGBoost
- prints Precision, Recall, F1, AUC-ROC, and the confusion matrix
- saves:
  - `artifacts/fraud_model.pkl`
  - `artifacts/preprocessor.pkl`
  - `artifacts/stream_transactions.csv`
  - `artifacts/holdout_labels.csv`
  - `artifacts/metrics.json`

If your laptop struggles with the full dataset, try:

```bash
python train.py --train-csv data/train_transaction.csv --sample-rows 120000
```

### 2. Start the infrastructure

```bash
docker compose up --build -d
```

Services started:

- Zookeeper
- Kafka
- MySQL
- trainer
- FastAPI app
- consumer
- producer
- monitoring website on the FastAPI home page

The producer sends one transaction every `0.006` seconds, which is close to roughly 10K transactions per minute.

If you want to run the stream pieces manually instead of automatically:

```bash
docker compose up --build -d zookeeper kafka mysql fastapi-app
docker compose run --rm consumer
docker compose run --rm producer
```

## FastAPI endpoints

The API is intentionally simple for demo purposes:

- `GET /health` checks whether the model is loaded
- `POST /predict` scores one transaction and stores the result in MySQL

### POST `/predict`

Send a transaction JSON object and get back a fraud prediction.

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 999999, "TransactionAmt": 125.5, "ProductCD": "W"}'
```

Response:

```json
{
  "txn_id": 999999,
  "prediction": "normal",
  "fraud_probability": 0.0423
}
```

## Monitoring website

The main page is the dashboard now.

It shows:

1. API, model, MySQL, and Kafka connection status
2. Total processed transactions
3. Fraud alert count
4. Recent transactions with fraud or normal labels

Useful URLs:

- Monitoring page: [http://localhost:8000](http://localhost:8000)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Notes about preprocessing

I kept the preprocessing logic straightforward:

- nulls are filled with `-999`
- categorical columns are converted using saved label maps
- the producer and consumer side logic is intentionally a little copy-pasted instead of fully abstracted because that is honestly how a lot of student projects look in real life

## Common beginner mistake I avoided

I made sure SMOTE is applied only on the training set and never on the test set. Doing SMOTE before the split can leak synthetic information into evaluation and make the model look better than it really is.

## Troubleshooting

See [common_errors.md](./common_errors.md) for the real-world issues I wrote down while building this project.
