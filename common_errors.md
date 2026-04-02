# Common Errors I Faced

## 1. Kafka connection refused

Problem:

- `NoBrokersAvailable` from `kafka-python`

Why it happened:

- Kafka container was still starting up, but I launched `producer.py` too quickly
- Sometimes I used `localhost:9092` instead of `localhost:29092` from the host machine

What fixed it:

- waited 15 to 20 seconds after `docker compose up -d`
- used `localhost:29092` on the host
- used `kafka:9092` only from inside Docker containers

## 2. SMOTE memory error on the full IEEE-CIS dataset

Problem:

- my laptop started freezing or Python got killed while SMOTE was creating synthetic samples

Why it happened:

- the IEEE-CIS dataset is big, and the `V` features make the matrix really wide
- doing full oversampling on a student laptop can get heavy fast

What fixed it:

- I added `--sample-rows` in `train.py` for smaller local experiments
- I also converted numeric columns to `float32` to reduce memory

## 3. Consumer kept replaying old messages

Problem:

- every time I restarted the consumer, it would process the same transactions again

Why it happened:

- I forgot that Kafka offsets are separate from database writes
- I had auto-commit behavior wrong while testing

What fixed it:

- disabled auto-commit
- committed the offset only after the MySQL insert succeeded

## 4. Prediction script crashed because training artifacts were missing

Problem:

- FastAPI or the consumer failed because `fraud_model.pkl` or `preprocessor.pkl` did not exist yet

Why it happened:

- I started Docker before running `train.py`

What fixed it:

- ran `python train.py --train-csv data/train_transaction.csv` first
- checked that the files were created inside `artifacts/`
