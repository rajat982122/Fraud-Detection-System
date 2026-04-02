CREATE DATABASE IF NOT EXISTS fraud_db;

CREATE USER IF NOT EXISTS 'fraud_user'@'%' IDENTIFIED BY 'fraud_pass';
GRANT ALL PRIVILEGES ON fraud_db.* TO 'fraud_user'@'%';
FLUSH PRIVILEGES;

USE fraud_db;

CREATE TABLE IF NOT EXISTS fraud_predictions (
    id BIGINT NOT NULL AUTO_INCREMENT,
    txn_id BIGINT NOT NULL,
    amount DECIMAL(18, 4) NOT NULL,
    is_fraud TINYINT(1) NOT NULL,
    fraud_prob DECIMAL(10, 6) NOT NULL,
    processed_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    INDEX idx_txn_id (txn_id),
    INDEX idx_processed_at (processed_at),
    INDEX idx_is_fraud (is_fraud)
);
