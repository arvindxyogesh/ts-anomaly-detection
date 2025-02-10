"""
Detect anomalies on a given dataset using all trained models.
Outputs timestamped anomaly events to reports/anomalies.csv.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["path"], index_col=0, parse_dates=True)
    feature_cols = config["data"]["feature_cols"]
    X = df[feature_cols].values.astype(np.float32)

    all_labels = {}

    if os.path.exists("artifacts/autoencoder.pt"):
        from src.detectors.autoencoder import AutoencoderDetector
        ae = AutoencoderDetector(config).load("artifacts/autoencoder.pt")
        _, labels = ae.detect(X)
        all_labels["lstm_ae"] = labels
        logger.info(f"LSTM-AE detected {labels.sum()} anomalies")

    if os.path.exists("artifacts/isolation_forest.pkl"):
        from src.detectors.isolation_forest import IsolationForestDetector
        ifd = IsolationForestDetector(config).load("artifacts/isolation_forest.pkl")
        _, labels = ifd.detect(X)
        all_labels["isolation_forest"] = labels
        logger.info(f"IsolationForest detected {labels.sum()} anomalies")

    # Ensemble: flag if majority of detectors agree
    if all_labels:
        stack = np.stack(list(all_labels.values()), axis=1)
        ensemble = (stack.mean(axis=1) >= 0.5).astype(int)
        all_labels["ensemble"] = ensemble

        result_df = pd.DataFrame(all_labels, index=df.index)
        anomaly_timestamps = result_df[result_df["ensemble"] == 1]

        os.makedirs("reports", exist_ok=True)
        anomaly_timestamps.to_csv("reports/anomalies.csv")
        print(f"\nDetected {len(anomaly_timestamps)} anomaly time steps")
        print(f"Saved to reports/anomalies.csv")
        print(anomaly_timestamps.head(10))


if __name__ == "__main__":
    main()
