"""
Train anomaly detectors on normal sensor data and evaluate on full dataset.
"""
import json
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

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = config["data"]["path"]
    if not os.path.exists(data_path):
        logger.info("Generating synthetic sensor data...")
        from data.generate_data import generate_sensor_data
        generate_sensor_data(output_path=data_path)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    feature_cols = config["data"]["feature_cols"]
    label_col    = config["data"]["label_col"]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)

    # Train only on clean data (no anomalies), test on full series
    train_ratio = config["data"]["train_ratio"]
    n_train = int(len(X) * train_ratio)
    X_train_clean = X[:n_train][y[:n_train] == 0]
    X_full, y_full = X, y

    logger.info(f"Train (clean): {len(X_train_clean)}  |  Test (full): {len(X_full)}  |  Anomaly rate: {y_full.mean():.2%}")

    from src.evaluation.metrics import evaluate_detector
    results = []
    os.makedirs("artifacts", exist_ok=True)

    # ── LSTM Autoencoder ───────────────────────────────────────────────────────
    logger.info("Training LSTM Autoencoder...")
    from src.detectors.autoencoder import AutoencoderDetector
    ae = AutoencoderDetector(config)
    ae.fit(X_train_clean)
    ae_scores, ae_labels = ae.detect(X_full)
    ae.save("artifacts/autoencoder.pt")
    ae_metrics = evaluate_detector(y_full, ae_labels, ae_scores, name="LSTM-AE")
    results.append(ae_metrics)
    logger.info(f"AE: {ae_metrics}")

    # ── Isolation Forest ───────────────────────────────────────────────────────
    logger.info("Training Isolation Forest...")
    from src.detectors.isolation_forest import IsolationForestDetector
    ifd = IsolationForestDetector(config)
    ifd.fit(X_train_clean)
    if_scores, if_labels = ifd.detect(X_full)
    ifd.save("artifacts/isolation_forest.pkl")
    if_metrics = evaluate_detector(y_full, if_labels, if_scores, name="IsolationForest")
    results.append(if_metrics)
    logger.info(f"IF: {if_metrics}")

    # ── Statistical (Z-Score on first feature) ────────────────────────────────
    from src.detectors.statistical import ZScoreDetector, RollingMADDetector
    zs = ZScoreDetector(window=config["statistical"]["zscore_window"], threshold=config["statistical"]["zscore_threshold"])
    zs_scores, zs_labels = zs.detect(X_full[:, 0])
    zs_metrics = evaluate_detector(y_full, zs_labels, zs_scores, name="Z-Score")
    results.append(zs_metrics)

    mad = RollingMADDetector(window=config["statistical"]["mad_window"], threshold=config["statistical"]["mad_threshold"])
    mad_scores, mad_labels = mad.detect(X_full[:, 0])
    mad_metrics = evaluate_detector(y_full, mad_labels, mad_scores, name="RollingMAD")
    results.append(mad_metrics)

    os.makedirs("reports", exist_ok=True)
    with open("reports/detection_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== ANOMALY DETECTION RESULTS ==========")
    print(f"{'Detector':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'PR-AUC':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['detector']:<20} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['pr_auc']:>8.4f}")
    print("\nReport → reports/detection_report.json")


if __name__ == "__main__":
    main()
