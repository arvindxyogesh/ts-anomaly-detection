# Time Series Anomaly Detection

A production-grade anomaly detection system for multivariate sensor streams, implementing and comparing **four detection approaches**: LSTM Autoencoder, Isolation Forest, Z-Score, and Rolling MAD (Median Absolute Deviation).

## Architecture

```
ts-anomaly-detection/
├── config/config.yaml              # All detector hyper-parameters
├── data/generate_data.py           # Industrial IoT sensor data (3 channels, injected anomalies)
├── src/
│   ├── detectors/
│   │   ├── autoencoder.py          # LSTM Autoencoder: reconstruction error threshold
│   │   ├── isolation_forest.py     # IF on sliding-window feature vectors
│   │   └── statistical.py          # Z-Score, IQR, Rolling MAD detectors
│   └── evaluation/
│       └── metrics.py              # Precision, Recall, F1, ROC-AUC, PR-AUC
├── scripts/
│   ├── train.py                    # Train all detectors + full comparison table
│   └── detect.py                   # Run detection on new data, output anomaly timestamps
├── artifacts/                      # Saved detector models
└── reports/                        # detection_report.json
```

## Anomaly Types in Dataset

| Type | Description | Sensor |
|---|---|---|
| **Point spike** | Sudden large deviation (4–8σ) | All channels |
| **Level shift** | Mean offset lasting 10–30 steps | All channels |
| **Slow drift** | Gradual ramp over 20–50 steps | All channels |

## Detection Methods

| Method | Type | Training required? | Multivariate? |
|---|---|---|---|
| LSTM Autoencoder | Deep learning | Yes (normal data) | Yes |
| Isolation Forest | Ensemble ML | Yes (normal data) | Yes |
| Z-Score | Statistical | No | Single channel |
| Rolling MAD | Statistical | No | Single channel |

## Quick Start

```bash
pip install -r requirements.txt

# Train all detectors and print comparison
python scripts/train.py

# Detect on existing data
python scripts/detect.py
```

## LSTM Autoencoder Design

```
Input window (seq_len × n_features)
    │
    ▼
LSTM Encoder → latent vector (hidden_size,)
    │
    ▼
LSTM Decoder → reconstructed window
    │
    ▼
MSE reconstruction error
    │
    ▼
Threshold = mean(train_errors) + k × std(train_errors)
Anomaly if error > threshold
```

**Key design choices:**
- Trained **only on normal data** — learns the manifold of typical behaviour
- Threshold calibrated by a configurable `threshold_sigma` (default 3σ)
- Per-timestep scores computed by averaging overlapping window errors

## Running Tests

```bash
pytest tests/ -v
```
