"""
Synthetic sensor / financial data generator with injected anomalies.
Generates a multivariate time series (temperature + pressure + vibration)
from an industrial IoT sensor array with realistic failure patterns.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def inject_anomalies(series: np.ndarray, rng: np.random.Generator, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject three types of anomalies:
      - Point spikes: sudden large deviations
      - Level shifts: sudden mean offset lasting several steps
      - Slow drifts: gradual growth over a window
    Returns (anomalous_series, binary_label_array).
    """
    labels = np.zeros(len(series), dtype=int)
    s = series.copy()

    n = len(s)
    n_point   = config.get("n_point", 15)
    n_shift   = config.get("n_shift", 4)
    n_drift   = config.get("n_drift", 3)
    std = np.std(series)

    # Point anomalies
    for idx in rng.choice(n, size=n_point, replace=False):
        s[idx] += rng.choice([-1, 1]) * rng.uniform(4, 8) * std
        labels[idx] = 1

    # Level shifts
    for idx in rng.choice(n // 2, size=n_shift, replace=False):
        duration = rng.integers(10, 30)
        end = min(idx + duration, n)
        shift = rng.choice([-1, 1]) * rng.uniform(2, 4) * std
        s[idx:end] += shift
        labels[idx:end] = 1

    # Slow drifts
    for idx in rng.choice(n // 3, size=n_drift, replace=False):
        duration = rng.integers(20, 50)
        end = min(idx + duration, n)
        drift = np.linspace(0, rng.uniform(3, 6) * std, end - idx)
        s[idx:end] += drift
        labels[idx:end] = 1

    return s, labels


def generate_sensor_data(
    n_steps: int = 2000,
    seed: int = 42,
    output_path: str = "data/sensor_data.csv",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # Temperature sensor
    temp_clean = (
        60
        + 5 * np.sin(2 * np.pi * t / 144)   # 144-step daily cycle
        + 2 * np.sin(2 * np.pi * t / 1008)  # weekly cycle
        + rng.normal(0, 1, n_steps)
    )

    # Pressure sensor (correlated with temperature)
    pressure_clean = (
        100
        + 0.4 * (temp_clean - 60)
        + 3 * np.sin(2 * np.pi * t / 144 + 0.5)
        + rng.normal(0, 0.8, n_steps)
    )

    # Vibration sensor (mostly noise with machine cycles)
    vibration_clean = (
        1.5
        + 0.3 * np.abs(np.sin(2 * np.pi * t / 48))
        + rng.normal(0, 0.2, n_steps)
    ).clip(min=0)

    anomaly_config = {"n_point": 20, "n_shift": 5, "n_drift": 3}

    temp,     temp_labels     = inject_anomalies(temp_clean,     rng, anomaly_config)
    pressure, pressure_labels = inject_anomalies(pressure_clean, rng, anomaly_config)
    vibration, vib_labels     = inject_anomalies(vibration_clean, rng, {"n_point": 25, "n_shift": 3, "n_drift": 2})

    # Union of anomaly labels across all sensors
    labels = ((temp_labels + pressure_labels + vib_labels) > 0).astype(int)

    idx = pd.date_range("2023-01-01", periods=n_steps, freq="10min")
    df = pd.DataFrame({
        "temperature": temp.round(3),
        "pressure":    pressure.round(3),
        "vibration":   vibration.round(4),
        "is_anomaly":  labels,
    }, index=idx)
    df.index.name = "timestamp"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Generated {n_steps} sensor readings → {output_path}")
    print(f"Anomaly rate: {labels.mean():.2%}")
    return df


if __name__ == "__main__":
    df = generate_sensor_data()
    print(df.head())
