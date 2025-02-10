"""Tests for anomaly detection components."""
import numpy as np
import pytest


@pytest.fixture
def normal_data():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (500, 3)).astype(np.float32)


@pytest.fixture
def anomaly_data(normal_data):
    data = normal_data.copy()
    data[100:105] += 10
    return data


def test_zscore_detector(normal_data):
    from src.detectors.statistical import ZScoreDetector
    det = ZScoreDetector(window=30, threshold=3.0)
    scores, labels = det.detect(normal_data[:, 0])
    assert len(scores) == len(normal_data)
    assert set(np.unique(labels)).issubset({0, 1})


def test_mad_detector(anomaly_data):
    from src.detectors.statistical import RollingMADDetector
    det = RollingMADDetector(window=30, threshold=3.5)
    scores, labels = det.detect(anomaly_data[:, 0])
    assert labels[100:105].sum() >= 1, "Should detect injected spike"


def test_iqr_detector(normal_data):
    from src.detectors.statistical import IQRDetector
    det = IQRDetector(window=50, k=1.5)
    scores, labels = det.detect(normal_data[:, 0])
    assert len(labels) == len(normal_data)


def test_isolation_forest(normal_data):
    from src.detectors.isolation_forest import IsolationForestDetector
    config = {
        "isolation_forest": {"n_estimators": 10, "contamination": 0.05},
        "detection": {"seq_len": 10},
    }
    det = IsolationForestDetector(config)
    det.fit(normal_data[:300])
    scores, labels = det.detect(normal_data)
    assert len(labels) == len(normal_data)


def test_evaluation_metrics():
    from src.evaluation.metrics import evaluate_detector
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.9).astype(int)
    scores = rng.random(200)
    pred = (scores > 0.5).astype(int)
    m = evaluate_detector(y, pred, scores, "test")
    assert "f1" in m
    assert "pr_auc" in m
