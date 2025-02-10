"""
Statistical anomaly detectors: Z-score, IQR, and rolling MAD.
Fast, interpretable baselines that need no training data.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ZScoreDetector:
    """Flag observations more than `threshold` standard deviations from rolling mean."""

    def __init__(self, window: int = 50, threshold: float = 3.0):
        self.window = window
        self.threshold = threshold

    def detect(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = pd.Series(series)
        rol_mean = s.rolling(self.window, min_periods=1, center=True).mean()
        rol_std  = s.rolling(self.window, min_periods=1, center=True).std().fillna(1e-6)
        scores = np.abs((s - rol_mean) / rol_std).values
        return scores, (scores > self.threshold).astype(int)


class IQRDetector:
    """
    Rolling IQR detector.
    Anomaly when point falls outside [Q1 - k*IQR, Q3 + k*IQR].
    """

    def __init__(self, window: int = 100, k: float = 1.5):
        self.window = window
        self.k = k

    def detect(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = pd.Series(series)
        q1 = s.rolling(self.window, min_periods=10).quantile(0.25)
        q3 = s.rolling(self.window, min_periods=10).quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - self.k * iqr, q3 + self.k * iqr
        anomaly = ((s < lower) | (s > upper)).fillna(False).astype(int).values
        distance = np.maximum(lower - s, s - upper).clip(lower=0).values
        return distance.astype(float), anomaly


class RollingMADDetector:
    """
    Median Absolute Deviation (MAD) detector — robust to existing outliers.
    Uses rolling window for non-stationarity adaptation.
    """

    def __init__(self, window: int = 50, threshold: float = 3.5):
        self.window = window
        self.threshold = threshold

    def detect(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = pd.Series(series)
        rol_median = s.rolling(self.window, min_periods=1, center=True).median()
        residuals = (s - rol_median).abs()
        rol_mad = residuals.rolling(self.window, min_periods=1, center=True).median().replace(0, 1e-6)
        # Modified Z-score (Iglewicz & Hoaglin 1993)
        scores = (0.6745 * residuals / rol_mad).values
        return scores, (scores > self.threshold).astype(int)
