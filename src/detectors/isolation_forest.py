"""Isolation Forest anomaly detector for multivariate time series."""
import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest on sliding-window feature vectors.
    Each window is summarized to (mean, std, min, max, range) per channel.
    """

    def __init__(self, config: dict):
        cfg = config.get("isolation_forest", {})
        self.n_estimators   = cfg.get("n_estimators", 200)
        self.contamination  = cfg.get("contamination", 0.05)
        self.window         = config["detection"].get("seq_len", 20)
        self._model         = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()

    def _featurize(self, X: np.ndarray) -> np.ndarray:
        """Convert each window into a flat feature vector."""
        features = []
        for i in range(len(X) - self.window + 1):
            w = X[i : i + self.window]
            feat = np.concatenate([
                w.mean(axis=0),
                w.std(axis=0),
                w.min(axis=0),
                w.max(axis=0),
                w.max(axis=0) - w.min(axis=0),
            ])
            features.append(feat)
        return np.array(features)

    def fit(self, X_normal: np.ndarray) -> "IsolationForestDetector":
        feats = self._featurize(X_normal)
        feats_scaled = self._scaler.fit_transform(feats)
        self._model.fit(feats_scaled)
        logger.info(f"IsolationForest fitted on {len(feats)} windows")
        return self

    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feats = self._featurize(X)
        feats_scaled = self._scaler.transform(feats)
        # IF scores: negative = more anomalous
        scores = -self._model.score_samples(feats_scaled)
        labels = (self._model.predict(feats_scaled) == -1).astype(int)

        # Map back to original time axis
        full_scores = np.zeros(len(X))
        full_labels = np.zeros(len(X), dtype=int)
        for i, (s, l) in enumerate(zip(scores, labels)):
            idx = i + self.window // 2
            if full_scores[idx] < s:
                full_scores[idx] = s
                full_labels[idx] = l
        return full_scores, full_labels

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler}, f)

    def load(self, path: str) -> "IsolationForestDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model  = data["model"]
        self._scaler = data["scaler"]
        return self
