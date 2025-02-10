"""
LSTM Autoencoder for unsupervised time series anomaly detection.
Reconstruction error above a learned threshold → anomaly.
"""
import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # last layer's hidden state


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, seq_len: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Repeat latent vector across time axis
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z)
        return self.fc(out)   # (batch, seq_len, output_size)


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder: encode to fixed latent → reconstruct sequence."""

    def __init__(self, input_size: int, hidden_size: int, seq_len: int, num_layers: int = 1):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMDecoder(hidden_size, input_size, seq_len, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector:
    """Wraps LSTMAutoencoder for training, threshold calibration, and detection."""

    def __init__(self, config: dict):
        self.config = config["autoencoder"]
        self.seq_len = config["detection"]["seq_len"]
        self.threshold: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMAutoencoder(
            input_size=config["data"].get("n_features", 3),
            hidden_size=self.config["hidden_size"],
            seq_len=self.seq_len,
            num_layers=self.config.get("num_layers", 1),
        ).to(self.device)

    def _windows(self, X: np.ndarray) -> np.ndarray:
        """Slide a window over X returning (N, seq_len, features)."""
        return np.array([X[i : i + self.seq_len] for i in range(len(X) - self.seq_len + 1)], dtype=np.float32)

    def fit(self, X_normal: np.ndarray) -> "AutoencoderDetector":
        """Train on normal (anomaly-free) windows only."""
        windows = self._windows(X_normal)
        tensor = torch.from_numpy(windows)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.config.get("batch_size", 64), shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 1e-3))
        criterion = nn.MSELoss()
        epochs = self.config.get("epochs", 50)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                recon = self.model(x_batch)
                loss = criterion(recon, x_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x_batch)
            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}  loss={total_loss/len(tensor):.4f}")

        # Calibrate threshold: mean + k*std of reconstruction errors on training data
        train_errors = self._reconstruction_errors(X_normal)
        k = self.config.get("threshold_sigma", 3.0)
        self.threshold = float(np.mean(train_errors) + k * np.std(train_errors))
        logger.info(f"Anomaly threshold set to {self.threshold:.4f} ({k}σ)")
        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        windows = self._windows(X)
        tensor = torch.from_numpy(windows).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor).cpu().numpy()
        errors = np.mean((windows - recon) ** 2, axis=(1, 2))
        return errors

    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (anomaly_scores, binary_labels) for each time step."""
        errors = self._reconstruction_errors(X)
        # Map window errors back to original time axis (take max over overlapping windows)
        scores = np.zeros(len(X))
        counts = np.zeros(len(X))
        for i, err in enumerate(errors):
            scores[i : i + self.seq_len] += err
            counts[i : i + self.seq_len] += 1
        counts = np.maximum(counts, 1)
        scores /= counts
        labels = (scores > self.threshold).astype(int)
        return scores, labels

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "threshold": self.threshold}, path)

    def load(self, path: str) -> "AutoencoderDetector":
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state"])
        self.threshold = ckpt["threshold"]
        return self
