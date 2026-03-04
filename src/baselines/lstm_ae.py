"""LSTM autoencoder baseline for financial regime detection.

Trains an LSTM autoencoder on return sequences, then uses reconstruction
error as a regime indicator: high error = stress/crisis regime.
Provides a modern deep learning baseline for comparison with KTND.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM encoder-decoder for sequence reconstruction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder(x)
        latent = self.fc_latent(h[-1])  # (batch, latent_dim)
        dec_input = self.fc_decode(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input)
        return dec_out, latent


class LSTMBaselineDetector:
    """LSTM autoencoder regime detector.

    Parameters
    ----------
    sequence_length : int
        Length of input sequences for the autoencoder.
    hidden_dim : int
        LSTM hidden dimension.
    latent_dim : int
        Bottleneck dimension.
    n_epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    high_recon_percentile : float
        Percentile of training reconstruction error above which a
        time step is classified as "stress" (regime 1).
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        n_epochs: int = 100,
        batch_size: int = 256,
        high_recon_percentile: float = 80,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.high_recon_percentile = high_recon_percentile
        self.model_: Optional[LSTMAutoencoder] = None
        self.threshold_: Optional[float] = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(self, data: np.ndarray) -> np.ndarray:
        """Sliding window sequences from (T, d) array."""
        seqs = []
        for i in range(len(data) - self.sequence_length + 1):
            seqs.append(data[i: i + self.sequence_length])
        return np.array(seqs)

    def fit(self, train_returns: np.ndarray) -> "LSTMBaselineDetector":
        """Fit the LSTM autoencoder on training returns."""
        if train_returns.ndim == 1:
            train_returns = train_returns[:, np.newaxis]

        input_dim = train_returns.shape[1]
        train_seqs = self._make_sequences(train_returns)

        self.model_ = LSTMAutoencoder(
            input_dim, self.hidden_dim, self.latent_dim,
        ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_tensor = torch.tensor(train_seqs, dtype=torch.float32)
        train_ds = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
        )

        self.model_.train()
        for _ in range(self.n_epochs):
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device_)
                recon, _ = self.model_(batch_x)
                loss = criterion(recon, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Set threshold from training reconstruction error
        self.model_.eval()
        with torch.no_grad():
            recon_train, _ = self.model_(train_tensor.to(self.device_))
        train_errors = torch.mean(
            (recon_train.cpu() - train_tensor) ** 2, dim=(1, 2),
        ).numpy()
        self.threshold_ = float(
            np.percentile(train_errors, self.high_recon_percentile)
        )

        return self

    def detect_regimes(self, returns: np.ndarray) -> np.ndarray:
        """Detect regimes on a full return series.

        Returns
        -------
        labels : ndarray of shape (T,)
            0 = normal, 1 = stress.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if returns.ndim == 1:
            returns = returns[:, np.newaxis]

        seqs = self._make_sequences(returns)
        if len(seqs) == 0:
            return np.zeros(len(returns), dtype=int)

        seq_tensor = torch.tensor(seqs, dtype=torch.float32)

        self.model_.eval()
        # Batched inference to avoid OOM on large sequences
        batch_size = 1024
        errors_list = []
        with torch.no_grad():
            for i in range(0, len(seq_tensor), batch_size):
                batch = seq_tensor[i:i + batch_size].to(self.device_)
                recon, _ = self.model_(batch)
                err = torch.mean(
                    (recon.cpu() - seq_tensor[i:i + batch_size]) ** 2,
                    dim=(1, 2),
                ).numpy()
                errors_list.append(err)
        errors = np.concatenate(errors_list)

        seq_labels = (errors > self.threshold_).astype(int)

        # Pad beginning with 0s to match full series length
        pad_len = len(returns) - len(seq_labels)
        return np.concatenate([np.zeros(pad_len, dtype=int), seq_labels])
