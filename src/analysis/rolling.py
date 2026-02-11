"""Rolling-window spectral analysis for time-varying Koopman decomposition.

Trains independent models on overlapping sub-windows of the full time series
and tracks how eigenvalues, the spectral gap, and entropy production evolve
over time.  This is the primary tool for detecting structural breaks and
time-varying market dynamics.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .spectral import SpectralAnalyzer

logger = logging.getLogger(__name__)


class RollingSpectralAnalyzer:
    """Rolling-window Koopman spectral analysis.

    Parameters
    ----------
    model_class : type
        The model class to instantiate for each window (e.g.
        ``NonEquilibriumVAMPNet``).  Must accept ``**model_kwargs`` at
        construction time.
    model_kwargs : dict
        Keyword arguments forwarded to ``model_class(...)`` for every window.
    tau : float
        Physical lag time (in time-step units, typically trading days).
    window_size : int
        Number of time steps in each rolling window.
    stride : int
        Step size between consecutive window starts.
    """

    def __init__(
        self,
        model_class: Type[torch.nn.Module],
        model_kwargs: Dict[str, Any],
        tau: float,
        window_size: int = 500,
        stride: int = 5,
    ) -> None:
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.tau: float = tau
        self.window_size: int = window_size
        self.stride: int = stride

    # ------------------------------------------------------------------
    # Core rolling fit
    # ------------------------------------------------------------------

    def fit_rolling(
        self,
        data: torch.Tensor,
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        dates: Optional[pd.DatetimeIndex] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Train a fresh model per window and collect spectral summaries.

        Parameters
        ----------
        data : torch.Tensor
            Full trajectory of shape ``(T, d)``.
        n_epochs : int
            Training epochs per window.
        learning_rate : float
            Optimizer learning rate for each window model.
        dates : pd.DatetimeIndex or None
            Optional date index aligned with *data*.  If provided, results
            are keyed by the centre date of each window.
        device : torch.device or None
            Device for training.  Defaults to CUDA if available.

        Returns
        -------
        dict
            ``eigenvalues``          -- list of complex eigenvalue arrays, one
                                        per window.
            ``spectral_gaps``        -- list of float spectral gaps.
            ``entropy_productions``  -- list of per-mode entropy arrays.
            ``window_centres``       -- list of int (or Timestamp) window
                                        centre indices.
            ``singular_values``      -- list of singular value arrays.
            ``decay_rates``          -- list of decay-rate arrays.
            ``frequencies``          -- list of frequency arrays.
            ``relaxation_times``     -- list of relaxation-time arrays.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        T = data.shape[0]
        window_starts = list(range(0, T - self.window_size + 1, self.stride))

        results: Dict[str, List[Any]] = {
            "eigenvalues": [],
            "spectral_gaps": [],
            "entropy_productions": [],
            "window_centres": [],
            "singular_values": [],
            "decay_rates": [],
            "frequencies": [],
            "relaxation_times": [],
        }

        logger.info(
            "Rolling analysis: %d windows (size=%d, stride=%d, T=%d)",
            len(window_starts),
            self.window_size,
            self.stride,
            T,
        )

        for start in tqdm(window_starts, desc="Rolling spectral analysis"):
            end = start + self.window_size
            window_data = data[start:end].to(device)

            centre_idx = start + self.window_size // 2
            if dates is not None:
                centre = dates[centre_idx]
            else:
                centre = centre_idx

            # --- Train a fresh model on this window ---
            model = self.model_class(**self.model_kwargs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            tau_int = int(self.tau)
            x_t = window_data[:-tau_int]
            x_t_tau = window_data[tau_int:]

            model.train()
            for _epoch in range(n_epochs):
                optimizer.zero_grad()
                loss = model(x_t, x_t_tau)
                loss.backward()
                optimizer.step()

            # --- Spectral analysis ---
            analyzer = SpectralAnalyzer(model, self.tau)
            spec = analyzer.analyze(window_data.cpu())

            # --- Entropy production per mode ---
            ep = self._estimate_mode_entropy(spec["eigenvalues"])

            results["eigenvalues"].append(spec["eigenvalues"])
            results["spectral_gaps"].append(spec["spectral_gap"])
            results["entropy_productions"].append(ep)
            results["window_centres"].append(centre)
            results["singular_values"].append(spec["singular_values"])
            results["decay_rates"].append(spec["decay_rates"])
            results["frequencies"].append(spec["frequencies"])
            results["relaxation_times"].append(spec["relaxation_times"])

        logger.info("Rolling analysis complete: %d windows processed", len(window_starts))
        return results

    # ------------------------------------------------------------------
    # Convenience time-series extractors
    # ------------------------------------------------------------------

    @staticmethod
    def spectral_gap_timeseries(results: Dict[str, Any]) -> pd.Series:
        """Extract the spectral gap as a pandas Series.

        Parameters
        ----------
        results : dict
            Output of :meth:`fit_rolling`.

        Returns
        -------
        pd.Series
            Spectral gap indexed by window centre (date or integer).
        """
        return pd.Series(
            results["spectral_gaps"],
            index=results["window_centres"],
            name="spectral_gap",
        )

    @staticmethod
    def entropy_timeseries(results: Dict[str, Any]) -> pd.DataFrame:
        """Extract mode-resolved entropy production as a DataFrame.

        Parameters
        ----------
        results : dict
            Output of :meth:`fit_rolling`.

        Returns
        -------
        pd.DataFrame
            Columns are mode indices; rows are indexed by window centre.
        """
        max_modes = max(len(ep) for ep in results["entropy_productions"])
        padded = []
        for ep in results["entropy_productions"]:
            row = np.full(max_modes, np.nan)
            row[: len(ep)] = ep
            padded.append(row)

        columns = [f"mode_{k}" for k in range(max_modes)]
        return pd.DataFrame(
            padded,
            index=results["window_centres"],
            columns=columns,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_mode_entropy(eigenvalues: np.ndarray) -> np.ndarray:
        """Estimate per-mode entropy production from eigenvalues.

        For a Koopman eigenvalue ``lambda_k``, detailed balance implies
        ``lambda_k`` is real and positive.  Departure from this --
        measured as ``|Im(lambda_k)| / |lambda_k|`` -- quantifies the
        contribution of mode *k* to time-reversal asymmetry (entropy
        production).

        A more rigorous estimator (following the VAMP-based entropy
        decomposition in the model module) would require the full
        singular value pairs; this lightweight proxy suffices for
        rolling monitoring.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Complex eigenvalue array.

        Returns
        -------
        np.ndarray
            Non-negative entropy proxy per mode.
        """
        magnitudes = np.abs(eigenvalues)
        # Avoid division by zero for vanishing eigenvalues
        safe_mag = np.where(magnitudes > 1e-15, magnitudes, 1e-15)

        # Entropy proxy: departure from the real positive axis
        # sigma_k = |Im(lambda_k)| / |lambda_k|  +  max(0, -Re(lambda_k)) / |lambda_k|
        imag_part = np.abs(eigenvalues.imag) / safe_mag
        neg_real_part = np.clip(-eigenvalues.real, 0, None) / safe_mag
        entropy_proxy = imag_part + neg_real_part

        return entropy_proxy
