"""Spectral analysis of Koopman operators for financial time series.

Extracts eigenvalues, eigenfunctions, singular values, spectral gaps,
decay rates, oscillation frequencies, and relaxation times from trained
VAMPNet models.  Provides visualisation of the eigenvalue spectrum on the
complex plane and utilities for sorting modes by physical timescale.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """Post-training spectral analysis of a Koopman / VAMPNet model.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model exposing a ``transform`` method that maps raw
        observations to the learned feature (eigenfunction) space.  Expected
        signature: ``model.transform(data: torch.Tensor) -> torch.Tensor``.
    tau : float
        Physical lag time used during training (in the same time units as the
        data -- typically trading days).
    """

    def __init__(self, model: torch.nn.Module, tau: float) -> None:
        self.model = model
        self.tau: float = tau

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, data: torch.Tensor) -> Dict[str, Any]:
        """Run full spectral decomposition on *data*.

        The method constructs time-lagged pairs ``(x_t, x_{t+tau})`` from the
        supplied trajectory, transforms them through the learned lobe
        networks, estimates covariance matrices, and performs the generalised
        eigenvalue / SVD analysis that yields the Koopman spectrum.

        Parameters
        ----------
        data : torch.Tensor
            Trajectory tensor of shape ``(T, d)`` where *T* is the number of
            time steps and *d* the observable dimensionality.

        Returns
        -------
        dict
            ``eigenvalues``      -- complex Koopman eigenvalues ``lambda_k``.
            ``eigenfunctions``   -- projected eigenfunction values ``(T-tau, K)``.
            ``singular_values``  -- real SVD singular values.
            ``spectral_gap``     -- gap between 1st and 2nd largest |lambda_k|.
            ``decay_rates``      -- ``gamma_k = -ln|lambda_k| / tau``.
            ``frequencies``      -- ``omega_k = angle(lambda_k) / tau``.
            ``relaxation_times`` -- ``1 / gamma_k`` (inf for |lambda_k|=1).
        """
        self.model.eval()
        tau_int = int(self.tau)

        # Build time-lagged pairs
        x_t = data[:-tau_int]
        x_t_tau = data[tau_int:]

        with torch.no_grad():
            chi_t = self.model.transform(x_t)       # (N, K)
            chi_t_tau = self.model.transform(x_t_tau)  # (N, K)

        chi_t_np: np.ndarray = chi_t.cpu().numpy()
        chi_t_tau_np: np.ndarray = chi_t_tau.cpu().numpy()

        # Covariance matrices (VAMP formulation)
        n_samples = chi_t_np.shape[0]
        mean_t = chi_t_np.mean(axis=0, keepdims=True)
        mean_t_tau = chi_t_tau_np.mean(axis=0, keepdims=True)

        chi_t_centered = chi_t_np - mean_t
        chi_t_tau_centered = chi_t_tau_np - mean_t_tau

        C_00 = (chi_t_centered.T @ chi_t_centered) / n_samples
        C_11 = (chi_t_tau_centered.T @ chi_t_tau_centered) / n_samples
        C_01 = (chi_t_centered.T @ chi_t_tau_centered) / n_samples

        # Regularise for numerical stability
        reg = 1e-6 * np.eye(C_00.shape[0])
        C_00_reg = C_00 + reg
        C_11_reg = C_11 + reg

        # SVD-based Koopman matrix estimation
        # K = C_00^{-1/2} C_01 C_11^{-1/2}  (symmetric form for SVD)
        # Then eigendecompose the Koopman matrix  K_full = C_00^{-1} C_01.
        L_00 = np.linalg.cholesky(C_00_reg)
        L_11 = np.linalg.cholesky(C_11_reg)
        L_00_inv = np.linalg.inv(L_00)
        L_11_inv = np.linalg.inv(L_11)

        # Koopman matrix in the original basis
        K_matrix = np.linalg.solve(C_00_reg, C_01)

        # SVD of the whitened cross-correlation for singular values
        M = L_00_inv @ C_01 @ L_11_inv.T
        U, singular_values, Vt = np.linalg.svd(M, full_matrices=False)

        # Eigendecomposition of K for complex eigenvalues
        # Sanitize K to prevent MKL SGEBAL segfaults on ill-conditioned matrices
        if not np.isfinite(K_matrix).all():
            K_matrix = np.where(np.isfinite(K_matrix), K_matrix, 0.0)
        eigenvalues_complex = np.linalg.eigvals(K_matrix)

        # Sort by magnitude descending
        order = np.argsort(-np.abs(eigenvalues_complex))
        eigenvalues = eigenvalues_complex[order]
        singular_values = np.sort(singular_values)[::-1]

        # Derived spectral quantities
        magnitudes = np.abs(eigenvalues)
        spectral_gap = float(magnitudes[0] - magnitudes[1]) if len(magnitudes) > 1 else 0.0

        # Decay rates: gamma_k = -ln|lambda_k| / tau
        with np.errstate(divide="ignore"):
            decay_rates = -np.log(np.clip(magnitudes, 1e-15, None)) / self.tau

        # Oscillation frequencies: omega_k = angle(lambda_k) / tau
        frequencies = np.angle(eigenvalues) / self.tau

        # Relaxation times: t_k = 1 / gamma_k
        with np.errstate(divide="ignore"):
            relaxation_times = np.where(
                decay_rates > 1e-15, 1.0 / decay_rates, np.inf
            )

        logger.info(
            "Spectral analysis: %d modes, spectral_gap=%.4f, "
            "leading relaxation_time=%.2f",
            len(eigenvalues),
            spectral_gap,
            relaxation_times[0] if len(relaxation_times) > 0 else float("nan"),
        )

        return {
            "eigenvalues": eigenvalues,
            "eigenfunctions": chi_t_np,
            "singular_values": singular_values,
            "spectral_gap": spectral_gap,
            "decay_rates": decay_rates,
            "frequencies": frequencies,
            "relaxation_times": relaxation_times,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def plot_eigenvalue_plane(
        eigenvalues: np.ndarray,
        *,
        title: str = "Koopman Eigenvalue Spectrum",
        figsize: Tuple[int, int] = (7, 7),
        annotate_top_k: int = 5,
    ) -> Figure:
        """Plot eigenvalues on the complex plane with the unit circle.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Complex eigenvalue array.
        title : str
            Plot title.
        figsize : tuple of int
            Figure dimensions in inches.
        annotate_top_k : int
            Number of leading eigenvalues to annotate with their index.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered figure (call ``plt.show()`` or ``fig.savefig(...)``
            to display / persist).
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.5, label="|$\\lambda$|=1")

        # Eigenvalues
        ax.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            c=np.abs(eigenvalues),
            cmap="viridis",
            edgecolors="k",
            linewidths=0.4,
            s=60,
            zorder=3,
        )

        # Annotate the top-k modes
        order = np.argsort(-np.abs(eigenvalues))
        for rank, idx in enumerate(order[:annotate_top_k]):
            ev = eigenvalues[idx]
            ax.annotate(
                f"$\\lambda_{{{rank}}}$",
                (ev.real, ev.imag),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

        ax.set_xlabel("Re($\\lambda$)")
        ax.set_ylabel("Im($\\lambda$)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Sorting utility
    # ------------------------------------------------------------------

    @staticmethod
    def sort_by_timescale(eigenvalues: np.ndarray) -> np.ndarray:
        """Return eigenvalues sorted by decreasing magnitude (slowest first).

        Parameters
        ----------
        eigenvalues : np.ndarray
            Complex eigenvalue array of arbitrary order.

        Returns
        -------
        np.ndarray
            Sorted copy of the eigenvalue array.
        """
        order = np.argsort(-np.abs(eigenvalues))
        return eigenvalues[order]
