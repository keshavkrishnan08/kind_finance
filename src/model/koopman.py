"""
KoopmanAnalyzer: high-level interface for extracting Koopman spectral
quantities from a trained NonEquilibriumVAMPNet.

Provides methods for:
    - Koopman matrix extraction
    - Eigenvalue / eigenfunction extraction
    - Spectral-gap computation
    - Regime persistence bounds (from the spectral gap)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Optional, Tuple

from .vampnet import NonEquilibriumVAMPNet


class KoopmanAnalyzer:
    """Stateless helper that wraps a trained :class:`NonEquilibriumVAMPNet`
    and exposes convenience methods for spectral analysis.

    All public methods run in ``torch.no_grad()`` context and return
    CPU tensors by default.

    Parameters
    ----------
    model : NonEquilibriumVAMPNet
        A trained (or partially trained) VAMPNet.
    device : torch.device or str or None
        Device on which to run inference.  If ``None``, the model's
        current device is used.
    """

    def __init__(
        self,
        model: NonEquilibriumVAMPNet,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Internal helper: run forward pass on arbitrary data
    # ------------------------------------------------------------------

    def _forward(
        self,
        x_t: Tensor,
        x_tau: Tensor,
    ) -> Dict[str, Tensor]:
        """Run the model forward pass, handling device placement."""
        self.model.eval()
        x_t = x_t.to(self.device)
        x_tau = x_tau.to(self.device)
        with torch.no_grad():
            return self.model(x_t, x_tau)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_koopman_matrix(
        self,
        x_t: Tensor,
        x_tau: Tensor,
    ) -> Tensor:
        """Extract the whitened Koopman matrix K.

        Parameters
        ----------
        x_t : Tensor, shape ``(N, input_dim)``
            Data at time *t*.
        x_tau : Tensor, shape ``(N, input_dim)``
            Data at time *t + tau*.

        Returns
        -------
        K : Tensor, shape ``(d, d)``
            Whitened Koopman approximation on CPU.
        """
        out = self._forward(x_t, x_tau)
        return out["koopman_matrix"].cpu()

    def extract_eigenvalues(
        self,
        x_t: Tensor,
        x_tau: Tensor,
    ) -> Tensor:
        """Extract complex eigenvalues of K.

        Parameters
        ----------
        x_t, x_tau : Tensor
            Time-lagged pair, each shape ``(N, input_dim)``.

        Returns
        -------
        eigenvalues : Tensor, shape ``(d,)`` complex128
            Eigenvalues on CPU.
        """
        out = self._forward(x_t, x_tau)
        return out["eigenvalues"].cpu()

    def extract_eigenfunctions(
        self,
        x: Tensor,
        x_t: Tensor,
        x_tau: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate left and right Koopman eigenfunctions at arbitrary points.

        Parameters
        ----------
        x : Tensor, shape ``(M, input_dim)``
            Points at which to evaluate eigenfunctions.
        x_t, x_tau : Tensor
            Time-lagged pair used to build the Koopman matrix (may be
            the full training set or a representative batch).

        Returns
        -------
        u : Tensor, shape ``(M, d)``
            Right eigenfunction values on CPU.
        v : Tensor, shape ``(M, d)``
            Left eigenfunction values on CPU.
        """
        out = self._forward(x_t, x_tau)
        x_dev = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            u, v = self.model.compute_eigenfunctions(x_dev, out)
        return u.cpu(), v.cpu()

    # ------------------------------------------------------------------
    # Spectral-gap analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compute_spectral_gap(
        eigenvalues: Tensor,
        tau: float = 1.0,
    ) -> Tensor:
        """Spectral gap: delta = |Re(ln(lambda_2))| / tau.

        The eigenvalues are sorted by descending modulus.  ``lambda_1``
        is the leading eigenvalue (closest to 1 in magnitude) and
        ``lambda_2`` is the second.  The spectral gap governs the rate
        at which the system relaxes to its slowest non-trivial mode.

        Parameters
        ----------
        eigenvalues : Tensor, shape ``(d,)`` complex
            Koopman eigenvalues.
        tau : float
            Lag time used to construct the Koopman operator.

        Returns
        -------
        delta : Tensor, scalar
            Spectral gap (non-negative real value).
        """
        # Sort by descending magnitude
        magnitudes = eigenvalues.abs()
        sorted_idx = torch.argsort(magnitudes, descending=True)
        sorted_eigs = eigenvalues[sorted_idx]

        if sorted_eigs.shape[0] < 2:
            return torch.tensor(0.0)

        lambda_2 = sorted_eigs[1]
        # ln(lambda_2)  --  complex log
        log_lambda_2 = torch.log(lambda_2.to(torch.complex128))
        delta = log_lambda_2.real.abs() / tau
        return delta.real  # guaranteed real after .abs()

    @staticmethod
    def regime_persistence_bound(
        spectral_gap: Tensor,
        tau: float = 1.0,
    ) -> Tensor:
        """Lower bound on regime duration implied by the spectral gap.

        T_persist >= 1 / delta

        where delta is the spectral gap in continuous-time units.
        This gives the *minimum* expected persistence time of the
        dominant dynamical regime.

        Parameters
        ----------
        spectral_gap : Tensor, scalar
            Output of :meth:`compute_spectral_gap`.
        tau : float
            Lag time (same units as spectral_gap).

        Returns
        -------
        T_persist : Tensor, scalar
            Lower bound on regime duration (in the same time units).
        """
        # Guard against near-zero spectral gap (infinite persistence)
        gap_clamped = spectral_gap.clamp(min=1e-12)
        return 1.0 / gap_clamped

    # ------------------------------------------------------------------
    # Convenience: full spectral summary
    # ------------------------------------------------------------------

    def spectral_summary(
        self,
        x_t: Tensor,
        x_tau: Tensor,
        tau: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Return a dictionary with all key spectral quantities.

        Returns
        -------
        dict with keys:
            ``koopman_matrix``, ``eigenvalues``, ``singular_values``,
            ``spectral_gap``, ``regime_persistence_bound``.
        """
        out = self._forward(x_t, x_tau)
        eigs = out["eigenvalues"].cpu()
        gap = self.compute_spectral_gap(eigs, tau)
        persist = self.regime_persistence_bound(gap, tau)

        return {
            "koopman_matrix": out["koopman_matrix"].cpu(),
            "eigenvalues": eigs,
            "singular_values": out["singular_values"].cpu(),
            "spectral_gap": gap,
            "regime_persistence_bound": persist,
        }
