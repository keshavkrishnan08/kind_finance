"""Dynamic Mode Decomposition baseline for financial time-series analysis.

Implements standard DMD per PRD Section 11.2. DMD extracts spatiotemporal
coherent structures from high-dimensional data by computing a best-fit
linear operator that advances the state forward by a fixed time lag.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DMDBaseline:
    """Standard Dynamic Mode Decomposition baseline.

    Given a returns matrix of shape (T, D), DMD constructs time-shifted
    snapshot pairs X and Y separated by lag ``tau``, then computes the
    rank-truncated best-fit linear operator

        A_tilde = U_r^T Y V_r S_r^{-1}

    whose eigenvalues and modes characterise the dominant dynamical patterns.

    Parameters
    ----------
    n_modes : int, default=10
        Number of SVD modes to retain (rank truncation parameter ``r``).

    Attributes
    ----------
    A_tilde_ : np.ndarray of shape (r, r)
        Reduced-rank linear operator in the SVD basis.
    eigenvalues_ : np.ndarray of shape (r,)
        Complex eigenvalues of ``A_tilde_``.
    modes_ : np.ndarray of shape (D, r)
        DMD modes (exact modes projected back to the full state space).
    U_r_ : np.ndarray of shape (D, r)
        Left singular vectors retained after truncation.
    S_r_ : np.ndarray of shape (r,)
        Singular values retained after truncation.
    V_r_ : np.ndarray of shape (T-tau, r)
        Right singular vectors retained after truncation.
    X_ : np.ndarray of shape (D, T-tau)
        Snapshot matrix (columns = states at times 0 .. T-tau-1).
    Y_ : np.ndarray of shape (D, T-tau)
        Time-shifted snapshot matrix (columns = states at times tau .. T-1).
    reconstruction_error_ : float
        Relative Frobenius-norm reconstruction error on training data.
    """

    def __init__(self, n_modes: int = 10) -> None:
        self.n_modes = n_modes

        # Fitted attributes (set by .fit())
        self.A_tilde_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.modes_: Optional[np.ndarray] = None
        self.U_r_: Optional[np.ndarray] = None
        self.S_r_: Optional[np.ndarray] = None
        self.V_r_: Optional[np.ndarray] = None
        self.X_: Optional[np.ndarray] = None
        self.Y_: Optional[np.ndarray] = None
        self.reconstruction_error_: Optional[float] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray, tau: int = 1) -> "DMDBaseline":
        """Fit DMD to observed returns.

        Parameters
        ----------
        returns : np.ndarray of shape (T, D)
            Return matrix where rows are time steps and columns are assets
            or features. A 1-D array of shape (T,) is treated as (T, 1).
        tau : int, default=1
            Time lag between snapshot pairs.

        Returns
        -------
        self
            The fitted baseline instance.

        Raises
        ------
        ValueError
            If ``tau`` is not positive or if the number of snapshots after
            lagging is less than ``n_modes``.
        """
        X_full = np.asarray(returns, dtype=np.float64)
        if X_full.ndim == 1:
            X_full = X_full.reshape(-1, 1)
        if X_full.ndim != 2:
            raise ValueError(
                f"Expected 1-D or 2-D array, got shape {X_full.shape}"
            )

        T, D = X_full.shape
        if tau < 1:
            raise ValueError(f"tau must be >= 1, got {tau}")
        if T - tau < self.n_modes:
            raise ValueError(
                f"Not enough snapshots ({T - tau}) for {self.n_modes} modes. "
                f"Reduce n_modes or increase data length."
            )

        # Snapshot matrices (columns = state vectors)
        # X columns: t = 0, 1, ..., T-tau-1
        # Y columns: t = tau, tau+1, ..., T-1
        self.X_ = X_full[:-tau].T  # (D, T-tau)
        self.Y_ = X_full[tau:].T   # (D, T-tau)

        # Step 1: Economy SVD of X
        U, S, Vt = np.linalg.svd(self.X_, full_matrices=False)

        # Step 2: Truncate to rank r
        r = min(self.n_modes, len(S))
        self.U_r_ = U[:, :r]       # (D, r)
        self.S_r_ = S[:r]          # (r,)
        self.V_r_ = Vt[:r].T       # (T-tau, r)

        # Step 3: Compute reduced operator  A_tilde = U_r^T Y V_r S_r^{-1}
        S_inv = np.diag(1.0 / self.S_r_)                       # (r, r)
        self.A_tilde_ = self.U_r_.T @ self.Y_ @ self.V_r_ @ S_inv  # (r, r)

        # Step 4: Eigendecomposition of A_tilde
        eigvals, eigvecs = np.linalg.eig(self.A_tilde_)         # (r,), (r, r)
        self.eigenvalues_ = eigvals

        # Step 5: Exact DMD modes  Phi = Y V_r S_r^{-1} W
        self.modes_ = self.Y_ @ self.V_r_ @ S_inv @ eigvecs     # (D, r)

        # Step 6: Reconstruction error
        self.reconstruction_error_ = self._compute_reconstruction_error()

        logger.info(
            "DMD fitted: n_modes=%d, tau=%d, reconstruction_error=%.6f",
            r,
            tau,
            self.reconstruction_error_,
        )
        return self

    def get_eigenvalues(self) -> np.ndarray:
        """Return the complex DMD eigenvalues.

        Returns
        -------
        np.ndarray of shape (r,)
            Complex eigenvalues of the reduced operator ``A_tilde_``.
        """
        self._check_fitted()
        return self.eigenvalues_.copy()

    def get_modes(self) -> np.ndarray:
        """Return the DMD modes.

        Returns
        -------
        np.ndarray of shape (D, r)
            Exact DMD modes (columns are individual modes).
        """
        self._check_fitted()
        return self.modes_.copy()

    def reconstruction_error(self) -> float:
        """Return the relative Frobenius-norm reconstruction error.

        The error is defined as

            ||Y - A_approx X|| / ||Y||

        where ``A_approx = U_r A_tilde U_r^T`` is the full-rank approximation
        of the linear operator.

        Returns
        -------
        float
            Relative reconstruction error in [0, inf).
        """
        self._check_fitted()
        return self.reconstruction_error_

    def get_metrics(self) -> Dict[str, object]:
        """Return a dictionary of DMD diagnostics.

        Returns
        -------
        dict
            Keys:
            - ``eigenvalues``          : complex DMD eigenvalues (ndarray)
            - ``modes``                : DMD mode matrix (ndarray)
            - ``A_tilde``              : reduced operator (ndarray)
            - ``reconstruction_error`` : relative Frobenius error (float)
        """
        self._check_fitted()
        return {
            "eigenvalues": self.eigenvalues_.copy(),
            "modes": self.modes_.copy(),
            "A_tilde": self.A_tilde_.copy(),
            "reconstruction_error": self.reconstruction_error_,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_reconstruction_error(self) -> float:
        """Relative Frobenius-norm reconstruction error."""
        # Full-space approximation: A_approx = U_r A_tilde U_r^T
        A_approx = self.U_r_ @ self.A_tilde_ @ self.U_r_.T  # (D, D)
        Y_hat = A_approx @ self.X_                           # (D, T-tau)

        num = np.linalg.norm(self.Y_ - Y_hat, "fro")
        denom = np.linalg.norm(self.Y_, "fro")
        if denom == 0.0:
            return 0.0
        return float(num / denom)

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if self.A_tilde_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )
