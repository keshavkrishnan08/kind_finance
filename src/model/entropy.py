"""
Entropy production estimation and spectral decomposition (PRD Sections 3.5
and 7.3).

Two complementary approaches:

1. **Empirical estimation** via kernel-density-based log-ratios of forward
   and time-reversed transition densities.
2. **Spectral decomposition** of the entropy production rate into
   contributions from individual Koopman modes, providing a
   frequency-resolved view of irreversibility.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Empirical entropy production via KDE
# ---------------------------------------------------------------------------

def _kde_entropy_production(
    x_t: np.ndarray,
    x_tau: np.ndarray,
    bw_method: str = "scott",
    return_samples: bool = False,
):
    """Core KDE log-ratio entropy production on pre-built transition pairs.

    Parameters
    ----------
    x_t : ndarray, shape ``(N, d)``
        Observations at time *t*.
    x_tau : ndarray, shape ``(N, d)``
        Observations at time *t + tau*.
    bw_method : str
        Bandwidth selection method for ``gaussian_kde``.
    return_samples : bool
        If True, return per-sample log-ratios instead of the mean.

    Returns
    -------
    float or ndarray
        If ``return_samples`` is False (default), returns the estimated
        entropy production rate (clamped >= 0).  If True, returns
        per-sample log-ratios as ndarray of shape ``(N,)``.
    """
    joint_fwd = np.concatenate([x_t, x_tau], axis=1).T   # (2d, N)
    joint_rev = np.concatenate([x_tau, x_t], axis=1).T   # (2d, N)

    try:
        kde_fwd = gaussian_kde(joint_fwd, bw_method=bw_method)
        kde_rev = gaussian_kde(joint_rev, bw_method=bw_method)

        log_p_fwd = kde_fwd.logpdf(joint_fwd)   # (N,)
        log_p_rev = kde_rev.logpdf(joint_fwd)    # (N,)

        per_sample = log_p_fwd - log_p_rev
        if return_samples:
            return per_sample
        sigma = float(np.mean(per_sample))
    except np.linalg.LinAlgError:
        if return_samples:
            return np.zeros(x_t.shape[0])
        sigma = 0.0

    return max(sigma, 0.0)


def estimate_empirical_entropy_production(
    returns: Tensor,
    tau: int,
    bandwidth: Optional[float] = None,
    n_samples: int = 5000,
) -> Tensor:
    """Estimate the entropy production rate from a univariate or multivariate
    return series using the KDE log-ratio method.

    The entropy production rate is related to the Kullback--Leibler
    divergence between the forward and time-reversed path measures.
    For a stationary process with transition density p(x_{t+tau} | x_t)
    and time-reversed density p_rev(x_{t+tau} | x_t), the per-step
    entropy production is

        sigma = <ln p(x_{t+tau}|x_t) - ln p_rev(x_{t+tau}|x_t)>

    We estimate the joint densities p(x_t, x_{t+tau}) and
    p(x_{t+tau}, x_t) with Gaussian KDE and compute the sample-mean
    log-ratio.

    Parameters
    ----------
    returns : Tensor, shape ``(T,)`` or ``(T, d)``
        Return time series.
    tau : int
        Lag (in discrete time steps).
    bandwidth : float or None
        KDE bandwidth.  If ``None``, Scott's rule is used.
    n_samples : int
        Maximum number of transition pairs to use (sub-sampled if the
        series is longer, for computational tractability).

    Returns
    -------
    sigma : Tensor, scalar
        Estimated entropy production rate (non-negative by construction
        when averaged, though individual log-ratios can be negative).
    """
    r = returns.detach().cpu().numpy()
    if r.ndim == 1:
        r = r[:, np.newaxis]

    T, d = r.shape
    if T <= tau:
        return torch.tensor(0.0)

    x_t = r[:-tau]
    x_tau = r[tau:]
    n_pairs = x_t.shape[0]

    if n_pairs > n_samples:
        idx = np.random.choice(n_pairs, size=n_samples, replace=False)
        x_t = x_t[idx]
        x_tau = x_tau[idx]

    bw_method = bandwidth if bandwidth is not None else "scott"
    sigma = _kde_entropy_production(x_t, x_tau, bw_method)

    return torch.tensor(sigma, dtype=torch.float32)


def estimate_per_sample_entropy_production(
    returns: Tensor,
    tau: int,
    bandwidth: Optional[float] = None,
    n_samples: int = 5000,
) -> np.ndarray:
    """Per-sample entropy production log-ratios for fluctuation theorem tests.

    Parameters
    ----------
    returns : Tensor, shape ``(T,)`` or ``(T, d)``
        Return time series.
    tau : int
        Lag (in discrete time steps).
    bandwidth : float or None
        KDE bandwidth.  If ``None``, Scott's rule is used.
    n_samples : int
        Maximum number of transition pairs (sub-sampled if longer).

    Returns
    -------
    ndarray, shape ``(N,)``
        Per-sample log p_fwd / p_rev values.
    """
    r = returns.detach().cpu().numpy()
    if r.ndim == 1:
        r = r[:, np.newaxis]

    T, d = r.shape
    if T <= tau:
        return np.zeros(0)

    x_t = r[:-tau]
    x_tau = r[tau:]
    n_pairs = x_t.shape[0]

    if n_pairs > n_samples:
        idx = np.random.choice(n_pairs, size=n_samples, replace=False)
        x_t = x_t[idx]
        x_tau = x_tau[idx]

    bw_method = bandwidth if bandwidth is not None else "scott"
    return _kde_entropy_production(x_t, x_tau, bw_method, return_samples=True)


def estimate_empirical_entropy_production_with_ci(
    returns: Tensor,
    tau: int,
    bandwidth: Optional[float] = None,
    n_samples: int = 5000,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    block_length: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Entropy production with block-bootstrap confidence intervals.

    Uses a moving-block bootstrap to preserve temporal autocorrelation
    in the resampled series (Kunsch 1989, Politis & Romano 1994).

    Parameters
    ----------
    returns : Tensor, shape ``(T,)`` or ``(T, d)``
        Return time series.
    tau : int
        Lag (in discrete time steps).
    bandwidth : float or None
        KDE bandwidth.  If ``None``, Scott's rule is used.
    n_samples : int
        Maximum number of transition pairs per bootstrap replicate.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95 -> 95% CI).
    block_length : int
        Length of contiguous blocks for the moving-block bootstrap.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``point_estimate`` : float
        ``ci_lower`` : float
        ``ci_upper`` : float
        ``std_error`` : float
        ``n_bootstrap`` : int
    """
    r = returns.detach().cpu().numpy()
    if r.ndim == 1:
        r = r[:, np.newaxis]

    T, d = r.shape
    if T <= tau:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
            "n_bootstrap": 0,
        }

    bw_method = bandwidth if bandwidth is not None else "scott"

    # Point estimate on full data
    x_t_full = r[:-tau]
    x_tau_full = r[tau:]
    n_pairs = x_t_full.shape[0]
    if n_pairs > n_samples:
        rng_sub = np.random.RandomState(seed)
        idx = rng_sub.choice(n_pairs, size=n_samples, replace=False)
        x_t_sub = x_t_full[idx]
        x_tau_sub = x_tau_full[idx]
    else:
        x_t_sub = x_t_full
        x_tau_sub = x_tau_full

    point_estimate = _kde_entropy_production(x_t_sub, x_tau_sub, bw_method)

    # Block bootstrap
    rng = np.random.RandomState(seed)
    boot_estimates = np.empty(n_bootstrap)
    block_length = min(block_length, T // 2)

    for b in range(n_bootstrap):
        # Draw blocks to reconstruct a series of length T
        n_blocks = int(np.ceil(T / block_length))
        starts = rng.randint(0, T - block_length, size=n_blocks)
        boot_idx = np.concatenate(
            [np.arange(s, s + block_length) for s in starts]
        )[:T]

        r_boot = r[boot_idx]
        x_t_b = r_boot[:-tau]
        x_tau_b = r_boot[tau:]

        if len(x_t_b) > n_samples:
            idx_b = rng.choice(len(x_t_b), size=n_samples, replace=False)
            x_t_b = x_t_b[idx_b]
            x_tau_b = x_tau_b[idx_b]

        boot_estimates[b] = _kde_entropy_production(x_t_b, x_tau_b, bw_method)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1 - alpha / 2)))
    std_error = float(np.std(boot_estimates, ddof=1))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Spectral decomposition of entropy production
# ---------------------------------------------------------------------------

class EntropyDecomposer:
    """Decompose total entropy production into per-mode contributions.

    Given the Koopman eigenvalues lambda_k and the eigenfunctions psi_k
    evaluated at data points, each mode contributes

        sigma_k = |omega_k|^2 * A_k

    where
        omega_k = angle(lambda_k) / tau    (angular frequency)
        A_k     = mean(|psi_k|^2)          (mean squared amplitude)

    The total spectral entropy production is  sum_k sigma_k.

    Parameters
    ----------
    tau : float
        Lag time used when constructing the Koopman operator.
    """

    def __init__(self, tau: float = 1.0) -> None:
        self.tau = tau

    def decompose(
        self,
        eigenvalues: Tensor,
        eigenfunctions: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute per-mode entropy production contributions.

        Parameters
        ----------
        eigenvalues : Tensor, shape ``(d,)`` complex
            Koopman eigenvalues.
        eigenfunctions : Tensor, shape ``(N, d)``
            Eigenfunction values at N data points (can be real; if
            complex, absolute values are used).

        Returns
        -------
        dict with keys:
            ``mode_contributions`` : Tensor (d,)
                sigma_k for each mode.
            ``frequencies`` : Tensor (d,)
                omega_k = angle(lambda_k) / tau.
            ``amplitudes`` : Tensor (d,)
                A_k = mean(|psi_k|^2).
            ``total`` : Tensor scalar
                Total spectral entropy production.
            ``cumulative_fraction`` : Tensor (d,)
                Cumulative fraction of total entropy production
                (sorted descending by sigma_k).
        """
        eigenvalues = eigenvalues.detach()
        eigenfunctions = eigenfunctions.detach()

        # Angular frequencies
        omega = torch.angle(eigenvalues).float() / self.tau  # (d,)

        # Mean squared amplitudes
        if eigenfunctions.is_complex():
            amplitudes = eigenfunctions.abs().pow(2).mean(dim=0).float()  # (d,)
        else:
            amplitudes = eigenfunctions.pow(2).mean(dim=0).float()  # (d,)

        # Per-mode entropy production
        mode_contributions = omega.pow(2) * amplitudes  # (d,)

        # Total
        total = mode_contributions.sum()

        # Cumulative fraction (sorted descending)
        sorted_sigma, sorted_idx = mode_contributions.sort(descending=True)
        cumsum = sorted_sigma.cumsum(dim=0)
        # Normalise; guard against zero total
        total_safe = total.clamp(min=1e-12)
        cumulative_fraction = cumsum / total_safe

        return {
            "mode_contributions": mode_contributions,
            "frequencies": omega,
            "amplitudes": amplitudes,
            "total": total,
            "cumulative_fraction": cumulative_fraction,
        }

    def rank_modes(
        self,
        eigenvalues: Tensor,
        eigenfunctions: Tensor,
    ) -> Tensor:
        """Return mode indices sorted by descending entropy-production
        contribution.

        Parameters
        ----------
        eigenvalues : Tensor, shape ``(d,)`` complex
        eigenfunctions : Tensor, shape ``(N, d)``

        Returns
        -------
        indices : Tensor, shape ``(d,)`` long
        """
        result = self.decompose(eigenvalues, eigenfunctions)
        _, idx = result["mode_contributions"].sort(descending=True)
        return idx
