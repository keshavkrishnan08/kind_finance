"""Chapman--Kolmogorov consistency test for Koopman / transfer operators.

Implements the test described in PRD Section 10.1: for a Markovian model
the propagator at lag ``n * tau`` should equal the *n*-th power of the
propagator at lag ``tau``.  Deviations are quantified in the Frobenius norm
and statistical significance is assessed via a block-bootstrap null
distribution.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ======================================================================
# Public API
# ======================================================================


def chapman_kolmogorov_test(
    model: torch.nn.Module,
    data: torch.Tensor,
    tau: float,
    max_multiple: int = 5,
    n_bootstrap: int = 200,
    block_size: int = 50,
    confidence_level: float = 0.95,
) -> Dict[int, Dict[str, Any]]:
    """Chapman--Kolmogorov self-consistency test (PRD Section 10.1).

    Compares the predicted propagator ``K(tau)^n`` with the directly
    estimated propagator ``K_direct(n * tau)`` for ``n = 2, ..., max_multiple``.
    A block bootstrap provides p-values and confidence intervals.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with a ``transform`` method.
    data : torch.Tensor
        Full trajectory ``(T, d)``.
    tau : float
        Base lag time.
    max_multiple : int
        Largest integer multiple to test (inclusive).
    n_bootstrap : int
        Number of bootstrap replicates for the null distribution.
    block_size : int
        Block length for the non-overlapping block bootstrap.
    confidence_level : float
        Confidence level for the reported interval (default 0.95).

    Returns
    -------
    dict[int, dict]
        Keyed by the integer multiple ``n`` (2 through *max_multiple*).
        Each value dict contains:

        ``statistic``  -- Frobenius norm  ``||K^n - K_direct(n*tau)||_F``.
        ``p_value``    -- fraction of bootstrap replicates with statistic
                          >= the observed one.
        ``ci_95``      -- ``(lower, upper)`` confidence interval for the
                          statistic under the null.
        ``K_power``    -- ``K(tau)^n`` matrix.
        ``K_direct``   -- ``K_direct(n*tau)`` matrix.
    """
    tau_int = int(tau)

    # Base Koopman matrix at lag tau
    K_base = model_koopman_matrix(model, data, tau_int)

    results: Dict[int, Dict[str, Any]] = {}

    for n in range(2, max_multiple + 1):
        lag_n = n * tau_int

        # Check that we have enough data for this lag
        if lag_n >= data.shape[0]:
            logger.warning(
                "Skipping n=%d: lag %d exceeds trajectory length %d",
                n,
                lag_n,
                data.shape[0],
            )
            continue

        # Predicted propagator: K(tau)^n  (use repeated @ to avoid
        # np.linalg.matrix_power which can segfault via MKL SGEBAL
        # on ill-conditioned matrices)
        K_power = _safe_matrix_power(K_base, n)

        # Direct propagator at lag n*tau
        K_direct = model_koopman_matrix(model, data, lag_n)

        # Observed Frobenius norm
        observed_stat = float(np.linalg.norm(K_power - K_direct, "fro"))

        # Block-bootstrap null distribution
        boot_stats = _block_bootstrap_ck(
            model=model,
            data=data,
            tau_int=tau_int,
            n=n,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
        )

        # p-value: fraction of bootstrap replicates >= observed
        p_value = float(np.mean(boot_stats >= observed_stat))

        # Confidence interval
        alpha = 1.0 - confidence_level
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        results[n] = {
            "statistic": observed_stat,
            "p_value": p_value,
            "ci_95": (ci_lower, ci_upper),
            "K_power": K_power,
            "K_direct": K_direct,
        }

        logger.info(
            "CK test n=%d: ||K^n - K_direct||_F = %.4f, p=%.3f, CI=(%.4f, %.4f)",
            n,
            observed_stat,
            p_value,
            ci_lower,
            ci_upper,
        )

    return results


# ======================================================================
# Helper: extract Koopman matrix
# ======================================================================


def model_koopman_matrix(
    model: torch.nn.Module,
    data: torch.Tensor,
    lag: int,
) -> np.ndarray:
    """Estimate the Koopman matrix ``K`` at a given integer lag.

    Uses the learned feature map to construct the time-lagged covariance
    matrices and solves for ``K = C_{00}^{-1} C_{01}``.

    Parameters
    ----------
    model : torch.nn.Module
        Model with a ``transform`` method.
    data : torch.Tensor
        Trajectory ``(T, d)``.
    lag : int
        Integer lag (number of time steps).

    Returns
    -------
    np.ndarray
        Square Koopman matrix of shape ``(K, K)`` where *K* is the feature
        dimensionality.
    """
    model.eval()

    x_t = data[:-lag]
    x_t_lag = data[lag:]

    with torch.no_grad():
        chi_t = model.transform(x_t).cpu().numpy()
        chi_t_lag = model.transform(x_t_lag).cpu().numpy()

    n_samples = chi_t.shape[0]
    mean_t = chi_t.mean(axis=0, keepdims=True)
    mean_lag = chi_t_lag.mean(axis=0, keepdims=True)

    chi_t_c = chi_t - mean_t
    chi_lag_c = chi_t_lag - mean_lag

    C_00 = (chi_t_c.T @ chi_t_c) / n_samples
    C_01 = (chi_t_c.T @ chi_lag_c) / n_samples

    # Tikhonov regularisation
    reg = 1e-6 * np.eye(C_00.shape[0])
    K = np.linalg.solve(C_00 + reg, C_01)

    return K


# ======================================================================
# Internal: block bootstrap
# ======================================================================


def _block_bootstrap_ck(
    model: torch.nn.Module,
    data: torch.Tensor,
    tau_int: int,
    n: int,
    n_bootstrap: int,
    block_size: int,
) -> np.ndarray:
    """Generate a null distribution for the CK statistic via block bootstrap.

    Non-overlapping blocks of length ``block_size`` are resampled with
    replacement from the transformed feature trajectory to create
    surrogate datasets.  For each surrogate the CK statistic is
    recomputed.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    data : torch.Tensor
        Original trajectory.
    tau_int : int
        Base lag in integer steps.
    n : int
        Multiple being tested.
    n_bootstrap : int
        Number of replicates.
    block_size : int
        Block length.

    Returns
    -------
    np.ndarray
        Array of bootstrap CK statistics of length ``n_bootstrap``.
    """
    model.eval()

    with torch.no_grad():
        features = model.transform(data).cpu().numpy()  # (T, K)

    T, K = features.shape
    n_blocks = T // block_size
    if n_blocks < 2:
        logger.warning(
            "Block size %d too large for trajectory length %d; "
            "falling back to block_size=max(1, T//4)",
            block_size,
            T,
        )
        block_size = max(1, T // 4)
        n_blocks = T // block_size

    rng = np.random.default_rng(seed=42)
    boot_stats = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        # Resample blocks with replacement
        chosen_blocks = rng.integers(0, n_blocks, size=n_blocks)
        surrogate_indices = np.concatenate(
            [np.arange(i * block_size, (i + 1) * block_size) for i in chosen_blocks]
        )
        surrogate = features[surrogate_indices]

        S = surrogate.shape[0]
        lag_n = n * tau_int

        if tau_int >= S or lag_n >= S:
            boot_stats[b] = 0.0
            continue

        # Koopman matrix at lag tau from surrogate features
        K_base_b = _koopman_from_features(surrogate, tau_int)

        # Koopman matrix at lag n*tau from surrogate features
        K_direct_b = _koopman_from_features(surrogate, lag_n)

        K_power_b = _safe_matrix_power(K_base_b, n)
        boot_stats[b] = float(np.linalg.norm(K_power_b - K_direct_b, "fro"))

    return boot_stats


def _safe_matrix_power(K: np.ndarray, n: int) -> np.ndarray:
    """Compute K^n via repeated multiplication, avoiding MKL SGEBAL segfaults.

    ``np.linalg.matrix_power`` internally uses eigendecomposition for large
    exponents, which calls MKL's SGEBAL (matrix balancing).  On
    ill-conditioned matrices this can trigger a segmentation fault.
    Repeated ``@`` is O(n) but safe for the small *n* values (2-5) used
    in the Chapman-Kolmogorov test.
    """
    if not np.isfinite(K).all():
        K = np.where(np.isfinite(K), K, 0.0)
    result = K.copy()
    for _ in range(n - 1):
        result = result @ K
        # Clamp to prevent overflow propagation
        if not np.isfinite(result).all():
            result = np.where(np.isfinite(result), result, 0.0)
    return result


def _koopman_from_features(
    features: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Estimate Koopman matrix directly from pre-computed feature arrays.

    Parameters
    ----------
    features : np.ndarray
        Feature array ``(T, K)``.
    lag : int
        Integer lag.

    Returns
    -------
    np.ndarray
        Koopman matrix ``(K, K)``.
    """
    chi_t = features[:-lag]
    chi_lag = features[lag:]

    n_samples = chi_t.shape[0]
    mean_t = chi_t.mean(axis=0, keepdims=True)
    mean_lag = chi_lag.mean(axis=0, keepdims=True)

    chi_t_c = chi_t - mean_t
    chi_lag_c = chi_lag - mean_lag

    C_00 = (chi_t_c.T @ chi_t_c) / n_samples
    C_01 = (chi_t_c.T @ chi_lag_c) / n_samples

    reg = 1e-6 * np.eye(C_00.shape[0])
    K = np.linalg.solve(C_00 + reg, C_01)

    # Sanitize: replace NaN/Inf (can occur from degenerate resamples)
    if not np.isfinite(K).all():
        K = np.where(np.isfinite(K), K, 0.0)

    return K
