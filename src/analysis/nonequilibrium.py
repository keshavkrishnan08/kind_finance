"""Post-hoc non-equilibrium diagnostics (zero GPU cost).

All functions operate on already-computed Koopman matrices, eigenvalues,
and entropy production estimates. No model inference or training is required.

Implements:
    - Detailed balance violation metric
    - Gallavotti-Cohen symmetry function
    - Fluctuation theorem ratio
    - Onsager regression test
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def detailed_balance_violation(K: NDArray[np.floating]) -> Dict[str, float]:
    """Quantify detailed balance violation from the Koopman matrix.

    For a reversible process K = K^T. The violation metric is
        D = ||K - K^T||_F / ||K||_F
    which is 0 for reversible dynamics and approaches sqrt(2) for
    maximally antisymmetric K.

    Parameters
    ----------
    K : ndarray, shape (d, d)
        Koopman matrix (whitened or un-whitened).

    Returns
    -------
    dict
        ``violation`` -- D metric in [0, sqrt(2)].
        ``symmetric_norm`` -- Frobenius norm of (K + K^T) / 2.
        ``antisymmetric_norm`` -- Frobenius norm of (K - K^T) / 2.
        ``ratio`` -- antisymmetric / symmetric norm.
    """
    K = np.asarray(K, dtype=np.float64)
    K_sym = (K + K.T) / 2.0
    K_anti = (K - K.T) / 2.0

    norm_K = np.linalg.norm(K, "fro")
    norm_sym = np.linalg.norm(K_sym, "fro")
    norm_anti = np.linalg.norm(K_anti, "fro")

    violation = norm_anti / max(norm_K, 1e-15)
    ratio = norm_anti / max(norm_sym, 1e-15)

    logger.info(
        "Detailed balance violation: D=%.4f, sym=%.4f, anti=%.4f, ratio=%.4f",
        violation, norm_sym, norm_anti, ratio,
    )

    return {
        "violation": float(violation),
        "symmetric_norm": float(norm_sym),
        "antisymmetric_norm": float(norm_anti),
        "ratio": float(ratio),
    }


def gallavotti_cohen_symmetry(
    entropy_production_samples: NDArray[np.floating],
    n_bins: int = 50,
    tau: float = 1.0,
) -> Dict[str, Any]:
    """Compute the Gallavotti-Cohen symmetry function.

    For a system obeying the steady-state fluctuation theorem,
        zeta(s) = (1/tau) * ln[P(+s) / P(-s)] = s
    i.e. the symmetry function is linear with unit slope.

    Parameters
    ----------
    entropy_production_samples : ndarray, shape (N,)
        Per-sample entropy production values (log-ratios of forward
        and backward transition densities).
    n_bins : int
        Number of histogram bins.
    tau : float
        Lag time.

    Returns
    -------
    dict
        ``s_values`` -- bin centers.
        ``zeta`` -- symmetry function values.
        ``slope`` -- linear fit slope (should be ~1 for FT).
        ``intercept`` -- linear fit intercept (should be ~0).
        ``r_squared`` -- R^2 of linear fit.
    """
    samples = np.asarray(entropy_production_samples, dtype=np.float64)

    # Symmetrize the histogram range
    s_max = np.percentile(np.abs(samples), 95)
    if s_max < 1e-12:
        return {
            "s_values": np.array([0.0]),
            "zeta": np.array([0.0]),
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
        }

    bins = np.linspace(-s_max, s_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    hist_pos, _ = np.histogram(samples, bins=bins)
    hist_neg, _ = np.histogram(-samples, bins=bins)

    # Avoid log(0)
    valid = (hist_pos > 0) & (hist_neg > 0)
    s_valid = bin_centers[valid]
    zeta = np.log(hist_pos[valid].astype(float) / hist_neg[valid].astype(float)) / tau

    # Linear fit
    if len(s_valid) >= 3:
        coeffs = np.polyfit(s_valid, zeta, 1)
        slope, intercept = coeffs
        zeta_pred = slope * s_valid + intercept
        ss_res = np.sum((zeta - zeta_pred) ** 2)
        ss_tot = np.sum((zeta - zeta.mean()) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)
    else:
        slope, intercept, r_squared = 0.0, 0.0, 0.0

    logger.info(
        "Gallavotti-Cohen: slope=%.4f, intercept=%.4f, R^2=%.4f",
        slope, intercept, r_squared,
    )

    return {
        "s_values": s_valid,
        "zeta": zeta,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def fluctuation_theorem_ratio(
    entropy_production_samples: NDArray[np.floating],
) -> Dict[str, float]:
    """Test the integral fluctuation theorem: <exp(-sigma)> = 1.

    For a process satisfying the integral fluctuation theorem (Jarzynski
    equality in the entropy production formulation), the ensemble average
    of exp(-sigma) equals 1. Deviations indicate finite-size effects or
    non-stationarity.

    Parameters
    ----------
    entropy_production_samples : ndarray, shape (N,)
        Per-sample entropy production values.

    Returns
    -------
    dict
        ``mean_exp_neg_sigma`` -- <exp(-sigma)>, should be ~1.
        ``log_deviation`` -- |ln(<exp(-sigma)>)|.
        ``n_samples`` -- number of samples used.
    """
    samples = np.asarray(entropy_production_samples, dtype=np.float64)
    exp_neg = np.exp(-samples)
    mean_val = float(np.mean(exp_neg))
    log_dev = float(np.abs(np.log(max(mean_val, 1e-15))))

    logger.info(
        "Fluctuation theorem: <exp(-sigma)>=%.4f, |ln(<exp(-sigma)>)|=%.4f",
        mean_val, log_dev,
    )

    return {
        "mean_exp_neg_sigma": mean_val,
        "log_deviation": log_dev,
        "n_samples": len(samples),
    }


def onsager_regression_test(
    eigenvalues: NDArray[np.complexfloating],
    autocorrelation_times: NDArray[np.floating],
    tau: float = 1.0,
) -> Dict[str, Any]:
    """Test the Onsager regression hypothesis.

    The Onsager regression hypothesis states that the regression of
    macroscopic fluctuations follows the same law as the microscopic
    relaxation. For the Koopman framework, this means the decay rates
    gamma_k = -ln|lambda_k| / tau should match the autocorrelation
    decay rates of the corresponding observables.

    Parameters
    ----------
    eigenvalues : ndarray, shape (K,), complex
        Koopman eigenvalues.
    autocorrelation_times : ndarray, shape (K,)
        Empirical autocorrelation times for the K dominant observables.
    tau : float
        Lag time.

    Returns
    -------
    dict
        ``koopman_rates`` -- decay rates from eigenvalues.
        ``acf_rates`` -- decay rates from autocorrelation times.
        ``correlation`` -- Pearson correlation between the two.
        ``relative_errors`` -- per-mode relative error.
    """
    eigs = np.asarray(eigenvalues)
    acf_times = np.asarray(autocorrelation_times, dtype=np.float64)

    magnitudes = np.abs(eigs)
    koopman_rates = -np.log(np.clip(magnitudes, 1e-15, None)) / tau
    acf_rates = 1.0 / np.clip(acf_times, 1e-15, None)

    # Match dimensions
    n = min(len(koopman_rates), len(acf_rates))
    kr = koopman_rates[:n]
    ar = acf_rates[:n]

    if n >= 2:
        correlation = float(np.corrcoef(kr, ar)[0, 1])
    else:
        correlation = 0.0

    relative_errors = np.abs(kr - ar) / np.clip(np.abs(ar), 1e-15, None)

    logger.info(
        "Onsager regression: correlation=%.4f, mean_rel_error=%.4f",
        correlation, float(np.mean(relative_errors)),
    )

    return {
        "koopman_rates": kr,
        "acf_rates": ar,
        "correlation": correlation,
        "relative_errors": relative_errors,
    }


def eigenvalue_complex_plane_statistics(
    eigenvalues: NDArray[np.complexfloating],
    tau: float = 1.0,
) -> Dict[str, Any]:
    """Compute summary statistics of the eigenvalue spectrum.

    Provides a comprehensive statistical summary of the Koopman spectrum
    including measures of non-equilibrium content (imaginary parts),
    timescale separation, and mode classification.

    Parameters
    ----------
    eigenvalues : ndarray, shape (K,), complex
        Koopman eigenvalues.
    tau : float
        Lag time.

    Returns
    -------
    dict
        Comprehensive spectrum statistics.
    """
    eigs = np.asarray(eigenvalues)
    magnitudes = np.abs(eigs)
    angles = np.angle(eigs)

    # Sort by magnitude
    order = np.argsort(-magnitudes)
    eigs_sorted = eigs[order]
    mags_sorted = magnitudes[order]
    angles_sorted = angles[order]

    # Timescale analysis
    decay_rates = -np.log(np.clip(mags_sorted, 1e-15, None)) / tau
    frequencies = angles_sorted / tau

    # Mode classification
    real_threshold = 0.05  # modes with |Im(lambda)| < threshold are "real"
    n_real = int(np.sum(np.abs(eigs.imag) < real_threshold))
    n_complex = len(eigs) - n_real

    # Irreversibility measure from complex content
    complex_content = float(np.sum(np.abs(eigs.imag) ** 2))
    total_content = float(np.sum(np.abs(eigs) ** 2))
    complex_fraction = complex_content / max(total_content, 1e-15)

    return {
        "eigenvalues_sorted": eigs_sorted,
        "magnitudes": mags_sorted,
        "angles": angles_sorted,
        "decay_rates": decay_rates,
        "frequencies": frequencies,
        "n_real_modes": n_real,
        "n_complex_modes": n_complex,
        "complex_fraction": float(complex_fraction),
        "spectral_radius": float(mags_sorted[0]) if len(mags_sorted) > 0 else 0.0,
        "condition_number": float(mags_sorted[0] / max(mags_sorted[-1], 1e-15))
            if len(mags_sorted) > 0 else 0.0,
    }
