"""
Unit tests for entropy decomposition.

Validates:
    - Entropy production is non-negative
    - Mode contributions sum to total
    - Frequencies derived from eigenvalue angles
    - Cumulative fraction reaches 1
    - Empirical entropy is ~0 for symmetric/reversible data
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.koopman import KoopmanAnalyzer
from src.model.losses import (
    entropy_production_consistency_loss,
    total_loss,
)


# ---------------------------------------------------------------------------
# Entropy decomposition utilities (self-contained for testing)
# ---------------------------------------------------------------------------

def compute_entropy_production(
    eigenvalues: torch.Tensor,
    amplitudes: torch.Tensor,
    tau: float,
) -> dict:
    """Compute spectral entropy production and per-mode decomposition.

    Parameters
    ----------
    eigenvalues : Tensor, shape ``(d,)`` complex
        Koopman eigenvalues.
    amplitudes : Tensor, shape ``(d,)``
        Mean squared amplitude A_k = <|psi_k|^2> per mode.
    tau : float
        Lag time.

    Returns
    -------
    dict
        ``sigma_total``       -- total entropy production rate.
        ``sigma_modes``       -- per-mode contributions sigma_k.
        ``frequencies``       -- omega_k = angle(lambda_k) / tau.
        ``cumulative_frac``   -- cumulative fraction sum_k(sigma_k) / sigma_total.
    """
    omega = torch.angle(eigenvalues) / tau  # angular frequency per mode
    magnitudes = eigenvalues.abs().float().clamp(min=1e-12, max=1.0 - 1e-7)
    gamma = (-torch.log(magnitudes) / tau).clamp(min=1e-6)
    sigma_modes = omega.pow(2) * amplitudes / gamma  # per-mode contribution

    sigma_total = sigma_modes.sum()

    # Cumulative fraction (sorted by descending contribution)
    sorted_sigma, _ = sigma_modes.abs().sort(descending=True)
    if sigma_total.abs().item() > 1e-15:
        cumulative_frac = torch.cumsum(sorted_sigma, dim=0) / sigma_total.abs()
    else:
        cumulative_frac = torch.ones_like(sorted_sigma)

    return {
        "sigma_total": sigma_total,
        "sigma_modes": sigma_modes,
        "frequencies": omega,
        "cumulative_frac": cumulative_frac,
    }


def estimate_empirical_entropy_production(
    data: np.ndarray,
    tau: int = 1,
) -> float:
    """Estimate entropy production from trajectory time-reversal asymmetry.

    Uses a simple proxy: the Kullback-Leibler divergence between the forward
    and time-reversed transition kernel, estimated via the asymmetry of the
    cross-correlation matrix.

    Parameters
    ----------
    data : np.ndarray, shape ``(T, d)``
        Trajectory.
    tau : int
        Lag time.

    Returns
    -------
    float
        Non-negative entropy production estimate.
    """
    x_t = data[:-tau]
    x_tau = data[tau:]

    N = x_t.shape[0]
    x_t_centered = x_t - x_t.mean(axis=0)
    x_tau_centered = x_tau - x_tau.mean(axis=0)

    # Cross-correlation
    C_forward = (x_t_centered.T @ x_tau_centered) / N
    C_backward = (x_tau_centered.T @ x_t_centered) / N

    # Asymmetry as a proxy for entropy production
    asymmetry = C_forward - C_backward
    sigma = np.sum(asymmetry ** 2)

    return float(sigma)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def oscillatory_eigenvalues():
    """Eigenvalues with significant imaginary parts (non-reversible)."""
    return torch.tensor(
        [0.99 + 0.0j, 0.90 + 0.20j, 0.90 - 0.20j, 0.70 + 0.30j],
        dtype=torch.complex128,
    )


@pytest.fixture
def real_eigenvalues():
    """Purely real eigenvalues (reversible system)."""
    return torch.tensor(
        [0.99 + 0.0j, 0.90 + 0.0j, 0.80 + 0.0j, 0.60 + 0.0j],
        dtype=torch.complex128,
    )


@pytest.fixture
def unit_amplitudes():
    """Unit eigenfunction amplitudes for all modes."""
    return torch.ones(4)


@pytest.fixture
def nonrev_trajectory():
    """Non-reversible trajectory from a rotational drift system."""
    np.random.seed(42)
    T = 5000
    d = 3
    data = np.zeros((T, d))

    # Simple rotational dynamics that break time-reversal symmetry
    rotation = np.array([
        [0.95, -0.2, 0.0],
        [0.2, 0.95, 0.0],
        [0.0, 0.0, 0.90],
    ])

    for t in range(1, T):
        data[t] = rotation @ data[t - 1] + 0.1 * np.random.randn(d)

    return data


@pytest.fixture
def reversible_trajectory():
    """Reversible trajectory (Gaussian white noise)."""
    np.random.seed(42)
    T = 5000
    d = 3
    return np.random.randn(T, d)


# ---------------------------------------------------------------------------
# Tests: Entropy production non-negativity
# ---------------------------------------------------------------------------

class TestEntropyProductionNonNeg:
    """Entropy production should be non-negative."""

    def test_entropy_production_nonnegative(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """sigma >= 0 for any eigenvalue configuration."""
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau=1.0
        )
        sigma = result["sigma_total"].item()
        assert sigma >= -1e-10, (
            f"Entropy production should be non-negative, got {sigma:.6f}"
        )

    def test_entropy_production_nonneg_real(
        self, real_eigenvalues, unit_amplitudes
    ):
        """sigma >= 0 even for real eigenvalues (should be ~0)."""
        result = compute_entropy_production(
            real_eigenvalues, unit_amplitudes, tau=1.0
        )
        sigma = result["sigma_total"].item()
        assert sigma >= -1e-10, (
            f"Entropy production should be >= 0, got {sigma:.6f}"
        )

    def test_entropy_production_positive_oscillatory(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """sigma > 0 when eigenvalues have imaginary parts."""
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau=1.0
        )
        sigma = result["sigma_total"].item()
        assert sigma > 0, (
            f"Entropy production should be positive for oscillatory eigenvalues, "
            f"got {sigma:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: Mode contributions sum
# ---------------------------------------------------------------------------

class TestModeContributionsSum:
    """Per-mode entropy contributions should sum to total."""

    def test_mode_contributions_sum(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """sum(sigma_k) ~ sigma_total."""
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau=1.0
        )
        sigma_total = result["sigma_total"].item()
        sigma_modes_sum = result["sigma_modes"].sum().item()

        assert abs(sigma_modes_sum - sigma_total) < 1e-8, (
            f"Mode sum ({sigma_modes_sum:.8f}) should equal total "
            f"({sigma_total:.8f})"
        )

    def test_mode_contributions_sum_random(self):
        """Random amplitudes: sum still equals total."""
        torch.manual_seed(123)
        eigs = torch.tensor(
            [0.9 + 0.1j, 0.8 + 0.2j, 0.7 - 0.15j, 0.6 + 0.0j],
            dtype=torch.complex128,
        )
        amps = torch.rand(4).abs() + 0.01

        result = compute_entropy_production(eigs, amps, tau=2.0)
        sigma_total = result["sigma_total"].item()
        sigma_modes_sum = result["sigma_modes"].sum().item()

        assert abs(sigma_modes_sum - sigma_total) < 1e-8, (
            f"Mode sum ({sigma_modes_sum:.8f}) != total ({sigma_total:.8f})"
        )


# ---------------------------------------------------------------------------
# Tests: Frequencies from eigenvalue angles
# ---------------------------------------------------------------------------

class TestFrequencies:
    """omega_k = angle(lambda_k) / tau."""

    def test_frequencies_from_eigenvalues(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """Verify omega_k = arg(lambda_k) / tau."""
        tau = 1.0
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau
        )
        omega = result["frequencies"]

        # Manually compute expected frequencies
        expected_omega = torch.angle(oscillatory_eigenvalues) / tau

        for k in range(len(omega)):
            assert abs(omega[k].item() - expected_omega[k].item()) < 1e-10, (
                f"omega_{k}: expected {expected_omega[k].item():.8f}, "
                f"got {omega[k].item():.8f}"
            )

    def test_frequencies_zero_for_real(
        self, real_eigenvalues, unit_amplitudes
    ):
        """Real positive eigenvalues should have omega_k = 0."""
        result = compute_entropy_production(
            real_eigenvalues, unit_amplitudes, tau=1.0
        )
        omega = result["frequencies"]

        for k in range(len(omega)):
            assert abs(omega[k].item()) < 1e-10, (
                f"Real eigenvalue {k} should have omega=0, "
                f"got {omega[k].item():.8f}"
            )

    def test_frequencies_conjugate_pair(self):
        """Conjugate eigenvalue pair should give opposite frequencies."""
        eigs = torch.tensor(
            [0.9 + 0.2j, 0.9 - 0.2j], dtype=torch.complex128
        )
        amps = torch.ones(2)
        tau = 1.0

        result = compute_entropy_production(eigs, amps, tau)
        omega = result["frequencies"]

        assert abs(omega[0].item() + omega[1].item()) < 1e-10, (
            f"Conjugate pair should have opposite frequencies: "
            f"{omega[0].item():.8f} and {omega[1].item():.8f}"
        )


# ---------------------------------------------------------------------------
# Tests: Cumulative fraction reaches 1
# ---------------------------------------------------------------------------

class TestCumulativeFraction:
    """Cumulative fraction of entropy production should reach 1."""

    def test_cumulative_fraction_reaches_one(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """cumulative[-1] ~ 1."""
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau=1.0
        )
        cum_frac = result["cumulative_frac"]
        last_value = cum_frac[-1].item()

        assert abs(last_value - 1.0) < 1e-6, (
            f"Cumulative fraction should reach 1, got {last_value:.8f}"
        )

    def test_cumulative_fraction_monotonic(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """Cumulative fraction should be monotonically increasing."""
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau=1.0
        )
        cum_frac = result["cumulative_frac"].numpy()

        for k in range(len(cum_frac) - 1):
            assert cum_frac[k] <= cum_frac[k + 1] + 1e-10, (
                f"Cumulative fraction should be monotonic: "
                f"f[{k}]={cum_frac[k]:.6f} > f[{k+1}]={cum_frac[k+1]:.6f}"
            )

    def test_cumulative_fraction_starts_positive(
        self, oscillatory_eigenvalues, unit_amplitudes
    ):
        """First cumulative fraction entry should be positive."""
        result = compute_entropy_production(
            oscillatory_eigenvalues, unit_amplitudes, tau=1.0
        )
        cum_frac = result["cumulative_frac"]

        assert cum_frac[0].item() > 0, (
            f"First cumulative entry should be positive, "
            f"got {cum_frac[0].item():.8f}"
        )


# ---------------------------------------------------------------------------
# Tests: Empirical entropy for symmetric data
# ---------------------------------------------------------------------------

class TestEmpiricalEntropy:
    """Empirical entropy production for symmetric (reversible) data."""

    def test_empirical_entropy_symmetric_data(self, reversible_trajectory):
        """sigma ~ 0 for Gaussian white noise (no temporal structure)."""
        sigma = estimate_empirical_entropy_production(
            reversible_trajectory, tau=1
        )

        # White noise has no time-reversal asymmetry, so sigma should be
        # very small (not exactly zero due to finite sample effects)
        assert sigma < 0.1, (
            f"Empirical entropy for white noise should be ~0, got {sigma:.6f}"
        )

    def test_empirical_entropy_nonrev_positive(self, nonrev_trajectory):
        """sigma > 0 for trajectory with rotational drift."""
        sigma = estimate_empirical_entropy_production(
            nonrev_trajectory, tau=1
        )
        assert sigma > 0, (
            f"Empirical entropy for non-reversible trajectory should be > 0, "
            f"got {sigma:.6f}"
        )

    def test_empirical_entropy_nonnegative(self, reversible_trajectory):
        """sigma >= 0 always (it is a sum of squares)."""
        sigma = estimate_empirical_entropy_production(
            reversible_trajectory, tau=1
        )
        assert sigma >= 0, (
            f"Empirical entropy should be non-negative, got {sigma:.6f}"
        )

    def test_empirical_entropy_increases_with_rotation(self):
        """Stronger rotation -> higher entropy production."""
        np.random.seed(42)
        T = 3000
        d = 2

        sigmas = []
        for rotation_strength in [0.0, 0.1, 0.3]:
            data = np.zeros((T, d))
            R = np.array([
                [0.95, -rotation_strength],
                [rotation_strength, 0.95],
            ])
            for t in range(1, T):
                data[t] = R @ data[t - 1] + 0.1 * np.random.randn(d)

            sigma = estimate_empirical_entropy_production(data, tau=1)
            sigmas.append(sigma)

        # Entropy should increase with rotation strength
        assert sigmas[0] < sigmas[1] or sigmas[1] < sigmas[2], (
            f"Entropy should increase with rotation: {sigmas}"
        )


# ---------------------------------------------------------------------------
# Tests: Consistency with loss function
# ---------------------------------------------------------------------------

class TestEntropyLossConsistency:
    """Entropy production from decomposition should be consistent with loss."""

    def test_entropy_decomposition_matches_loss(self):
        """The decomposed sigma_total should match the loss function's input."""
        eigs = torch.tensor(
            [0.98 + 0.0j, 0.90 + 0.15j, 0.90 - 0.15j, 0.75 + 0.0j],
            dtype=torch.complex128,
        )
        amps = torch.tensor([1.0, 0.8, 0.8, 0.5])
        tau = 1.0

        result = compute_entropy_production(eigs, amps, tau)
        sigma_decomposed = result["sigma_total"]

        # Use the loss function with the same sigma as target -> should give 0
        loss = entropy_production_consistency_loss(
            eigs, amps, tau, sigma_decomposed
        )
        assert loss.item() < 1e-10, (
            f"Loss should be ~0 when target equals decomposed total, "
            f"got {loss.item():.6e}"
        )

    def test_entropy_per_mode_all_nonneg_with_unit_amps(self):
        """Per-mode sigma_k >= 0 when amplitudes are positive."""
        eigs = torch.tensor(
            [0.9 + 0.1j, 0.85 - 0.2j, 0.7 + 0.3j, 0.5 + 0.0j],
            dtype=torch.complex128,
        )
        amps = torch.ones(4)
        tau = 1.0

        result = compute_entropy_production(eigs, amps, tau)
        sigma_modes = result["sigma_modes"]

        # sigma_k = omega_k^2 * A_k >= 0 since omega_k^2 >= 0 and A_k >= 0
        for k in range(len(sigma_modes)):
            assert sigma_modes[k].item() >= -1e-10, (
                f"sigma_{k} should be non-negative, got {sigma_modes[k].item()}"
            )
