"""
Unit tests for Koopman analysis (KoopmanAnalyzer).

Validates:
    - Spectral gap positivity
    - Decay rate positivity for non-trivial modes
    - Regime persistence lower bound
    - Eigenvalue sorting by decreasing magnitude
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.model.vampnet import NonEquilibriumVAMPNet
from src.model.koopman import KoopmanAnalyzer
from src.model.losses import total_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train a small model on synthetic AR data for Koopman analysis."""
    torch.manual_seed(42)
    np.random.seed(42)

    d = 5
    N = 1000
    tau = 1

    # Generate an AR(1) process with known structure
    data = np.zeros((N, d))
    A = np.diag([0.95, 0.85, 0.70, 0.50, 0.30])  # known decay rates
    for t in range(1, N):
        data[t] = A @ data[t - 1] + 0.1 * np.random.randn(d)

    x_t = torch.tensor(data[:-tau], dtype=torch.float32)
    x_tau = torch.tensor(data[tau:], dtype=torch.float32)

    model = NonEquilibriumVAMPNet(
        input_dim=d,
        hidden_dims=[32, 16],
        output_dim=4,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(x_t, x_tau)
        loss, _ = total_loss(out, tau=float(tau))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output_dict = model(x_t, x_tau)

    analyzer = KoopmanAnalyzer(model)
    return analyzer, model, output_dict, x_t, x_tau, float(tau)


@pytest.fixture
def known_eigenvalues():
    """Eigenvalues with known spectral properties."""
    # lambda_0 close to 1 (stationary mode), lambda_1 smaller, etc.
    return torch.tensor(
        [0.98 + 0.0j, 0.85 + 0.1j, 0.85 - 0.1j, 0.60 + 0.0j],
        dtype=torch.complex128,
    )


@pytest.fixture
def real_eigenvalues():
    """Purely real eigenvalues (reversible system)."""
    return torch.tensor(
        [0.99 + 0.0j, 0.90 + 0.0j, 0.75 + 0.0j, 0.50 + 0.0j],
        dtype=torch.complex128,
    )


# ---------------------------------------------------------------------------
# Tests: Spectral gap positivity
# ---------------------------------------------------------------------------

class TestSpectralGap:
    """Spectral gap should be positive for a non-trivial system."""

    def test_spectral_gap_positive(self, known_eigenvalues):
        """Delta > 0 for eigenvalues with distinct magnitudes."""
        gap = KoopmanAnalyzer.compute_spectral_gap(known_eigenvalues, tau=1.0)
        assert gap.item() > 0, (
            f"Spectral gap should be positive, got {gap.item():.6f}"
        )

    def test_spectral_gap_value(self, real_eigenvalues):
        """Verify spectral gap value for known real eigenvalues."""
        gap = KoopmanAnalyzer.compute_spectral_gap(real_eigenvalues, tau=1.0)

        # lambda_2 = 0.90, spectral gap = |Re(ln(0.90))| / 1.0 = |ln(0.90)|
        expected = abs(np.log(0.90))
        assert abs(gap.item() - expected) < 1e-4, (
            f"Expected spectral gap = {expected:.6f}, got {gap.item():.6f}"
        )

    def test_spectral_gap_trained_model(self, trained_model_and_data):
        """Trained model should have a positive spectral gap."""
        analyzer, _, output_dict, _, _, tau = trained_model_and_data
        eigs = output_dict["eigenvalues"].cpu()
        gap = KoopmanAnalyzer.compute_spectral_gap(eigs, tau)
        assert gap.item() > 0, (
            f"Trained model spectral gap should be positive, "
            f"got {gap.item():.6f}"
        )

    def test_spectral_gap_single_eigenvalue(self):
        """Single eigenvalue should give gap = 0."""
        eigs = torch.tensor([0.9 + 0.0j], dtype=torch.complex128)
        gap = KoopmanAnalyzer.compute_spectral_gap(eigs, tau=1.0)
        assert gap.item() == 0.0, (
            f"Single eigenvalue gap should be 0, got {gap.item()}"
        )


# ---------------------------------------------------------------------------
# Tests: Decay rates positive
# ---------------------------------------------------------------------------

class TestDecayRates:
    """Decay rates gamma_k = -ln|lambda_k| / tau should be positive for |lambda| < 1."""

    def test_decay_rates_positive(self, known_eigenvalues):
        """gamma_k > 0 for k > 0 (all |lambda_k| < 1)."""
        tau = 1.0
        magnitudes = known_eigenvalues.abs()

        # gamma_k = -ln(|lambda_k|) / tau
        decay_rates = -torch.log(magnitudes) / tau
        decay_rates_np = decay_rates.numpy()

        # All magnitudes are < 1, so all decay rates should be positive
        assert np.all(decay_rates_np > 0), (
            f"All decay rates should be positive for |lambda| < 1, "
            f"got {decay_rates_np}"
        )

    def test_decay_rates_ordering(self, real_eigenvalues):
        """Larger |lambda| -> smaller decay rate (slower mode)."""
        tau = 1.0
        magnitudes = real_eigenvalues.abs()
        sorted_idx = torch.argsort(magnitudes, descending=True)
        sorted_mags = magnitudes[sorted_idx]

        decay_rates = -torch.log(sorted_mags) / tau
        decay_np = decay_rates.numpy()

        # Decay rates should be increasing (sorted by decreasing magnitude)
        for k in range(len(decay_np) - 1):
            assert decay_np[k] <= decay_np[k + 1] + 1e-10, (
                f"Decay rate {k} ({decay_np[k]:.6f}) should be <= "
                f"decay rate {k+1} ({decay_np[k+1]:.6f})"
            )

    def test_decay_rates_from_trained(self, trained_model_and_data):
        """Trained model should have positive decay rates."""
        _, _, output_dict, _, _, tau = trained_model_and_data
        eigs = output_dict["eigenvalues"].cpu()
        magnitudes = eigs.abs()

        # Only consider modes with |lambda| < 1
        valid = magnitudes < 1.0
        if valid.any():
            decay_rates = -torch.log(magnitudes[valid]) / tau
            assert (decay_rates > 0).all(), (
                f"Decay rates should be positive, got {decay_rates.numpy()}"
            )


# ---------------------------------------------------------------------------
# Tests: Regime persistence bound
# ---------------------------------------------------------------------------

class TestRegimePersistence:
    """Regime persistence T >= 1 / spectral_gap."""

    def test_regime_persistence_bound(self, known_eigenvalues):
        """T_regime >= 1 / Delta."""
        gap = KoopmanAnalyzer.compute_spectral_gap(known_eigenvalues, tau=1.0)
        persist = KoopmanAnalyzer.regime_persistence_bound(gap, tau=1.0)

        expected = 1.0 / gap.item()
        assert abs(persist.item() - expected) < 1e-4, (
            f"Persistence bound = {persist.item():.4f}, expected 1/gap = "
            f"{expected:.4f}"
        )
        assert persist.item() >= 1.0 / gap.item() - 1e-10, (
            f"Persistence should be >= 1/gap"
        )

    def test_regime_persistence_large_gap(self):
        """Large spectral gap -> short persistence time."""
        gap = torch.tensor(2.0)
        persist = KoopmanAnalyzer.regime_persistence_bound(gap, tau=1.0)
        assert persist.item() == pytest.approx(0.5, abs=1e-6), (
            f"1/2.0 = 0.5, got {persist.item()}"
        )

    def test_regime_persistence_small_gap(self):
        """Small spectral gap -> long persistence time."""
        gap = torch.tensor(0.01)
        persist = KoopmanAnalyzer.regime_persistence_bound(gap, tau=1.0)
        assert persist.item() == pytest.approx(100.0, abs=1e-2), (
            f"1/0.01 = 100, got {persist.item()}"
        )

    def test_regime_persistence_zero_gap(self):
        """Near-zero gap -> very large persistence (clamped)."""
        gap = torch.tensor(0.0)
        persist = KoopmanAnalyzer.regime_persistence_bound(gap, tau=1.0)
        assert persist.item() > 1e10, (
            f"Zero gap should give very large persistence, "
            f"got {persist.item()}"
        )


# ---------------------------------------------------------------------------
# Tests: Eigenvalue sorting
# ---------------------------------------------------------------------------

class TestEigenvalueSorting:
    """Eigenvalues should be sorted by decreasing magnitude."""

    def test_eigenvalue_sorting(self, known_eigenvalues):
        """After sorting, |lambda_k| >= |lambda_{k+1}|."""
        magnitudes = known_eigenvalues.abs()
        sorted_idx = torch.argsort(magnitudes, descending=True)
        sorted_eigs = known_eigenvalues[sorted_idx]
        sorted_mags = sorted_eigs.abs().numpy()

        for k in range(len(sorted_mags) - 1):
            assert sorted_mags[k] >= sorted_mags[k + 1] - 1e-10, (
                f"|lambda_{k}| = {sorted_mags[k]:.6f} should be >= "
                f"|lambda_{k+1}| = {sorted_mags[k+1]:.6f}"
            )

    def test_eigenvalue_sorting_preserves_count(self, known_eigenvalues):
        """Sorting should not change the number of eigenvalues."""
        magnitudes = known_eigenvalues.abs()
        sorted_idx = torch.argsort(magnitudes, descending=True)
        sorted_eigs = known_eigenvalues[sorted_idx]
        assert len(sorted_eigs) == len(known_eigenvalues), (
            f"Sorting changed eigenvalue count: {len(sorted_eigs)} vs "
            f"{len(known_eigenvalues)}"
        )

    def test_eigenvalue_sorting_trained(self, trained_model_and_data):
        """Trained model eigenvalues can be sorted by magnitude."""
        _, _, output_dict, _, _, _ = trained_model_and_data
        eigs = output_dict["eigenvalues"].cpu()
        magnitudes = eigs.abs()
        sorted_idx = torch.argsort(magnitudes, descending=True)
        sorted_mags = magnitudes[sorted_idx].numpy()

        for k in range(len(sorted_mags) - 1):
            assert sorted_mags[k] >= sorted_mags[k + 1] - 1e-10, (
                f"|lambda_{k}| = {sorted_mags[k]:.6f} should be >= "
                f"|lambda_{k+1}| = {sorted_mags[k+1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Tests: Spectral summary
# ---------------------------------------------------------------------------

class TestSpectralSummary:
    """KoopmanAnalyzer.spectral_summary() returns all expected keys."""

    def test_spectral_summary_keys(self, trained_model_and_data):
        """spectral_summary should return all expected keys."""
        analyzer, _, _, x_t, x_tau, tau = trained_model_and_data
        summary = analyzer.spectral_summary(x_t, x_tau, tau)

        expected_keys = {
            "koopman_matrix",
            "eigenvalues",
            "singular_values",
            "spectral_gap",
            "regime_persistence_bound",
        }
        assert expected_keys.issubset(summary.keys()), (
            f"Missing keys: {expected_keys - set(summary.keys())}"
        )

    def test_spectral_summary_gap_matches(self, trained_model_and_data):
        """spectral_gap in summary should match compute_spectral_gap."""
        analyzer, _, _, x_t, x_tau, tau = trained_model_and_data
        summary = analyzer.spectral_summary(x_t, x_tau, tau)

        gap_direct = KoopmanAnalyzer.compute_spectral_gap(
            summary["eigenvalues"], tau
        )
        gap_summary = summary["spectral_gap"]

        assert abs(gap_direct.item() - gap_summary.item()) < 1e-6, (
            f"Spectral gap mismatch: direct={gap_direct.item():.6f}, "
            f"summary={gap_summary.item():.6f}"
        )
