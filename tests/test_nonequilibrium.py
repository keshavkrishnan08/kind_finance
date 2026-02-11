"""Tests for post-hoc non-equilibrium diagnostics."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.analysis.nonequilibrium import (
    detailed_balance_violation,
    gallavotti_cohen_symmetry,
    fluctuation_theorem_ratio,
    onsager_regression_test,
    eigenvalue_complex_plane_statistics,
)


class TestDetailedBalanceViolation:
    """Detailed balance violation metric."""

    def test_symmetric_matrix_zero_violation(self):
        """Symmetric K should have D = 0."""
        K = np.array([[0.9, 0.1], [0.1, 0.8]])
        result = detailed_balance_violation(K)
        assert result["violation"] < 1e-10
        assert result["antisymmetric_norm"] < 1e-10

    def test_antisymmetric_matrix_max_violation(self):
        """Pure antisymmetric K should have high violation."""
        K = np.array([[0.0, 1.0], [-1.0, 0.0]])
        result = detailed_balance_violation(K)
        assert result["violation"] > 0.5
        assert result["symmetric_norm"] < 1e-10

    def test_general_matrix_bounded(self):
        """Violation metric should be in [0, sqrt(2)]."""
        rng = np.random.default_rng(42)
        K = rng.standard_normal((10, 10))
        result = detailed_balance_violation(K)
        assert 0.0 <= result["violation"] <= np.sqrt(2) + 1e-10

    def test_output_keys(self):
        """All expected keys present."""
        K = np.eye(3)
        result = detailed_balance_violation(K)
        assert set(result.keys()) == {
            "violation", "symmetric_norm", "antisymmetric_norm", "ratio"
        }


class TestGallavottiCohenSymmetry:
    """Gallavotti-Cohen symmetry function tests."""

    def test_symmetric_samples_zero_slope(self):
        """Symmetric distribution should give slope ~0."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(10000)
        result = gallavotti_cohen_symmetry(samples)
        assert abs(result["slope"]) < 0.5

    def test_output_keys(self):
        """All expected keys present."""
        samples = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        result = gallavotti_cohen_symmetry(samples, n_bins=5)
        assert "slope" in result
        assert "intercept" in result
        assert "r_squared" in result

    def test_zero_samples(self):
        """Near-zero samples should not crash."""
        samples = np.zeros(100)
        result = gallavotti_cohen_symmetry(samples)
        assert result["slope"] == 0.0

    def test_biased_samples_positive_slope(self):
        """Positively biased samples should give positive slope."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(10000) + 1.0  # shifted positive
        result = gallavotti_cohen_symmetry(samples, n_bins=30)
        assert result["slope"] > 0


class TestFluctuationTheoremRatio:
    """Integral fluctuation theorem tests."""

    def test_zero_entropy_production(self):
        """Zero entropy production -> <exp(-sigma)> = 1."""
        samples = np.zeros(1000)
        result = fluctuation_theorem_ratio(samples)
        assert abs(result["mean_exp_neg_sigma"] - 1.0) < 1e-10

    def test_output_keys(self):
        """All expected keys present."""
        samples = np.array([0.1, 0.2, -0.1])
        result = fluctuation_theorem_ratio(samples)
        assert set(result.keys()) == {
            "mean_exp_neg_sigma", "log_deviation", "n_samples"
        }

    def test_n_samples_correct(self):
        """n_samples matches input length."""
        samples = np.ones(42)
        result = fluctuation_theorem_ratio(samples)
        assert result["n_samples"] == 42


class TestOnsagerRegressionTest:
    """Onsager regression hypothesis test."""

    def test_perfect_match(self):
        """When rates match perfectly, correlation = 1."""
        eigs = np.array([0.9, 0.8, 0.7]) + 0j
        tau = 1.0
        rates = -np.log(np.abs(eigs)) / tau
        acf_times = 1.0 / rates
        result = onsager_regression_test(eigs, acf_times, tau=tau)
        assert result["correlation"] > 0.99

    def test_output_keys(self):
        """All expected keys present."""
        eigs = np.array([0.9 + 0.1j, 0.8 + 0j])
        acf_times = np.array([10.0, 5.0])
        result = onsager_regression_test(eigs, acf_times)
        assert "correlation" in result
        assert "koopman_rates" in result
        assert "acf_rates" in result
        assert "relative_errors" in result


class TestEigenvalueStatistics:
    """Eigenvalue complex plane statistics."""

    def test_real_eigenvalues(self):
        """Real eigenvalues should have zero complex fraction."""
        eigs = np.array([0.95, 0.8, 0.6, 0.3]) + 0j
        result = eigenvalue_complex_plane_statistics(eigs)
        assert result["n_complex_modes"] == 0
        assert result["complex_fraction"] < 1e-10

    def test_complex_eigenvalues(self):
        """Complex eigenvalues should have nonzero complex fraction."""
        eigs = np.array([0.9 + 0.3j, 0.9 - 0.3j, 0.7 + 0j, 0.5 + 0j])
        result = eigenvalue_complex_plane_statistics(eigs)
        assert result["n_complex_modes"] >= 2
        assert result["complex_fraction"] > 0

    def test_sorted_by_magnitude(self):
        """Eigenvalues should be sorted by descending magnitude."""
        eigs = np.array([0.3, 0.9, 0.5, 0.7]) + 0j
        result = eigenvalue_complex_plane_statistics(eigs)
        mags = result["magnitudes"]
        assert all(mags[i] >= mags[i + 1] for i in range(len(mags) - 1))

    def test_output_keys(self):
        """All expected keys present."""
        eigs = np.array([0.9, 0.5]) + 0j
        result = eigenvalue_complex_plane_statistics(eigs)
        expected = {
            "eigenvalues_sorted", "magnitudes", "angles", "decay_rates",
            "frequencies", "n_real_modes", "n_complex_modes",
            "complex_fraction", "spectral_radius", "condition_number",
        }
        assert set(result.keys()) == expected
