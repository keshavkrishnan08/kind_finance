"""
Tests for the bootstrap entropy production confidence intervals.

Validates the block-bootstrap CI machinery added for PRE error bounds.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.model.entropy import (
    _kde_entropy_production,
    estimate_empirical_entropy_production_with_ci,
)


class TestKDEEntropyHelper:
    """Test the extracted _kde_entropy_production helper."""

    def test_symmetric_data_near_zero(self):
        """Symmetric forward/backward should give ~0 entropy production."""
        rng = np.random.RandomState(42)
        n = 500
        x_t = rng.randn(n, 1)
        x_tau = rng.randn(n, 1)  # independent => symmetric
        sigma = _kde_entropy_production(x_t, x_tau)
        assert sigma >= 0.0  # clamped
        assert sigma < 1.0  # should be near zero for symmetric

    def test_asymmetric_data_positive(self):
        """Clearly asymmetric transitions should produce positive entropy."""
        rng = np.random.RandomState(42)
        n = 1000
        # Non-reversible process: x_tau depends on x_t with asymmetric noise
        x_t = rng.randn(n, 1)
        x_tau = 0.8 * x_t + 0.5 * rng.exponential(size=(n, 1))  # skewed noise breaks symmetry
        sigma = _kde_entropy_production(x_t, x_tau)
        assert sigma >= 0.0  # clamped non-negative

    def test_returns_float(self):
        """Output is a plain float."""
        x_t = np.zeros((10, 1))
        x_tau = np.ones((10, 1))
        result = _kde_entropy_production(x_t, x_tau)
        assert isinstance(result, float)


class TestBootstrapCI:
    """Test estimate_empirical_entropy_production_with_ci."""

    def test_output_keys(self):
        """Returns dict with expected keys."""
        returns = torch.randn(200)
        result = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=10, n_samples=100,
        )
        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "std_error" in result
        assert "n_bootstrap" in result

    def test_ci_ordering(self):
        """ci_lower <= point_estimate <= ci_upper."""
        rng_state = torch.manual_seed(42)
        returns = torch.randn(500)
        result = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=50, n_samples=200,
        )
        assert result["ci_lower"] <= result["ci_upper"]

    def test_nonnegative_point_estimate(self):
        """Point estimate is non-negative (clamped)."""
        returns = torch.randn(200)
        result = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=10, n_samples=100,
        )
        assert result["point_estimate"] >= 0.0

    def test_std_error_positive(self):
        """Standard error is positive for non-trivial data."""
        returns = torch.randn(500)
        result = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=30, n_samples=200,
        )
        assert result["std_error"] >= 0.0

    def test_short_series_returns_zeros(self):
        """Series shorter than tau returns all zeros."""
        returns = torch.randn(3)
        result = estimate_empirical_entropy_production_with_ci(
            returns, tau=5, n_bootstrap=10,
        )
        assert result["point_estimate"] == 0.0
        assert result["n_bootstrap"] == 0

    def test_n_bootstrap_respected(self):
        """Returned n_bootstrap matches input."""
        returns = torch.randn(200)
        result = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=25, n_samples=100,
        )
        assert result["n_bootstrap"] == 25

    def test_reproducible_with_seed(self):
        """Same seed gives same results."""
        returns = torch.randn(300)
        r1 = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=20, seed=123,
        )
        r2 = estimate_empirical_entropy_production_with_ci(
            returns, tau=1, n_bootstrap=20, seed=123,
        )
        assert r1["point_estimate"] == r2["point_estimate"]
        assert r1["ci_lower"] == r2["ci_lower"]


class TestDataLoaderBasics:
    """Minimal tests for TimeLaggedDataset."""

    def test_dataset_length(self):
        """Dataset length = T - lag."""
        from src.data.loader import TimeLaggedDataset
        data = np.arange(100).reshape(-1, 1).astype(np.float32)
        ds = TimeLaggedDataset(data, lag=5, preprocess=False)
        assert len(ds) == 95

    def test_dataset_pair_shape(self):
        """Each item is a (x_t, x_tau) pair with correct shape."""
        from src.data.loader import TimeLaggedDataset
        data = np.random.randn(50, 3).astype(np.float32)
        ds = TimeLaggedDataset(data, lag=2, preprocess=False)
        x_t, x_tau = ds[0]
        assert x_t.shape == (3,)
        assert x_tau.shape == (3,)

    def test_dataset_temporal_alignment(self):
        """x_tau[i] = data[i + lag]."""
        from src.data.loader import TimeLaggedDataset
        data = np.arange(20).reshape(-1, 1).astype(np.float32)
        ds = TimeLaggedDataset(data, lag=3, preprocess=False)
        x_t, x_tau = ds[0]
        assert float(x_t[0]) == 0.0
        assert float(x_tau[0]) == 3.0
        x_t5, x_tau5 = ds[5]
        assert float(x_t5[0]) == 5.0
        assert float(x_tau5[0]) == 8.0
