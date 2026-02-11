"""
Tests for the data preprocessing pipeline (PRD Section 5.3).

Validates log-return computation, standardization with train-only
statistics, time-lagged pair construction, time-delay embedding,
rolling windows, and data leakage detection.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    compute_log_returns,
    standardize_returns,
    create_time_lagged_pairs,
    time_delay_embedding,
    create_rolling_windows,
    validate_no_leakage,
)


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------


class TestLogReturns:
    """Validate log-return computation."""

    def test_log_returns_shape(self):
        """Output has T-1 rows when drop_first=True."""
        prices = np.array([[100.0], [105.0], [103.0], [110.0]])
        ret = compute_log_returns(prices, drop_first=True)
        assert ret.shape == (3, 1)

    def test_log_returns_values(self):
        """r_t = ln(P_t / P_{t-1})."""
        prices = np.array([[100.0], [200.0], [100.0]])
        ret = compute_log_returns(prices, drop_first=True)
        expected = np.array([[np.log(2.0)], [-np.log(2.0)]])
        np.testing.assert_allclose(ret, expected, atol=1e-12)

    def test_log_returns_dataframe(self):
        """Works with pandas DataFrame and preserves column names."""
        df = pd.DataFrame({"SPY": [100.0, 110.0, 105.0]})
        ret = compute_log_returns(df, drop_first=True)
        assert isinstance(ret, pd.DataFrame)
        assert list(ret.columns) == ["SPY"]
        assert len(ret) == 2

    def test_log_returns_no_drop(self):
        """First row is NaN when drop_first=False."""
        prices = np.array([[100.0], [110.0]])
        ret = compute_log_returns(prices, drop_first=False)
        assert ret.shape == (2, 1)
        assert np.isnan(ret[0, 0])

    def test_log_returns_multivariate(self):
        """Handles multiple columns independently."""
        prices = np.array([[100.0, 50.0], [110.0, 55.0], [105.0, 60.0]])
        ret = compute_log_returns(prices, drop_first=True)
        assert ret.shape == (2, 2)
        np.testing.assert_allclose(ret[0, 0], np.log(110 / 100), atol=1e-12)
        np.testing.assert_allclose(ret[0, 1], np.log(55 / 50), atol=1e-12)


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


class TestStandardization:
    """Validate train-only standardization."""

    def test_zscore_train_only(self):
        """Stats computed only on train slice."""
        data = np.array([[1.0], [2.0], [3.0], [100.0], [200.0]])
        std, stats = standardize_returns(data, method="zscore", train_end_idx=3)
        # Mean/std should be from [1, 2, 3] only
        np.testing.assert_allclose(stats["center"], [2.0], atol=1e-12)
        np.testing.assert_allclose(stats["scale"], [1.0], atol=1e-12)
        # Train slice should be standardized
        np.testing.assert_allclose(std[:3, 0], [-1.0, 0.0, 1.0], atol=1e-12)

    def test_robust_method(self):
        """Robust uses median/IQR."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        std, stats = standardize_returns(data, method="robust", train_end_idx=5)
        assert stats["center"][0] == pytest.approx(3.0, abs=1e-12)  # median
        assert stats["scale"][0] > 0  # IQR

    def test_precomputed_stats(self):
        """Pre-computed stats override train_end_idx."""
        data = np.array([[10.0], [20.0], [30.0]])
        stats = {"center": np.array([0.0]), "scale": np.array([10.0])}
        std, _ = standardize_returns(data, stats=stats)
        np.testing.assert_allclose(std[:, 0], [1.0, 2.0, 3.0], atol=1e-12)

    def test_zero_scale_guard(self):
        """Constant columns get scale=1.0 (no division by zero)."""
        data = np.array([[5.0], [5.0], [5.0]])
        std, stats = standardize_returns(data, method="zscore")
        assert stats["scale"][0] == 1.0


# ---------------------------------------------------------------------------
# Time-lagged pairs
# ---------------------------------------------------------------------------


class TestTimeLaggedPairs:
    """Validate Koopman pair construction."""

    def test_lag_1(self):
        """Lag=1 gives consecutive pairs."""
        data = np.arange(10).reshape(-1, 1).astype(float)
        x_t, x_tau = create_time_lagged_pairs(data, lag=1)
        assert x_t.shape == (9, 1)
        assert x_tau.shape == (9, 1)
        np.testing.assert_array_equal(x_t[:, 0], np.arange(9))
        np.testing.assert_array_equal(x_tau[:, 0], np.arange(1, 10))

    def test_lag_5(self):
        """Lag=5 gives pairs separated by 5 steps."""
        data = np.arange(20).reshape(-1, 1).astype(float)
        x_t, x_tau = create_time_lagged_pairs(data, lag=5)
        assert x_t.shape == (15, 1)
        np.testing.assert_array_equal(x_tau[0, 0], 5.0)

    def test_invalid_lag(self):
        """Lag=0 or lag >= T raises."""
        data = np.zeros((10, 1))
        with pytest.raises(ValueError):
            create_time_lagged_pairs(data, lag=0)
        with pytest.raises(ValueError):
            create_time_lagged_pairs(data, lag=10)

    def test_copies_not_views(self):
        """Returned arrays are independent copies."""
        data = np.arange(10).reshape(-1, 1).astype(float)
        x_t, x_tau = create_time_lagged_pairs(data, lag=1)
        x_t[0, 0] = -999
        assert data[0, 0] == 0.0  # original unchanged


# ---------------------------------------------------------------------------
# Time-delay embedding
# ---------------------------------------------------------------------------


class TestTimeDelayEmbedding:
    """Validate Takens embedding."""

    def test_embedding_shape(self):
        """Output shape: (T - (d-1)*delay, D*d)."""
        x = np.arange(100).astype(float)
        emb = time_delay_embedding(x, embedding_dim=3, delay=1)
        assert emb.shape == (98, 3)

    def test_embedding_values(self):
        """Columns are shifted copies of the input."""
        x = np.arange(10).astype(float)
        emb = time_delay_embedding(x, embedding_dim=3, delay=1)
        # Row 0 should be [0, 1, 2]
        np.testing.assert_array_equal(emb[0], [0, 1, 2])
        # Row 1 should be [1, 2, 3]
        np.testing.assert_array_equal(emb[1], [1, 2, 3])

    def test_embedding_delay_2(self):
        """Delay=2 skips every other step."""
        x = np.arange(20).astype(float)
        emb = time_delay_embedding(x, embedding_dim=3, delay=2)
        # Row 0: [0, 2, 4]
        np.testing.assert_array_equal(emb[0], [0, 2, 4])

    def test_multivariate_embedding(self):
        """Multivariate input: each column embedded independently."""
        x = np.column_stack([np.arange(20), np.arange(20) * 10]).astype(float)
        emb = time_delay_embedding(x, embedding_dim=3, delay=1)
        assert emb.shape == (18, 6)  # 2 cols * 3 dims

    def test_invalid_params(self):
        """Invalid embedding_dim or delay raises."""
        x = np.arange(10).astype(float)
        with pytest.raises(ValueError):
            time_delay_embedding(x, embedding_dim=1)
        with pytest.raises(ValueError):
            time_delay_embedding(x, embedding_dim=2, delay=0)


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------


class TestRollingWindows:
    """Validate sliding window construction."""

    def test_window_shape(self):
        """Correct number of windows and window size."""
        data = np.arange(100).reshape(-1, 1).astype(float)
        windows = create_rolling_windows(data, window_size=10, stride=1)
        assert windows.shape == (91, 10, 1)

    def test_window_stride(self):
        """Stride > 1 reduces window count."""
        data = np.arange(100).reshape(-1, 1).astype(float)
        windows = create_rolling_windows(data, window_size=10, stride=5)
        assert windows.shape == (19, 10, 1)

    def test_window_too_large(self):
        """Window larger than data raises."""
        data = np.zeros((5, 1))
        with pytest.raises(ValueError):
            create_rolling_windows(data, window_size=10)


# ---------------------------------------------------------------------------
# Leakage validation
# ---------------------------------------------------------------------------


class TestLeakageValidation:
    """Validate the data leakage detection utility."""

    def test_clean_splits_pass(self):
        """Non-overlapping chronological splits pass all checks."""
        dates = pd.date_range("2000-01-01", periods=2000, freq="B")
        date_ranges = {
            "train": ("2000-01-01", "2002-12-31"),
            "val": ("2003-01-01", "2004-12-31"),
            "test": ("2005-01-01", "2006-12-31"),
        }
        stats = {"center": np.array([0.0]), "scale": np.array([1.0])}
        report = validate_no_leakage(dates, date_ranges, stats, train_end_idx=500)
        assert report["passed"] is True

    def test_overlapping_splits_fail(self):
        """Overlapping date ranges are detected."""
        dates = pd.date_range("2000-01-01", periods=1000, freq="B")
        date_ranges = {
            "train": ("2000-01-01", "2003-06-30"),
            "val": ("2003-01-01", "2003-12-31"),  # overlaps train
            "test": ("2004-01-01", "2004-12-31"),
        }
        stats = {"center": np.array([0.0]), "scale": np.array([1.0])}
        report = validate_no_leakage(dates, date_ranges, stats, train_end_idx=500)
        assert report["passed"] is False

    def test_stats_mismatch_detected(self):
        """Wrong standardization stats are caught."""
        dates = pd.date_range("2000-01-01", periods=100, freq="B")
        date_ranges = {
            "train": ("2000-01-01", "2000-03-31"),
            "test": ("2000-04-01", "2000-06-30"),
        }
        rng = np.random.RandomState(42)
        returns = rng.randn(100, 1)
        # Give wrong stats
        bad_stats = {"center": np.array([999.0]), "scale": np.array([0.001])}
        report = validate_no_leakage(
            dates, date_ranges, bad_stats, train_end_idx=50, returns=returns,
        )
        assert report["passed"] is False
