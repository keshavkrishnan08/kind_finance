"""Preprocessing pipeline for KTND-Finance (PRD Section 5.3).

All functions operate on numpy arrays or pandas DataFrames with vectorized
operations for maximum throughput. The pipeline converts raw close prices
into log-returns, standardises them using train-only statistics, builds
time-lagged observation pairs for the Koopman operator, and provides
utilities for time-delay embedding and false-nearest-neighbor analysis.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. Log-return computation
# ---------------------------------------------------------------------------

def compute_log_returns(
    prices: Union[pd.DataFrame, NDArray[np.floating]],
    drop_first: bool = True,
) -> Union[pd.DataFrame, NDArray[np.floating]]:
    """Compute log-returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : DataFrame or ndarray of shape (T, D)
        Raw close prices.  Each column is an asset; rows are time-ordered.
    drop_first : bool
        If True, drop the first row (which is NaN by construction).

    Returns
    -------
    Same type as *prices*, shape (T-1, D) if *drop_first* else (T, D).
    """
    is_df = isinstance(prices, pd.DataFrame)
    arr = prices.values if is_df else np.asarray(prices, dtype=np.float64)

    # Vectorised log-return: ln(P_t) - ln(P_{t-1})
    log_prices = np.log(arr)
    log_ret = np.empty_like(log_prices)
    log_ret[0] = np.nan
    log_ret[1:] = log_prices[1:] - log_prices[:-1]

    if drop_first:
        log_ret = log_ret[1:]
        if is_df:
            return pd.DataFrame(log_ret, columns=prices.columns, index=prices.index[1:])
    else:
        if is_df:
            return pd.DataFrame(log_ret, columns=prices.columns, index=prices.index)

    return log_ret


# ---------------------------------------------------------------------------
# 2. Standardisation (train-only statistics)
# ---------------------------------------------------------------------------

def standardize_returns(
    returns: Union[pd.DataFrame, NDArray[np.floating]],
    method: Literal["zscore", "robust"] = "zscore",
    train_end_idx: Optional[int] = None,
    stats: Optional[dict] = None,
) -> Tuple[Union[pd.DataFrame, NDArray[np.floating]], dict]:
    """Standardise returns using statistics computed *only* on the training set.

    Parameters
    ----------
    returns : DataFrame or ndarray of shape (T, D)
        Log-returns to standardise.
    method : {"zscore", "robust"}
        ``"zscore"`` -> (x - mean) / std
        ``"robust"`` -> (x - median) / IQR
    train_end_idx : int, optional
        Index (exclusive) marking the end of the training period.  When
        *stats* is ``None`` the function computes statistics on
        ``returns[:train_end_idx]``.  If ``None``, all rows are treated as
        training data.
    stats : dict, optional
        Pre-computed ``{"center": ..., "scale": ...}`` arrays.  When
        supplied, *train_end_idx* and *method* are ignored.

    Returns
    -------
    standardised : same type as *returns*
        Standardised array / DataFrame.
    stats : dict
        ``{"center": ndarray, "scale": ndarray}`` used for the transform.
    """
    is_df = isinstance(returns, pd.DataFrame)
    arr = returns.values.astype(np.float64) if is_df else np.asarray(returns, dtype=np.float64)
    train = arr[:train_end_idx] if train_end_idx is not None else arr

    if stats is None:
        if method == "zscore":
            center = np.nanmean(train, axis=0)
            scale = np.nanstd(train, axis=0, ddof=1)
        elif method == "robust":
            center = np.nanmedian(train, axis=0)
            q75 = np.nanpercentile(train, 75, axis=0)
            q25 = np.nanpercentile(train, 25, axis=0)
            scale = q75 - q25
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'zscore' or 'robust'.")
        # Guard against zero / near-zero scale
        scale = np.where(scale < 1e-12, 1.0, scale)
        stats = {"center": center, "scale": scale}

    standardised = (arr - stats["center"]) / stats["scale"]

    if is_df:
        standardised = pd.DataFrame(standardised, columns=returns.columns, index=returns.index)

    return standardised, stats


# ---------------------------------------------------------------------------
# 3. Time-lagged pair construction (Koopman operator)
# ---------------------------------------------------------------------------

def create_time_lagged_pairs(
    data: NDArray[np.floating],
    lag: int = 1,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Create (x_t, x_{t+lag}) observation pairs for the Koopman formulation.

    Parameters
    ----------
    data : ndarray of shape (T, D)
        Standardised observable matrix.
    lag : int
        Number of time-steps between the pair elements.

    Returns
    -------
    x_t : ndarray of shape (T - lag, D)
    x_t_lag : ndarray of shape (T - lag, D)
    """
    data = np.asarray(data, dtype=np.float64)
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if lag >= data.shape[0]:
        raise ValueError(f"lag ({lag}) must be < number of samples ({data.shape[0]})")
    return data[:-lag].copy(), data[lag:].copy()


# ---------------------------------------------------------------------------
# 4. Time-delay embedding (Takens' theorem)
# ---------------------------------------------------------------------------

def time_delay_embedding(
    x: NDArray[np.floating],
    embedding_dim: int,
    delay: int = 1,
) -> NDArray[np.floating]:
    """Construct a time-delay embedding matrix (Takens' reconstruction).

    For a univariate series of length T the result has shape
    ``(T - (embedding_dim - 1) * delay, embedding_dim)``.
    For a multivariate series of shape (T, D) each column is embedded
    independently and the results are concatenated along axis-1, giving
    shape ``(T - (embedding_dim - 1) * delay, D * embedding_dim)``.

    Parameters
    ----------
    x : ndarray of shape (T,) or (T, D)
        Input time series.
    embedding_dim : int
        Number of delay coordinates (>= 2).
    delay : int
        Spacing between successive delay coordinates.

    Returns
    -------
    embedded : ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    if embedding_dim < 2:
        raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    if x.ndim == 1:
        x = x[:, np.newaxis]
    T, D = x.shape
    n_rows = T - (embedding_dim - 1) * delay
    if n_rows <= 0:
        raise ValueError(
            f"Time series too short (T={T}) for embedding_dim={embedding_dim}, delay={delay}"
        )

    # Build column indices for each lag dimension: shape (embedding_dim,)
    col_offsets = np.arange(embedding_dim) * delay  # [0, delay, 2*delay, ...]
    # Row indices: (n_rows, 1) + (1, embedding_dim) -> (n_rows, embedding_dim)
    row_idx = np.arange(n_rows)[:, np.newaxis] + col_offsets[np.newaxis, :]

    # Advanced indexing – one gather per feature column, then hstack
    embedded = np.empty((n_rows, D * embedding_dim), dtype=np.float64)
    for d in range(D):
        embedded[:, d * embedding_dim:(d + 1) * embedding_dim] = x[row_idx, d]

    return embedded


# ---------------------------------------------------------------------------
# 5. False nearest neighbours (FNN) for embedding-dim selection
# ---------------------------------------------------------------------------

def false_nearest_neighbors(
    x: NDArray[np.floating],
    max_dim: int = 10,
    delay: int = 1,
    rtol: float = 15.0,
    atol: float = 2.0,
    n_neighbours: int = 1,
) -> NDArray[np.floating]:
    """Estimate the fraction of false nearest neighbours for dims 1..max_dim.

    Uses the Kennel-Brown-Abarbanel criterion with both relative and
    absolute tolerance thresholds.

    Parameters
    ----------
    x : ndarray of shape (T,) or (T, D)
        Input time series (typically univariate).
    max_dim : int
        Maximum embedding dimension to test.
    delay : int
        Time delay for the embedding.
    rtol : float
        Relative distance tolerance (default 15).
    atol : float
        Absolute distance tolerance as multiple of data std (default 2).
    n_neighbours : int
        Number of nearest neighbours to evaluate per point (default 1).

    Returns
    -------
    fnn_fractions : ndarray of shape (max_dim,)
        Fraction of false nearest neighbours for embedding dimensions
        1, 2, ..., *max_dim*.
    """
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]

    sigma = np.std(x)
    atol_thresh = atol * sigma
    fnn_fractions = np.empty(max_dim, dtype=np.float64)

    for dim in range(1, max_dim + 1):
        # Embedding at dim and dim+1.  dim=1 is just the raw series
        # (time_delay_embedding requires embedding_dim >= 2).
        if dim == 1:
            n_rows_d1 = x.shape[0] - delay
            emb_d = x[:n_rows_d1].reshape(n_rows_d1, -1)
        else:
            emb_d = time_delay_embedding(x, embedding_dim=dim, delay=delay)
        emb_d1 = time_delay_embedding(x, embedding_dim=dim + 1, delay=delay)
        n_pts = emb_d1.shape[0]
        emb_d = emb_d[:n_pts]  # trim to same length

        tree = cKDTree(emb_d)
        # k = n_neighbours + 1 because query includes the point itself
        dists, idxs = tree.query(emb_d, k=n_neighbours + 1, workers=-1)

        # Extract neighbour distances and indices (skip self at position 0)
        nn_dists = dists[:, 1:]  # (n_pts, n_neighbours)
        nn_idxs = idxs[:, 1:]

        false_count = 0
        total_count = 0
        for k in range(n_neighbours):
            d_k = nn_dists[:, k]
            idx_k = nn_idxs[:, k]

            # Distance in the (dim+1)-th coordinate
            extra_dist = np.abs(emb_d1[:, -1] - emb_d1[idx_k, -1])

            # Avoid division by zero
            safe_d_k = np.where(d_k < 1e-15, 1e-15, d_k)

            # Criterion 1: relative increase
            crit1 = (extra_dist / safe_d_k) > rtol
            # Criterion 2: absolute distance
            new_dist = np.sqrt(d_k ** 2 + extra_dist ** 2)
            crit2 = new_dist > atol_thresh

            false_count += np.sum(crit1 | crit2)
            total_count += n_pts

        fnn_fractions[dim - 1] = false_count / max(total_count, 1)

    return fnn_fractions


# ---------------------------------------------------------------------------
# 6. Rolling windows
# ---------------------------------------------------------------------------

def create_rolling_windows(
    data: NDArray[np.floating],
    window_size: int,
    stride: int = 1,
) -> NDArray[np.floating]:
    """Create rolling (sliding) windows using stride tricks (zero-copy).

    Parameters
    ----------
    data : ndarray of shape (T, D)
        Input matrix.
    window_size : int
        Number of time-steps per window.
    stride : int
        Step between successive windows.

    Returns
    -------
    windows : ndarray of shape (N, window_size, D)
        View into *data* (no copies unless the caller writes to it).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    T, D = data.shape
    if window_size > T:
        raise ValueError(f"window_size ({window_size}) > number of samples ({T})")

    n_windows = (T - window_size) // stride + 1
    # Use np.lib.stride_tricks for a zero-copy sliding window view
    byte_stride = data.strides  # (T-stride-bytes, D-stride-bytes)
    shape = (n_windows, window_size, D)
    strides = (byte_stride[0] * stride, byte_stride[0], byte_stride[1])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windows


# ---------------------------------------------------------------------------
# 7. Data leakage validation
# ---------------------------------------------------------------------------

def validate_no_leakage(
    dates: pd.DatetimeIndex,
    date_ranges: dict[str, tuple[str, str]],
    stats: dict,
    train_end_idx: int,
    returns: Union[pd.DataFrame, NDArray[np.floating], None] = None,
) -> dict:
    """Validate that there is no data leakage across train/val/test splits.

    Checks
    ------
    1. Splits are strictly chronological with no overlap.
    2. Standardization stats were computed on train data only.
    3. No temporal overlap between any split pair.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index of the full dataset.
    date_ranges : dict
        Mapping of split name -> (start, end) date strings.
    stats : dict
        Standardization statistics (center, scale).
    train_end_idx : int
        Index marking end of training data in the returns array.
    returns : DataFrame or ndarray, optional
        If provided, re-computes stats on train slice and verifies match.

    Returns
    -------
    dict with 'passed' (bool) and detailed check results.
    """
    report: dict = {"passed": True, "checks": []}

    # Check 1: Chronological ordering
    split_names = list(date_ranges.keys())
    for i in range(len(split_names) - 1):
        curr_end = pd.Timestamp(date_ranges[split_names[i]][1])
        next_start = pd.Timestamp(date_ranges[split_names[i + 1]][0])
        gap_days = (next_start - curr_end).days
        ok = gap_days >= 1
        report["checks"].append({
            "check": f"chronological_{split_names[i]}_{split_names[i+1]}",
            "passed": ok,
            "gap_days": gap_days,
        })
        if not ok:
            report["passed"] = False

    # Check 2: No temporal overlap
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s_i = pd.Timestamp(date_ranges[split_names[i]][0])
            e_i = pd.Timestamp(date_ranges[split_names[i]][1])
            s_j = pd.Timestamp(date_ranges[split_names[j]][0])
            e_j = pd.Timestamp(date_ranges[split_names[j]][1])
            overlap = s_i <= e_j and s_j <= e_i
            report["checks"].append({
                "check": f"no_overlap_{split_names[i]}_{split_names[j]}",
                "passed": not overlap,
            })
            if overlap:
                report["passed"] = False

    # Check 3: Standardization uses only train data
    if returns is not None and stats is not None:
        arr = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
        train_slice = arr[:train_end_idx]
        recomputed_center = np.nanmean(train_slice, axis=0)
        recomputed_scale = np.nanstd(train_slice, axis=0, ddof=1)
        recomputed_scale = np.where(recomputed_scale < 1e-12, 1.0, recomputed_scale)

        center_match = np.allclose(stats["center"], recomputed_center, atol=1e-10)
        scale_match = np.allclose(stats["scale"], recomputed_scale, atol=1e-10)
        report["checks"].append({
            "check": "stats_train_only",
            "passed": center_match and scale_match,
            "center_match": bool(center_match),
            "scale_match": bool(scale_match),
        })
        if not (center_match and scale_match):
            report["passed"] = False

    # Check 4: Split sizes
    for name, (start, end) in date_ranges.items():
        mask = (dates >= start) & (dates <= end)
        n = int(mask.sum())
        report["checks"].append({
            "check": f"split_size_{name}",
            "n_samples": n,
            "passed": n > 0,
        })
        if n == 0:
            report["passed"] = False

    return report
