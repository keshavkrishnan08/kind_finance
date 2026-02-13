"""Data augmentation for statistical validation (KTND-Finance).

Provides augmentation strategies that preserve the relevant
statistical structure of the original time series while generating
synthetic variants for bootstrap confidence intervals and robustness
checks:

- ``block_bootstrap``: resamples contiguous blocks to maintain
  short-range autocorrelation.
- ``random_time_reversal``: reverses randomly selected sub-segments
  to test time-asymmetry (a hallmark of non-equilibrium dynamics).
- ``iaaft_surrogate``: Iterative Amplitude Adjusted Fourier Transform
  surrogates that preserve power spectrum and amplitude distribution
  while destroying temporal asymmetry (Schreiber & Schmitz, 2000).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def block_bootstrap(
    data: NDArray[np.floating],
    block_size: int,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Generate block-bootstrap resamples of a multivariate time series.

    Contiguous blocks of length *block_size* are drawn uniformly at
    random (with replacement) and concatenated to form a new series of
    the same length as *data*.  This preserves short-range temporal
    dependence up to the block scale.

    Parameters
    ----------
    data : ndarray of shape (T, D)
        Input time series (rows = time, columns = features).
    block_size : int
        Length of each contiguous block.  A common heuristic is
        ``int(T ** (1/3))`` (cube-root rule).
    n_samples : int
        Number of bootstrap resamples to generate.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.  If ``None`` the
        default Generator is used.

    Returns
    -------
    resamples : ndarray of shape (n_samples, T, D)
        Stack of bootstrap resamples.  When ``n_samples == 1`` the
        leading dimension is still present for consistency.

    Raises
    ------
    ValueError
        If *block_size* is < 1 or > T.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    T, D = data.shape

    if block_size < 1 or block_size > T:
        raise ValueError(
            f"block_size must be in [1, {T}], got {block_size}"
        )

    if rng is None:
        rng = np.random.default_rng()

    # Number of blocks required to cover T samples (may overshoot)
    n_blocks = int(np.ceil(T / block_size))
    max_start = T - block_size  # inclusive upper bound for block starts

    resamples = np.empty((n_samples, T, D), dtype=np.float64)

    for i in range(n_samples):
        # Draw random block starting indices
        starts = rng.integers(0, max_start + 1, size=n_blocks)

        # Build index array by concatenating block ranges
        # Vectorised: (n_blocks, block_size) offsets + starts[:, None]
        offsets = np.arange(block_size)[np.newaxis, :]  # (1, block_size)
        indices = (starts[:, np.newaxis] + offsets).ravel()[:T]

        resamples[i] = data[indices]

    return resamples


def random_time_reversal(
    data: NDArray[np.floating],
    n_segments: int = 5,
    min_segment_frac: float = 0.05,
    max_segment_frac: float = 0.25,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Generate augmented series by reversing random sub-segments.

    For each sample, *n_segments* non-overlapping sub-intervals are
    chosen at random and their temporal order is reversed in place.
    This creates data that shares the same marginal distribution but
    breaks the causal / irreversible (non-equilibrium) structure,
    making it useful as a null-model for entropy-production tests.

    Parameters
    ----------
    data : ndarray of shape (T, D)
        Input time series.
    n_segments : int
        Number of segments to reverse per augmented sample.
    min_segment_frac : float
        Minimum segment length as a fraction of T.
    max_segment_frac : float
        Maximum segment length as a fraction of T.
    n_samples : int
        Number of augmented copies to produce.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    augmented : ndarray of shape (n_samples, T, D)
        Augmented copies with reversed sub-segments.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    T, D = data.shape

    min_len = max(2, int(T * min_segment_frac))
    max_len = max(min_len + 1, int(T * max_segment_frac))

    if rng is None:
        rng = np.random.default_rng()

    augmented = np.empty((n_samples, T, D), dtype=np.float64)

    for i in range(n_samples):
        sample = data.copy()

        # Greedily place non-overlapping reversed segments
        occupied = np.zeros(T, dtype=bool)
        placed = 0
        max_attempts = n_segments * 10  # bound retries to avoid infinite loop
        attempts = 0

        while placed < n_segments and attempts < max_attempts:
            attempts += 1
            seg_len = rng.integers(min_len, max_len + 1)
            start = rng.integers(0, T - seg_len + 1)
            end = start + seg_len

            # Check for overlap with already-reversed regions
            if np.any(occupied[start:end]):
                continue

            # Reverse the segment in-place
            sample[start:end] = sample[start:end][::-1]
            occupied[start:end] = True
            placed += 1

        augmented[i] = sample

    return augmented


def _iaaft_1d(
    x: NDArray[np.floating],
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.floating]:
    """Single-channel IAAFT surrogate (Schreiber & Schmitz, 2000).

    Iteratively adjusts a randomized-phase Fourier surrogate to match
    both the power spectrum AND the amplitude distribution of *x*.

    Parameters
    ----------
    x : ndarray of shape (T,)
        Original univariate time series.
    rng : numpy.random.Generator
    max_iter : int
        Maximum IAAFT iterations.
    tol : float
        Convergence tolerance on the spectrum matching.

    Returns
    -------
    surrogate : ndarray of shape (T,)
    """
    T = len(x)
    sorted_x = np.sort(x)
    target_fft = np.fft.rfft(x)
    target_amplitudes = np.abs(target_fft)

    # Step 1: initial shuffle (random rank ordering)
    surrogate = x.copy()
    rng.shuffle(surrogate)

    prev_spec_diff = np.inf
    for _ in range(max_iter):
        # Step 2: impose target power spectrum
        surr_fft = np.fft.rfft(surrogate)
        surr_phases = np.angle(surr_fft)
        adjusted_fft = target_amplitudes * np.exp(1j * surr_phases)
        surrogate_spectral = np.fft.irfft(adjusted_fft, n=T)

        # Step 3: impose target amplitude distribution via rank ordering
        ranks = np.argsort(np.argsort(surrogate_spectral))
        surrogate = sorted_x[ranks]

        # Check convergence
        spec_diff = np.mean((np.abs(np.fft.rfft(surrogate)) - target_amplitudes) ** 2)
        if abs(prev_spec_diff - spec_diff) < tol:
            break
        prev_spec_diff = spec_diff

    # Final spectrum adjustment (keep spectrum exact, amplitude approximate)
    surr_fft = np.fft.rfft(surrogate)
    surr_phases = np.angle(surr_fft)
    surrogate = np.fft.irfft(target_amplitudes * np.exp(1j * surr_phases), n=T)

    return surrogate


def iaaft_surrogate(
    data: NDArray[np.floating],
    n_samples: int = 1,
    max_iter: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """Generate IAAFT surrogates of a multivariate time series.

    Iterative Amplitude Adjusted Fourier Transform (Schreiber & Schmitz,
    2000) preserves the power spectrum and marginal amplitude distribution
    of each channel but destroys all nonlinear temporal structure including
    time-reversal asymmetry.  This is the gold-standard null model for
    testing irreversibility / broken detailed balance.

    Each channel is treated independently (channel-wise IAAFT), which is
    the standard approach for multivariate time series.

    Parameters
    ----------
    data : ndarray of shape (T, D)
        Input time series (rows = time, columns = features).
    n_samples : int
        Number of surrogate realisations to generate.
    max_iter : int
        Maximum IAAFT iterations per channel.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    surrogates : ndarray of shape (n_samples, T, D)

    References
    ----------
    Schreiber, T. & Schmitz, A. (2000). Surrogate time series.
    Physica D, 142(3-4), 346-382.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    T, D = data.shape

    if rng is None:
        rng = np.random.default_rng()

    surrogates = np.empty((n_samples, T, D), dtype=np.float64)

    for i in range(n_samples):
        for d in range(D):
            surrogates[i, :, d] = _iaaft_1d(data[:, d], rng, max_iter=max_iter)

    return surrogates
