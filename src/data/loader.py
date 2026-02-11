"""PyTorch Dataset classes for KTND-Finance (PRD Section 5.4).

Provides two datasets:
- ``TimeLaggedDataset``: yields (x_t, x_{t+lag}) pairs for the Koopman
  operator formulation.
- ``RollingWindowDataset``: yields fixed-length windows for sequence
  models and rolling analysis.

Both classes accept raw price DataFrames or pre-processed numpy arrays
and handle the full preprocessing chain internally when needed.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.typing import NDArray

from .preprocessing import (
    compute_log_returns,
    create_rolling_windows,
    create_time_lagged_pairs,
    standardize_returns,
)


class TimeLaggedDataset(Dataset):
    """Dataset of (x_t, x_{t+lag}) pairs for Koopman-style training.

    Parameters
    ----------
    data : DataFrame or ndarray of shape (T, D)
        If a DataFrame of raw prices is supplied, log-returns and
        standardisation are applied automatically.  If an ndarray is
        supplied it is assumed to be already preprocessed.
    lag : int
        Time lag between observation pairs.
    preprocess : bool
        Whether to compute log-returns and standardise.  Ignored when
        *data* is an ndarray (assumed preprocessed).
    standardize_method : {"zscore", "robust"}
        Standardisation method forwarded to ``standardize_returns``.
    train_end_idx : int, optional
        Training set cutoff for computing standardisation statistics.
    dtype : torch.dtype
        Tensor dtype (default ``torch.float32``).
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, NDArray[np.floating]],
        lag: int = 1,
        preprocess: bool = True,
        standardize_method: Literal["zscore", "robust"] = "zscore",
        train_end_idx: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        arr = self._prepare(data, preprocess, standardize_method, train_end_idx)
        self.x_t, self.x_t_lag = create_time_lagged_pairs(arr, lag=lag)
        self.x_t = torch.as_tensor(self.x_t, dtype=dtype)
        self.x_t_lag = torch.as_tensor(self.x_t_lag, dtype=dtype)

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare(
        data: Union[pd.DataFrame, NDArray[np.floating]],
        preprocess: bool,
        method: str,
        train_end_idx: Optional[int],
    ) -> NDArray[np.floating]:
        if isinstance(data, pd.DataFrame) and preprocess:
            returns = compute_log_returns(data, drop_first=True)
            standardised, _ = standardize_returns(
                returns, method=method, train_end_idx=train_end_idx,
            )
            return standardised.values if isinstance(standardised, pd.DataFrame) else standardised
        if isinstance(data, pd.DataFrame):
            return data.values.astype(np.float64)
        return np.asarray(data, dtype=np.float64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.x_t.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_t[idx], self.x_t_lag[idx]


class RollingWindowDataset(Dataset):
    """Dataset of fixed-length rolling windows for sequence modelling.

    Parameters
    ----------
    data : DataFrame or ndarray of shape (T, D)
        Raw prices (DataFrame) or preprocessed array.
    window_size : int
        Number of time-steps per window.
    stride : int
        Step between successive windows.
    preprocess : bool
        If True and *data* is a DataFrame, apply log-returns and
        standardisation.
    standardize_method : {"zscore", "robust"}
        Forwarded to ``standardize_returns``.
    train_end_idx : int, optional
        Training-set cutoff for standardisation statistics.
    return_target : bool
        If True each sample is ``(window[:-1], window[-1])`` so the last
        step serves as the prediction target.  If False the full window
        is returned.
    dtype : torch.dtype
        Tensor dtype (default ``torch.float32``).
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, NDArray[np.floating]],
        window_size: int = 50,
        stride: int = 1,
        preprocess: bool = True,
        standardize_method: Literal["zscore", "robust"] = "zscore",
        train_end_idx: Optional[int] = None,
        return_target: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        arr = TimeLaggedDataset._prepare(data, preprocess, standardize_method, train_end_idx)
        # create_rolling_windows returns a stride-trick *view*; copy for safety
        # before wrapping in a tensor (tensors must own their data).
        self.windows = torch.as_tensor(
            np.ascontiguousarray(create_rolling_windows(arr, window_size, stride)),
            dtype=dtype,
        )
        self.return_target = return_target

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        window = self.windows[idx]  # (window_size, D)
        if self.return_target:
            return window[:-1], window[-1]
        return window
