"""Regime detection from Koopman eigenfunctions.

Identifies market regimes (e.g. expansion / contraction, risk-on / risk-off)
by clustering the dominant eigenfunctions of the learned Koopman operator.
Provides comparison against NBER recession dates and empirical transition
matrix estimation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect market regimes from Koopman eigenfunction values.

    The core idea follows the Perron--Frobenius / Koopman duality:
    the sign structure of the *second* eigenfunction of a reversible
    transfer operator naturally partitions state space into two
    metastable sets.  For higher-dimensional partitions the
    dominant *K-1* non-trivial eigenfunctions are clustered with
    K-means.
    """

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_from_eigenfunctions(
        eigenfunctions: np.ndarray,
        n_regimes: Optional[int] = None,
        spectral_gap_threshold: float = 0.10,
    ) -> np.ndarray:
        """Assign regime labels using the dominant eigenfunction(s).

        Parameters
        ----------
        eigenfunctions : np.ndarray
            Eigenfunction array of shape ``(T, K)`` where *K* is the number
            of learned basis functions.  The columns should be ordered by
            decreasing eigenvalue magnitude (i.e. column 0 is the slowest
            mode).
        n_regimes : int or None
            Number of regimes.  If ``None`` the method infers the number from
            the eigenvalue spectrum via the spectral gap heuristic: it finds
            the largest gap between consecutive singular value magnitudes and
            sets ``n_regimes`` to the index *before* the gap (minimum 2).
        spectral_gap_threshold : float
            Minimum absolute gap required when auto-detecting ``n_regimes``.

        Returns
        -------
        np.ndarray
            Integer label array of length *T* with values in
            ``{0, 1, ..., n_regimes - 1}``.
        """
        T, K = eigenfunctions.shape

        if n_regimes is None:
            n_regimes = RegimeDetector._infer_n_regimes(
                eigenfunctions, spectral_gap_threshold
            )
            logger.info("Auto-detected n_regimes=%d from spectral gap", n_regimes)

        if n_regimes == 2:
            # Fast path: sign of the first non-trivial eigenfunction
            psi = eigenfunctions[:, 1] if K > 1 else eigenfunctions[:, 0]
            labels = (psi >= 0).astype(int)
        else:
            # General path: K-means on the first (n_regimes - 1) non-trivial
            # eigenfunctions.
            n_features = min(n_regimes - 1, K - 1) if K > 1 else K
            features = eigenfunctions[:, 1 : 1 + n_features] if K > 1 else eigenfunctions[:, :n_features]
            km = KMeans(n_clusters=n_regimes, n_init=20, random_state=42)
            labels = km.fit_predict(features)

        return labels

    # ------------------------------------------------------------------
    # Duration statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_regime_durations(labels: np.ndarray) -> np.ndarray:
        """Compute durations of consecutive regime segments.

        Parameters
        ----------
        labels : np.ndarray
            Integer regime labels of length *T*.

        Returns
        -------
        np.ndarray
            1-D array whose entries are the lengths of each contiguous run
            of the same label.
        """
        if len(labels) == 0:
            return np.array([], dtype=int)

        diffs = np.diff(labels)
        change_points = np.nonzero(diffs)[0] + 1
        boundaries = np.concatenate(([0], change_points, [len(labels)]))
        durations = np.diff(boundaries)
        return durations

    # ------------------------------------------------------------------
    # NBER comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_with_nber(
        labels: np.ndarray,
        dates: pd.DatetimeIndex,
        nber_recessions: Optional[List[Tuple[str, str]]] = None,
        train_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare detected regimes against NBER recession dates.

        The label-to-recession mapping is learned on training data only
        (dates <= ``train_end``) to avoid data snooping.  A naive
        frequency baseline (always predict the majority class) is
        included for comparison.

        Parameters
        ----------
        labels : np.ndarray
            Predicted regime labels (length *T*).
        dates : pd.DatetimeIndex
            Date index aligned with *labels*.
        nber_recessions : list of (start, end) date strings, optional
            NBER recession date ranges.  If ``None`` a canonical set of
            post-2000 recessions is used.
        train_end : str or None
            End date of training period for label mapping.  If ``None``,
            defaults to "2017-12-31".

        Returns
        -------
        dict
            ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``confusion``, ``naive_accuracy``, ``recession_label``,
            ``mapping_learned_on``.
        """
        if nber_recessions is None:
            nber_recessions = [
                ("2001-03-01", "2001-11-30"),
                ("2007-12-01", "2009-06-30"),
                ("2020-02-01", "2020-04-30"),
            ]
        if train_end is None:
            train_end = "2017-12-31"

        # Build NBER binary indicator
        nber_binary = np.zeros(len(dates), dtype=int)
        for start, end in nber_recessions:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            nber_binary[mask] = 1

        # Learn label mapping on training data only
        train_mask = dates <= pd.Timestamp(train_end)
        labels_train = labels[train_mask]
        nber_train = nber_binary[train_mask]

        unique_labels = np.unique(labels)
        best_recession_label = unique_labels[0]
        best_train_acc = -1.0
        invert = False

        for rl in unique_labels:
            pred_train = (labels_train == rl).astype(int)
            acc = float(np.mean(pred_train == nber_train))
            if acc > best_train_acc:
                best_train_acc = acc
                best_recession_label = rl
                invert = False
            # Also check inverted mapping
            inv_acc = float(np.mean((1 - pred_train) == nber_train))
            if inv_acc > best_train_acc:
                best_train_acc = inv_acc
                best_recession_label = rl
                invert = True

        # Apply mapping to FULL dataset
        pred_binary = (labels == best_recession_label).astype(int)
        if invert:
            pred_binary = 1 - pred_binary

        # Naive baseline: always predict majority class
        recession_rate = float(np.mean(nber_binary))
        naive_accuracy = max(recession_rate, 1.0 - recession_rate)

        # Metrics on full dataset
        tp = int(np.sum((pred_binary == 1) & (nber_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (nber_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (nber_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (nber_binary == 0)))

        accuracy = float(np.mean(pred_binary == nber_binary))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        confusion = np.array([[tn, fp], [fn, tp]])

        logger.info(
            "NBER comparison: accuracy=%.3f (naive=%.3f), precision=%.3f, "
            "recall=%.3f, F1=%.3f",
            accuracy, naive_accuracy, precision, recall, f1,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
            "naive_accuracy": naive_accuracy,
            "recession_label": int(best_recession_label),
            "mapping_learned_on": f"train (dates <= {train_end})",
            "mapping_inverted": invert,
        }

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    @staticmethod
    def transition_matrix_from_labels(labels: np.ndarray) -> np.ndarray:
        """Estimate an empirical transition probability matrix from labels.

        Parameters
        ----------
        labels : np.ndarray
            Integer regime labels of length *T*.

        Returns
        -------
        np.ndarray
            Row-stochastic transition matrix of shape ``(n_regimes, n_regimes)``
            where entry ``(i, j)`` is the empirical probability of transitioning
            from regime *i* to regime *j* in one time step.
        """
        unique = np.unique(labels)
        n = len(unique)
        label_to_idx = {int(lbl): idx for idx, lbl in enumerate(unique)}

        counts = np.zeros((n, n), dtype=float)
        for t in range(len(labels) - 1):
            i = label_to_idx[int(labels[t])]
            j = label_to_idx[int(labels[t + 1])]
            counts[i, j] += 1.0

        # Normalise rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        transition_matrix = counts / row_sums

        return transition_matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_n_regimes(
        eigenfunctions: np.ndarray,
        gap_threshold: float = 0.10,
    ) -> int:
        """Heuristic: choose n_regimes from eigenfunction variance gaps.

        We compute the variance of each eigenfunction column.  A large
        drop in variance after column *k* suggests that the first *k+1*
        eigenfunctions are dynamically relevant, giving ``n_regimes = k + 1``
        (the trivial constant eigenfunction counts as one).  Minimum 2.
        """
        variances = np.var(eigenfunctions, axis=0)
        if len(variances) < 2:
            return 2

        # Normalise so the largest is 1
        variances = variances / (variances.max() + 1e-15)
        gaps = np.abs(np.diff(variances))

        # Find the first gap exceeding the threshold
        significant = np.where(gaps > gap_threshold)[0]
        if len(significant) == 0:
            return 2

        # n_regimes = index of gap + 1 (counting the constant mode)
        n_regimes = int(significant[0]) + 1
        return max(n_regimes, 2)
