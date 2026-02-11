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
    ) -> Dict[str, Any]:
        """Compare detected regimes against NBER recession dates.

        Parameters
        ----------
        labels : np.ndarray
            Predicted regime labels (length *T*).  The method assumes that
            the regime with *lower mean return* (or equivalently the minority
            regime, in practice) corresponds to NBER recessions.
        dates : pd.DatetimeIndex
            Date index aligned with *labels*.
        nber_recessions : list of (start, end) date strings, optional
            NBER recession date ranges.  If ``None`` a canonical set of
            post-2000 recessions is used:
            2001-03 to 2001-11, 2007-12 to 2009-06, 2020-02 to 2020-04.

        Returns
        -------
        dict
            ``accuracy``   -- fraction of days correctly classified.
            ``precision``  -- precision for recession detection.
            ``recall``     -- recall for recession detection.
            ``f1``         -- F-1 score.
            ``confusion``  -- 2x2 confusion matrix as np.ndarray.
        """
        if nber_recessions is None:
            nber_recessions = [
                ("2001-03-01", "2001-11-30"),
                ("2007-12-01", "2009-06-30"),
                ("2020-02-01", "2020-04-30"),
            ]

        # Build NBER binary indicator
        nber_binary = np.zeros(len(dates), dtype=int)
        for start, end in nber_recessions:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            nber_binary[mask] = 1

        # Determine which label maps to "recession"
        unique_labels = np.unique(labels)
        best_mapping: Optional[Dict[int, int]] = None
        best_accuracy = -1.0

        # Try each label as the recession label to find optimal mapping
        for recession_label in unique_labels:
            pred_binary = (labels == recession_label).astype(int)
            acc = float(np.mean(pred_binary == nber_binary))
            if acc > best_accuracy:
                best_accuracy = acc
                best_mapping = {recession_label: 1}

        # Build predicted binary with best mapping
        recession_label = list(best_mapping.keys())[0]  # type: ignore[union-attr]
        pred_binary = (labels == recession_label).astype(int)

        # Also try the inverse mapping
        inv_pred = 1 - pred_binary
        inv_accuracy = float(np.mean(inv_pred == nber_binary))
        if inv_accuracy > best_accuracy:
            pred_binary = inv_pred
            best_accuracy = inv_accuracy

        # Metrics
        tp = int(np.sum((pred_binary == 1) & (nber_binary == 1)))
        fp = int(np.sum((pred_binary == 1) & (nber_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (nber_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (nber_binary == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2 * precision * recall / max(precision + recall, 1e-12)
        )

        confusion = np.array([[tn, fp], [fn, tp]])

        logger.info(
            "NBER comparison: accuracy=%.3f, precision=%.3f, recall=%.3f, F1=%.3f",
            best_accuracy,
            precision,
            recall,
            f1,
        )

        return {
            "accuracy": best_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
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
