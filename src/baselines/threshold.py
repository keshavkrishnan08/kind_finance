"""VIX threshold baseline for market regime classification.

Implements the deterministic VIX-based regime classifier per PRD Section 11.3.
This is the simplest baseline: regimes are assigned purely from whether the
VIX level falls below, between, or above predefined thresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class VIXThresholdBaseline:
    """Deterministic VIX-threshold regime classifier.

    Assigns each observation to a regime based on which interval its VIX
    value falls into.  With the default thresholds ``[20, 30, 40]`` the
    mapping is:

    ======  =================  ===========
    Label   VIX range          Regime
    ======  =================  ===========
    0       VIX < 20           Low vol
    1       20 <= VIX < 30     Elevated
    2       VIX >= 30          Crisis
    ======  =================  ===========

    More thresholds produce more regimes (``len(thresholds) + 1`` total when
    all thresholds are used, but the default mapping collapses the top two
    bins into a single "crisis" regime to maintain exactly 3 classes).

    Parameters
    ----------
    thresholds : list of float, default=[20, 30, 40]
        Sorted VIX level boundaries. The number of output regimes is
        ``min(len(thresholds), 2) + 1`` = 3 by default (low / elevated /
        crisis).
    """

    # Canonical regime names for the default 3-regime setup.
    REGIME_NAMES: Dict[int, str] = {
        0: "low_volatility",
        1: "elevated",
        2: "crisis",
    }

    def __init__(self, thresholds: Optional[List[float]] = None) -> None:
        if thresholds is None:
            thresholds = [20.0, 30.0, 40.0]
        self.thresholds: List[float] = sorted(thresholds)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def predict(self, vix_data: np.ndarray) -> np.ndarray:
        """Classify VIX levels into regime labels.

        Parameters
        ----------
        vix_data : np.ndarray of shape (T,) or (T, 1)
            VIX index values (non-negative floats).

        Returns
        -------
        np.ndarray of shape (T,)
            Integer regime labels:
            - 0 = low volatility  (VIX < first threshold)
            - 1 = elevated        (first threshold <= VIX < second threshold)
            - 2 = crisis          (VIX >= second threshold)

            When more than two thresholds are provided, values above the
            second threshold are all mapped to label 2 (crisis).
        """
        vix = self._validate(vix_data)

        # np.digitize returns bin index; clip to [0, 2] for the 3-regime case.
        raw_bins = np.digitize(vix, bins=self.thresholds, right=False)
        labels = np.clip(raw_bins, 0, 2).astype(np.intp)

        logger.debug(
            "VIX threshold predict: T=%d, regime counts=%s",
            len(labels),
            {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        )
        return labels

    def get_metrics(self, vix_data: np.ndarray) -> Dict[str, object]:
        """Compute regime statistics for a VIX series.

        Parameters
        ----------
        vix_data : np.ndarray of shape (T,) or (T, 1)
            VIX index values.

        Returns
        -------
        dict
            Keys:
            - ``regime_labels``        : integer labels per timestep (ndarray)
            - ``regime_durations``     : dict mapping label -> list of
              consecutive-run lengths (in timesteps).
            - ``fraction_per_regime``  : dict mapping label -> fraction of
              total observations in that regime.
        """
        labels = self.predict(vix_data)
        T = len(labels)

        # --- Fraction per regime ---
        unique, counts = np.unique(labels, return_counts=True)
        fraction_per_regime: Dict[int, float] = {
            int(u): float(c / T) for u, c in zip(unique, counts)
        }
        # Ensure all three canonical regimes are present in the output.
        for regime_id in range(3):
            fraction_per_regime.setdefault(regime_id, 0.0)

        # --- Regime durations (consecutive run lengths) ---
        regime_durations: Dict[int, List[int]] = {r: [] for r in range(3)}
        if T > 0:
            current_label = int(labels[0])
            run_length = 1
            for i in range(1, T):
                if labels[i] == current_label:
                    run_length += 1
                else:
                    regime_durations[current_label].append(run_length)
                    current_label = int(labels[i])
                    run_length = 1
            regime_durations[current_label].append(run_length)

        logger.info(
            "VIX threshold metrics: fractions=%s",
            {k: f"{v:.3f}" for k, v in sorted(fraction_per_regime.items())},
        )

        return {
            "regime_labels": labels,
            "regime_durations": regime_durations,
            "fraction_per_regime": fraction_per_regime,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(vix_data: np.ndarray) -> np.ndarray:
        """Flatten and validate VIX data."""
        vix = np.asarray(vix_data, dtype=np.float64).ravel()
        if vix.size == 0:
            raise ValueError("vix_data must be non-empty.")
        if np.any(np.isnan(vix)):
            raise ValueError("vix_data contains NaN values.")
        return vix
