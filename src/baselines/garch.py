"""GARCH(1,1) baseline for financial regime detection.

Fits a GARCH(1,1) model to log-returns, extracts conditional volatility,
and classifies regimes by volatility thresholds (high-vol = crisis).
This is a standard econophysics baseline that PRE reviewers expect.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GARCHBaseline:
    """GARCH(1,1) volatility regime detector.

    Fits a GARCH(1,1) model to univariate returns, extracts conditional
    volatility sigma_t, and classifies regimes using percentile thresholds.

    Parameters
    ----------
    high_vol_percentile : float, default=80
        Percentile of conditional volatility above which a day is
        classified as "crisis" (regime 1). Below is "normal" (regime 0).
    """

    def __init__(self, high_vol_percentile: float = 80) -> None:
        self.high_vol_percentile = high_vol_percentile
        self.model_ = None
        self.result_ = None
        self.conditional_vol_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None

    def fit(self, returns: np.ndarray) -> "GARCHBaseline":
        """Fit GARCH(1,1) to returns.

        Parameters
        ----------
        returns : np.ndarray of shape (T,) or (T, 1)
            Log-return series. If multivariate, uses only the first column.

        Returns
        -------
        self
        """
        from arch import arch_model

        r = np.asarray(returns, dtype=np.float64).ravel()
        # arch expects returns scaled by 100 for numerical stability
        r_pct = r * 100.0

        am = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal",
                        mean="Constant", rescale=False)
        self.result_ = am.fit(disp="off", show_warning=False)
        self.model_ = am

        # Conditional volatility (back to original scale)
        self.conditional_vol_ = self.result_.conditional_volatility / 100.0

        # Threshold from training data
        self.threshold_ = float(np.percentile(
            self.conditional_vol_, self.high_vol_percentile
        ))

        logger.info(
            "GARCH(1,1) fitted: vol threshold=%.6f (p%d), AIC=%.1f, BIC=%.1f",
            self.threshold_, self.high_vol_percentile,
            self.result_.aic, self.result_.bic,
        )
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict regime labels from conditional volatility.

        Parameters
        ----------
        returns : np.ndarray of shape (T,) or (T, 1)
            Log-return series.

        Returns
        -------
        np.ndarray of shape (T,)
            0 = normal, 1 = high-volatility (crisis).
        """
        from arch import arch_model

        r = np.asarray(returns, dtype=np.float64).ravel()
        r_pct = r * 100.0

        am = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal",
                        mean="Constant", rescale=False)
        # Fix parameters from training and filter
        res = am.fit(
            disp="off", show_warning=False,
            starting_values=self.result_.params.values,
            options={"maxiter": 0},
        )
        vol = res.conditional_volatility / 100.0
        return (vol > self.threshold_).astype(int)

    def get_metrics(self) -> Dict[str, object]:
        """Return model diagnostics."""
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        return {
            "aic": float(self.result_.aic),
            "bic": float(self.result_.bic),
            "log_likelihood": float(self.result_.loglikelihood),
            "params": {k: float(v) for k, v in self.result_.params.items()},
            "conditional_vol": self.conditional_vol_.copy(),
            "threshold": self.threshold_,
        }
