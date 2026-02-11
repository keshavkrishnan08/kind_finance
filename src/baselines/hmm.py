"""Hidden Markov Model baseline for financial regime detection.

Implements a Gaussian HMM baseline per PRD Section 11.1. This serves as the
primary statistical benchmark against which the Koopman-Thermodynamic model
is compared for regime identification accuracy and transition timing.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class HMMBaseline:
    """Gaussian Hidden Markov Model baseline for financial regime detection.

    Fits a Gaussian HMM with full covariance to asset return series and
    extracts regime labels, transition matrices, and information criteria.

    Parameters
    ----------
    n_states : int, default=3
        Number of hidden states (regimes). The default of 3 corresponds to
        the canonical low-volatility / normal / crisis taxonomy.

    Attributes
    ----------
    model_ : GaussianHMM
        Fitted hmmlearn model instance.
    states_ : np.ndarray of shape (T,)
        Most-probable state sequence from the Viterbi algorithm.
    log_likelihood_ : float
        Log-likelihood of the fitted model on training data.
    n_samples_ : int
        Number of observations used during fitting.
    n_features_ : int
        Number of features per observation.
    """

    def __init__(self, n_states: int = 3) -> None:
        self.n_states = n_states
        self.model_: Optional[GaussianHMM] = None
        self.states_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "HMMBaseline":
        """Fit the Gaussian HMM to observed returns.

        Parameters
        ----------
        returns : np.ndarray of shape (T,) or (T, D)
            Return series. A 1-D array is automatically reshaped to (T, 1).

        Returns
        -------
        self
            The fitted baseline instance (for method chaining).
        """
        X = self._validate_and_reshape(returns)
        self.n_samples_, self.n_features_ = X.shape

        self.model_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self.model_.fit(X)

        self.log_likelihood_ = self.model_.score(X)
        self.states_ = self.model_.predict(X)

        logger.info(
            "HMM fitted: n_states=%d, log_likelihood=%.4f",
            self.n_states,
            self.log_likelihood_,
        )
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Decode the most-probable state sequence via the Viterbi algorithm.

        Parameters
        ----------
        returns : np.ndarray of shape (T,) or (T, D)
            Return series to decode.

        Returns
        -------
        np.ndarray of shape (T,)
            Integer state labels in ``[0, n_states)``.
        """
        self._check_fitted()
        X = self._validate_and_reshape(returns)
        return self.model_.predict(X)

    def score(self, returns: np.ndarray) -> float:
        """Compute the log-likelihood of *returns* under the fitted model.

        Parameters
        ----------
        returns : np.ndarray of shape (T,) or (T, D)
            Return series to score.

        Returns
        -------
        float
            Total log-likelihood.
        """
        self._check_fitted()
        X = self._validate_and_reshape(returns)
        return float(self.model_.score(X))

    def get_metrics(self) -> Dict[str, object]:
        """Return a dictionary of model diagnostics.

        Returns
        -------
        dict
            Keys:
            - ``states``             : most-probable state labels (ndarray)
            - ``log_likelihood``     : training log-likelihood (float)
            - ``aic``                : Akaike information criterion (float)
            - ``bic``                : Bayesian information criterion (float)
            - ``transition_matrix``  : row-stochastic transition matrix (ndarray)
        """
        self._check_fitted()

        n_free_params = self._count_free_params()
        aic = -2.0 * self.log_likelihood_ + 2.0 * n_free_params
        bic = (
            -2.0 * self.log_likelihood_
            + n_free_params * np.log(self.n_samples_)
        )

        return {
            "states": self.states_.copy(),
            "log_likelihood": self.log_likelihood_,
            "aic": float(aic),
            "bic": float(bic),
            "transition_matrix": self.model_.transmat_.copy(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_reshape(self, returns: np.ndarray) -> np.ndarray:
        """Ensure *returns* is a 2-D float array."""
        X = np.asarray(returns, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"Expected 1-D or 2-D array, got shape {X.shape}"
            )
        return X

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )

    def _count_free_params(self) -> int:
        """Count the number of free parameters in the Gaussian HMM.

        Free parameters:
        - Initial state distribution: (n_states - 1)
        - Transition matrix:          n_states * (n_states - 1)
        - Means:                       n_states * n_features
        - Full covariances:            n_states * n_features * (n_features + 1) / 2
        """
        k = self.n_states
        d = self.n_features_
        n_startprob = k - 1
        n_transmat = k * (k - 1)
        n_means = k * d
        n_covars = k * d * (d + 1) // 2
        return n_startprob + n_transmat + n_means + n_covars
