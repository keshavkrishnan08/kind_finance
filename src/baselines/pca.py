"""PCA baseline for dimensionality reduction and regime clustering.

Provides a principal component analysis baseline that projects return data
into a low-dimensional subspace and optionally clusters the projections
to detect market regimes via KMeans.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCABaseline:
    """PCA-based dimensionality reduction and regime detection baseline.

    Wraps ``sklearn.decomposition.PCA`` with a convenience API for
    financial return matrices and adds a KMeans-based regime detection
    step on the PCA-projected space.

    Parameters
    ----------
    n_components : int, default=10
        Number of principal components to retain.

    Attributes
    ----------
    model_ : PCA
        Fitted scikit-learn PCA instance.
    n_samples_ : int
        Number of observations used during fitting.
    n_features_ : int
        Number of features (assets) per observation.
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components
        self.model_: Optional[PCA] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "PCABaseline":
        """Fit PCA to observed returns.

        Parameters
        ----------
        returns : np.ndarray of shape (T, D)
            Return matrix where rows are time steps and columns are assets.
            A 1-D array of shape (T,) is treated as (T, 1).

        Returns
        -------
        self
            The fitted baseline instance.
        """
        X = self._validate_and_reshape(returns)
        self.n_samples_, self.n_features_ = X.shape

        # Clamp n_components to min(T, D) to prevent sklearn errors
        effective_components = min(
            self.n_components, self.n_samples_, self.n_features_
        )

        self.model_ = PCA(n_components=effective_components, random_state=42)
        self.model_.fit(X)

        cumulative = np.cumsum(self.model_.explained_variance_ratio_)
        logger.info(
            "PCA fitted: n_components=%d, cumulative variance explained=%.4f",
            effective_components,
            cumulative[-1],
        )
        return self

    def transform(self, returns: np.ndarray) -> np.ndarray:
        """Project returns onto the principal component subspace.

        Parameters
        ----------
        returns : np.ndarray of shape (T, D)
            Return matrix to project.

        Returns
        -------
        np.ndarray of shape (T, n_components)
            Projected data.
        """
        self._check_fitted()
        X = self._validate_and_reshape(returns)
        return self.model_.transform(X)

    def detect_regimes(
        self,
        returns: np.ndarray,
        n_clusters: int = 3,
    ) -> np.ndarray:
        """Detect regimes by clustering in PCA-projected space.

        Projects *returns* via the fitted PCA, then applies KMeans clustering
        to the low-dimensional representation.

        Parameters
        ----------
        returns : np.ndarray of shape (T, D)
            Return matrix.
        n_clusters : int, default=3
            Number of regime clusters.

        Returns
        -------
        np.ndarray of shape (T,)
            Integer cluster labels in ``[0, n_clusters)``.
        """
        self._check_fitted()
        projected = self.transform(returns)

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        labels = kmeans.fit_predict(projected)

        logger.info(
            "Regime detection: n_clusters=%d, inertia=%.4f",
            n_clusters,
            kmeans.inertia_,
        )
        return labels

    def get_metrics(self) -> Dict[str, object]:
        """Return a dictionary of PCA diagnostics.

        Returns
        -------
        dict
            Keys:
            - ``explained_variance_ratio`` : per-component variance ratio (ndarray)
            - ``components``               : principal axes in feature space (ndarray)
            - ``singular_values``          : singular values from the SVD (ndarray)
        """
        self._check_fitted()
        return {
            "explained_variance_ratio": self.model_.explained_variance_ratio_.copy(),
            "components": self.model_.components_.copy(),
            "singular_values": self.model_.singular_values_.copy(),
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
