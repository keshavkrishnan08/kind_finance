"""
Loss functions for NonEquilibriumVAMPNet training (PRD Section 7.1).

All losses operate on the output dictionary returned by
``NonEquilibriumVAMPNet.forward()``.  The primary training objective is the
VAMP-2 score (negated for minimisation), augmented with physically motivated
regularisers for orthogonality, spectral boundedness, and consistency of the
learned entropy production rate with an empirical estimate.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def vamp2_loss(singular_values: Tensor) -> Tensor:
    """Negative VAMP-2 score: L = -sum_k sigma_k^2.

    Minimising this is equivalent to maximising the VAMP-2 variational
    bound on the Koopman approximation quality.

    Parameters
    ----------
    singular_values : Tensor, shape ``(d,)``
        Singular values of the whitened Koopman matrix K.

    Returns
    -------
    Tensor, scalar
        Negative sum of squared singular values.
    """
    return -singular_values.pow(2).sum()


def orthogonality_loss(koopman_matrix: Tensor) -> Tensor:
    """Penalise off-diagonal entries of K^T K.

    Encourages the learned Koopman modes to be approximately orthogonal,
    which improves interpretability of the spectral decomposition.

    Parameters
    ----------
    koopman_matrix : Tensor, shape ``(d, d)``
        Whitened Koopman matrix.

    Returns
    -------
    Tensor, scalar
        Frobenius norm of the off-diagonal part of K^T K.
    """
    KtK = koopman_matrix.T @ koopman_matrix  # (d, d)
    # Zero out the diagonal -> keep only off-diagonal
    d = KtK.shape[0]
    mask = ~torch.eye(d, dtype=torch.bool, device=KtK.device)
    off_diag = KtK[mask]
    return off_diag.pow(2).sum()


def entropy_production_consistency_loss(
    eigenvalues: Tensor,
    eigenfunc_amplitudes: Tensor,
    tau: float,
    sigma_empirical: Tensor,
) -> Tensor:
    """Consistency between spectral entropy production and an empirical estimate.

    The spectral entropy production rate is
        sigma_spectral = sum_k omega_k^2 * A_k / gamma_k
    where omega_k = angle(lambda_k) / tau, gamma_k = -ln|lambda_k| / tau,
    and A_k is the mean squared amplitude of the k-th eigenfunction.

    This loss penalises the squared deviation from an externally
    supplied empirical entropy-production rate.

        L = |sigma_spectral - sigma_empirical|^2

    Parameters
    ----------
    eigenvalues : Tensor, shape ``(d,)`` complex
        Complex eigenvalues of K.
    eigenfunc_amplitudes : Tensor, shape ``(d,)``
        A_k = mean(|psi_k|^2) for each mode k.
    tau : float
        Lag time (in the same units as the return series).
    sigma_empirical : Tensor, scalar
        Empirical entropy-production estimate.

    Returns
    -------
    Tensor, scalar
        Squared difference.
    """
    omega = torch.angle(eigenvalues) / tau  # (d,)
    magnitudes = eigenvalues.abs().float().clamp(min=1e-12, max=1.0 - 1e-7)
    gamma = (-torch.log(magnitudes) / tau).clamp(min=1e-6)  # (d,)
    sigma_spectral = (omega.pow(2) * eigenfunc_amplitudes / gamma).sum()
    return (sigma_spectral - sigma_empirical).pow(2)


def spectral_penalty(singular_values: Tensor) -> Tensor:
    """Penalise singular values that exceed 1.

    Physical Koopman operators on L^2 are contractions, so all singular
    values should satisfy sigma_k <= 1.  This soft constraint penalises
    any violation via a one-sided squared hinge.

    Parameters
    ----------
    singular_values : Tensor, shape ``(d,)``

    Returns
    -------
    Tensor, scalar
        sum_k max(0, sigma_k - 1)^2.
    """
    excess = torch.clamp(singular_values - 1.0, min=0.0)
    return excess.pow(2).sum()


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

def total_loss(
    output_dict: Dict[str, Tensor],
    *,
    tau: float = 1.0,
    sigma_empirical: Optional[Tensor] = None,
    eigenfunc_amplitudes: Optional[Tensor] = None,
    w_vamp2: float = 1.0,
    w_orthogonality: float = 0.01,
    w_entropy: float = 0.1,
    w_spectral: float = 0.1,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Weighted combination of all loss terms.

    Parameters
    ----------
    output_dict : dict
        Return value of ``NonEquilibriumVAMPNet.forward()``.
    tau : float
        Lag time.
    sigma_empirical : Tensor or None
        Empirical entropy-production estimate.  If ``None`` the
        entropy-consistency term is skipped (weight forced to 0).
    eigenfunc_amplitudes : Tensor or None
        A_k for each mode.  Required when ``sigma_empirical`` is
        provided.
    w_vamp2 : float
        Weight for the VAMP-2 score term (default 1.0).
    w_orthogonality : float
        Weight for the orthogonality regulariser (default 0.01).
    w_entropy : float
        Weight for the entropy-consistency term (default 0.1).
    w_spectral : float
        Weight for the spectral penalty (default 0.1).

    Returns
    -------
    loss : Tensor, scalar
        The total weighted loss (for ``loss.backward()``).
    loss_dict : dict[str, Tensor]
        Individual (unweighted) loss components for logging.
    """
    sigma = output_dict["singular_values"]
    K = output_dict["koopman_matrix"]
    eigs = output_dict["eigenvalues"]

    # --- individual terms --------------------------------------------------
    l_vamp2 = vamp2_loss(sigma)
    l_ortho = orthogonality_loss(K)
    l_spectral = spectral_penalty(sigma)

    loss = w_vamp2 * l_vamp2 + w_orthogonality * l_ortho + w_spectral * l_spectral

    loss_dict: Dict[str, Tensor] = {
        "vamp2": l_vamp2.detach(),
        "orthogonality": l_ortho.detach(),
        "spectral_penalty": l_spectral.detach(),
    }

    # Entropy consistency (optional -- requires empirical estimate)
    if sigma_empirical is not None and eigenfunc_amplitudes is not None:
        l_entropy = entropy_production_consistency_loss(
            eigs, eigenfunc_amplitudes, tau, sigma_empirical
        )
        loss = loss + w_entropy * l_entropy
        loss_dict["entropy_consistency"] = l_entropy.detach()

    loss_dict["total"] = loss.detach()

    return loss, loss_dict
