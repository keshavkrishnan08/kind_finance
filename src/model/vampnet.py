"""
Non-Equilibrium VAMPNet for financial time-series Koopman analysis.

Implements the full NonEquilibriumVAMPNet architecture from PRD Section 6.1.
The network learns non-reversible Koopman eigenfunctions via two separate
encoder lobes (chi_t, chi_tau), enabling detection of broken detailed balance
and irreversibility in financial market dynamics.

References:
    - Mardt et al., "VAMPnets for deep learning of molecular kinetics", 2018.
    - Wu & Noe, "Variational approach for learning Markov processes from
      time series data", 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Building block: a single MLP lobe
# ---------------------------------------------------------------------------

class VAMPNetLobe(nn.Module):
    """MLP encoder lobe for VAMPNet.

    Architecture per layer:
        Linear -> BatchNorm -> ELU -> Dropout
    The *final* projection is a plain Linear (no activation, no batchnorm,
    no dropout) so that the output lives in an unconstrained real space
    suitable for covariance estimation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dims : list[int]
        Widths of the hidden layers.
    output_dim : int
        Number of output components (Koopman basis functions).
    dropout : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_features = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_features = h_dim

        # Final linear projection -- no activation, no batchnorm, no dropout
        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)

    # -- Init helpers -------------------------------------------------------
    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters (Xavier uniform for Linear
        layers, default for BatchNorm)."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Map input features to Koopman basis coefficients.

        Parameters
        ----------
        x : Tensor, shape ``(batch, input_dim)``

        Returns
        -------
        Tensor, shape ``(batch, output_dim)``
        """
        return self.network(x)


# ---------------------------------------------------------------------------
# Numerically stable matrix square-root inverse
# ---------------------------------------------------------------------------

def matrix_sqrt_inv(
    C: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute C^{-1/2} via eigendecomposition with clamped eigenvalues.

    For a symmetric positive-semi-definite matrix C we decompose
        C = Q diag(lambda) Q^T
    clamp eigenvalues from below by ``epsilon`` to avoid division by zero,
    then return
        C^{-1/2} = Q diag(lambda_clamped^{-1/2}) Q^T.

    Parameters
    ----------
    C : Tensor, shape ``(d, d)``
        Symmetric (semi-)positive-definite matrix.
    epsilon : float
        Lower clamp for eigenvalues.  Values below this are set to
        ``epsilon`` before inversion, ensuring numerical stability.

    Returns
    -------
    Tensor, shape ``(d, d)``
        The matrix inverse square root C^{-1/2}.
    """
    # Symmetric eigendecomposition (guaranteed real eigenvalues)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    eigenvalues_clamped = eigenvalues.clamp(min=epsilon)
    inv_sqrt_eigenvalues = eigenvalues_clamped.pow(-0.5)
    # Q @ diag(1/sqrt(lam)) @ Q^T
    return eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Full Non-Equilibrium VAMPNet
# ---------------------------------------------------------------------------

class NonEquilibriumVAMPNet(nn.Module):
    """Non-equilibrium VAMPNet with two *separate* (non-reversible) lobes.

    The model maps time-lagged pairs ``(x_t, x_{t+tau})`` through two
    independent encoder MLPs producing basis expansions ``chi_t`` and
    ``chi_tau``.  From the batch it estimates the covariance matrices

        C00      = (1/N) chi_t^T  chi_t
        C0tau    = (1/N) chi_t^T  chi_tau
        Ctautau  = (1/N) chi_tau^T chi_tau

    and builds the whitened Koopman approximation

        K = C00^{-1/2}  C0tau  Ctautau^{-1/2}

    whose SVD yields singular values sigma_k and left/right singular
    vectors.  A subsequent complex eigendecomposition of K recovers
    potentially complex eigenvalues that encode oscillatory (non-
    equilibrium) modes.

    Parameters
    ----------
    input_dim : int
        Number of input features per time point.
    hidden_dims : list[int]
        Hidden layer widths shared by both lobes (architecture is
        identical; *weights* are separate).
    output_dim : int
        Number of Koopman basis functions (rank of the approximation).
    dropout : float
        Dropout probability for both lobes.
    epsilon : float
        Eigenvalue clamp for ``matrix_sqrt_inv``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        # Two *independent* lobes (non-reversible setting)
        self.lobe_t = VAMPNetLobe(input_dim, hidden_dims, output_dim, dropout)
        self.lobe_tau = VAMPNetLobe(input_dim, hidden_dims, output_dim, dropout)

        # Xavier init
        self.lobe_t.reset_parameters()
        self.lobe_tau.reset_parameters()

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: Tensor,
        x_tau: Tensor,
    ) -> Dict[str, Tensor]:
        """Full forward pass: encode, estimate covariances, Koopman, SVD, eig.

        Parameters
        ----------
        x_t : Tensor, shape ``(batch, input_dim)``
            Features at time *t*.
        x_tau : Tensor, shape ``(batch, input_dim)``
            Features at time *t + tau*.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``chi_t``, ``chi_tau``, ``C00``, ``C0tau``, ``Ctautau``,
            ``C00_sqrt_inv``, ``Ctautau_sqrt_inv``, ``koopman_matrix``,
            ``singular_values``, ``svd_U``, ``svd_V``, ``eigenvalues``.
        """
        # --- 1. Encode ---------------------------------------------------
        chi_t: Tensor = self.lobe_t(x_t)      # (N, d)
        chi_tau: Tensor = self.lobe_tau(x_tau)  # (N, d)
        N = chi_t.shape[0]

        # --- 2. Covariance matrices ---------------------------------------
        # Center (optional but improves stability; mean-free covariances)
        chi_t_centered = chi_t - chi_t.mean(dim=0, keepdim=True)
        chi_tau_centered = chi_tau - chi_tau.mean(dim=0, keepdim=True)

        C00 = (chi_t_centered.T @ chi_t_centered) / N          # (d, d)
        C0tau = (chi_t_centered.T @ chi_tau_centered) / N       # (d, d)
        Ctautau = (chi_tau_centered.T @ chi_tau_centered) / N   # (d, d)

        # Small ridge for positive-definiteness
        ridge = self.epsilon * torch.eye(
            self.output_dim, device=chi_t.device, dtype=chi_t.dtype
        )
        C00 = C00 + ridge
        Ctautau = Ctautau + ridge

        # --- 3. Whitened Koopman matrix -----------------------------------
        C00_sqrt_inv = matrix_sqrt_inv(C00, epsilon=self.epsilon)
        Ctautau_sqrt_inv = matrix_sqrt_inv(Ctautau, epsilon=self.epsilon)

        K = C00_sqrt_inv @ C0tau @ Ctautau_sqrt_inv  # (d, d)

        # --- 4. SVD of K -------------------------------------------------
        U, sigma, Vh = torch.linalg.svd(K, full_matrices=False)
        V = Vh.T  # convention: K = U diag(sigma) V^T

        # --- 5. Complex eigendecomposition of K ---------------------------
        # K may be non-symmetric -> complex eigenvalues capture oscillations
        eigenvalues = torch.linalg.eigvals(K)  # complex (d,)

        return {
            "chi_t": chi_t,
            "chi_tau": chi_tau,
            "C00": C00,
            "C0tau": C0tau,
            "Ctautau": Ctautau,
            "C00_sqrt_inv": C00_sqrt_inv,
            "Ctautau_sqrt_inv": Ctautau_sqrt_inv,
            "koopman_matrix": K,
            "singular_values": sigma,
            "svd_U": U,
            "svd_V": V,
            "eigenvalues": eigenvalues,
        }

    # ------------------------------------------------------------------
    # Eigenfunction evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_eigenfunctions(
        self,
        x: Tensor,
        output_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate left and right Koopman eigenfunctions at data points.

        The *right* eigenfunctions (associated with lobe_t) are
            psi_right(x) = C00^{-1/2} @ lobe_t(x)    projected onto SVD cols of U
        and the *left* eigenfunctions (associated with lobe_tau) are
            psi_left(x)  = Ctautau^{-1/2} @ lobe_tau(x) projected onto SVD cols of V

        These are the ``u_k`` and ``v_k`` functions used in the
        irreversibility field.

        Parameters
        ----------
        x : Tensor, shape ``(N, input_dim)``
            Data points at which to evaluate.
        output_dict : dict
            Output of :meth:`forward` (needs ``C00_sqrt_inv``,
            ``Ctautau_sqrt_inv``, ``svd_U``, ``svd_V``).

        Returns
        -------
        u : Tensor, shape ``(N, d)``
            Right eigenfunction values (one column per mode).
        v : Tensor, shape ``(N, d)``
            Left eigenfunction values.
        """
        C00_si = output_dict["C00_sqrt_inv"]        # (d, d)
        Ctautau_si = output_dict["Ctautau_sqrt_inv"]  # (d, d)
        U = output_dict["svd_U"]                     # (d, d)
        V = output_dict["svd_V"]                     # (d, d)

        chi_t_x = self.lobe_t(x)    # (N, d)
        chi_tau_x = self.lobe_tau(x)  # (N, d)

        # Center using running batch stats (use same mean as training)
        chi_t_x = chi_t_x - chi_t_x.mean(dim=0, keepdim=True)
        chi_tau_x = chi_tau_x - chi_tau_x.mean(dim=0, keepdim=True)

        # Whitened representations projected onto SVD basis
        u = (chi_t_x @ C00_si) @ U         # (N, d)  right eigenfunctions
        v = (chi_tau_x @ Ctautau_si) @ V    # (N, d)  left eigenfunctions

        return u, v

    # ------------------------------------------------------------------
    # Eigendecomposition-based eigenfunctions (theory-correct for non-reversible)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_eigenfunctions_eig(
        self,
        x: Tensor,
        output_dict: Dict[str, Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Evaluate left/right Koopman eigenfunctions via eigendecomposition.

        Unlike :meth:`compute_eigenfunctions` which uses SVD singular
        vectors, this method uses the proper eigendecomposition of K for
        non-reversible systems where K is non-symmetric.

        Falls back to ``(None, None)`` if eigendecomposition fails
        (ill-conditioned K).

        Parameters
        ----------
        x : Tensor, shape ``(N, input_dim)``
        output_dict : dict
            Output of :meth:`forward`.

        Returns
        -------
        u : Tensor or None, shape ``(N, d)``
            Right eigenfunctions projected onto eigenvector basis.
        v : Tensor or None, shape ``(N, d)``
            Left eigenfunctions projected onto eigenvector basis.
        """
        K = output_dict["koopman_matrix"]
        C00_si = output_dict["C00_sqrt_inv"]
        Ctautau_si = output_dict["Ctautau_sqrt_inv"]

        try:
            # Right eigenvectors: K W_R = W_R diag(lambda)
            _, W_right = torch.linalg.eig(K)
            # Left eigenvectors: K^T W_L = W_L diag(lambda*)
            _, W_left = torch.linalg.eig(K.T)

            # Use real parts for projection (complex eigenvectors come in
            # conjugate pairs; real part captures the physical mode shape)
            W_right_real = W_right.real.float()
            W_left_real = W_left.real.float()

            chi_t_x = self.lobe_t(x)
            chi_tau_x = self.lobe_tau(x)
            chi_t_x = chi_t_x - chi_t_x.mean(dim=0, keepdim=True)
            chi_tau_x = chi_tau_x - chi_tau_x.mean(dim=0, keepdim=True)

            u = (chi_t_x @ C00_si) @ W_right_real
            v = (chi_tau_x @ Ctautau_si) @ W_left_real
            return u, v
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Irreversibility field
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_irreversibility_field(
        self,
        x: Tensor,
        output_dict: Dict[str, Tensor],
    ) -> Tensor:
        """Pointwise irreversibility indicator (SVD-based).

        I(x) = sum_k sigma_k * |u_k(x) - v_k(x)|^2

        Large I(x) signals that the dynamics near x strongly break
        detailed balance.  Uses SVD singular vectors and values.

        Parameters
        ----------
        x : Tensor, shape ``(N, input_dim)``
        output_dict : dict
            Output of :meth:`forward`.

        Returns
        -------
        I : Tensor, shape ``(N,)``
            Non-negative irreversibility field evaluated at each point.
        """
        u, v = self.compute_eigenfunctions(x, output_dict)
        sigma = output_dict["singular_values"]  # (d,)

        # |u_k - v_k|^2 per sample per mode -> (N, d)
        diff_sq = (u - v).pow(2)
        # Weight by singular values and sum over modes
        I = (diff_sq * sigma.unsqueeze(0)).sum(dim=1)  # (N,)
        return I

    @torch.no_grad()
    def compute_irreversibility_field_eig(
        self,
        x: Tensor,
        output_dict: Dict[str, Tensor],
    ) -> Optional[Tensor]:
        """Pointwise irreversibility via eigendecomposition (theory-correct).

        I(x) = sum_k |lambda_k| * |u_k(x) - v_k(x)|^2

        Uses the proper eigendecomposition of K rather than SVD, which
        is the correct formulation for non-reversible systems.  Falls
        back to ``None`` if eigendecomposition fails.

        Parameters
        ----------
        x : Tensor, shape ``(N, input_dim)``
        output_dict : dict
            Output of :meth:`forward`.

        Returns
        -------
        I : Tensor or None, shape ``(N,)``
        """
        result = self.compute_eigenfunctions_eig(x, output_dict)
        if result[0] is None:
            return None

        u, v = result
        eigenvalues = output_dict["eigenvalues"]
        weights = eigenvalues.abs().float()  # |lambda_k|

        diff_sq = (u - v).pow(2)
        I = (diff_sq * weights.unsqueeze(0)).sum(dim=1)
        return I
