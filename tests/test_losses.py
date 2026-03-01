"""
Unit tests for all loss functions (PRD Section 7.1).

Validates:
    - VAMP-2 loss is negative (we minimise)
    - Identity Koopman gives maximal VAMP-2
    - Orthogonality loss is zero for orthogonal K
    - Entropy loss behaviour with/without empirical estimate
    - Spectral penalty zero/positive semantics
    - Total loss is the correct weighted combination
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.model.losses import (
    entropy_production_consistency_loss,
    orthogonality_loss,
    spectral_penalty,
    total_loss,
    vamp2_loss,
)
from src.model.vampnet import NonEquilibriumVAMPNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_singular_values():
    """Singular values that are all 1 (perfect Koopman operator)."""
    return torch.ones(4)


@pytest.fixture
def orthogonal_koopman():
    """An orthogonal matrix (K^T K = I)."""
    # Construct via QR decomposition of a random matrix
    torch.manual_seed(42)
    A = torch.randn(4, 4)
    Q, _ = torch.linalg.qr(A)
    return Q


@pytest.fixture
def model_output():
    """A mock output_dict from NonEquilibriumVAMPNet.forward()."""
    torch.manual_seed(42)
    d = 4
    N = 128

    model = NonEquilibriumVAMPNet(
        input_dim=5,
        hidden_dims=[16],
        output_dim=d,
        dropout=0.0,
    )
    x_t = torch.randn(N, 5)
    x_tau = torch.randn(N, 5) * 0.9 + x_t * 0.1

    model.train()
    out = model(x_t, x_tau)
    return out


# ---------------------------------------------------------------------------
# Tests: VAMP-2 loss
# ---------------------------------------------------------------------------

class TestVAMP2Loss:
    """VAMP-2 loss should be negative (we minimise to maximise score)."""

    def test_vamp2_loss_negative(self):
        """Loss is negative of sum of squared singular values."""
        sv = torch.tensor([0.9, 0.7, 0.5, 0.3])
        loss = vamp2_loss(sv)
        assert loss.item() < 0, (
            f"VAMP-2 loss should be negative, got {loss.item():.4f}"
        )

    def test_vamp2_loss_equals_negative_sum_sq(self):
        """Loss = -sum(sigma_k^2)."""
        sv = torch.tensor([0.9, 0.7, 0.5, 0.3])
        loss = vamp2_loss(sv)
        expected = -(0.9 ** 2 + 0.7 ** 2 + 0.5 ** 2 + 0.3 ** 2)
        assert abs(loss.item() - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {loss.item():.6f}"
        )

    def test_vamp2_loss_identity(self, identity_singular_values):
        """Identity K gives max VAMP-2 = K (number of modes)."""
        sv = identity_singular_values
        loss = vamp2_loss(sv)
        n_modes = len(sv)
        expected_loss = -float(n_modes)  # -sum(1^2) = -K

        assert abs(loss.item() - expected_loss) < 1e-6, (
            f"Identity singular values: expected loss = {expected_loss}, "
            f"got {loss.item():.6f}"
        )

    def test_vamp2_loss_zero_sv(self):
        """Zero singular values give zero VAMP-2 loss."""
        sv = torch.zeros(4)
        loss = vamp2_loss(sv)
        assert abs(loss.item()) < 1e-10, (
            f"Zero singular values should give zero loss, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Tests: Orthogonality loss
# ---------------------------------------------------------------------------

class TestOrthogonalityLoss:
    """Orthogonality loss penalises off-diagonal entries of K^T K."""

    def test_orthogonality_zero_for_orthogonal(self, orthogonal_koopman):
        """L = 0 when K has orthogonal columns (K^T K = I)."""
        loss = orthogonality_loss(orthogonal_koopman)
        assert loss.item() < 1e-8, (
            f"Orthogonality loss should be ~0 for orthogonal K, "
            f"got {loss.item():.6e}"
        )

    def test_orthogonality_positive_for_nonorthogonal(self):
        """L > 0 when K is not orthogonal."""
        K = torch.tensor([
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.3],
            [0.0, 0.0, 1.0],
        ])
        loss = orthogonality_loss(K)
        assert loss.item() > 0, (
            f"Non-orthogonal K should have positive orthogonality loss, "
            f"got {loss.item()}"
        )

    def test_orthogonality_identity(self):
        """Identity matrix is orthogonal -> loss = 0."""
        K = torch.eye(4)
        loss = orthogonality_loss(K)
        assert loss.item() < 1e-10, (
            f"Identity is orthogonal, loss should be ~0, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Tests: Entropy production consistency loss
# ---------------------------------------------------------------------------

class TestEntropyConsistencyLoss:
    """Entropy consistency loss penalises mismatch with empirical estimate."""

    def test_entropy_loss_zero_when_disabled(self, model_output):
        """L = 0 when sigma_empirical is None (term is skipped)."""
        _, loss_dict = total_loss(
            model_output,
            tau=1.0,
            sigma_empirical=None,
            eigenfunc_amplitudes=None,
        )
        assert "entropy_consistency" not in loss_dict, (
            "Entropy consistency should be absent when sigma_empirical=None"
        )

    def test_entropy_loss_matches(self):
        """L ~ 0 when learned and empirical entropy match."""
        # Create eigenvalues with known angle
        angles = torch.tensor([0.0, 0.1, -0.1, 0.2])
        magnitudes = torch.tensor([0.99, 0.95, 0.95, 0.90])
        eigenvalues = magnitudes * torch.exp(1j * angles)

        amplitudes = torch.ones(4)
        tau = 1.0

        # Compute sigma_spectral manually: omega^2 * A_k / gamma_k
        omega = angles / tau
        gamma = (-torch.log(magnitudes.clamp(min=1e-12, max=1.0 - 1e-7)) / tau).clamp(min=1e-6)
        sigma_target = (omega.pow(2) * amplitudes / gamma).sum()

        loss = entropy_production_consistency_loss(
            eigenvalues, amplitudes, tau, sigma_target
        )
        assert loss.item() < 1e-10, (
            f"Entropy loss should be ~0 when learned matches empirical, "
            f"got {loss.item():.6e}"
        )

    def test_entropy_loss_positive_when_mismatch(self):
        """L > 0 when learned and empirical entropy disagree."""
        eigenvalues = torch.tensor(
            [0.99 + 0.0j, 0.9 + 0.1j, 0.9 - 0.1j, 0.8 + 0.2j]
        )
        amplitudes = torch.ones(4)
        tau = 1.0
        sigma_empirical = torch.tensor(999.0)  # clearly mismatched

        loss = entropy_production_consistency_loss(
            eigenvalues, amplitudes, tau, sigma_empirical
        )
        assert loss.item() > 0, (
            f"Entropy loss should be positive when mismatched, "
            f"got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Tests: Spectral penalty
# ---------------------------------------------------------------------------

class TestSpectralPenalty:
    """Spectral penalty penalises sigma_k > 1."""

    def test_spectral_penalty_zero_when_valid(self):
        """L = 0 when all sigma_k <= 1."""
        sv = torch.tensor([0.99, 0.85, 0.70, 0.50])
        loss = spectral_penalty(sv)
        assert loss.item() < 1e-10, (
            f"Spectral penalty should be 0 when all sigma_k <= 1, "
            f"got {loss.item():.6e}"
        )

    def test_spectral_penalty_positive_when_violated(self):
        """L > 0 when sigma_k > 1."""
        sv = torch.tensor([1.2, 0.9, 0.8, 1.1])
        loss = spectral_penalty(sv)
        assert loss.item() > 0, (
            f"Spectral penalty should be positive when sigma_k > 1, "
            f"got {loss.item()}"
        )

        # Check exact value: sum(max(0, sv-1)^2) = 0.2^2 + 0 + 0 + 0.1^2
        expected = 0.2 ** 2 + 0.1 ** 2
        assert abs(loss.item() - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {loss.item():.6f}"
        )

    def test_spectral_penalty_exact_one(self):
        """L = 0 when sigma_k == 1 exactly."""
        sv = torch.ones(4)
        loss = spectral_penalty(sv)
        assert loss.item() < 1e-10, (
            f"Spectral penalty at sigma_k=1 exactly should be 0, "
            f"got {loss.item()}"
        )

    def test_spectral_penalty_all_above(self):
        """Large penalty when all sigma_k > 1."""
        sv = torch.tensor([2.0, 3.0, 4.0])
        loss = spectral_penalty(sv)
        expected = 1.0 ** 2 + 2.0 ** 2 + 3.0 ** 2  # (2-1)^2 + (3-1)^2 + (4-1)^2
        assert abs(loss.item() - expected) < 1e-5, (
            f"Expected {expected:.4f}, got {loss.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Total loss composition
# ---------------------------------------------------------------------------

class TestTotalLoss:
    """Total loss should be the weighted combination of components."""

    def test_total_loss_is_sum(self, model_output):
        """total = w_vamp2 * L_vamp2 + w_ortho * L_ortho + w_spectral * L_spectral."""
        w_vamp2 = 1.0
        w_ortho = 0.01
        w_spectral = 0.1

        total, loss_dict = total_loss(
            model_output,
            tau=1.0,
            w_vamp2=w_vamp2,
            w_orthogonality=w_ortho,
            w_spectral=w_spectral,
            sigma_empirical=None,
        )

        # Reconstruct the expected total from individual components
        expected = (
            w_vamp2 * loss_dict["vamp2"].item()
            + w_ortho * loss_dict["orthogonality"].item()
            + w_spectral * loss_dict["spectral_penalty"].item()
        )

        assert abs(total.item() - expected) < 1e-4, (
            f"Total loss ({total.item():.6f}) should equal weighted sum "
            f"({expected:.6f})"
        )

    def test_total_loss_with_entropy(self, model_output):
        """Total includes entropy term when sigma_empirical is provided."""
        # Compute eigenfunc amplitudes from the output
        chi_t = model_output["chi_t"]
        amplitudes = chi_t.pow(2).mean(dim=0).detach()

        sigma_emp = torch.tensor(0.1)

        total, loss_dict = total_loss(
            model_output,
            tau=1.0,
            sigma_empirical=sigma_emp,
            eigenfunc_amplitudes=amplitudes,
            w_vamp2=1.0,
            w_orthogonality=0.01,
            w_entropy=0.1,
            w_spectral=0.1,
        )

        assert "entropy_consistency" in loss_dict, (
            "Entropy consistency should be in loss_dict when sigma_empirical "
            "is provided"
        )
        assert "total" in loss_dict, "Total should always be in loss_dict"

    def test_total_loss_all_weights_zero(self, model_output):
        """All weights zero gives total = 0."""
        total, _ = total_loss(
            model_output,
            tau=1.0,
            w_vamp2=0.0,
            w_orthogonality=0.0,
            w_spectral=0.0,
            sigma_empirical=None,
        )
        assert abs(total.item()) < 1e-10, (
            f"All weights zero should give total = 0, got {total.item()}"
        )

    def test_total_loss_differentiable(self, model_output):
        """Total loss should be differentiable (no detach on the returned loss)."""
        total, _ = total_loss(model_output, tau=1.0)
        assert total.requires_grad, "Total loss should require grad for backprop"
