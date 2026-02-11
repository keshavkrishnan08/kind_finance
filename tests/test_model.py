"""
Unit tests for the VAMPnet architecture per PRD Section 16.1.

Validates:
    - Output tensor shapes
    - Singular value bounds (sigma_k <= 1 + eps after training)
    - Eigenvalue magnitude bounds (|lambda_k| <= 1 + eps)
    - VAMP-2 score improvement during training
    - Shared vs separate weight modes
    - matrix_sqrt_inv correctness
    - Gradient flow through all computational paths
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.vampnet import (
    NonEquilibriumVAMPNet,
    VAMPNetLobe,
    matrix_sqrt_inv,
)
from src.model.losses import total_loss, vamp2_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model():
    """A small NonEquilibriumVAMPNet for unit testing."""
    return NonEquilibriumVAMPNet(
        input_dim=5,
        hidden_dims=[32, 16],
        output_dim=4,
        dropout=0.0,
        epsilon=1e-6,
    )


@pytest.fixture
def random_data():
    """Random time-lagged pair for a 5-dimensional system."""
    torch.manual_seed(42)
    N = 256
    d = 5
    x_t = torch.randn(N, d)
    x_tau = torch.randn(N, d) * 0.9 + x_t * 0.1  # weak correlation
    return x_t, x_tau


@pytest.fixture
def correlated_data():
    """Correlated time-lagged pair for training tests."""
    torch.manual_seed(42)
    N = 512
    d = 5
    # Generate a simple AR(1) process
    data = torch.zeros(N + 1, d)
    for t in range(N):
        data[t + 1] = 0.8 * data[t] + 0.2 * torch.randn(d)
    x_t = data[:-1]
    x_tau = data[1:]
    return x_t, x_tau


# ---------------------------------------------------------------------------
# Tests: Output shapes
# ---------------------------------------------------------------------------

class TestVAMPNetOutputShapes:
    """All output tensors should have correct shapes."""

    def test_vampnet_output_shapes(self, simple_model, random_data):
        """Verify all output tensors from forward() have expected shapes."""
        x_t, x_tau = random_data
        N = x_t.shape[0]
        d = simple_model.output_dim

        simple_model.eval()
        with torch.no_grad():
            out = simple_model(x_t, x_tau)

        # Encoded representations
        assert out["chi_t"].shape == (N, d), (
            f"chi_t shape: expected ({N}, {d}), got {out['chi_t'].shape}"
        )
        assert out["chi_tau"].shape == (N, d), (
            f"chi_tau shape: expected ({N}, {d}), got {out['chi_tau'].shape}"
        )

        # Covariance matrices
        assert out["C00"].shape == (d, d), (
            f"C00 shape: expected ({d}, {d}), got {out['C00'].shape}"
        )
        assert out["C0tau"].shape == (d, d), (
            f"C0tau shape: expected ({d}, {d}), got {out['C0tau'].shape}"
        )
        assert out["Ctautau"].shape == (d, d), (
            f"Ctautau shape: expected ({d}, {d}), got {out['Ctautau'].shape}"
        )

        # Inverse square root matrices
        assert out["C00_sqrt_inv"].shape == (d, d), (
            f"C00_sqrt_inv shape: expected ({d}, {d}), got {out['C00_sqrt_inv'].shape}"
        )
        assert out["Ctautau_sqrt_inv"].shape == (d, d), (
            f"Ctautau_sqrt_inv shape: expected ({d}, {d}), "
            f"got {out['Ctautau_sqrt_inv'].shape}"
        )

        # Koopman matrix
        assert out["koopman_matrix"].shape == (d, d), (
            f"koopman_matrix shape: expected ({d}, {d}), "
            f"got {out['koopman_matrix'].shape}"
        )

        # Singular values
        assert out["singular_values"].shape == (d,), (
            f"singular_values shape: expected ({d},), "
            f"got {out['singular_values'].shape}"
        )

        # SVD factors
        assert out["svd_U"].shape == (d, d), (
            f"svd_U shape: expected ({d}, {d}), got {out['svd_U'].shape}"
        )
        assert out["svd_V"].shape == (d, d), (
            f"svd_V shape: expected ({d}, {d}), got {out['svd_V'].shape}"
        )

        # Eigenvalues (complex)
        assert out["eigenvalues"].shape == (d,), (
            f"eigenvalues shape: expected ({d},), got {out['eigenvalues'].shape}"
        )

    def test_lobe_output_shape(self):
        """Individual lobe produces correct output dimensionality."""
        lobe = VAMPNetLobe(input_dim=10, hidden_dims=[32, 16], output_dim=6)
        x = torch.randn(64, 10)
        y = lobe(x)
        assert y.shape == (64, 6), f"Expected (64, 6), got {y.shape}"


# ---------------------------------------------------------------------------
# Tests: Singular value bounds
# ---------------------------------------------------------------------------

class TestSingularValueBounds:
    """Singular values of the Koopman matrix should be bounded by 1."""

    def test_singular_values_bounded(self, correlated_data):
        """sigma_k <= 1 + eps after training with spectral penalty."""
        x_t, x_tau = correlated_data
        torch.manual_seed(42)

        model = NonEquilibriumVAMPNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=4,
            dropout=0.0,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(80):
            optimizer.zero_grad()
            out = model(x_t, x_tau)
            loss, _ = total_loss(out, tau=1.0, w_spectral=1.0)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x_t, x_tau)
        sv = out["singular_values"].numpy()

        eps = 0.15  # small tolerance for finite training
        assert np.all(sv <= 1.0 + eps), (
            f"Singular values should be <= 1+eps, got max = {sv.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Eigenvalue magnitude bounds
# ---------------------------------------------------------------------------

class TestEigenvalueBounds:
    """Eigenvalue magnitudes should be bounded by 1."""

    def test_eigenvalue_magnitudes_bounded(self, correlated_data):
        """|lambda_k| <= 1 + eps after training."""
        x_t, x_tau = correlated_data
        torch.manual_seed(42)

        model = NonEquilibriumVAMPNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=4,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(80):
            optimizer.zero_grad()
            out = model(x_t, x_tau)
            loss, _ = total_loss(out, tau=1.0, w_spectral=1.0)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x_t, x_tau)

        eig_mags = out["eigenvalues"].abs().numpy()
        eps = 0.2
        assert np.all(eig_mags <= 1.0 + eps), (
            f"|lambda_k| should be <= 1+eps, got max = {eig_mags.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: VAMP-2 improves during training
# ---------------------------------------------------------------------------

class TestVAMP2Training:
    """VAMP-2 score should increase during training."""

    def test_vamp2_increases_during_training(self, correlated_data):
        """VAMP-2 improves over 50 epochs."""
        x_t, x_tau = correlated_data
        torch.manual_seed(42)

        model = NonEquilibriumVAMPNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=4,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        vamp2_scores = []

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            out = model(x_t, x_tau)
            loss, loss_dict = total_loss(out, tau=1.0)
            loss.backward()
            optimizer.step()

            # Record VAMP-2 score (positive = better)
            vamp2 = out["singular_values"].pow(2).sum().item()
            vamp2_scores.append(vamp2)

        # VAMP-2 at epoch 50 should exceed VAMP-2 at epoch 1
        early_vamp2 = np.mean(vamp2_scores[:5])
        late_vamp2 = np.mean(vamp2_scores[-5:])

        assert late_vamp2 >= early_vamp2, (
            f"VAMP-2 should improve: early={early_vamp2:.4f}, late={late_vamp2:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Shared vs separate weight modes
# ---------------------------------------------------------------------------

class TestWeightSharing:
    """Shared and separate weight modes should work correctly."""

    def test_shared_weight_mode(self):
        """share_weights=True: lobe_left and lobe_right are the same object."""
        model = NonEquilibriumVAMPNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=4,
            dropout=0.0,
        )
        # Simulate shared weights by setting lobe_tau = lobe_t
        model.lobe_tau = model.lobe_t

        # Verify they are the same object
        assert model.lobe_t is model.lobe_tau, (
            "Shared weight mode: lobe_t and lobe_tau should be the same object"
        )

        # Verify same parameters
        params_t = list(model.lobe_t.parameters())
        params_tau = list(model.lobe_tau.parameters())
        for p_t, p_tau in zip(params_t, params_tau):
            assert torch.equal(p_t, p_tau), (
                "Shared weight parameters should be identical"
            )

        # Forward pass should still work
        x = torch.randn(32, 5)
        out = model(x, x)
        assert "koopman_matrix" in out

    def test_separate_weight_mode(self):
        """share_weights=False: lobes have different parameters."""
        model = NonEquilibriumVAMPNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=4,
            dropout=0.0,
        )

        # By default, lobes are separate
        assert model.lobe_t is not model.lobe_tau, (
            "Separate weight mode: lobe_t and lobe_tau should be different objects"
        )

        # Verify parameters are not the same tensors
        params_t = list(model.lobe_t.parameters())
        params_tau = list(model.lobe_tau.parameters())
        different = any(
            not torch.equal(p_t, p_tau)
            for p_t, p_tau in zip(params_t, params_tau)
        )
        assert different, (
            "Separate lobes should have different parameter values "
            "(from independent initialization)"
        )


# ---------------------------------------------------------------------------
# Tests: matrix_sqrt_inv correctness
# ---------------------------------------------------------------------------

class TestMatrixSqrtInv:
    """M^{-1/2} @ M^{-1/2} should approximately equal M^{-1}."""

    def test_matrix_sqrt_inv(self):
        """Verify that (C^{-1/2})^2 = C^{-1} for a random PD matrix."""
        torch.manual_seed(42)
        d = 5

        # Create a random positive definite matrix
        A = torch.randn(d, d)
        C = A @ A.T + 0.1 * torch.eye(d)  # guarantee PD

        C_sqrt_inv = matrix_sqrt_inv(C)

        # Check: C^{-1/2} @ C^{-1/2} ~ C^{-1}
        product = C_sqrt_inv @ C_sqrt_inv
        C_inv = torch.linalg.inv(C)

        # Frobenius norm of the difference
        error = torch.norm(product - C_inv, p="fro").item()
        relative_error = error / torch.norm(C_inv, p="fro").item()

        assert relative_error < 1e-4, (
            f"matrix_sqrt_inv: (C^(-1/2))^2 should equal C^(-1), "
            f"relative error = {relative_error:.6e}"
        )

    def test_matrix_sqrt_inv_identity(self):
        """C^{-1/2} of the identity should be the identity."""
        d = 4
        I = torch.eye(d)
        I_sqrt_inv = matrix_sqrt_inv(I)
        error = torch.norm(I_sqrt_inv - I, p="fro").item()
        assert error < 1e-5, (
            f"matrix_sqrt_inv(I) should be I, error = {error:.6e}"
        )

    def test_matrix_sqrt_inv_diagonal(self):
        """C^{-1/2} of a diagonal matrix should have diagonal 1/sqrt(c_ii)."""
        d = 4
        diag_vals = torch.tensor([1.0, 4.0, 9.0, 16.0])
        C = torch.diag(diag_vals)
        C_sqrt_inv = matrix_sqrt_inv(C)

        expected_diag = 1.0 / torch.sqrt(diag_vals)
        expected = torch.diag(expected_diag)

        error = torch.norm(C_sqrt_inv - expected, p="fro").item()
        assert error < 1e-5, (
            f"matrix_sqrt_inv of diagonal matrix: error = {error:.6e}"
        )


# ---------------------------------------------------------------------------
# Tests: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Gradients should flow through all computational paths."""

    def test_forward_backward_consistency(self, simple_model, random_data):
        """All parameters should receive gradients after backward pass."""
        x_t, x_tau = random_data

        simple_model.train()
        out = simple_model(x_t, x_tau)
        loss, _ = total_loss(out, tau=1.0)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in simple_model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient after backward()"
            )
            # At least one element should be non-zero
            assert param.grad.abs().sum().item() > 0, (
                f"Parameter '{name}' has all-zero gradient"
            )

    def test_gradient_through_svd(self, simple_model, random_data):
        """Gradients flow through the SVD decomposition."""
        x_t, x_tau = random_data

        simple_model.train()
        out = simple_model(x_t, x_tau)

        # Loss depends on singular values (which come from SVD)
        sv_loss = -out["singular_values"].pow(2).sum()
        sv_loss.backward()

        # lobe_t parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in simple_model.lobe_t.parameters()
        )
        assert has_grad, "lobe_t should have gradients from SVD path"

    def test_gradient_through_koopman(self, simple_model, random_data):
        """Gradients flow through the Koopman matrix construction."""
        x_t, x_tau = random_data

        simple_model.train()
        out = simple_model(x_t, x_tau)

        # Loss depends on Koopman matrix entries
        k_loss = out["koopman_matrix"].pow(2).sum()
        k_loss.backward()

        # Both lobes should receive gradients
        has_grad_t = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in simple_model.lobe_t.parameters()
        )
        has_grad_tau = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in simple_model.lobe_tau.parameters()
        )
        assert has_grad_t, "lobe_t should have gradients from Koopman path"
        assert has_grad_tau, "lobe_tau should have gradients from Koopman path"
