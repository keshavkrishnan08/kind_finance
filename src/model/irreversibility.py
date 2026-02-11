"""
Irreversibility field analysis.

The irreversibility field I(x) quantifies how strongly the dynamics at
state x deviate from detailed balance.  It is constructed from the
singular values of the whitened Koopman matrix and the mismatch between
left and right Koopman eigenfunctions:

    I(x) = sum_k sigma_k * |u_k(x) - v_k(x)|^2

where u_k, v_k are the right and left eigenfunctions respectively.

This module provides utilities for computing I(x) at arbitrary data
points and on regular grids (for 1-D / 2-D visualization).
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from .vampnet import NonEquilibriumVAMPNet


class IrreversibilityAnalyzer:
    """Compute and cache the irreversibility field from a trained
    :class:`NonEquilibriumVAMPNet`.

    Parameters
    ----------
    model : NonEquilibriumVAMPNet
        A trained VAMPNet instance.
    device : torch.device or str or None
        Device for inference.  Defaults to the model's device.
    """

    def __init__(
        self,
        model: NonEquilibriumVAMPNet,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Core: compute I(x) at given data points
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_field(
        self,
        x: Tensor,
        output_dict: Dict[str, Tensor],
        batch_size: int = 4096,
    ) -> Tensor:
        """Evaluate the irreversibility field at data points.

        I(x) = sum_k sigma_k * |u_k(x) - v_k(x)|^2

        Parameters
        ----------
        x : Tensor, shape ``(N, input_dim)``
            Points at which to evaluate.
        output_dict : dict
            Output of ``model.forward(x_t, x_tau)`` on a representative
            data batch (provides covariance inverses, SVD vectors, and
            singular values).
        batch_size : int
            Process ``x`` in mini-batches of this size to manage memory.

        Returns
        -------
        I : Tensor, shape ``(N,)``
            Non-negative irreversibility field on CPU.
        """
        self.model.eval()
        x = x.to(self.device)

        # Move output_dict tensors to device
        out_dev = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in output_dict.items()
        }

        N = x.shape[0]
        I_chunks: List[Tensor] = []

        for start in range(0, N, batch_size):
            x_batch = x[start : start + batch_size]
            I_batch = self.model.compute_irreversibility_field(x_batch, out_dev)
            I_chunks.append(I_batch.cpu())

        return torch.cat(I_chunks, dim=0)

    # ------------------------------------------------------------------
    # Grid evaluation (1-D and 2-D visualisation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_on_grid(
        self,
        output_dict: Dict[str, Tensor],
        grid_ranges: List[Tuple[float, float]],
        n_points: Union[int, List[int]] = 100,
        fixed_values: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Tensor]:
        """Evaluate I(x) on a regular grid for low-dimensional visualisation.

        For 1-D: provide one range, get I on a line.
        For 2-D: provide two ranges, get I on a meshgrid.
        Higher-dimensional inputs are supported by clamping unspecified
        dimensions to fixed values via ``fixed_values``.

        Parameters
        ----------
        output_dict : dict
            Output of ``model.forward()`` on a representative batch.
        grid_ranges : list of (lo, hi)
            Min/max for each grid dimension.  Length determines the grid
            dimensionality (1 or 2 for practical visualisation).
        n_points : int or list[int]
            Number of grid points per dimension.  If a scalar, all
            dimensions get the same resolution.
        fixed_values : dict[int, float] or None
            For input dimensions *not* in the grid: ``{dim_idx: value}``.
            All non-grid dimensions default to 0.0 if not specified.

        Returns
        -------
        dict with keys:
            ``grid_coords`` : list[Tensor]
                1-D coordinate tensors for each grid dimension.
            ``meshgrid`` : list[Tensor]
                Meshgrid tensors (for 2-D contour plots).
            ``I`` : Tensor
                Irreversibility field values.  Shape matches the
                meshgrid (squeezed for 1-D).
        """
        self.model.eval()
        grid_dim = len(grid_ranges)
        input_dim = self.model.input_dim

        if isinstance(n_points, int):
            n_points_list = [n_points] * grid_dim
        else:
            n_points_list = list(n_points)

        if fixed_values is None:
            fixed_values = {}

        # Build 1-D coordinate vectors
        coords = [
            torch.linspace(lo, hi, n_pts)
            for (lo, hi), n_pts in zip(grid_ranges, n_points_list)
        ]

        # Build meshgrid
        if grid_dim == 1:
            grid_points = coords[0].unsqueeze(1)  # (n, 1)
        else:
            mesh = torch.meshgrid(*coords, indexing="ij")
            # Flatten to (n_total, grid_dim)
            grid_points = torch.stack(
                [m.reshape(-1) for m in mesh], dim=1
            )

        n_total = grid_points.shape[0]

        # Embed grid points into the full input space
        x_full = torch.zeros(n_total, input_dim, dtype=torch.float32)

        # Set fixed dimensions
        for dim_idx, val in fixed_values.items():
            x_full[:, dim_idx] = val

        # Determine which input dimensions the grid covers.
        # By convention, the grid dimensions are the *first* dimensions
        # not in fixed_values, in order.
        free_dims = sorted(
            set(range(input_dim)) - set(fixed_values.keys())
        )
        for g_idx in range(grid_dim):
            if g_idx < len(free_dims):
                x_full[:, free_dims[g_idx]] = grid_points[:, g_idx]

        # Compute irreversibility field
        I_flat = self.compute_field(x_full, output_dict)

        # Reshape to grid
        if grid_dim == 1:
            I_grid = I_flat
        else:
            shape = tuple(n_points_list)
            I_grid = I_flat.reshape(shape)

        result: Dict[str, Tensor] = {
            "grid_coords": coords,
            "I": I_grid,
        }

        if grid_dim >= 2:
            result["meshgrid"] = list(
                torch.meshgrid(*coords, indexing="ij")
            )

        return result

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(
        self,
        x: Tensor,
        output_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute summary statistics of the irreversibility field.

        Returns
        -------
        dict with keys ``mean``, ``std``, ``max``, ``median``,
        ``fraction_above_mean`` (fraction of points with I > mean I).
        """
        I = self.compute_field(x, output_dict)
        mean_I = I.mean()

        return {
            "mean": mean_I,
            "std": I.std(),
            "max": I.max(),
            "median": I.median(),
            "fraction_above_mean": (I > mean_I).float().mean(),
            "field": I,
        }
