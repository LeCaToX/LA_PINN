"""Reusable cubic-B-spline KAN layers for the LA_PINN examples.

The basis is degree-3, clamped, and uniform on ``[-1, 1]``.  Hidden-layer
spline inputs are smoothly bounded with ``tanh``; the first layer can receive
coordinates already normalized to the spline domain.  Grid refinement projects
the learned edge curves to a finer basis instead of reinitializing them.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CubicBSplineBasis(nn.Module):
    """Clamped cubic B-spline basis with compact support."""

    degree = 3

    def __init__(self, grid_size: int = 5, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        self.grid_size = int(grid_size)
        basis_dtype = dtype or torch.get_default_dtype()
        self.register_buffer("knots", self._make_knots(self.grid_size, basis_dtype))

    @staticmethod
    def _make_knots(grid_size: int, dtype: torch.dtype) -> Tensor:
        inner = torch.linspace(-1.0, 1.0, grid_size + 1, dtype=dtype)
        return torch.cat(
            (
                torch.full((CubicBSplineBasis.degree + 1,), -1.0, dtype=dtype),
                inner[1:-1],
                torch.full((CubicBSplineBasis.degree + 1,), 1.0, dtype=dtype),
            )
        )

    @property
    def n_basis(self) -> int:
        return self.grid_size + self.degree

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"KAN expects [batch, channels], got shape {tuple(x.shape)}")

        eps = torch.finfo(x.dtype).eps * 8.0
        x_right = x >= 1.0
        x_eval = x.clamp(-1.0, 1.0 - eps)
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        x_expanded = x_eval.unsqueeze(-1)

        basis = (
            (x_expanded >= knots[:-1].view(1, 1, -1))
            & (x_expanded < knots[1:].view(1, 1, -1))
        ).to(dtype=x.dtype)

        for order in range(1, self.degree + 1):
            left_den = knots[order:-1] - knots[: -(order + 1)]
            right_den = knots[order + 1 :] - knots[1:-order]
            left_num = x_expanded - knots[: -(order + 1)].view(1, 1, -1)
            right_num = knots[order + 1 :].view(1, 1, -1) - x_expanded

            left = torch.where(
                left_den.view(1, 1, -1) > 0.0,
                left_num / left_den.clamp_min(eps).view(1, 1, -1),
                torch.zeros_like(left_num),
            )
            right = torch.where(
                right_den.view(1, 1, -1) > 0.0,
                right_num / right_den.clamp_min(eps).view(1, 1, -1),
                torch.zeros_like(right_num),
            )
            basis = left * basis[..., :-1] + right * basis[..., 1:]

        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.0
        return torch.where(x_right.unsqueeze(-1), endpoint, basis)


class KANLayer(nn.Module):
    """KAN edge functions followed by summation at output nodes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 5,
        bounded_spline_input: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.bounded_spline_input = bool(bounded_spline_input)
        parameter_dtype = dtype or torch.get_default_dtype()
        self.basis = CubicBSplineBasis(grid_size, dtype=parameter_dtype)
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim, dtype=parameter_dtype))
        self.spline_weight = nn.Parameter(
            torch.empty(out_dim, in_dim, self.basis.n_basis, dtype=parameter_dtype)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=parameter_dtype))
        self.reset_parameters()

    @property
    def grid_size(self) -> int:
        return self.basis.grid_size

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        x_grid = torch.tanh(x) if self.bounded_spline_input else x
        basis = self.basis(x_grid)
        spline_out = torch.einsum("nib,oib->no", basis, self.spline_weight)
        base_out = F.linear(F.silu(x), self.base_weight)
        return base_out + spline_out + self.bias

    @torch.no_grad()
    def refine_grid(self, new_grid_size: int, fit_points: int = 256) -> None:
        if new_grid_size <= self.grid_size:
            raise ValueError(
                f"new_grid_size must exceed {self.grid_size}; got {new_grid_size}."
            )
        if fit_points < new_grid_size + self.basis.degree + 2:
            raise ValueError("fit_points is too small for the requested grid.")

        device = self.spline_weight.device
        dtype = self.spline_weight.dtype
        samples = torch.linspace(-1.0, 1.0, fit_points, dtype=dtype, device=device).unsqueeze(1)
        old_basis = self.basis(samples)
        old_values = torch.einsum(
            "nib,oib->nio", old_basis, self.spline_weight
        ).reshape(fit_points, self.in_dim * self.out_dim)

        new_basis_module = CubicBSplineBasis(new_grid_size, dtype=dtype).to(
            device=device
        )
        new_basis = new_basis_module(samples).squeeze(1)
        new_values = torch.linalg.lstsq(new_basis, old_values).solution
        new_weight = new_values.reshape(
            new_basis_module.n_basis, self.in_dim, self.out_dim
        ).permute(2, 1, 0).contiguous()

        self.basis = new_basis_module
        self.spline_weight = nn.Parameter(new_weight)


class KAN(nn.Module):
    """KAN velocity-field approximator for any spatial dimension."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        grid_size: int = 5,
        input_scale: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if input_scale <= 0.0:
            raise ValueError("input_scale must be positive.")

        self.input_scale = float(input_scale)
        layers: List[nn.Module] = [
            KANLayer(
                in_dim,
                width,
                grid_size,
                bounded_spline_input=False,
                dtype=dtype,
            )
        ]
        for _ in range(1, depth):
            layers.append(KANLayer(width, width, grid_size, dtype=dtype))
        layers.append(KANLayer(width, out_dim, grid_size, dtype=dtype))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        z = 2.0 * x / self.input_scale - 1.0
        for layer in self.layers:
            z = layer(z)
        return z

    @torch.no_grad()
    def refine_grid(self, new_grid_size: int, fit_points: int = 256) -> None:
        for layer in self.layers:
            assert isinstance(layer, KANLayer)
            layer.refine_grid(new_grid_size, fit_points)


def build_kan(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    grid_size: int = 5,
    input_scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> KAN:
    """Construct a KAN and optionally move it to a device/dtype."""
    model = KAN(
        in_dim,
        out_dim,
        width,
        depth,
        grid_size=grid_size,
        input_scale=input_scale,
        dtype=dtype,
    )
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model
