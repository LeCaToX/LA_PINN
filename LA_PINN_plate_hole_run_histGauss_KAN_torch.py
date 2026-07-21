"""Limit analysis using KAN with multiple high-order Gauss integration.

PyTorch conversion of LA_PINN_plate_hole_run_histGauss_KAN.m.
The geometry, Q4 quadrature, hard boundary conditions, external-work
normalization, dissipation, Adam/L-BFGS stages, and plotting are retained.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from kan_layers import build_kan


# ============================================================
# REPRODUCIBILITY AND TORCH SETTINGS
# ============================================================
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Problem:
    nx: int = 80
    ny: int = 80
    R: float = 0.2
    a: float = 1.0
    p1: float = 1.0
    p2: float = 1.0
    numGauss: int = 2

    coords: np.ndarray | None = None
    elem: np.ndarray | None = None
    Xg: Tensor | None = None
    Wg: Tensor | None = None
    Xr: Tensor | None = None
    Wr: Tensor | None = None
    Xt: Tensor | None = None
    Wt: Tensor | None = None


# ============================================================
# KOLMOGOROV-ARNOLD NETWORK
# ============================================================
class CubicBSplineBasis(nn.Module):
    """Clamped, uniform cubic B-spline basis on ``[-1, 1]``.

    ``grid_size`` is the number of intervals, so the number of basis
    functions is ``grid_size + 3`` for a degree-3 spline.  Cox--de Boor is
    evaluated directly with tensor operations, which keeps the basis fully
    differentiable with respect to the activation values.

    The clamped knot vector is important here.  The previous prototype used
    isolated cardinal splines centred at fixed points; near the endpoints
    those basis functions did not form a partition of unity.  Clamping gives
    stable endpoint behaviour while retaining compact support.
    """

    degree = 3

    def __init__(self, grid_size: int = 5) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        self.grid_size = int(grid_size)
        self.register_buffer("knots", self._make_knots(self.grid_size))

    @staticmethod
    def _make_knots(grid_size: int) -> Tensor:
        inner = torch.linspace(-1.0, 1.0, grid_size + 1, dtype=DTYPE)
        return torch.cat(
            (
                torch.full((CubicBSplineBasis.degree + 1,), -1.0, dtype=DTYPE),
                inner[1:-1],
                torch.full((CubicBSplineBasis.degree + 1,), 1.0, dtype=DTYPE),
            )
        )

    @property
    def n_basis(self) -> int:
        return self.grid_size + self.degree

    def forward(self, x: Tensor) -> Tensor:
        # x is [batch, channels], and the output is [batch, channels, n_basis].
        eps = torch.finfo(x.dtype).eps * 8.0
        x_right = x >= 1.0
        x_eval = x.clamp(-1.0, 1.0 - eps)
        knots = self.knots.to(dtype=x.dtype, device=x.device)

        # Degree-zero basis functions.
        basis = (
            (x_eval.unsqueeze(-1) >= knots[:-1].view(1, 1, -1))
            & (x_eval.unsqueeze(-1) < knots[1:].view(1, 1, -1))
        ).to(dtype=x.dtype)

        # Cox--de Boor recursion.  Zero-width intervals are present because
        # the knot vector is clamped; their contribution is defined as zero.
        for order in range(1, self.degree + 1):
            left_den = knots[order:-1] - knots[: -(order + 1)]
            right_den = knots[order + 1 :] - knots[1:-order]

            left_num = x_eval.unsqueeze(-1) - knots[: -(order + 1)].view(1, 1, -1)
            right_num = knots[order + 1 :].view(1, 1, -1) - x_eval.unsqueeze(-1)

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

        # The right endpoint belongs to the final clamped basis function.
        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.0
        return torch.where(x_right.unsqueeze(-1), endpoint, basis)


class KANLayer(nn.Module):
    """One KAN layer with a smooth base branch and cubic spline edges.

    For each edge ``i -> j`` the learned function is

        ``phi_ji(x) = w_base[j,i] * SiLU(x) +
                      sum_m w_spline[j,i,m] * B_m(tanh(x))``.

    The base branch gives a useful extrapolating path outside the spline grid;
    the spline branch provides local resolution where the collapse mechanism
    changes rapidly around the hole.  The first layer receives the explicitly
    normalized physical coordinates and therefore uses them directly for the
    spline grid; hidden layers use ``tanh`` to keep their spline inputs in the
    bounded grid domain.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 5,
        bounded_spline_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.bounded_spline_input = bool(bounded_spline_input)
        self.basis = CubicBSplineBasis(grid_size)
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim, dtype=DTYPE))
        self.spline_weight = nn.Parameter(
            torch.empty(out_dim, in_dim, self.basis.n_basis, dtype=DTYPE)
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=DTYPE))
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
        base_out = torch.nn.functional.linear(torch.nn.functional.silu(x), self.base_weight)
        return base_out + spline_out + self.bias

    @torch.no_grad()
    def refine_grid(self, new_grid_size: int, fit_points: int = 256) -> None:
        """Refine the spline grid while approximately preserving each edge.

        This is the grid-extension step used by the original KAN work.  A
        least-squares projection maps the current spline curves to the finer
        basis, so refinement does not restart training from random functions.
        """
        if new_grid_size <= self.grid_size:
            raise ValueError(
                f"new_grid_size must exceed {self.grid_size}; got {new_grid_size}."
            )
        if fit_points < new_grid_size + self.basis.degree + 2:
            raise ValueError("fit_points is too small for the requested grid.")

        device = self.spline_weight.device
        dtype = self.spline_weight.dtype
        samples = torch.linspace(-1.0, 1.0, fit_points, dtype=dtype, device=device)
        samples = samples.unsqueeze(1)

        old_basis = self.basis(samples)
        old_values = torch.einsum(
            "nib,oib->nio", old_basis, self.spline_weight
        ).reshape(fit_points, self.in_dim * self.out_dim)

        new_basis_module = CubicBSplineBasis(new_grid_size).to(device=device, dtype=dtype)
        new_basis = new_basis_module(samples).squeeze(1)
        new_values = torch.linalg.lstsq(new_basis, old_values).solution
        new_weight = new_values.reshape(
            new_basis_module.n_basis, self.in_dim, self.out_dim
        ).permute(2, 1, 0).contiguous()

        self.basis = new_basis_module
        self.spline_weight = nn.Parameter(new_weight)


class KAN(nn.Module):
    """KAN for the 2-D velocity field in the plate-with-hole problem."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        grid_size: int = 5,
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if input_scale <= 0.0:
            raise ValueError("input_scale must be positive.")

        self.input_scale = float(input_scale)
        layers: List[nn.Module] = [
            KANLayer(in_dim, width, grid_size, bounded_spline_input=False)
        ]
        for _ in range(1, depth):
            layers.append(KANLayer(width, width, grid_size))
        layers.append(KANLayer(width, out_dim, grid_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        # The geometry is [0, a] x [0, a].  Normalize only the physical input;
        # hidden activations are bounded inside each spline branch by tanh.
        z = 2.0 * x / self.input_scale - 1.0
        for layer in self.layers:
            z = layer(z)
        return z

    @torch.no_grad()
    def refine_grid(self, new_grid_size: int, fit_points: int = 256) -> None:
        for layer in self.layers:
            assert isinstance(layer, KANLayer)
            layer.refine_grid(new_grid_size, fit_points)


def build_net(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    grid_size: int = 5,
    input_scale: float = 1.0,
) -> KAN:
    # Use the shared implementation so every Python KAN entry point has the
    # same cubic basis and grid-extension behaviour.
    return build_kan(
        in_dim,
        out_dim,
        width,
        depth,
        grid_size=grid_size,
        input_scale=input_scale,
        device=DEVICE,
        dtype=DTYPE,
    )


# ============================================================
# PROBLEM CONSTRUCTION
# ============================================================
def build_problem(prob: Problem) -> Problem:
    coords = formnode_pla(prob.nx, prob.ny, prob.R, prob.a)
    elem = build_q4(prob.nx, prob.ny)

    Xg, Wg = domain_quad_q4(coords, elem, prob.numGauss)
    Xr, Wr = edge_quad(coords, prob.a, "right", prob.numGauss)
    Xt, Wt = edge_quad(coords, prob.a, "top", prob.numGauss)

    prob.coords = coords
    prob.elem = elem
    prob.Xg = torch.as_tensor(Xg, dtype=DTYPE, device=DEVICE)
    prob.Wg = torch.as_tensor(Wg, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    prob.Xr = torch.as_tensor(Xr, dtype=DTYPE, device=DEVICE)
    prob.Wr = torch.as_tensor(Wr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    prob.Xt = torch.as_tensor(Xt, dtype=DTYPE, device=DEVICE)
    prob.Wt = torch.as_tensor(Wt, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    return prob


# ============================================================
# LOSS AND KINEMATICS
# ============================================================
def hard_bc(X: Tensor, uv_raw: Tensor) -> Tensor:
    x = X[:, 0:1]
    y = X[:, 1:2]
    uh = uv_raw[:, 0:1]
    vh = uv_raw[:, 1:2]
    return torch.cat((x * uh, y * vh), dim=1)


def external_work(net: nn.Module, prob: Problem) -> Tensor:
    assert prob.Xr is not None and prob.Wr is not None
    assert prob.Xt is not None and prob.Wt is not None

    dr = hard_bc(prob.Xr, net(prob.Xr))
    dt = hard_bc(prob.Xt, net(prob.Xt))

    wext_r = torch.sum(prob.p1 * dr[:, 0:1] * prob.Wr)
    wext_t = torch.sum(prob.p2 * dt[:, 1:2] * prob.Wt)
    return wext_r + wext_t


def strain_rate(X: Tensor, d: Tensor) -> Tensor:
    u = d[:, 0:1]
    v = d[:, 1:2]

    du = torch.autograd.grad(
        u.sum(), X, create_graph=True, retain_graph=True, allow_unused=False
    )[0]
    dv = torch.autograd.grad(
        v.sum(), X, create_graph=True, retain_graph=True, allow_unused=False
    )[0]

    exx = du[:, 0:1]
    eyy = dv[:, 1:2]
    gxy = du[:, 1:2] + dv[:, 0:1]
    return torch.cat((exx, eyy, gxy), dim=1)


def dissipation_density(eps: Tensor, sigma0: float) -> Tensor:
    exx = eps[:, 0:1]
    eyy = eps[:, 1:2]
    gxy = eps[:, 2:3]
    quad = (4.0 / 3.0) * (exx.square() + eyy.square() + exx * eyy) + (
        1.0 / 3.0
    ) * gxy.square()
    return sigma0 * torch.sqrt(torch.clamp(quad, min=1.0e-18))


def compute_loss(net: nn.Module, prob: Problem, sigma0: float) -> Tuple[Tensor, Dict[str, Tensor]]:
    assert prob.Xg is not None and prob.Wg is not None

    wraw = external_work(net, prob)
    alpha = 1.0 / (torch.abs(wraw) + 1.0e-12)

    # A fresh leaf tensor is needed for spatial automatic differentiation.
    X = prob.Xg.detach().clone().requires_grad_(True)
    uv_raw = net(X)
    d = alpha * hard_bc(X, uv_raw)
    eps = strain_rate(X, d)
    D = dissipation_density(eps, sigma0)

    dint = torch.sum(D * prob.Wg)
    wnorm = alpha * torch.abs(wraw)
    info = {"Wraw": wraw, "Wnorm": wnorm, "alpha": alpha}
    return dint, info


@torch.no_grad()
def scalar_info(loss: Tensor, info: Dict[str, Tensor]) -> Tuple[float, float, float, float]:
    return (
        float(loss.detach().cpu()),
        float(info["Wnorm"].detach().cpu()),
        float(info["Wraw"].detach().cpu()),
        float(info["alpha"].detach().cpu()),
    )


# ============================================================
# TRAINING
# ============================================================
def train_one_gauss(
    prob: Problem,
    sigma0: float,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    grid_size: int = 5,
    grid_schedule: Sequence[Tuple[int, int]] = (),
) -> Tuple[nn.Module, List[int], List[float]]:
    net = build_net(2, 2, 64, 4, grid_size=grid_size, input_scale=prob.a)
    lambda_hist: List[float] = []
    iter_hist: List[int] = []

    scheduled_grids = dict(grid_schedule)

    print("Adam training...")
    adam = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(1, n_adam + 1):
        adam.zero_grad(set_to_none=True)
        loss, info = compute_loss(net, prob, sigma0)
        loss.backward()
        adam.step()

        lambda_hist.append(float(loss.detach().cpu()))
        iter_hist.append(epoch)

        if epoch % 100 == 0:
            lam, wnorm, wraw, alpha = scalar_info(loss, info)
            print(
                f"Gauss {prob.numGauss} | Adam {epoch:5d} | "
                f"lambda = {lam:.6f} | Wnorm = {wnorm:.6f} | "
                f"Wraw = {wraw:.6e} | alpha = {alpha:.6e}"
            )

        if epoch in scheduled_grids:
            new_grid_size = scheduled_grids[epoch]
            print(
                f"Refining cubic B-spline grid: {net.layers[0].grid_size} "
                f"-> {new_grid_size} intervals"
            )
            net.refine_grid(new_grid_size)
            # The parameter tensors are replaced by their least-squares
            # projections during refinement, so recreate Adam's state.
            adam = torch.optim.Adam(net.parameters(), lr=lr)

    print("LBFGS training...")
    # max_iter=1 preserves the original outer-loop count and reporting cadence.
    lbfgs = torch.optim.LBFGS(
        net.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=2,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    for iteration in range(1, n_lbfgs + 1):
        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = compute_loss(net, prob, sigma0)
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)

        if iteration % 25 == 0:
            # Evaluation must retain gradient tracking because spatial derivatives
            # are part of the loss even though no parameter update is performed here.
            loss, info = compute_loss(net, prob, sigma0)
            lambda_hist.append(float(loss.detach().cpu()))
            iter_hist.append(n_adam + iteration)

            lam, wnorm, wraw, alpha = scalar_info(loss, info)
            print(
                f"Gauss {prob.numGauss} | LBFGS {iteration:5d} | "
                f"lambda = {lam:.6f} | Wnorm = {wnorm:.6f} | "
                f"Wraw = {wraw:.6e} | alpha = {alpha:.6e}"
            )

    return net, iter_hist, lambda_hist


# ============================================================
# MESH GENERATION
# ============================================================
def formnode_pla(nx: int, ny: int, R: float, a: float) -> np.ndarray:
    if nx % 2 == 1:
        raise ValueError("NX must be even.")

    dd = 0.25
    ang = math.pi / (2.0 * nx)
    nk = ny + ny * (ny - 1) * dd / 2.0
    coords = np.zeros(((nx + 1) * (ny + 1), 2), dtype=np.float64)
    c = 0

    for ip in range(nx + 1):
        angi = math.pi / 2.0 - ip * ang
        xi = R * math.cos(angi)
        yi = R * math.sin(angi)

        if ip <= nx // 2:
            ye = a
            # tan(pi/2) is large but finite numerically; this reproduces MATLAB.
            xe = ye / math.tan(angi)
        else:
            xe = a
            ye = xe * math.tan(angi)

        dx = 0.0
        dy = 0.0
        for iq in range(ny + 1):
            if iq > 0:
                k = ny - iq
                dx += (1.0 + k * dd) * (xe - xi) / nk
                dy += (1.0 + k * dd) * (ye - yi) / nk
            coords[c, :] = (xe - dx, ye - dy)
            c += 1

    return coords


def build_q4(nx: int, ny: int) -> np.ndarray:
    elem = np.zeros((nx * ny, 4), dtype=np.int64)
    e = 0
    for i in range(nx):
        for j in range(ny):
            # MATLAB node numbers converted from one-based to zero-based indexing.
            n1 = (ny + 1) * i + (j + 1)
            n2 = (ny + 1) * (i + 1) + (j + 1)
            n3 = (ny + 1) * (i + 1) + j
            n4 = (ny + 1) * i + j
            elem[e, :] = (n1, n2, n3, n4)
            e += 1
    return elem


def shape4_q(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    dNdxi = 0.25 * np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=np.float64,
    )
    dNdeta = 0.25 * np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=np.float64,
    )
    return N, dNdxi, dNdeta


def gauss_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n == 2:
        gp = np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
        wg = np.array([1.0, 1.0])
    elif n == 3:
        gp = np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)])
        wg = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif n == 4:
        gp = np.array(
            [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
        )
        wg = np.array(
            [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
        )
    elif n == 5:
        gp = np.array(
            [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
        )
        wg = np.array(
            [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
             0.4786286704993665, 0.2369268850561891]
        )
    else:
        raise ValueError("Unsupported Gauss order.")
    return gp, wg


def domain_quad_q4(
    coords: np.ndarray, elem: np.ndarray, ngauss: int
) -> Tuple[np.ndarray, np.ndarray]:
    gp, wg = gauss_1d(ngauss)
    ng = elem.shape[0] * gp.size * gp.size
    Xg = np.zeros((ng, 2), dtype=np.float32)
    Wg = np.zeros(ng, dtype=np.float32)
    c = 0

    for e, nodes in enumerate(elem):
        Xe = coords[nodes, :]
        for i, xi in enumerate(gp):
            for j, eta in enumerate(gp):
                N, dNdxi, dNdeta = shape4_q(float(xi), float(eta))
                J = np.vstack((dNdxi, dNdeta)) @ Xe
                detJ = float(np.linalg.det(J))
                if detJ <= 0.0:
                    raise RuntimeError(f"Negative or zero Jacobian in element {e + 1}.")
                Xg[c, :] = N @ Xe
                Wg[c] = wg[i] * wg[j] * detJ
                c += 1
    return Xg, Wg


def edge_quad(
    coords: np.ndarray, a: float, edge: str, ngauss: int
) -> Tuple[np.ndarray, np.ndarray]:
    tol = 1.0e-6
    x = coords[:, 0]
    y = coords[:, 1]

    if edge == "right":
        ids = np.where(np.abs(x - a) < tol)[0]
        ids = ids[np.argsort(y[ids])]
    elif edge == "top":
        ids = np.where(np.abs(y - a) < tol)[0]
        ids = ids[np.argsort(x[ids])]
    else:
        raise ValueError("Unknown edge.")

    pts = coords[ids, :]
    gp, wg = gauss_1d(ngauss)
    ng = gp.size * (pts.shape[0] - 1)
    Xg = np.zeros((ng, 2), dtype=np.float32)
    Wg = np.zeros(ng, dtype=np.float32)
    c = 0

    for k in range(pts.shape[0] - 1):
        P0 = pts[k, :]
        P1 = pts[k + 1, :]
        Jedge = np.linalg.norm(P1 - P0) / 2.0
        for i, s in enumerate(gp):
            Xg[c, :] = 0.5 * (1.0 - s) * P0 + 0.5 * (1.0 + s) * P1
            Wg[c] = wg[i] * Jedge
            c += 1
    return Xg, Wg


# ============================================================
# POSTPROCESSING
# ============================================================
def nodal_dissipation(net: nn.Module, prob: Problem, sigma0: float) -> np.ndarray:
    assert prob.coords is not None
    wraw = external_work(net, prob)
    alpha = 1.0 / (torch.abs(wraw) + 1.0e-12)
    X = torch.as_tensor(prob.coords, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    d = alpha * hard_bc(X, net(X))
    eps = strain_rate(X, d)
    D = dissipation_density(eps, sigma0)
    return D.detach().cpu().numpy().ravel()


def plot_dissipation(prob: Problem, Dnode: np.ndarray) -> None:
    assert prob.coords is not None and prob.elem is not None
    coords = prob.coords
    elem = prob.elem
    tris = np.empty((2 * elem.shape[0], 3), dtype=np.int64)
    tris[0::2, :] = elem[:, [0, 1, 2]]
    tris[1::2, :] = elem[:, [0, 2, 3]]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    contour = ax.tripcolor(
        coords[:, 0], coords[:, 1], tris, Dnode, shading="gouraud"
    )
    fig.colorbar(contour, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Normalized upper-bound dissipation, {prob.numGauss} x {prob.numGauss} Gauss"
    )
    th = np.linspace(0.0, math.pi / 2.0, 200)
    ax.plot(prob.R * np.cos(th), prob.R * np.sin(th), "k--", linewidth=1.2)
    fig.tight_layout()
    fig.savefig(f"UB_dissipation_gauss_{prob.numGauss}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lambda_history_all(
    all_iter_hist: Sequence[Sequence[int]],
    all_lambda_hist: Sequence[Sequence[float]],
    num_gauss_list: Sequence[int],
    n_adam: int,
    nx: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for iterations, values, ngauss in zip(all_iter_hist, all_lambda_hist, num_gauss_list):
        iterations_np = np.asarray(iterations)
        values_np = np.asarray(values)
        n_adam_local = min(n_adam, iterations_np.size)

        line = ax.plot(
            iterations_np[:n_adam_local],
            values_np[:n_adam_local],
            "-",
            linewidth=2.0,
            label=f"{ngauss} x {ngauss} Gauss - Adam",
        )[0]

        if iterations_np.size > n_adam_local:
            ax.plot(
                iterations_np[n_adam_local:],
                values_np[n_adam_local:],
                "--",
                linewidth=2.0,
                color=line.get_color(),
                label=f"{ngauss} x {ngauss} Gauss - LBFGS",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^{+}$")
    ax.grid(True)
    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"Hist_gaussp2_{nx}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_initial_nodes(prob: Problem) -> None:
    assert prob.coords is not None
    coords = prob.coords
    tol = 1.0e-8
    Lx = np.max(coords[:, 0])
    Ly = np.max(coords[:, 1])
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    id_hole = np.abs(r - prob.R) < 1.0e-6
    id_left = np.abs(coords[:, 0]) < tol
    id_bottom = np.abs(coords[:, 1]) < tol
    id_right = np.abs(coords[:, 0] - Lx) < tol
    id_top = np.abs(coords[:, 1] - Ly) < tol
    id_bnd = id_hole | id_left | id_bottom | id_right | id_top
    id_int = ~id_bnd

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(coords[id_int, 0], coords[id_int, 1], "k.", markersize=2.5)
    ax.plot(coords[id_hole, 0], coords[id_hole, 1], "co", markersize=4)
    ax.plot(coords[id_left, 0], coords[id_left, 1], "bo", markersize=3)
    ax.plot(coords[id_bottom, 0], coords[id_bottom, 1], "gs", markersize=3)
    ax.plot(coords[id_right, 0], coords[id_right, 1], "ro", markersize=3)
    ax.plot(coords[id_top, 0], coords[id_top, 1], "md", markersize=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    fig.tight_layout()
    fig.savefig(f"Data_gauss_{prob.nx}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    sigma0 = 1.0
    base_prob = Problem(nx=80, ny=80, R=0.2, a=1.0, p1=1.0, p2=1.0)
    num_gauss_list = [2, 3, 5]
    n_adam = 4000
    n_lbfgs = 900
    lr = 1.0e-3
    # Start coarse for stable optimization, then extend the grid after the
    # mechanism has formed.  Cubic degree remains fixed throughout.
    grid_size = 5
    grid_schedule = ((1500, 9), (3000, 13))

    print(f"Using device: {DEVICE}")
    all_iter_hist: List[List[int]] = []
    all_lambda_hist: List[List[float]] = []

    for ngauss in num_gauss_list:
        print("\n========================================")
        print(f"Running numGauss = {ngauss}")
        print("========================================")

        prob = Problem(
            nx=base_prob.nx,
            ny=base_prob.ny,
            R=base_prob.R,
            a=base_prob.a,
            p1=base_prob.p1,
            p2=base_prob.p2,
            numGauss=ngauss,
        )
        prob = build_problem(prob)
        if ig == 0:
            plot_initial_nodes(prob)

        net, iter_hist, lambda_hist = train_one_gauss(
            prob,
            sigma0,
            n_adam,
            n_lbfgs,
            lr,
            grid_size=grid_size,
            grid_schedule=grid_schedule,
        )
        all_iter_hist.append(iter_hist)
        all_lambda_hist.append(lambda_hist)

        # Save the representative dissipation field for the finest Gauss case.
        if ngauss == num_gauss_list[-1]:
            Dnode = nodal_dissipation(net, prob, sigma0)
            plot_dissipation(prob, Dnode)

    plot_lambda_history_all(
        all_iter_hist, all_lambda_hist, num_gauss_list, n_adam, base_prob.nx
    )


if __name__ == "__main__":
    main()
