"""Python counterparts for the MATLAB KAN entry points.

This module keeps the MATLAB case names and numerical settings in Python.  The
plate cases reuse the Q4 geometry and cubic B-spline KAN used by the existing
PyTorch implementation.  The thick and thin cylinder cases are implemented
here directly because their integration domains differ from the plate case.

Every training routine writes a ``*.progress.pt`` checkpoint.  If a process is
interrupted, rerunning the same entry point resumes that case automatically.
Use ``--output-dir`` and the optional ``--adam``/``--lbfgs`` overrides on the
small wrapper scripts when testing before a full GPU run.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import torch
from torch import Tensor, nn

import LA_PINN_plate_hole_run_histGauss_KAN_torch as plate
from kan_layers import build_kan


DTYPE = plate.DTYPE
DEVICE = plate.DEVICE
SIGMA0 = 1.0
SEED = 1234


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_tree(value):
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to_device(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device=DEVICE)


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(cpu_tree(payload), temporary)
    os.replace(temporary, path)


def build_mlp(in_dim: int, out_dim: int, width: int, depth: int) -> nn.Module:
    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, width), nn.Tanh()))
        current = width
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers).to(device=DEVICE, dtype=DTYPE)


def build_model(kind: str, in_dim: int, out_dim: int, width: int, depth: int, scale: float, grid_size: int = 5) -> nn.Module:
    if kind == "kan":
        return build_kan(
            in_dim,
            out_dim,
            width,
            depth,
            grid_size=grid_size,
            input_scale=scale,
            device=DEVICE,
            dtype=DTYPE,
        )
    if kind == "mlp":
        return build_mlp(in_dim, out_dim, width, depth)
    raise ValueError(f"Unknown model kind: {kind}")


LossFactory = Callable[[], tuple[Tensor, dict[str, Tensor]]]


def train_with_checkpoint(
    net: nn.Module,
    loss_factory: LossFactory,
    output_dir: Path,
    checkpoint_name: str,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    epoch_offset: int = 0,
    save_every: int = 100,
) -> tuple[nn.Module, list[int], list[float]]:
    """Train Adam then L-BFGS and resume from a compact checkpoint."""

    checkpoint = output_dir / checkpoint_name
    history_iter: list[int] = []
    history_loss: list[float] = []
    phase = "adam"
    step = 0
    payload = {}
    loaded_phase: str | None = None
    adam: torch.optim.Optimizer | None = None
    lbfgs: torch.optim.Optimizer | None = None

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        net.load_state_dict(payload["model"])
        history_iter = list(payload.get("iter", []))
        history_loss = list(payload.get("loss", []))
        phase = payload.get("phase", "adam")
        step = int(payload.get("step", 0))
        loaded_phase = phase
        print(f"Resuming {checkpoint.name}: {phase} step {step}")

    if phase == "done":
        return net, history_iter, history_loss

    if phase == "adam":
        adam = torch.optim.Adam(net.parameters(), lr=lr)
        if loaded_phase == "adam" and checkpoint.exists() and payload.get("optimizer") is not None:
            adam.load_state_dict(payload["optimizer"])
            optimizer_to_device(adam)
        for epoch in range(step + 1, n_adam + 1):
            adam.zero_grad(set_to_none=True)
            loss, info = loss_factory()
            loss.backward()
            adam.step()
            history_iter.append(epoch_offset + epoch)
            history_loss.append(float(loss.detach().cpu()))
            if epoch % 100 == 0 or epoch == 1:
                print(
                    f"Adam {epoch:5d} | lambda = {float(loss.detach().cpu()):.8f} | "
                    f"Wext = {float(info.get('Wraw', info.get('Wext', torch.zeros(())).detach()).detach().cpu()):.4e}"
                )
            if epoch % save_every == 0 or epoch == n_adam:
                atomic_save(
                    {"model": net.state_dict(), "optimizer": adam.state_dict(),
                     "phase": "adam", "step": epoch, "iter": history_iter,
                     "loss": history_loss}, checkpoint
                )
        phase = "lbfgs"
        step = 0
        atomic_save(
            {"model": net.state_dict(), "optimizer": None, "phase": phase,
             "step": step, "iter": history_iter, "loss": history_loss}, checkpoint
        )

    if phase == "lbfgs":
        lbfgs = torch.optim.LBFGS(
            net.parameters(), lr=1.0, max_iter=1, max_eval=2,
            tolerance_grad=1.0e-7, tolerance_change=1.0e-9,
            history_size=50, line_search_fn="strong_wolfe",
        )
        if loaded_phase == "lbfgs" and checkpoint.exists() and payload.get("optimizer") is not None:
            lbfgs.load_state_dict(payload["optimizer"])
            optimizer_to_device(lbfgs)
        for iteration in range(step + 1, n_lbfgs + 1):
            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                value, _ = loss_factory()
                value.backward()
                return value

            lbfgs.step(closure)
            if iteration % 25 == 0 or iteration == n_lbfgs:
                value, _ = loss_factory()
                history_iter.append(epoch_offset + n_adam + iteration)
                history_loss.append(float(value.detach().cpu()))
                print(f"LBFGS {iteration:5d} | lambda = {float(value.detach().cpu()):.8f}")
            if iteration % save_every == 0 or iteration == n_lbfgs:
                atomic_save(
                    {"model": net.state_dict(), "optimizer": lbfgs.state_dict(),
                     "phase": "lbfgs", "step": iteration, "iter": history_iter,
                     "loss": history_loss}, checkpoint
                )

    atomic_save(
        {"model": net.state_dict(), "optimizer": None, "phase": "done",
         "step": n_lbfgs, "iter": history_iter, "loss": history_loss}, checkpoint
    )
    return net, history_iter, history_loss


def set_plate_integration(prob, Xg: np.ndarray, Wg: np.ndarray) -> None:
    prob.Xg = torch.as_tensor(Xg, dtype=DTYPE, device=DEVICE)
    prob.Wg = torch.as_tensor(Wg, dtype=DTYPE, device=DEVICE).reshape(-1, 1)


def plate_loss(net: nn.Module, prob, shear_factor: float = 1.0 / 3.0) -> tuple[Tensor, dict[str, Tensor]]:
    wraw = plate.external_work(net, prob)
    alpha = 1.0 / (torch.abs(wraw) + 1.0e-12)
    X = prob.Xg.detach().clone().requires_grad_(True)
    d = alpha * plate.hard_bc(X, net(X))
    eps = plate.strain_rate(X, d)
    exx, eyy, gxy = eps[:, 0:1], eps[:, 1:2], eps[:, 2:3]
    quad = (4.0 / 3.0) * (exx.square() + eyy.square() + exx * eyy) + shear_factor * gxy.square()
    D = SIGMA0 * torch.sqrt(torch.clamp(quad, min=1.0e-18))
    loss = torch.sum(D * prob.Wg)
    return loss, {"Wraw": wraw, "Wnorm": alpha * torch.abs(wraw), "alpha": alpha}


def plate_nodes_dissipation(net: nn.Module, prob, shear_factor: float) -> np.ndarray:
    wraw = plate.external_work(net, prob)
    alpha = 1.0 / (torch.abs(wraw) + 1.0e-12)
    X = torch.as_tensor(prob.coords, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    d = alpha * plate.hard_bc(X, net(X))
    eps = plate.strain_rate(X, d)
    exx, eyy, gxy = eps[:, 0:1], eps[:, 1:2], eps[:, 2:3]
    quad = (4.0 / 3.0) * (exx.square() + eyy.square() + exx * eyy) + shear_factor * gxy.square()
    return (SIGMA0 * torch.sqrt(torch.clamp(quad, min=1.0e-18))).detach().cpu().numpy().ravel()


def save_plate_plots(prob, Dnode: np.ndarray, iterations: Sequence[int], values: Sequence[float], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tris = np.empty((2 * prob.elem.shape[0], 3), dtype=np.int64)
    tris[0::2] = prob.elem[:, [0, 1, 2]]
    tris[1::2] = prob.elem[:, [0, 2, 3]]
    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.tripcolor(prob.coords[:, 0], prob.coords[:, 1], tris, Dnode, shading="gouraud")
    fig.colorbar(c, ax=ax)
    th = np.linspace(0, math.pi / 2, 300)
    ax.plot(prob.R * np.cos(th), prob.R * np.sin(th), "k--", linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Plastic dissipation density")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_dissipation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    it = np.asarray(iterations)
    val = np.asarray(values)
    adam_mask = it <= (max(it) if len(it) else 0)
    ax.plot(it[adam_mask], val[adam_mask], "b-", linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^+$")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_history.pdf", bbox_inches="tight")
    plt.close(fig)


def run_plate_case(
    name: str,
    output_dir: Path,
    nx: int,
    ngauss: int,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    width: int = 64,
    depth: int = 4,
    kind: str = "kan",
    shear_factor: float = 1.0 / 3.0,
    p2: float = 0.0,
    grid_size: int = 5,
) -> tuple[nn.Module, object, list[int], list[float]]:
    seed_everything()
    output_dir.mkdir(parents=True, exist_ok=True)
    prob = plate.Problem(nx=nx, ny=nx, R=0.2, a=1.0, p1=1.0, p2=p2, numGauss=ngauss)
    prob = plate.build_problem(prob)
    net = build_model(kind, 2, 2, width, depth, prob.a, grid_size)
    loss_factory = lambda: plate_loss(net, prob, shear_factor)
    net, iterations, values = train_with_checkpoint(
        net, loss_factory, output_dir, f"{name}.progress.pt", n_adam, n_lbfgs, lr
    )
    Dnode = plate_nodes_dissipation(net, prob, shear_factor)
    atomic_save(
        {"model": net.state_dict(), "iter": iterations, "loss": values,
         "config": {"nx": nx, "gauss": ngauss, "kind": kind}},
        output_dir / f"{name}.pt",
    )
    save_plate_plots(prob, Dnode, iterations, values, output_dir, name)
    return net, prob, iterations, values


def run_fast(output_dir: Path, n_adam: int = 3000, n_lbfgs: int = 0, lr: float = 2.0e-3) -> None:
    run_plate_case("fast_KAN", output_dir, 40, 2, n_adam, n_lbfgs, lr, 32, 3, "kan", 1.0)


def run_standard(output_dir: Path, n_adam: int = 4000, n_lbfgs: int = 300, lr: float = 1.0e-3) -> None:
    run_plate_case("plate_hole_KAN", output_dir, 20, 2, n_adam, n_lbfgs, lr)


def run_high_gauss(output_dir: Path, n_adam: int = 4000, n_lbfgs: int = 300, lr: float = 1.0e-3) -> None:
    name = "UB_dissipation_high_order_gauss_enhanced"
    run_plate_case(name, output_dir, 20, 3, n_adam, n_lbfgs, lr)
    shutil.copyfile(output_dir / f"{name}_dissipation.pdf", output_dir / f"{name}.pdf")
    shutil.copyfile(output_dir / f"{name}_history.pdf", output_dir / "UB_lambda_history_high_order_gauss.pdf")


def run_hist_gauss(output_dir: Path, n_adam: int = 4000, n_lbfgs: int = 900, lr: float = 1.0e-3) -> None:
    all_hist: list[tuple[list[int], list[float], int]] = []
    for ngauss in (2, 3, 5):
        _, _, it, val = run_plate_case(f"plate_g{ngauss}_KAN", output_dir, 80, ngauss, n_adam, n_lbfgs, lr, p2=1.0, grid_size=8)
        all_hist.append((it, val, ngauss))
        if ngauss == 5:
            shutil.copyfile(output_dir / "plate_g5_KAN_dissipation.pdf", output_dir / "UB_dissipation_gauss_5.pdf")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for it, val, ngauss in all_hist:
        ax.plot(it, val, linewidth=1.8, label=f"{ngauss} x {ngauss} Gauss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^+$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "Hist_gaussp2_80.pdf", bbox_inches="tight")
    plt.close(fig)


def run_activation(output_dir: Path, n_adam: int = 4000, n_lbfgs: int = 500, lr: float = 1.0e-3) -> None:
    seed_everything()
    prob = plate.build_problem(plate.Problem(nx=20, ny=20, R=0.2, a=1.0, p1=1.0, p2=0.0, numGauss=3))
    results = []
    for kind, label in (("mlp", "MLP-tanh"), ("kan", "cubic B-spline KAN")):
        net = build_model(kind, 2, 2, 64, 4, prob.a)
        factory = lambda net=net: plate_loss(net, prob, 1.0 / 3.0)
        net, it, val = train_with_checkpoint(net, factory, output_dir, f"activation_{kind}.progress.pt", n_adam, n_lbfgs, lr)
        atomic_save({"model": net.state_dict(), "iter": it, "loss": val, "kind": kind}, output_dir / f"activation_{kind}.pt")
        results.append((label, net, it, val))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, _, it, val in results:
        ax.plot(it, val, linewidth=1.8, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^+$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "activation_comparison_history.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    tris = np.empty((2 * prob.elem.shape[0], 3), dtype=np.int64)
    tris[0::2] = prob.elem[:, [0, 1, 2]]
    tris[1::2] = prob.elem[:, [0, 2, 3]]
    for ax, (label, net, _, _) in zip(axes, results):
        D = plate_nodes_dissipation(net, prob, 1.0 / 3.0)
        c = ax.tripcolor(prob.coords[:, 0], prob.coords[:, 1], tris, D, shading="gouraud")
        ax.set_title(label)
        ax.set_aspect("equal")
        fig.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "activation_comparison_dissipation.pdf", bbox_inches="tight")
    plt.close(fig)


def adaptive_points(prob, hot: np.ndarray, base_order: int, hot_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_parts: list[np.ndarray] = []
    W_parts: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for eid, nodes in enumerate(prob.elem):
        order = hot_order if bool(hot[eid]) else base_order
        X, W = plate.domain_quad_q4(prob.coords, prob.elem[eid:eid + 1], order)
        X_parts.append(X)
        W_parts.append(W)
        ids.append(np.full(W.size, eid, dtype=np.int64))
    return np.vstack(X_parts), np.concatenate(W_parts), np.concatenate(ids)


def run_adaptive(output_dir: Path, n_adam1: int = 2000, n_adam2: int = 2000, n_lbfgs: int = 500, lr: float = 1.0e-3) -> None:
    seed_everything()
    prob = plate.build_problem(plate.Problem(nx=40, ny=40, R=0.2, a=1.0, p1=1.0, p2=0.0, numGauss=2))
    net = build_model("kan", 2, 2, 64, 4, prob.a)
    factory = lambda: plate_loss(net, prob, 1.0 / 3.0)
    net, it1, val1 = train_with_checkpoint(net, factory, output_dir, "adaptive_stage1.progress.pt", n_adam1, 0, lr)
    with torch.enable_grad():
        wraw = plate.external_work(net, prob)
        alpha = 1.0 / (torch.abs(wraw) + 1.0e-12)
        X = prob.Xg.detach().clone().requires_grad_(True)
        eps = plate.strain_rate(X, alpha * plate.hard_bc(X, net(X)))
        exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
        D = torch.sqrt(torch.clamp((4.0 / 3.0) * (exx.square() + eyy.square() + exx * eyy) + (1.0 / 3.0) * gxy.square(), min=1.0e-18)).detach().cpu().numpy()
    nqp = 2 ** 2
    cell_D = D.reshape(-1, nqp).mean(axis=1)
    threshold = np.percentile(cell_D, 80.0)
    hot = cell_D >= threshold
    Xint, Wint, elem_id = adaptive_points(prob, hot, 2, 3)
    set_plate_integration(prob, Xint, Wint)
    net, it2, val2 = train_with_checkpoint(net, factory, output_dir, "adaptive_stage2.progress.pt", n_adam2, n_lbfgs, lr, epoch_offset=n_adam1)
    iterations = it1 + it2
    values = val1 + val2
    Xeval, Weval = plate.domain_quad_q4(prob.coords, prob.elem, 5)
    eval_prob = SimpleNamespace(**vars(prob))
    set_plate_integration(eval_prob, Xeval, Weval)
    final_lambda, _ = plate_loss(net, eval_prob, 1.0 / 3.0)
    print(f"Final fixed 5 x 5 Gauss lambda = {float(final_lambda.detach().cpu()):.8f}")
    Dnode = plate_nodes_dissipation(net, prob, 1.0 / 3.0)
    atomic_save({"model": net.state_dict(), "iter": iterations, "loss": values, "hot": hot,
                 "final_fixed_gauss5": float(final_lambda.detach().cpu())}, output_dir / "adaptive_Gauss_KAN.pt")
    save_plate_plots(prob, Dnode, iterations, values, output_dir, "adaptive_Gauss")
    shutil.copyfile(output_dir / "adaptive_Gauss_dissipation.pdf", output_dir / "plastic_dissipation_density.pdf")
    fig, ax = plt.subplots(figsize=(7, 6))
    polys = [prob.coords[nodes] for nodes in prob.elem[hot]]
    ax.add_collection(PolyCollection(polys, facecolor="crimson", edgecolor="none", alpha=0.7))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(f"Adaptive hot elements: {int(hot.sum())}/{hot.size}")
    fig.tight_layout(); fig.savefig(output_dir / "adaptive_hot_elements.pdf", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(iterations, values, "b-"); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\lambda^+$"); ax.grid(True); fig.tight_layout(); fig.savefig(output_dir / "adaptive_Gauss_history.pdf", bbox_inches="tight"); plt.close(fig)


def thin_loss(net: nn.Module, theta: Tensor, a: float, t: float, p: float, beta: float) -> tuple[Tensor, dict[str, Tensor]]:
    theta_var = theta.detach().clone().requires_grad_(True)
    raw = net(theta_var)
    u = torch.cos(theta_var) * raw[:, 0:1]
    v = torch.sin(theta_var) * raw[:, 1:2]
    wraw = torch.trapezoid(p * a * (u * torch.cos(theta_var) + v * torch.sin(theta_var))[:, 0], theta_var[:, 0])
    alpha = torch.where(wraw < 0.0, -1.0, 1.0) / (torch.abs(wraw) + 1.0e-12)
    u = alpha * u; v = alpha * v
    dv = torch.autograd.grad(v.sum(), theta_var, create_graph=True)[0]
    eps = (dv + u) / a
    D = SIGMA0 * torch.sqrt(eps.square() + 1.0e-18)
    wint = torch.trapezoid((D * t)[:, 0], theta_var[:, 0])
    linc = torch.trapezoid(eps.square()[:, 0], theta_var[:, 0])
    return wint + beta * 0.0 * linc, {"Wraw": wraw, "Linc": linc}


def run_thin(output_dir: Path, n_adam: int = 4000, n_lbfgs: int = 400, lr: float = 1.0e-3) -> None:
    seed_everything(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    a, t, p = 1.0, 0.05, 1.0
    theta = torch.linspace(0.0, math.pi / 2.0, 120, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    net = build_kan(1, 2, 64, 4, grid_size=5, input_scale=math.pi / 2.0, device=DEVICE, dtype=DTYPE)
    factory = lambda: thin_loss(net, theta, a, t, p, 1000.0)
    net, it, val = train_with_checkpoint(net, factory, output_dir, "thinwall.progress.pt", n_adam, n_lbfgs, lr)
    loss, _ = thin_loss(net, theta, a, t, p, 1000.0)
    theta_eval = theta.detach().clone().requires_grad_(True)
    raw = net(theta_eval); u = torch.cos(theta_eval) * raw[:, 0:1]; v = torch.sin(theta_eval) * raw[:, 1:2]
    wraw = torch.trapezoid((p * a * (u * torch.cos(theta_eval) + v * torch.sin(theta_eval)))[:, 0], theta_eval[:, 0])
    alpha = torch.where(wraw < 0.0, -1.0, 1.0) / (torch.abs(wraw) + 1.0e-12)
    u = alpha * u; v = alpha * v; dv = torch.autograd.grad(v.sum(), theta_eval, create_graph=True)[0]
    D = (SIGMA0 * torch.sqrt(((dv + u) / a).square() + 1.0e-18)).detach().cpu().numpy().ravel()
    theta_np = theta[:, 0].detach().cpu().numpy(); u_np = u.detach().cpu().numpy().ravel(); v_np = v.detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(); ax.plot(theta_np, D, linewidth=2); ax.set_xlabel("theta"); ax.set_ylabel("Plastic dissipation"); ax.grid(True); fig.tight_layout(); fig.savefig(output_dir / "thinwall_dissipation.pdf"); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(a * np.cos(theta_np), a * np.sin(theta_np), "k-"); ax.quiver(a * np.cos(theta_np), a * np.sin(theta_np), u_np, v_np, color="r"); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(output_dir / "thinwall_velocity.pdf"); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(it, val, "b-"); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\lambda^+$"); ax.grid(True); fig.tight_layout(); fig.savefig(output_dir / "thinwall_history.pdf"); plt.close(fig)
    atomic_save({"model": net.state_dict(), "iter": it, "loss": val, "final_loss": float(loss.detach().cpu())}, output_dir / "thinwall_KAN.pt")


def annulus_mesh(a: float, b: float, nr: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, nr + 1); radii = a + (b - a) * s**1.8; theta = np.linspace(0.0, math.pi / 2.0, nt + 1)
    nodes = np.zeros(((nr + 1) * (nt + 1), 2), dtype=np.float64)
    def node_id(i: int, j: int) -> int: return j * (nr + 1) + i
    for j, th in enumerate(theta):
        for i, r in enumerate(radii): nodes[node_id(i, j)] = (r * math.cos(th), r * math.sin(th))
    elem = np.zeros((nr * nt, 4), dtype=np.int64); e = 0
    for j in range(nt):
        for i in range(nr):
            elem[e] = (node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)); e += 1
    return nodes, elem


def annulus_domain_quad(nodes: np.ndarray, elem: np.ndarray, ngauss: int) -> tuple[np.ndarray, np.ndarray]:
    gp, wg = plate.gauss_1d(ngauss); Xs: list[np.ndarray] = []; Ws: list[float] = []
    for eid in elem:
        Xe = nodes[eid]
        for i, xi in enumerate(gp):
            for j, eta in enumerate(gp):
                N, dxi, deta = plate.shape4_q(float(xi), float(eta)); J = np.vstack((dxi, deta)) @ Xe; det = float(np.linalg.det(J))
                if det <= 0.0: raise RuntimeError("Annulus element has non-positive Jacobian")
                Xs.append(N @ Xe); Ws.append(float(wg[i] * wg[j] * det))
    return np.asarray(Xs, dtype=np.float32), np.asarray(Ws, dtype=np.float32)


def inner_pressure_quad(a: float, nt: int, ngauss: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gp, wg = plate.gauss_1d(ngauss); angles = np.linspace(0.0, math.pi / 2.0, nt + 1); Xs=[]; Ws=[]; Ns=[]
    for k in range(nt):
        for s, w in zip(gp, wg):
            th = 0.5 * (1 - s) * angles[k] + 0.5 * (1 + s) * angles[k + 1]
            Xs.append((a * math.cos(th), a * math.sin(th))); Ws.append(float(w * (angles[k + 1] - angles[k]) * a)); Ns.append((math.cos(th), math.sin(th)))
    return np.asarray(Xs, dtype=np.float32), np.asarray(Ws, dtype=np.float32), np.asarray(Ns, dtype=np.float32)


def thick_loss(net: nn.Module, prob, sigma0: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    wraw = thick_external(net, prob); sign = torch.where(wraw < 0.0, -1.0, 1.0); alpha = sign / (torch.abs(wraw) + 1.0e-12)
    X = prob.Xg.detach().clone().requires_grad_(True); d = alpha * plate.hard_bc(X, net(X)); eps = plate.strain_rate(X, d)
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    D = sigma0 * torch.sqrt(torch.clamp((exx - eyy).square() + gxy.square(), min=1.0e-18))
    wint = torch.sum(D.reshape(-1, 1) * prob.Wg); div = eps[:, 0] + eps[:, 1]; linc = torch.sum(div.square().reshape(-1, 1) * prob.Wg)
    return wint + prob.betaInc * linc, {"Wraw": wraw, "Linc": linc}


def thick_external(net: nn.Module, prob) -> Tensor:
    d = plate.hard_bc(prob.Xi, net(prob.Xi)); un = torch.sum(d * prob.Ni, dim=1, keepdim=True)
    return torch.sum(prob.p * un * prob.Wi)


def run_thick(output_dir: Path, ratios: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8), n_adam: int = 20000, n_lbfgs: int = 200, lr: float = 1.0e-3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True); all_results=[]
    for ratio in ratios:
        seed_everything(); b, a = 2.0, ratio * 2.0; tag = f"ab_{round(100 * ratio):03d}"
        nodes, elem = annulus_mesh(a, b, 24, 48); Xg, Wg = annulus_domain_quad(nodes, elem, 5); Xi, Wi, Ni = inner_pressure_quad(a, 48, 5)
        prob = SimpleNamespace(node=nodes, elem=elem, a=a, b=b, p=1.0, betaInc=1.0,
                               Xg=torch.as_tensor(Xg, dtype=DTYPE, device=DEVICE), Wg=torch.as_tensor(Wg, dtype=DTYPE, device=DEVICE).reshape(-1, 1),
                               Xi=torch.as_tensor(Xi, dtype=DTYPE, device=DEVICE), Wi=torch.as_tensor(Wi, dtype=DTYPE, device=DEVICE).reshape(-1, 1), Ni=torch.as_tensor(Ni, dtype=DTYPE, device=DEVICE))
        net = build_kan(2, 2, 64, 4, grid_size=5, input_scale=b, device=DEVICE, dtype=DTYPE)
        factory = lambda: thick_loss(net, prob, SIGMA0)
        net, it, val = train_with_checkpoint(net, factory, output_dir, f"{tag}.progress.pt", n_adam, n_lbfgs, lr)
        Dnode = thick_node_dissipation(net, prob)
        atomic_save({"model": net.state_dict(), "iter": it, "loss": val, "ratio": ratio}, output_dir / f"{tag}_KAN.pt")
        save_thick_plots(prob, net, Dnode, it, val, output_dir, tag)
        all_results.append((f"a/b = {ratio:.1f}", it, val))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, it, val in all_results: ax.plot(it, val, linewidth=1.5, label=label)
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\lambda^+$"); ax.grid(True); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "thick_history_all_ab_ratios.pdf"); plt.close(fig)


def thick_node_dissipation(net: nn.Module, prob) -> np.ndarray:
    wraw = thick_external(net, prob); alpha = torch.where(wraw < 0.0, -1.0, 1.0) / (torch.abs(wraw) + 1.0e-12)
    X = torch.as_tensor(prob.node, dtype=DTYPE, device=DEVICE).requires_grad_(True); eps = plate.strain_rate(X, alpha * plate.hard_bc(X, net(X)))
    return torch.sqrt(torch.clamp((eps[:, 0] - eps[:, 1]).square() + eps[:, 2].square(), min=1.0e-18)).detach().cpu().numpy().ravel()


def save_thick_plots(prob, net: nn.Module, Dnode: np.ndarray, it: Sequence[int], val: Sequence[float], output_dir: Path, tag: str) -> None:
    tris = np.empty((2 * prob.elem.shape[0], 3), dtype=np.int64); tris[0::2] = prob.elem[:, [0, 1, 2]]; tris[1::2] = prob.elem[:, [0, 2, 3]]
    fig, ax = plt.subplots(); ax.triplot(prob.node[:, 0], prob.node[:, 1], tris, color="0.35", linewidth=0.25); th=np.linspace(0, math.pi/2, 300); ax.plot(prob.a*np.cos(th), prob.a*np.sin(th), "r-"); ax.plot(prob.b*np.cos(th), prob.b*np.sin(th), "k-"); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(output_dir / f"mesh_{tag}.pdf"); plt.close(fig)
    fig, ax = plt.subplots(); c=ax.tripcolor(prob.node[:,0], prob.node[:,1], tris, Dnode, shading="gouraud"); fig.colorbar(c, ax=ax); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(output_dir / f"thick_diss_{tag}.pdf"); plt.close(fig)
    X = prob.node
    fig, ax = plt.subplots(); ax.scatter(X[:,0], X[:,1], s=4, c=Dnode, cmap="viridis"); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(output_dir / f"thick_nodes_{tag}.pdf"); plt.close(fig)
    wraw = thick_external(net, prob)
    alpha = torch.where(wraw < 0.0, -1.0, 1.0) / (torch.abs(wraw) + 1.0e-12)
    Xtensor = torch.as_tensor(prob.node, dtype=DTYPE, device=DEVICE)
    fields = (alpha * plate.hard_bc(Xtensor, net(Xtensor))).detach().cpu().numpy()
    fig, ax = plt.subplots(); ax.tripcolor(X[:,0], X[:,1], tris, Dnode, shading="gouraud", alpha=0.25); ax.quiver(X[:,0], X[:,1], fields[:,0], fields[:,1], color="k", width=0.0015); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(output_dir / f"velocity_{tag}.pdf"); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(it, val); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\lambda^+$"); ax.grid(True); fig.tight_layout(); fig.savefig(output_dir / f"thick_history_{tag}.pdf"); plt.close(fig)


def parse_common() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("matlab_kan_results"))
    parser.add_argument("--adam", type=int, default=None)
    parser.add_argument("--lbfgs", type=int, default=None)
    return parser.parse_args()
