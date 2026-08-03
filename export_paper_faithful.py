"""Recreate the original MATLAB-style figures from completed checkpoints.

This module is postprocessing only.  It uses the same meshes, boundary
conditions, KAN architectures, and final checkpoints as the completed runs,
but follows the plotting conventions in the MATLAB files: trisurf-like
interpolation, jet/GYR colormaps, percentile clipping, axis visibility,
boundary overlays, and phase-colored histories.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import torch

import matlab_kan_cases as cases
from paper_style import configure_paper_style


configure_paper_style()
DTYPE = cases.DTYPE
DEVICE = cases.DEVICE


def load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_local_model(
    path: Path,
    kind: str,
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    scale: float,
    grid_size: int = 5,
):
    payload = load_payload(path)
    net = cases.build_model(kind, in_dim, out_dim, width, depth, scale, grid_size)
    net.load_state_dict(payload["model"])
    net.eval()
    return net, payload


def triangles(elem: np.ndarray) -> np.ndarray:
    out = np.empty((2 * elem.shape[0], 3), dtype=np.int64)
    out[0::2] = elem[:, [0, 1, 2]]
    out[1::2] = elem[:, [0, 2, 3]]
    return out


def local_gyr():
    n = 128
    green = np.array([0.0, 0.75, 0.0])
    yellow = np.array([1.0, 1.0, 0.0])
    red = np.array([1.0, 0.0, 0.0])
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "localGYR", np.vstack((np.linspace(green, yellow, n), np.linspace(yellow, red, n)))
    )


def save_plate_dissipation(prob, dnode: np.ndarray, path: Path, mode: str = "main", title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tris = triangles(prob.elem)
    values = np.nan_to_num(np.asarray(dnode, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    if mode == "activation":
        vmin, vmax = np.percentile(values, [5, 97])
        values = np.clip(values, vmin, vmax)
        cmap = local_gyr()
        axis_off = True
    else:
        vmin = vmax = None
        cmap = plt.get_cmap("jet", 256)
        axis_off = mode == "adaptive" or mode == "thick"
    artist = ax.tripcolor(
        prob.coords[:, 0], prob.coords[:, 1], tris, values,
        shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale(enable=True, axis="both", tight=True)
    if axis_off:
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if mode == "adaptive":
        theta = np.linspace(0.0, math.pi / 2.0, 300)
        ax.fill(prob.R * np.cos(theta), prob.R * np.sin(theta), "white", edgecolor="black", linewidth=1.2)
    elif mode == "main":
        theta = np.linspace(0.0, math.pi / 2.0, 200)
        ax.plot(prob.R * np.cos(theta), prob.R * np.sin(theta), "k--", linewidth=1.2)
    if mode != "activation":
        fig.colorbar(artist, ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_history(iterations: Iterable[float], values: Iterable[float], path: Path, n_adam: int, title: str | None = None, phase_breaks: tuple[int, ...] = ()) -> None:
    iterations = np.asarray(list(iterations), dtype=float)
    values = np.asarray(list(values), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor="white")
    breaks = (n_adam,) if not phase_breaks else phase_breaks
    if len(breaks) == 1:
        adam = iterations <= breaks[0]
        other = iterations > breaks[0]
        ax.plot(iterations[adam], values[adam], "b-", linewidth=2)
        if np.any(other):
            ax.plot(iterations[other], values[other], "r-", linewidth=2)
            ax.axvline(breaks[0], color="k", linestyle="--", linewidth=1.2)
            ax.legend(["Adam", "L-BFGS"], loc="best")
        else:
            ax.legend(["Adam"], loc="best")
    else:
        colors = ("b-", "m-", "r-")
        labels = ("Adam before adaptive", "Adam after adaptive", "L-BFGS")
        previous = -np.inf
        handles = []
        for index, end in enumerate((*breaks, np.inf)):
            mask = (iterations > previous) & (iterations <= end)
            if np.any(mask):
                handle, = ax.plot(iterations[mask], values[mask], colors[index], linewidth=2)
                handles.append(handle)
            previous = end
        for boundary in breaks:
            ax.axvline(boundary, color="k", linestyle="--", linewidth=1.2)
        if handles:
            ax.legend(handles, labels[:len(handles)], loc="best")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^+$")
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plate_prob(nx: int, gauss: int, p2: float = 0.0):
    return cases.plate.build_problem(cases.plate.Problem(nx=nx, ny=nx, R=0.2, a=1.0, p1=1.0, p2=p2, numGauss=gauss))


def export_plate_cases(input_dir: Path, output_dir: Path) -> None:
    definitions = (
        ("fast_plate", "fast_KAN.pt", "fast_KAN", 40, 2, 32, 3, 0.0, 1.0, 3000),
        ("standard_plate", "plate_hole_KAN.pt", "plate_hole_KAN", 20, 2, 64, 4, 0.0, 1.0 / 3.0, 4000),
        ("high_gauss_plate", "UB_dissipation_high_order_gauss_enhanced.pt", "high_gauss_KAN", 20, 3, 64, 4, 0.0, 1.0 / 3.0, 4000),
    )
    for folder, filename, prefix, nx, gauss, width, depth, p2, shear, n_adam in definitions:
        path = input_dir / folder / filename
        if not path.exists():
            print(f"Missing {path}; skipping")
            continue
        prob = plate_prob(nx, gauss, p2)
        net, payload = load_local_model(path, "kan", 2, 2, width, depth, prob.a, 5)
        dnode = cases.plate_nodes_dissipation(net, prob, shear)
        save_plate_dissipation(prob, dnode, output_dir / f"{prefix}_dissipation.pdf", "main", "Normalized upper-bound dissipation")
        if "iter" in payload and "loss" in payload:
            save_history(payload["iter"], payload["loss"], output_dir / f"{prefix}_history.pdf", n_adam)
        del net

    history_dir = input_dir / "history_gauss_plate"
    hist = []
    for gauss in (2, 3, 5):
        path = history_dir / f"plate_g{gauss}_KAN.pt"
        if not path.exists():
            continue
        prob = plate_prob(80, gauss, 1.0)
        net, payload = load_local_model(path, "kan", 2, 2, 64, 4, prob.a, 8)
        dnode = cases.plate_nodes_dissipation(net, prob, 1.0 / 3.0)
        save_plate_dissipation(prob, dnode, output_dir / f"hist_gauss_{gauss}_dissipation.pdf", "main", f"Normalized upper-bound dissipation, {gauss} x {gauss} Gauss")
        if "iter" in payload and "loss" in payload:
            hist.append((payload["iter"], payload["loss"], gauss))
        del net
    if hist:
        fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="white")
        colors = plt.get_cmap("tab10")
        for i, (iterations, values, gauss) in enumerate(hist):
            iterations = np.asarray(iterations); values = np.asarray(values)
            adam = iterations <= 4000
            ax.plot(iterations[adam], values[adam], "-", color=colors(i), linewidth=2, label=f"{gauss} x {gauss} Gauss - Adam")
            if np.any(~adam):
                ax.plot(iterations[~adam], values[~adam], "--", color=colors(i), linewidth=2, label=f"{gauss} x {gauss} Gauss - L-BFGS")
        ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\lambda^+$"); ax.grid(True); ax.legend(loc="best", fontsize=9); fig.tight_layout(); fig.savefig(output_dir / "Hist_gaussp2_80.pdf", bbox_inches="tight"); plt.close(fig)


def export_adaptive(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "adaptive_plate" / "adaptive_Gauss_KAN.pt"
    if not path.exists():
        return
    prob = plate_prob(40, 2, 0.0)
    net, payload = load_local_model(path, "kan", 2, 2, 64, 4, prob.a, 5)
    dnode = cases.plate_nodes_dissipation(net, prob, 1.0 / 3.0)
    save_plate_dissipation(prob, dnode, output_dir / "adaptive_Gauss_dissipation.pdf", "adaptive", "Plastic dissipation density")
    hot = np.asarray(payload.get("hot", []), dtype=bool)
    if hot.size == prob.elem.shape[0]:
        fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
        polygons = []
        colors = []
        for eid, elem in enumerate(prob.elem):
            polygons.append(prob.coords[elem])
            colors.append([1.0, 0.35, 0.25] if hot[eid] else [0.85, 0.85, 0.85])
        ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors=[0.4, 0.4, 0.4], linewidths=0.2))
        ax.autoscale(); ax.set_aspect("equal"); ax.set_title("Adaptive hot elements based on cell-averaged dissipation"); ax.set_box_aspect(1); fig.tight_layout(); fig.savefig(output_dir / "adaptive_hot_elements.pdf", bbox_inches="tight"); plt.close(fig)
    if "iter" in payload and "loss" in payload:
        save_history(payload["iter"], payload["loss"], output_dir / "adaptive_Gauss_history.pdf", 2000, phase_breaks=(2000, 4000))
    del net


def export_activation(input_dir: Path, output_dir: Path) -> None:
    folder = input_dir / "activation_plate"
    mlp_path = folder / "activation_mlp.pt"; kan_path = folder / "activation_kan.pt"
    if not mlp_path.exists() or not kan_path.exists():
        return
    prob = plate_prob(20, 3, 0.0)
    mlp, mlp_payload = load_local_model(mlp_path, "mlp", 2, 2, 64, 4, prob.a)
    kan, kan_payload = load_local_model(kan_path, "kan", 2, 2, 64, 4, prob.a, 5)
    records = (("MLP-tanh", mlp, mlp_payload), ("cubic B-spline KAN", kan, kan_payload))
    tris = triangles(prob.elem)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), facecolor="white")
    cmap = local_gyr()
    for index, (label, net, _) in enumerate(records):
        dnode = cases.plate_nodes_dissipation(net, prob, 1.0 / 3.0)
        values = np.clip(np.nan_to_num(dnode), *np.percentile(np.nan_to_num(dnode), [5, 97]))
        ax = axes[0, index]
        ax.tripcolor(prob.coords[:, 0], prob.coords[:, 1], tris, values, shading="gouraud", cmap=cmap)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(label)
    for ax in axes.flat:
        if ax not in axes[0, :2]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(output_dir / "activation_comparison_dissipation.pdf", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor="white")
    for label, _, payload in records:
        if "iter" not in payload: continue
        it = np.asarray(payload["iter"]); val = np.asarray(payload["loss"]); adam = it <= 4000
        line, = ax.plot(it[adam], val[adam], linewidth=1.8, label=label)
        if np.any(~adam): ax.plot(it[~adam], val[~adam], "--", color=line.get_color(), linewidth=1.8)
    ax.axvline(4000, color="k", linestyle="--", linewidth=1.2); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\lambda^+$"); ax.grid(True); ax.legend(loc="best"); fig.tight_layout(); fig.savefig(output_dir / "activation_comparison_history.pdf", bbox_inches="tight"); plt.close(fig)
    del mlp, kan


def thin_fields(net):
    theta = torch.linspace(0.0, math.pi / 2.0, 120, dtype=DTYPE, device=DEVICE).reshape(-1, 1).requires_grad_(True)
    raw = net(theta); u = torch.cos(theta) * raw[:, 0:1]; v = torch.sin(theta) * raw[:, 1:2]
    wraw = torch.trapezoid((u * torch.cos(theta) + v * torch.sin(theta))[:, 0], theta[:, 0])
    alpha = torch.where(wraw < 0.0, -1.0, 1.0) / (torch.abs(wraw) + 1.0e-12)
    u = alpha * u; v = alpha * v; dv = torch.autograd.grad(v.sum(), theta, create_graph=True)[0]
    D = torch.sqrt(((dv + u)).square() + 1.0e-18)
    return theta[:, 0].detach().cpu().numpy(), u.detach().cpu().numpy()[:, 0], v.detach().cpu().numpy()[:, 0], D.detach().cpu().numpy()[:, 0]


def export_thin(input_dir: Path, output_dir: Path) -> None:
    path = input_dir / "thin_cylinder" / "thinwall_KAN.pt"
    if not path.exists(): return
    net, payload = load_local_model(path, "kan", 1, 2, 64, 4, math.pi / 2.0, 5)
    theta, u, v, D = thin_fields(net); a = 1.0
    fig, ax = plt.subplots(facecolor="white"); ax.plot(theta, D, linewidth=2); ax.set_xlabel(r"$\theta$"); ax.set_ylabel("Plastic dissipation"); ax.grid(True); ax.set_box_aspect(1); fig.tight_layout(); fig.savefig(output_dir / "thinwall_dissipation.pdf", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(facecolor="white"); t = np.linspace(0, math.pi/2, 400); ax.plot(a*np.cos(t), a*np.sin(t), "k-", linewidth=2); ax.quiver(a*np.cos(theta), a*np.sin(theta), u, v, color="r", width=0.003); ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True); ax.set_title("Thin-wall collapse mechanism"); fig.tight_layout(); fig.savefig(output_dir / "thinwall_velocity.pdf", bbox_inches="tight"); plt.close(fig)
    if "iter" in payload: save_history(payload["iter"], payload["loss"], output_dir / "thinwall_history.pdf", 4000)
    del net


def make_thick_problem(ratio: float):
    b = 2.0; a = ratio * b; nodes, elem = cases.annulus_mesh(a, b, 24, 48); Xg, Wg = cases.annulus_domain_quad(nodes, elem, 5); Xi, Wi, Ni = cases.inner_pressure_quad(a, 48, 5)
    return SimpleNamespace(node=nodes, elem=elem, a=a, b=b, p=1.0, betaInc=1.0, Xg=torch.as_tensor(Xg, dtype=DTYPE, device=DEVICE), Wg=torch.as_tensor(Wg, dtype=DTYPE, device=DEVICE).reshape(-1,1), Xi=torch.as_tensor(Xi, dtype=DTYPE, device=DEVICE), Wi=torch.as_tensor(Wi, dtype=DTYPE, device=DEVICE).reshape(-1,1), Ni=torch.as_tensor(Ni, dtype=DTYPE, device=DEVICE))


def export_thick(input_dir: Path, output_dir: Path) -> None:
    folder = input_dir / "thick_cylinder"
    for ratio in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        tag = f"ab_{round(100*ratio):03d}"; path = folder / f"{tag}_KAN.pt"
        if not path.exists(): continue
        prob = make_thick_problem(ratio); net, payload = load_local_model(path, "kan", 2, 2, 64, 4, prob.b, 5); dnode = cases.thick_node_dissipation(net, prob)
        tris = triangles(prob.elem); vmin, vmax = np.percentile(np.nan_to_num(dnode), [5,97]); values=np.clip(np.nan_to_num(dnode),vmin,vmax)
        fig, ax = plt.subplots(facecolor="white"); ax.tripcolor(prob.node[:,0], prob.node[:,1], tris, values, shading="gouraud", cmap=plt.get_cmap("jet",256), vmin=vmin, vmax=vmax); ax.set_aspect("equal"); ax.axis("off"); fig.colorbar(ax.collections[0], ax=ax); fig.tight_layout(); fig.savefig(output_dir / f"thick_diss_{tag}.pdf", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(facecolor="white"); ax.add_collection(PolyCollection([prob.node[e] for e in prob.elem], facecolor=[0.85,0.85,0.85], edgecolor="none", alpha=0.7)); wraw=cases.thick_external(net,prob); alpha=torch.where(wraw<0,-1.,1.)/(torch.abs(wraw)+1e-12); X=torch.as_tensor(prob.node,dtype=DTYPE,device=DEVICE); fields=(alpha*cases.plate.hard_bc(X,net(X))).detach().cpu().numpy(); node_def=prob.node+0.2*fields; ax.add_collection(PolyCollection([node_def[e] for e in prob.elem], facecolor="none", edgecolor="r", linewidth=0.4)); ax.quiver(prob.node[:,0],prob.node[:,1],fields[:,0],fields[:,1],color="k",width=0.0015); ax.autoscale(); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"Velocity mechanism, a/b = {ratio:.1f}"); fig.tight_layout(); fig.savefig(output_dir / f"velocity_{tag}.pdf", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(facecolor="white"); r=np.sqrt(prob.node[:,0]**2+prob.node[:,1]**2); traction=np.abs(r-prob.a)<1e-6; boundary=traction|(np.abs(prob.node[:,0])<1e-6)|(np.abs(prob.node[:,1])<1e-6)|(np.abs(r-prob.b)<1e-6); ax.scatter(prob.node[~boundary,0],prob.node[~boundary,1],s=18,c=[[0,0.447,0.741]]); ax.scatter(prob.node[boundary & ~traction,0],prob.node[boundary & ~traction,1],s=28,c="k"); ax.scatter(prob.node[traction,0],prob.node[traction,1],s=36,c="r"); th=np.linspace(0,math.pi/2,400); ax.plot(prob.a*np.cos(th),prob.a*np.sin(th),"r-",linewidth=1.5); ax.plot(prob.b*np.cos(th),prob.b*np.sin(th),"k-",linewidth=1.5); ax.set_aspect("equal"); ax.set_box_aspect(1); fig.tight_layout(); fig.savefig(output_dir / f"thick_nodes_{tag}.pdf", bbox_inches="tight"); plt.close(fig)
        if "iter" in payload: save_history(payload["iter"], payload["loss"], output_dir / f"thick_history_{tag}.pdf", 20000)
        del net


def export_standalone(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_plate_cases(input_dir, output_dir)
    export_adaptive(input_dir, output_dir)
    export_activation(input_dir, output_dir)
    export_thin(input_dir, output_dir)
    export_thick(input_dir, output_dir)
    print(f"Paper-faithful standalone figures saved under {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("all_kan_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_faithful_figures"))
    args = parser.parse_args()
    export_standalone(args.input_dir.resolve(), args.output_dir.resolve())
