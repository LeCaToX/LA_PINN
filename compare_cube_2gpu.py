"""Two-GPU sharded comparison for the spherical-hole cube problem.

Launch with ``torchrun --nproc_per_node=2``. Each rank owns a shard of the
Gauss and boundary points, so the large autograd graph is split between the
two GPUs. Gradients are synchronized by DistributedDataParallel.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from kan_layers import KAN, build_kan


DTYPE = torch.float32
DEFAULT_SEED = 1234
DEFAULT_WIDTH = 64
DEFAULT_DEPTH = 4
DEFAULT_LR = 1.0e-3
DEFAULT_CHECKPOINT_INTERVAL = 100


def is_rank_zero(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(cpu_copy(payload), temporary_path)
    temporary_path.replace(path)


def atomic_json_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    parameter_dtype = None
    if optimizer.param_groups and optimizer.param_groups[0]["params"]:
        parameter_dtype = optimizer.param_groups[0]["params"][0].dtype
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if parameter_dtype is not None and value.is_floating_point():
                    state[key] = value.to(device=device, dtype=parameter_dtype)
                else:
                    state[key] = value.to(device=device)


def build_mlp(in_dim: int, out_dim: int, width: int, depth: int, device: torch.device) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(1, depth):
        layers.extend((nn.Linear(width, width), nn.Tanh()))
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers).to(device=device, dtype=DTYPE)


def restore_model_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    if isinstance(model, KAN):
        spline_weights = [
            value for key, value in state_dict.items() if key.endswith("spline_weight")
        ]
        if spline_weights:
            saved_grid = int(spline_weights[0].shape[-1]) - 3
            current_grid = model.layers[0].grid_size
            if saved_grid > current_grid:
                model.refine_grid(saved_grid)
    model.load_state_dict(state_dict)


def make_full_problem(args: argparse.Namespace, cube: Any) -> Any:
    """Build geometry and quadrature on CPU before sharding to the GPUs."""
    cube.device = torch.device("cpu")
    problem = cube.Problem()
    problem.a = 1.0
    problem.R = 0.2
    problem.px = 1.0
    problem.py = 0.0
    problem.pz = 0.0
    problem.nx = args.cube_n
    problem.ny = args.cube_n
    problem.nz = args.cube_n
    problem.numGauss = args.cube_gauss
    return cube.buildProblem(problem)


def make_local_problem(
    full_problem: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    cube: Any,
) -> Any:
    local = cube.Problem()
    for name in ("a", "R", "px", "py", "pz", "nx", "ny", "nz", "numGauss"):
        setattr(local, name, getattr(full_problem, name))
    local.coords = full_problem.coords
    local.elem = full_problem.elem

    for name in ("XgDL", "WgDL", "XxDL", "WxDL", "XyDL", "WyDL", "XzDL", "WzDL"):
        full_tensor = getattr(full_problem, name)
        shard = full_tensor[rank::world_size].contiguous()
        setattr(local, name, shard.to(device=device, dtype=DTYPE))
    return local


def sharded_loss(
    net: nn.Module,
    problem: Any,
    cube: Any,
    world_size: int,
    beta_inc: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the global cube loss while retaining only a local graph.

    The detached all-reduces make the scalar loss identical on every rank,
    while the local terms retain their parameter gradients. DDP then averages
    the scaled local gradients into the gradient of the global integral.
    """
    wraw_local = cube.externalWork(net, problem)
    wraw_global_detached = wraw_local.detach().clone()
    dist.all_reduce(wraw_global_detached, op=dist.ReduceOp.SUM)

    # Keep the derivative of alpha with respect to this rank's local external
    # work, while treating the other rank's contribution as a constant.
    wraw_global_local_graph = (
        wraw_global_detached - wraw_local.detach() + wraw_local
    )
    alpha = 1.0 / (torch.abs(wraw_global_local_graph) + 1.0e-12)

    X = problem.XgDL.detach().clone().requires_grad_(True)
    W = problem.WgDL
    uvw_raw = net(X)
    d = alpha * cube.hardBC(X, uvw_raw)
    eps = cube.strainRate3D(X, d)
    dissipation = cube.dissipationDensity3D(eps, sigma0=1.0)
    wint_local = torch.sum(dissipation * W)
    divp = eps[:, 0:1] + eps[:, 1:2] + eps[:, 2:3]
    linc_local = torch.sum(divp.square() * W)
    local_loss = wint_local + beta_inc * linc_local

    # Value is the global sum; derivative is world_size times the local
    # derivative, which DDP averages across ranks.
    global_loss_detached = local_loss.detach().clone()
    dist.all_reduce(global_loss_detached, op=dist.ReduceOp.SUM)
    loss = world_size * local_loss + (
        global_loss_detached - world_size * local_loss.detach()
    )
    info = {
        "Wint": wint_local.detach(),
        "Linc": linc_local.detach(),
        "Wext": wraw_global_detached.detach(),
        "alpha": (1.0 / (torch.abs(wraw_global_detached) + 1.0e-12)).detach(),
    }
    return loss, info


def save_progress(
    path: Path,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    phase: str,
    adam_epoch: int,
    lbfgs_iteration: int,
    histories: Dict[str, List[float]],
    best_loss: float,
    best_step: int,
    elapsed_seconds: float,
    complete: bool = False,
    result: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpoint_version": 1,
        "complete": complete,
        "config": config,
        "phase": phase,
        "adam_epoch": adam_epoch,
        "lbfgs_iteration": lbfgs_iteration,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "histories": histories,
        "best_loss": best_loss,
        "best_step": best_step,
        "elapsed_seconds": elapsed_seconds,
    }
    if result is not None:
        payload["result"] = result
    atomic_torch_save(payload, path)


def train_model(
    kind: str,
    args: argparse.Namespace,
    local_problem: Any,
    cube: Any,
    device: torch.device,
    local_rank: int,
    rank: int,
    world_size: int,
    output_dir: Path,
) -> Dict[str, Any]:
    label = f"cube/{kind}"
    final_path = output_dir / f"cube_{kind}.pt"
    progress_path = output_dir / f"cube_{kind}.progress.pt"
    config = {
        "label": label,
        "kind": kind,
        "cube_n": args.cube_n,
        "cube_gauss": args.cube_gauss,
        "width": args.width,
        "depth": args.depth,
        "n_adam": args.adam,
        "n_lbfgs": args.lbfgs,
        "lr": args.lr,
        "grid_schedule": [[1500, 9], [3500, 13]],
        "dtype": str(DTYPE),
        "world_size": world_size,
    }

    progress: Dict[str, Any] | None = None
    progress_source: Path | None = None
    if args.resume:
        checkpoint_candidates = [progress_path]
        if args.legacy_dir is not None:
            checkpoint_candidates.append(args.legacy_dir / f"cube_{kind}.progress.pt")
        for candidate_path in checkpoint_candidates:
            if not candidate_path.exists():
                continue
            candidate = load_checkpoint(candidate_path)
            is_legacy = candidate_path != progress_path
            if candidate.get("config") == config or is_legacy:
                progress = candidate
                progress_source = candidate_path
                if candidate.get("complete"):
                    if is_rank_zero(rank):
                        print(f"{label}: completed checkpoint found; skipping.")
                    return candidate["result"]
                if is_rank_zero(rank):
                    source_note = " (legacy checkpoint)" if is_legacy else ""
                    print(
                        f"{label}: restoring{source_note} "
                        f"phase={candidate.get('phase')} "
                        f"Adam={candidate.get('adam_epoch', 0)} "
                        f"L-BFGS={candidate.get('lbfgs_iteration', 0)}"
                    )
                break
            if is_rank_zero(rank):
                print(f"{label}: checkpoint settings changed; starting fresh.")

    seed_everything(args.seed)
    if kind == "MLP":
        raw_model = build_mlp(3, 3, args.width, args.depth, device)
    else:
        raw_model = build_kan(
            3,
            3,
            args.width,
            args.depth,
            grid_size=5,
            input_scale=local_problem.a,
            device=device,
            dtype=DTYPE,
        )

    histories: Dict[str, List[float]] = {"adam": [], "lbfgs": []}
    best_loss = float("inf")
    best_step = 0
    elapsed_before_restart = 0.0
    phase = "adam"
    adam_epoch = 0
    lbfgs_iteration = 0

    if progress is not None:
        restore_model_state(raw_model, progress["model_state_dict"])
        histories = {
            key: list(value) for key, value in progress.get("histories", histories).items()
        }
        best_loss = float(progress.get("best_loss", best_loss))
        best_step = int(progress.get("best_step", best_step))
        elapsed_before_restart = float(progress.get("elapsed_seconds", 0.0))
        phase = str(progress.get("phase", "adam"))
        adam_epoch = int(progress.get("adam_epoch", 0))
        lbfgs_iteration = int(progress.get("lbfgs_iteration", 0))

    ddp_model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
    start_time = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before_restart + time.perf_counter() - start_time

    def loss_fn() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return sharded_loss(ddp_model, local_problem, cube, world_size, args.beta_inc)

    optimizer: torch.optim.Optimizer | None = None
    if phase == "adam":
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
        if progress is not None and progress.get("optimizer_state_dict"):
            optimizer.load_state_dict(progress["optimizer_state_dict"])
            optimizer_to_device(optimizer, device)

        if is_rank_zero(rank):
            print(f"\n--- {label}: Adam ({args.adam} steps, starting at {adam_epoch}) ---")
        grid_schedule = {1500: 9, 3500: 13} if kind == "KAN" else {}
        for epoch in range(adam_epoch + 1, args.adam + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, info = loss_fn()
            loss.backward()
            optimizer.step()

            if epoch in grid_schedule:
                dist.barrier()
                raw_model = ddp_model.module
                del ddp_model
                raw_model.refine_grid(grid_schedule[epoch])
                dist.barrier()
                ddp_model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
                optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
                if is_rank_zero(rank):
                    print(f"{label}: refined cubic B-spline grid to {grid_schedule[epoch]}")

            should_save = epoch % args.checkpoint_interval == 0 or epoch == args.adam
            should_print = epoch % 100 == 0 or epoch == 1
            if should_save or should_print:
                loss_value = float(loss.detach().cpu())
                histories["adam"].append(loss_value)
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_step = epoch
                if is_rank_zero(rank) and should_print:
                    print(
                        f"{label} | Adam {epoch:5d} | lambda = {loss_value:.8f} | "
                        f"elapsed = {elapsed():.1f}s"
                    )
                if should_save:
                    dist.barrier()
                    if is_rank_zero(rank):
                        save_progress(
                            progress_path,
                            ddp_model.module,
                            optimizer,
                            config,
                            "adam",
                            epoch,
                            0,
                            histories,
                            best_loss,
                            best_step,
                            elapsed(),
                        )
                    dist.barrier()
            del loss, info

        phase = "lbfgs"
        adam_epoch = args.adam
        lbfgs_iteration = 0
    else:
        optimizer = None

    if is_rank_zero(rank):
        print(f"--- {label}: L-BFGS ({args.lbfgs} steps) ---")
    lbfgs = torch.optim.LBFGS(
        ddp_model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=2,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    if progress is not None and progress.get("phase") == "lbfgs":
        lbfgs.load_state_dict(progress["optimizer_state_dict"])
        optimizer_to_device(lbfgs, device)
    else:
        dist.barrier()
        if is_rank_zero(rank):
            save_progress(
                progress_path,
                ddp_model.module,
                lbfgs,
                config,
                "lbfgs",
                adam_epoch,
                0,
                histories,
                best_loss,
                best_step,
                elapsed(),
            )
        dist.barrier()

    lbfgs_checkpoint_interval = max(1, min(25, args.checkpoint_interval))
    for iteration in range(lbfgs_iteration + 1, args.lbfgs + 1):
        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = loss_fn()
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)
        should_save = iteration % lbfgs_checkpoint_interval == 0 or iteration == args.lbfgs
        should_print = iteration % 25 == 0 or iteration == args.lbfgs
        if should_save or should_print:
            loss, _ = loss_fn()
            loss_value = float(loss.detach().cpu())
            histories["lbfgs"].append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = args.adam + iteration
            if is_rank_zero(rank) and should_print:
                print(
                    f"{label} | L-BFGS {iteration:5d} | lambda = {loss_value:.8f} | "
                    f"elapsed = {elapsed():.1f}s"
                )
            if should_save:
                dist.barrier()
                if is_rank_zero(rank):
                    save_progress(
                        progress_path,
                        ddp_model.module,
                        lbfgs,
                        config,
                        "lbfgs",
                        args.adam,
                        iteration,
                        histories,
                        best_loss,
                        best_step,
                        elapsed(),
                    )
                dist.barrier()
            del loss

    final_loss, final_info = loss_fn()
    final_loss_value = float(final_loss.detach().cpu())
    if final_loss_value < best_loss:
        best_loss = final_loss_value
        best_step = args.adam + args.lbfgs
    elapsed_seconds = elapsed()
    result = {
        "label": label,
        "parameters": sum(parameter.numel() for parameter in ddp_model.module.parameters()),
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": final_loss_value,
        "seconds": elapsed_seconds,
        "final_info": {key: float(value.detach().cpu()) for key, value in final_info.items()},
        "history": histories,
        "checkpoint": str(final_path),
        "progress_checkpoint": str(progress_path),
        "world_size": world_size,
    }
    dist.barrier()
    if is_rank_zero(rank):
        save_progress(
            final_path,
            ddp_model.module,
            lbfgs,
            config,
            "complete",
            args.adam,
            args.lbfgs,
            histories,
            best_loss,
            best_step,
            elapsed_seconds,
            complete=True,
            result=result,
        )
        save_progress(
            progress_path,
            ddp_model.module,
            lbfgs,
            config,
            "complete",
            args.adam,
            args.lbfgs,
            histories,
            best_loss,
            best_step,
            elapsed_seconds,
            complete=True,
            result=result,
        )
    dist.barrier()
    del final_loss, final_info, ddp_model
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-n", type=int, default=20)
    parser.add_argument("--cube-gauss", type=int, default=2)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--adam", type=int, default=5000)
    parser.add_argument("--lbfgs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--beta-inc", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_cube_2gpu"))
    parser.add_argument("--legacy-dir", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.resume = not args.fresh
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch this script with torchrun on two GPUs.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --nproc_per_node=2 compare_cube_2gpu.py")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"Expected exactly 2 processes, got {world_size}.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # The original cube module sets float64 at import time. Override it here
    # so the sharded runner uses half the per-element memory by default.
    import Main_UB_PINN_cube_spherical_hole_KAN_torch as cube

    torch.set_default_dtype(DTYPE)
    full_problem = make_full_problem(args, cube)
    local_problem = make_local_problem(full_problem, rank, world_size, device, cube)
    del full_problem
    dist.barrier()

    if is_rank_zero(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    settings = {
        "problem": "cube",
        "cube_n": args.cube_n,
        "cube_gauss": args.cube_gauss,
        "width": args.width,
        "depth": args.depth,
        "adam": args.adam,
        "lbfgs": args.lbfgs,
        "lr": args.lr,
        "beta_inc": args.beta_inc,
        "seed": args.seed,
        "checkpoint_interval": args.checkpoint_interval,
        "dtype": str(DTYPE),
        "world_size": world_size,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    report_path = args.output_dir / "comparison_report.json"
    report: Dict[str, Any] = {"settings": settings, "results": {}}
    if is_rank_zero(rank) and args.resume and not args.fresh and report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("settings") == settings:
                report = previous
                print(f"Restoring report state from {report_path}")
        except (OSError, json.JSONDecodeError):
            pass
    if is_rank_zero(rank):
        atomic_json_save(report, report_path)
    dist.barrier()

    try:
        mlp_result = train_model(
            "MLP", args, local_problem, cube, device, local_rank, rank, world_size, args.output_dir
        )
        if is_rank_zero(rank):
            report["results"]["MLP"] = mlp_result
            atomic_json_save(report, report_path)
        dist.barrier()

        kan_result = train_model(
            "KAN", args, local_problem, cube, device, local_rank, rank, world_size, args.output_dir
        )
        if is_rank_zero(rank):
            report["results"]["KAN"] = kan_result
            report["comparison"] = {
                "final_loss_change_percent": 100.0
                * (kan_result["final_loss"] - mlp_result["final_loss"])
                / abs(mlp_result["final_loss"]),
                "best_loss_change_percent": 100.0
                * (kan_result["best_loss"] - mlp_result["best_loss"])
                / abs(mlp_result["best_loss"]),
                "runtime_ratio_KAN_over_MLP": kan_result["seconds"] / mlp_result["seconds"],
            }
            atomic_json_save(report, report_path)
            print(f"Saved two-GPU comparison report to {report_path.resolve()}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
