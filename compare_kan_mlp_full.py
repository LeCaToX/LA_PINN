"""Full GPU comparison of the original MLP and cubic-B-spline KAN.

Examples
--------
Plate-with-hole comparison at all production Gauss orders::

    python compare_kan_mlp_full.py --problem plate

Plate plus the one-octant spherical-hole problem::

    python compare_kan_mlp_full.py --problem both

The runner intentionally requires CUDA.  It uses the production problem
settings, Adam followed by L-BFGS, and saves JSON histories plus model
checkpoints in ``comparison_full_results``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

import LA_PINN_plate_hole_run_histGauss_KAN_torch as plate
import Main_UB_PINN_cube_spherical_hole_KAN_torch as cube
from kan_layers import KAN, build_kan


DEFAULT_SEED = 1234
DEFAULT_WIDTH = 64
DEFAULT_DEPTH = 4
DEFAULT_LR = 1.0e-3
DEFAULT_CHECKPOINT_INTERVAL = 100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(1, depth):
        layers.extend((nn.Linear(width, width), nn.Tanh()))
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers).to(device=device, dtype=dtype)


def scalar(value: Tensor) -> float:
    return float(value.detach().cpu())


def cpu_copy(value: Any) -> Any:
    """Copy checkpoint data to CPU so saving does not require extra GPU memory."""
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically, preserving the previous one on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(cpu_copy(payload), temporary_path)
    temporary_path.replace(path)


def atomic_json_save(payload: Dict[str, Any], path: Path) -> None:
    """Write JSON atomically so a server crash cannot leave a partial report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensors back to the active GPU after loading a checkpoint."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device=device)


def restore_model_state(model: nn.Module, state_dict: Dict[str, Tensor]) -> None:
    """Restore a model, including a KAN that was saved after grid refinement."""
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


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
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
        "model_state_dict": model.state_dict(),
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
    model: nn.Module,
    loss_fn: Callable[[], Tuple[Tensor, Dict[str, Tensor]]],
    label: str,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    grid_schedule: Sequence[Tuple[int, int]],
    checkpoint_path: Path,
    progress_path: Path,
    resume: bool,
    checkpoint_interval: int,
) -> Dict[str, Any]:
    """Train one model with crash-safe Adam/L-BFGS checkpoints."""
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    histories: Dict[str, List[float]] = {"adam": [], "lbfgs": []}
    best_loss = float("inf")
    best_step = 0
    elapsed_before_restart = 0.0
    phase = "adam"
    adam_epoch = 0
    lbfgs_iteration = 0
    progress: Dict[str, Any] | None = None
    config = {
        "label": label,
        "parameters": parameter_count,
        "n_adam": n_adam,
        "n_lbfgs": n_lbfgs,
        "lr": lr,
        "grid_schedule": list(grid_schedule),
    }

    if resume and progress_path.exists():
        candidate = load_checkpoint(progress_path)
        if candidate.get("config") == config:
            progress = candidate
            if candidate.get("complete"):
                print(f"{label}: completed checkpoint found; skipping.")
                return candidate["result"]
            print(
                f"{label}: restoring from {progress_path.name} "
                f"(phase={candidate.get('phase')}, "
                f"Adam={candidate.get('adam_epoch', 0)}, "
                f"L-BFGS={candidate.get('lbfgs_iteration', 0)})"
            )
        else:
            print(f"{label}: checkpoint settings changed; starting a new run.")

    if progress is not None:
        restore_model_state(model, progress["model_state_dict"])
        histories = {
            key: list(values) for key, values in progress.get("histories", histories).items()
        }
        best_loss = float(progress.get("best_loss", best_loss))
        best_step = int(progress.get("best_step", best_step))
        elapsed_before_restart = float(progress.get("elapsed_seconds", 0.0))
        phase = str(progress.get("phase", "adam"))
        adam_epoch = int(progress.get("adam_epoch", 0))
        lbfgs_iteration = int(progress.get("lbfgs_iteration", 0))

    start_time = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before_restart + time.perf_counter() - start_time

    adam: torch.optim.Optimizer | None = None
    grid_schedule_dict = dict(grid_schedule)

    if phase == "adam":
        adam = torch.optim.Adam(model.parameters(), lr=lr)
        if progress is not None and progress.get("optimizer_state_dict"):
            adam.load_state_dict(progress["optimizer_state_dict"])
            optimizer_to_device(adam, next(model.parameters()).device)

        print(
            f"\n--- {label}: Adam ({n_adam} steps, "
            f"starting at {adam_epoch}) ---"
        )
        for epoch in range(adam_epoch + 1, n_adam + 1):
            adam.zero_grad(set_to_none=True)
            loss, info = loss_fn()
            loss.backward()
            adam.step()

            loss_value = scalar(loss)
            histories["adam"].append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = epoch

            # Save after refinement. If a crash happens before this point,
            # the preceding checkpoint will replay this epoch and refine safely.
            if epoch in grid_schedule_dict and isinstance(model, KAN):
                new_grid = grid_schedule_dict[epoch]
                print(f"{label}: refining cubic B-spline grid to {new_grid} intervals")
                model.refine_grid(new_grid)
                # Refinement replaces spline tensors, so restart Adam's moments.
                adam = torch.optim.Adam(model.parameters(), lr=lr)

            if epoch % 100 == 0 or epoch == 1:
                print(
                    f"{label} | Adam {epoch:5d} | lambda = {loss_value:.8f} | "
                    f"elapsed = {elapsed():.1f}s"
                )

            if epoch % checkpoint_interval == 0 or epoch == n_adam:
                save_training_checkpoint(
                    progress_path,
                    model,
                    adam,
                    config,
                    "adam",
                    epoch,
                    0,
                    histories,
                    best_loss,
                    best_step,
                    elapsed(),
                )

            del loss, info

        phase = "lbfgs"
        adam_epoch = n_adam
        lbfgs_iteration = 0

    print(f"--- {label}: L-BFGS ({n_lbfgs} steps) ---")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=2,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    if progress is not None and phase == "lbfgs" and progress.get("optimizer_state_dict"):
        lbfgs.load_state_dict(progress["optimizer_state_dict"])
        optimizer_to_device(lbfgs, next(model.parameters()).device)

    # Checkpointing L-BFGS more often than Adam limits lost work without
    # writing a large optimizer history to disk on every iteration.
    lbfgs_checkpoint_interval = max(1, min(25, checkpoint_interval))
    if progress is None or progress.get("phase") != "lbfgs":
        save_training_checkpoint(
            progress_path,
            model,
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

    for iteration in range(lbfgs_iteration + 1, n_lbfgs + 1):
        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = loss_fn()
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)

        should_evaluate = (
            iteration % 25 == 0
            or iteration % lbfgs_checkpoint_interval == 0
            or iteration == n_lbfgs
        )
        if should_evaluate:
            loss, _ = loss_fn()
            loss_value = scalar(loss)
            histories["lbfgs"].append(loss_value)
            global_step = n_adam + iteration
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = global_step
            print(
                f"{label} | L-BFGS {iteration:5d} | lambda = {loss_value:.8f} | "
                f"elapsed = {elapsed():.1f}s"
            )

            if iteration % lbfgs_checkpoint_interval == 0 or iteration == n_lbfgs:
                save_training_checkpoint(
                    progress_path,
                    model,
                    lbfgs,
                    config,
                    "lbfgs",
                    adam_epoch,
                    iteration,
                    histories,
                    best_loss,
                    best_step,
                    elapsed(),
                )
            del loss

    final_loss, final_info = loss_fn()
    final_loss_value = scalar(final_loss)
    if final_loss_value < best_loss:
        best_loss = final_loss_value
        best_step = n_adam + n_lbfgs
    elapsed_seconds = elapsed()

    result: Dict[str, Any] = {
        "label": label,
        "parameters": parameter_count,
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": final_loss_value,
        "seconds": elapsed_seconds,
        "final_info": {key: scalar(value) for key, value in final_info.items()},
        "history": histories,
        "checkpoint": str(checkpoint_path),
        "progress_checkpoint": str(progress_path),
    }

    save_training_checkpoint(
        checkpoint_path,
        model,
        lbfgs,
        config,
        "complete",
        n_adam,
        n_lbfgs,
        histories,
        best_loss,
        best_step,
        elapsed_seconds,
        complete=True,
        result=result,
    )
    save_training_checkpoint(
        progress_path,
        model,
        lbfgs,
        config,
        "complete",
        n_adam,
        n_lbfgs,
        histories,
        best_loss,
        best_step,
        elapsed_seconds,
        complete=True,
        result=result,
    )
    del final_loss, final_info
    return result


def compare_pair(
    name: str,
    mlp: nn.Module,
    kan: nn.Module,
    loss_builder: Callable[[nn.Module], Callable[[], Tuple[Tensor, Dict[str, Tensor]]]],
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    grid_schedule: Sequence[Tuple[int, int]],
    output_dir: Path,
    resume: bool,
    checkpoint_interval: int,
) -> Dict[str, Any]:
    mlp_checkpoint = output_dir / f"{name}_MLP.pt"
    mlp_progress = output_dir / f"{name}_MLP.progress.pt"
    mlp_result = train_model(
        mlp,
        loss_builder(mlp),
        f"{name}/MLP",
        n_adam,
        n_lbfgs,
        lr,
        (),
        mlp_checkpoint,
        mlp_progress,
        resume,
        checkpoint_interval,
    )
    del mlp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    kan_checkpoint = output_dir / f"{name}_KAN.pt"
    kan_progress = output_dir / f"{name}_KAN.progress.pt"
    kan_result = train_model(
        kan,
        loss_builder(kan),
        f"{name}/KAN",
        n_adam,
        n_lbfgs,
        lr,
        grid_schedule,
        kan_checkpoint,
        kan_progress,
        resume,
        checkpoint_interval,
    )
    del kan
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mlp_result_loss = mlp_result["final_loss"]
    kan_result_loss = kan_result["final_loss"]
    mlp_best = mlp_result["best_loss"]
    kan_best = kan_result["best_loss"]
    return {
        "MLP": mlp_result,
        "KAN": kan_result,
        "comparison": {
            "final_loss_change_percent": 100.0 * (kan_result_loss - mlp_result_loss) / abs(mlp_result_loss),
            "best_loss_change_percent": 100.0 * (kan_best - mlp_best) / abs(mlp_best),
            "runtime_ratio_KAN_over_MLP": kan_result["seconds"] / mlp_result["seconds"],
            "parameter_ratio_KAN_over_MLP": kan_result["parameters"] / mlp_result["parameters"],
        },
    }


def run_plate(
    args: argparse.Namespace,
    output_dir: Path,
    resume: bool,
    report: Dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    dtype = plate.DTYPE
    device = plate.DEVICE
    grid_schedule = ((1500, 9), (3000, 13))

    for num_gauss in args.plate_gauss:
        seed_everything(args.seed)
        problem = plate.build_problem(
            plate.Problem(
                nx=args.plate_nx,
                ny=args.plate_nx,
                R=0.2,
                a=1.0,
                p1=1.0,
                p2=1.0,
                numGauss=num_gauss,
            )
        )

        seed_everything(args.seed)
        mlp = build_mlp(2, 2, args.width, args.depth, device, dtype)
        seed_everything(args.seed)
        kan = build_kan(
            2,
            2,
            args.width,
            args.depth,
            grid_size=5,
            input_scale=problem.a,
            device=device,
            dtype=dtype,
        )

        results[f"gauss_{num_gauss}"] = compare_pair(
            f"plate_g{num_gauss}",
            mlp,
            kan,
            lambda model: lambda: plate.compute_loss(model, problem, sigma0=1.0),
            args.plate_adam,
            args.plate_lbfgs,
            args.lr,
            grid_schedule,
            output_dir,
            resume,
            args.checkpoint_interval,
        )
        if report is not None and report_path is not None:
            report["results"]["plate"] = results
            atomic_json_save(report, report_path)
    return results


def run_cube(
    args: argparse.Namespace,
    output_dir: Path,
    resume: bool,
    report: Dict[str, Any] | None = None,
    report_path: Path | None = None,
) -> Dict[str, Any]:
    seed_everything(args.seed)
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
    problem = cube.buildProblem(problem)

    dtype = torch.get_default_dtype()
    device = cube.device
    grid_schedule = ((1500, 9), (3500, 13))

    seed_everything(args.seed)
    mlp = build_mlp(3, 3, args.width, args.depth, device, dtype)
    seed_everything(args.seed)
    kan = build_kan(
        3,
        3,
        args.width,
        args.depth,
        grid_size=5,
        input_scale=problem.a,
        device=device,
        dtype=dtype,
    )

    result = compare_pair(
        "cube",
        mlp,
        kan,
        lambda model: lambda: cube.computeLoss(model, problem, sigma0=1.0, betaInc=1.0),
        args.cube_adam,
        args.cube_lbfgs,
        args.lr,
        grid_schedule,
        output_dir,
        resume,
        args.checkpoint_interval,
    )
    if report is not None and report_path is not None:
        report["results"]["cube"] = result
        atomic_json_save(report, report_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("plate", "cube", "both"), default="plate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--plate-nx", type=int, default=80)
    parser.add_argument("--plate-gauss", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--plate-adam", type=int, default=4000)
    parser.add_argument("--plate-lbfgs", type=int, default=900)
    parser.add_argument("--cube-n", type=int, default=20)
    parser.add_argument("--cube-gauss", type=int, default=2)
    parser.add_argument("--cube-adam", type=int, default=5000)
    parser.add_argument("--cube-lbfgs", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Save a restart checkpoint every N Adam steps and up to every 25 L-BFGS steps.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_full_results"))
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing progress checkpoints and start the comparison from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the full comparison. Run this script on the GPU machine."
        )

    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison_report.json"
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "fresh"
    }
    settings["device"] = str(torch.cuda.get_device_name(0))

    report: Dict[str, Any] = {"settings": settings, "results": {}}
    if not args.fresh and report_path.exists():
        try:
            previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_report = None
        if isinstance(previous_report, dict) and previous_report.get("settings") == settings:
            report = previous_report
            report.setdefault("results", {})
            print(f"Restoring progress recorded in {report_path}")

    # Make even the initial empty report durable before expensive training starts.
    atomic_json_save(report, report_path)

    if args.problem in ("plate", "both"):
        report["results"]["plate"] = run_plate(
            args,
            args.output_dir,
            resume=not args.fresh,
            report=report,
            report_path=report_path,
        )
    if args.problem in ("cube", "both"):
        report["results"]["cube"] = run_cube(
            args,
            args.output_dir,
            resume=not args.fresh,
            report=report,
            report_path=report_path,
        )

    atomic_json_save(report, report_path)
    print(f"\nSaved comparison report to {report_path.resolve()}")


if __name__ == "__main__":
    main()
