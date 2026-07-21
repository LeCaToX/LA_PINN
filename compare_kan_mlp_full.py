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


def train_model(
    model: nn.Module,
    loss_fn: Callable[[], Tuple[Tensor, Dict[str, Tensor]]],
    label: str,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    grid_schedule: Sequence[Tuple[int, int]],
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Train one model with the production Adam/L-BFGS schedule."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    histories: Dict[str, List[float]] = {"adam": [], "lbfgs": []}
    best_loss = float("inf")
    best_step = 0
    start_time = time.perf_counter()

    adam = torch.optim.Adam(model.parameters(), lr=lr)
    grid_schedule_dict = dict(grid_schedule)

    print(f"\n--- {label}: Adam ({n_adam} steps) ---")
    for epoch in range(1, n_adam + 1):
        adam.zero_grad(set_to_none=True)
        loss, info = loss_fn()
        loss.backward()
        adam.step()

        loss_value = scalar(loss)
        histories["adam"].append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = epoch

        if epoch in grid_schedule_dict and isinstance(model, KAN):
            new_grid = grid_schedule_dict[epoch]
            print(f"{label}: refining cubic B-spline grid to {new_grid} intervals")
            model.refine_grid(new_grid)
            # Refinement replaces spline tensors, so restart Adam's moments.
            adam = torch.optim.Adam(model.parameters(), lr=lr)

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"{label} | Adam {epoch:5d} | lambda = {loss_value:.8f} | "
                f"elapsed = {time.perf_counter() - start_time:.1f}s"
            )

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

    for iteration in range(1, n_lbfgs + 1):
        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            closure_loss, _ = loss_fn()
            closure_loss.backward()
            return closure_loss

        lbfgs.step(closure)

        if iteration % 25 == 0 or iteration == n_lbfgs:
            loss, _ = loss_fn()
            loss_value = scalar(loss)
            histories["lbfgs"].append(loss_value)
            global_step = n_adam + iteration
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = global_step
            print(
                f"{label} | L-BFGS {iteration:5d} | lambda = {loss_value:.8f} | "
                f"elapsed = {time.perf_counter() - start_time:.1f}s"
            )

    final_loss, final_info = loss_fn()
    elapsed = time.perf_counter() - start_time
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "parameters": parameter_count,
            "best_loss": best_loss,
            "best_step": best_step,
        },
        checkpoint_path,
    )

    result: Dict[str, Any] = {
        "label": label,
        "parameters": parameter_count,
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": scalar(final_loss),
        "seconds": elapsed,
        "final_info": {key: scalar(value) for key, value in final_info.items()},
        "history": histories,
        "checkpoint": str(checkpoint_path),
    }
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
) -> Dict[str, Any]:
    mlp_result = train_model(
        mlp,
        loss_builder(mlp),
        f"{name}/MLP",
        n_adam,
        n_lbfgs,
        lr,
        (),
        output_dir / f"{name}_MLP.pt",
    )
    kan_result = train_model(
        kan,
        loss_builder(kan),
        f"{name}/KAN",
        n_adam,
        n_lbfgs,
        lr,
        grid_schedule,
        output_dir / f"{name}_KAN.pt",
    )
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


def run_plate(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
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
        )
    return results


def run_cube(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
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

    return compare_pair(
        "cube",
        mlp,
        kan,
        lambda model: lambda: cube.computeLoss(model, problem, sigma0=1.0, betaInc=1.0),
        args.cube_adam,
        args.cube_lbfgs,
        args.lr,
        grid_schedule,
        output_dir,
    )


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
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_full_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the full comparison. Run this script on the GPU machine."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "settings": vars(args) | {"device": str(torch.cuda.get_device_name(0))},
        "results": {},
    }
    report["settings"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["settings"].items()
    }

    if args.problem in ("plate", "both"):
        report["results"]["plate"] = run_plate(args, args.output_dir)
    if args.problem in ("cube", "both"):
        report["results"]["cube"] = run_cube(args, args.output_dir)

    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comparison report to {report_path.resolve()}")


if __name__ == "__main__":
    main()
