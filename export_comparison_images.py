"""Create paper-style plots from completed comparison checkpoints.

This script does not retrain. It loads ``*_MLP.pt`` and ``*_KAN.pt`` files
written by ``compare_kan_mlp_full.py`` and evaluates their nodal dissipation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_payload(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setting(report: Dict[str, Any], name: str, default: Any) -> Any:
    return report.get("settings", {}).get(name, default)


def load_report(input_dir: Path) -> Dict[str, Any]:
    candidates = [input_dir / "comparison_report.json"]
    candidates.extend(sorted(input_dir.glob("*/comparison_report.json")))
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return {}


def model_from_checkpoint(
    checkpoint_path: Path,
    model_kind: str,
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    input_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    compare: Any,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = load_payload(checkpoint_path)
    if model_kind == "MLP":
        model = compare.build_mlp(in_dim, out_dim, width, depth, device, dtype)
        model.load_state_dict(payload["model_state_dict"])
    else:
        model = compare.build_kan(
            in_dim,
            out_dim,
            width,
            depth,
            grid_size=5,
            input_scale=input_scale,
            device=device,
            dtype=dtype,
        )
        compare.restore_model_state(model, payload["model_state_dict"])
    model.eval()
    return model, payload


def call_plot_in_directory(
    output_dir: Path,
    generated_name: str,
    target_name: str,
    plotter: Any,
) -> None:
    previous_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(output_dir)
        plotter()
    finally:
        os.chdir(previous_dir)
    generated_path = output_dir / generated_name
    target_path = output_dir / target_name
    if generated_path.exists() and generated_path != target_path:
        generated_path.replace(target_path)


def plot_history(
    output_dir: Path,
    case_name: str,
    mlp_result: Dict[str, Any],
    kan_result: Dict[str, Any],
    n_adam: int,
    n_lbfgs: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for result, label, color in (
        (mlp_result, "MLP", "tab:blue"),
        (kan_result, "KAN", "tab:orange"),
    ):
        history = result.get("history", {})
        adam = np.asarray(history.get("adam", []), dtype=float)
        lbfgs = np.asarray(history.get("lbfgs", []), dtype=float)
        if adam.size:
            ax.plot(np.arange(1, adam.size + 1), adam, color=color, label=f"{label} Adam")
        if lbfgs.size:
            x = np.linspace(n_adam + 1, n_adam + n_lbfgs, lbfgs.size)
            ax.plot(x, lbfgs, "--", color=color, label=f"{label} L-BFGS")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda^{+}$")
    ax.set_title(f"{case_name}: MLP versus KAN")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{case_name}_loss_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def export_plate(
    input_dir: Path,
    output_dir: Path,
    report: Dict[str, Any],
    device: torch.device,
    compare: Any,
    plate: Any,
) -> None:
    nx = int(setting(report, "plate_nx", 80))
    width = int(setting(report, "width", 64))
    depth = int(setting(report, "depth", 4))
    plate_output = output_dir / "plate"
    plate_output.mkdir(parents=True, exist_ok=True)

    mlp_paths = sorted(input_dir.rglob("plate_g*_MLP.pt"))
    if not mlp_paths:
        print("No plate MLP checkpoints found.")
        return

    plotted_mesh = False
    for mlp_path in mlp_paths:
        case_name = mlp_path.stem.removesuffix("_MLP")
        kan_path = mlp_path.with_name(f"{case_name}_KAN.pt")
        if not kan_path.exists():
            print(f"Skipping {case_name}: missing {kan_path.name}")
            continue
        gauss = int(case_name.removeprefix("plate_g"))
        prob = plate.build_problem(
            plate.Problem(
                nx=nx,
                ny=nx,
                R=0.2,
                a=1.0,
                p1=1.0,
                p2=1.0,
                numGauss=gauss,
            )
        )
        dtype = plate.DTYPE
        mlp, mlp_payload = model_from_checkpoint(
            mlp_path, "MLP", 2, 2, width, depth, prob.a, device, dtype, compare
        )
        kan, kan_payload = model_from_checkpoint(
            kan_path, "KAN", 2, 2, width, depth, prob.a, device, dtype, compare
        )

        if not plotted_mesh:
            call_plot_in_directory(
                plate_output,
                f"Data_gauss_{nx}.pdf",
                "plate_mesh.pdf",
                lambda: plate.plot_initial_nodes(prob),
            )
            plotted_mesh = True

        for model, label, payload in (
            (mlp, "MLP", mlp_payload),
            (kan, "KAN", kan_payload),
        ):
            dissipation = plate.nodal_dissipation(model, prob, sigma0=1.0)
            call_plot_in_directory(
                plate_output,
                f"UB_dissipation_gauss_{gauss}.pdf",
                f"{case_name}_{label}_dissipation.pdf",
                lambda d=dissipation: plate.plot_dissipation(prob, d),
            )

        mlp_result = mlp_payload.get("result", {})
        kan_result = kan_payload.get("result", {})
        if mlp_result and kan_result:
            n_adam = int(setting(report, "plate_adam", 4000))
            n_lbfgs = int(setting(report, "plate_lbfgs", 900))
            plot_history(plate_output, case_name, mlp_result, kan_result, n_adam, n_lbfgs)

        del mlp, kan
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Exported images for {case_name}")


def export_cube(
    input_dir: Path,
    output_dir: Path,
    report: Dict[str, Any],
    device: torch.device,
    compare: Any,
    cube: Any,
) -> None:
    mlp_paths = sorted(input_dir.rglob("cube_MLP.pt"))
    if not mlp_paths:
        print("No cube MLP checkpoint found.")
        return
    mlp_path = mlp_paths[0]
    kan_path = mlp_path.with_name("cube_KAN.pt")
    if not kan_path.exists():
        print(f"Skipping cube: missing {kan_path.name}")
        return

    n = int(setting(report, "cube_n", 20))
    width = int(setting(report, "width", 64))
    depth = int(setting(report, "depth", 4))
    problem = cube.Problem()
    problem.a = 1.0
    problem.R = 0.2
    problem.px = 1.0
    problem.py = 0.0
    problem.pz = 0.0
    problem.nx = n
    problem.ny = n
    problem.nz = n
    problem.numGauss = int(setting(report, "cube_gauss", 2))
    problem = cube.buildProblem(problem)
    dtype = torch.get_default_dtype()

    mlp, mlp_payload = model_from_checkpoint(
        mlp_path, "MLP", 3, 3, width, depth, problem.a, device, dtype, compare
    )
    kan, kan_payload = model_from_checkpoint(
        kan_path, "KAN", 3, 3, width, depth, problem.a, device, dtype, compare
    )

    cube_output = output_dir / "cube"
    cube_output.mkdir(parents=True, exist_ok=True)
    call_plot_in_directory(
        cube_output,
        "cube_mesh.pdf",
        "cube_mesh.pdf",
        lambda: cube.plotH8Mesh(problem.coords, problem.elem),
    )
    for model, label in ((mlp, "MLP"), (kan, "KAN")):
        xnode = torch.tensor(problem.coords, device=device)
        dissipation = cube.nodalDissipation(model, problem, xnode, sigma0=1.0)
        call_plot_in_directory(
            cube_output,
            "cube_dissipation_3d.pdf",
            f"cube_{label}_dissipation_3d.pdf",
            lambda d=dissipation: cube.plotDissipation3D(problem, d),
        )
        call_plot_in_directory(
            cube_output,
            "cube_dissipation_3views.pdf",
            f"cube_{label}_dissipation_3views.pdf",
            lambda d=dissipation: cube.plotDissipation3Views(problem, d),
        )

    mlp_result = mlp_payload.get("result", {})
    kan_result = kan_payload.get("result", {})
    if mlp_result and kan_result:
        plot_history(
            cube_output,
            "cube",
            mlp_result,
            kan_result,
            int(setting(report, "cube_adam", 5000)),
            int(setting(report, "cube_lbfgs", 1000)),
        )
    del mlp, kan
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Exported images for cube")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("comparison_full_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_images"))
    parser.add_argument("--problem", choices=("plate", "cube", "both"), default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Export device: {device}")
    report = load_report(args.input_dir)

    # Imports are delayed until after argument parsing and backend selection.
    import compare_kan_mlp_full as compare
    import LA_PINN_plate_hole_run_histGauss_KAN_torch as plate
    import Main_UB_PINN_cube_spherical_hole_KAN_torch as cube

    if args.problem in ("plate", "both"):
        export_plate(args.input_dir, args.output_dir, report, device, compare, plate)
    if args.problem in ("cube", "both"):
        export_cube(args.input_dir, args.output_dir, report, device, compare, cube)
    print(f"Images saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
