"""Export paper figures from completed KAN runs without retraining.

The training pipeline already writes individual PDFs.  This postprocessor
collects those PDFs, creates combined history/final-lambda figures, and calls
the existing comparison exporter for the plate and cube MLP/KAN reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from paper_style import configure_paper_style
from export_paper_faithful import export_standalone


PAPER_FONT = configure_paper_style()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def relative_to(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def collect_pdfs(input_dir: Path, output_dir: Path) -> list[Path]:
    """Copy all training PDFs into a stable paper-figure raw directory."""

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in sorted(input_dir.rglob("*.pdf")):
        if output_dir.resolve() in source.resolve().parents:
            continue
        destination = raw_dir / relative_to(source, input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def history_from_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if "iter" in payload and "loss" in payload:
        return np.asarray(payload["iter"], dtype=float), np.asarray(payload["loss"], dtype=float)

    result = payload.get("result", {})
    histories = result.get("history", payload.get("histories", {})) if isinstance(result, dict) else {}
    if not isinstance(histories, dict):
        return None
    adam = np.asarray(histories.get("adam", []), dtype=float)
    lbfgs = np.asarray(histories.get("lbfgs", []), dtype=float)
    if not adam.size and not lbfgs.size:
        return None
    n_adam = int(payload.get("config", {}).get("n_adam", adam.size))
    adam_x = np.arange(1, adam.size + 1, dtype=float)
    lbfgs_x = np.linspace(n_adam + 1, n_adam + max(1, lbfgs.size), lbfgs.size) if lbfgs.size else np.array([])
    return np.concatenate((adam_x, lbfgs_x)), np.concatenate((adam, lbfgs))


def collect_histories(input_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    for checkpoint in sorted(input_dir.rglob("*.pt")):
        if checkpoint.name.endswith(".progress.pt"):
            continue
        try:
            payload = load_payload(checkpoint)
            history = history_from_payload(payload)
        except Exception as exc:  # corrupt/incompatible checkpoints should not block other figures
            print(f"Skipping history {checkpoint}: {exc}")
            continue
        if history is None:
            continue
        iterations, values = history
        if values.size:
            label = str(relative_to(checkpoint, input_dir).with_suffix(""))
            records.append((label, iterations, values))
    return records


def plot_histories(records: list[tuple[str, np.ndarray, np.ndarray]], output_dir: Path) -> Path | None:
    if not records:
        print("No standalone training histories found.")
        return None
    ncols = 2
    nrows = (len(records) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(4.0, 3.5 * nrows)), squeeze=False)
    for index, (label, iterations, values) in enumerate(records):
        ax = axes[index // ncols][index % ncols]
        ax.plot(iterations, values, linewidth=1.4)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\lambda^+$")
        ax.grid(True, alpha=0.3)
        if np.all(values > 0.0):
            ax.set_yscale("log")
    for index in range(len(records), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle("Training histories of translated KAN cases", fontsize=14)
    fig.tight_layout()
    path = output_dir / "all_case_training_histories.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_lambdas(records: list[tuple[str, np.ndarray, np.ndarray]], output_dir: Path) -> Path | None:
    if not records:
        return None
    labels = [record[0] for record in records]
    values = [float(record[2][-1]) for record in records]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(labels))))
    ax.barh(np.arange(len(labels)), np.asarray(values)[order], color="tab:orange")
    ax.set_yticks(np.arange(len(labels)), np.asarray(labels)[order])
    ax.set_xlabel(r"Final $\lambda^+$")
    ax.set_title("Final normalized limit loads")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "all_case_final_lambdas.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_comparison_export(root_dir: Path, input_dir: Path, output_dir: Path, pair_dir: Path | None, cube_dir: Path | None) -> list[Path]:
    generated: list[Path] = []
    exporter = root_dir / "export_comparison_images.py"
    if not exporter.exists():
        return generated
    jobs = []
    if pair_dir is not None and (pair_dir / "comparison_report.json").exists():
        jobs.append((pair_dir, output_dir / "plate_comparison", "plate"))
    if cube_dir is not None and (cube_dir / "comparison_report.json").exists():
        jobs.append((cube_dir, output_dir / "cube_comparison", "cube"))
    for source, destination, problem in jobs:
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(exporter),
            "--input-dir",
            str(source),
            "--output-dir",
            str(destination),
            "--problem",
            problem,
        ]
        print("Running:", " ".join(command))
        completed = subprocess.run(command, cwd=root_dir, check=False)
        if completed.returncode != 0:
            print(f"Comparison image export failed for {problem}; continuing with standalone figures.")
        else:
            generated.extend(sorted(destination.rglob("*.pdf")))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("all_kan_results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pair-dir", type=Path, default=None)
    parser.add_argument("--cube-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "paper_figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_dir is not None:
        pair_dir = args.pair_dir.resolve()
    elif (input_dir / "plate_pair_comparison" / "comparison_report.json").exists():
        pair_dir = input_dir / "plate_pair_comparison"
    elif (input_dir.parent / "comparison_full_results" / "comparison_report.json").exists():
        pair_dir = input_dir.parent / "comparison_full_results"
    else:
        pair_dir = None

    if args.cube_dir is not None:
        cube_dir = args.cube_dir.resolve()
    elif (input_dir / "cube_2gpu" / "comparison_report.json").exists():
        cube_dir = input_dir / "cube_2gpu"
    elif (input_dir.parent / "comparison_cube_2gpu" / "comparison_report.json").exists():
        cube_dir = input_dir.parent / "comparison_cube_2gpu"
    else:
        cube_dir = None

    print(f"Reading completed results from {input_dir}")
    print(f"Paper font selected: {PAPER_FONT}")
    copied = collect_pdfs(input_dir, output_dir)
    records = collect_histories(input_dir)
    summary_paths = [path for path in (plot_histories(records, output_dir), plot_final_lambdas(records, output_dir)) if path]
    comparison_paths = run_comparison_export(Path(__file__).resolve().parent, input_dir, output_dir, pair_dir, cube_dir)
    faithful_dir = output_dir / "paper_faithful"
    export_standalone(input_dir, faithful_dir)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "standalone_pdf_count": len(copied),
        "history_count": len(records),
        "generated_files": [str(path.relative_to(output_dir)) for path in summary_paths + comparison_paths]
        + [str(path.relative_to(output_dir)) for path in faithful_dir.rglob("*.pdf")],
    }
    (output_dir / "paper_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported paper figures to {output_dir}")
    print(f"Collected {len(copied)} existing PDFs and {len(records)} training histories.")


if __name__ == "__main__":
    main()
