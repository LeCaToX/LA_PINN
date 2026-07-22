"""Merge the plate report and the sharded two-GPU cube report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plate-dir", type=Path, default=Path("comparison_full_2gpu/plate"))
    parser.add_argument("--cube-dir", type=Path, default=Path("comparison_full_2gpu/cube"))
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_full_2gpu"))
    args = parser.parse_args()

    plate_path = args.plate_dir / "comparison_report.json"
    cube_path = args.cube_dir / "comparison_report.json"
    plate = read_json(plate_path)
    cube = read_json(cube_path)

    settings = dict(plate.get("settings", {}))
    settings.update(
        {
            "multi_gpu_cube": True,
            "cube_settings": cube.get("settings", {}),
        }
    )
    merged: Dict[str, Any] = {
        "settings": settings,
        "results": {
            "plate": plate.get("results", {}).get("plate", {}),
            "cube": cube.get("results", {}),
        },
        "source_reports": [str(plate_path.resolve()), str(cube_path.resolve())],
    }
    output_path = args.output_dir / "comparison_report.json"
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"Saved full comparison report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
