"""Python counterpart of ``LA_PINN_VonMises_3D_hole_KAN.m``.

The maintained 3-D implementation is shared with the comparison and
two-GPU runners.  This entrypoint preserves the MATLAB case name while
delegating to that PyTorch implementation; no MATLAB file is executed.
"""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("matlab_kan_results/cube"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_dir = Path.cwd()
    try:
        os.chdir(args.output_dir)
        runpy.run_path(
            str(Path(__file__).with_name("Main_UB_PINN_cube_spherical_hole_KAN_torch.py")),
            run_name="__main__",
        )
    finally:
        os.chdir(old_dir)
