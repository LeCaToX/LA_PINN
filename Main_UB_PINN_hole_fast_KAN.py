"""Python counterpart of ``Main_UB_PINN_hole_fast_KAN.m``."""

from matlab_kan_cases import parse_common, run_fast


if __name__ == "__main__":
    args = parse_common()
    run_fast(args.output_dir, args.adam if args.adam is not None else 3000, args.lbfgs if args.lbfgs is not None else 0, 2.0e-3)
