"""Python counterpart of ``LA_PINN_plate_hole_run_adaptive_Gauss_KAN.m``."""

from matlab_kan_cases import parse_common, run_adaptive


if __name__ == "__main__":
    args = parse_common()
    adam = args.adam if args.adam is not None else 2000
    lbfgs = args.lbfgs if args.lbfgs is not None else 500
    run_adaptive(args.output_dir, adam, adam, lbfgs, 1.0e-3)
