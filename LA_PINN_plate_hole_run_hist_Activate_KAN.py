"""Python counterpart of ``LA_PINN_plate_hole_run_hist_Activate_KAN.m``."""

from matlab_kan_cases import parse_common, run_activation


if __name__ == "__main__":
    args = parse_common()
    run_activation(args.output_dir, args.adam if args.adam is not None else 4000, args.lbfgs if args.lbfgs is not None else 500, 1.0e-3)
