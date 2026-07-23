"""Python counterpart of ``LA_PINN_plate_hole_run_histGauss_KAN.m``."""

from matlab_kan_cases import parse_common, run_hist_gauss


if __name__ == "__main__":
    args = parse_common()
    run_hist_gauss(args.output_dir, args.adam if args.adam is not None else 4000, args.lbfgs if args.lbfgs is not None else 900, 1.0e-3)
