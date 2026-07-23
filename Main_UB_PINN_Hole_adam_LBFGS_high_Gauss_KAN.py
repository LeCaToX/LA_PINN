"""Python counterpart of ``Main_UB_PINN_Hole_adam_LBFGS_high_Gauss_KAN.m``."""

from matlab_kan_cases import parse_common, run_high_gauss


if __name__ == "__main__":
    args = parse_common()
    run_high_gauss(args.output_dir, args.adam if args.adam is not None else 4000, args.lbfgs if args.lbfgs is not None else 300, 1.0e-3)
