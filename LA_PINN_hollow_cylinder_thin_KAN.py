"""Python counterpart of ``LA_PINN_hollow_cylinder_thin_KAN.m``."""

from matlab_kan_cases import parse_common, run_thin


if __name__ == "__main__":
    args = parse_common()
    run_thin(args.output_dir, args.adam if args.adam is not None else 4000, args.lbfgs if args.lbfgs is not None else 400, 1.0e-3)
