"""Python counterpart of ``LA_PINN_hollow_cylinder_hist_KAN.m``."""

from matlab_kan_cases import parse_common, run_thick


if __name__ == "__main__":
    args = parse_common()
    run_thick(output_dir=args.output_dir, n_adam=args.adam if args.adam is not None else 20000, n_lbfgs=args.lbfgs if args.lbfgs is not None else 200, lr=1.0e-3)
