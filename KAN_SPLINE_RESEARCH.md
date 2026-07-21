# Spline choice for the plate-with-hole limit-analysis KAN

## Decision

The PyTorch KAN uses **degree-3 (cubic), clamped, uniform B-splines** with a
coarse-to-fine grid schedule:

```text
5 intervals -> 9 intervals -> 13 intervals
```

The spline coefficients are projected to the refined basis by least squares,
so grid extension does not discard the mechanism already learned by Adam.

## Why this basis fits this problem

The network represents a velocity field and the objective evaluates first
spatial derivatives of that field through the strain rates. During parameter
optimization, those derivatives are differentiated again through the network.
For that setting, cubic B-splines are a good default because they are locally
supported, have a compact parameterization, and are `C2` across interior
knots. The local support is also useful for the sharp spatial change in the
collapse mechanism near the circular hole.

The clamped knot vector fixes an issue in the earlier prototype: isolated
cardinal splines did not form a partition of unity near `-1` and `1`. The
current basis has stable endpoint behaviour, while a smooth SiLU base branch
provides an extrapolating path when hidden activations leave the spline grid.

## Alternatives considered

- **Gaussian RBF / FastKAN:** substantially faster and smooth, but it is an
  approximation to the cubic B-spline KAN and loses compact support. It is a
  useful speed option if the full `80 x 80` / high-order quadrature run is
  memory-bound, but not the default accuracy choice here.
- **Chebyshev or other global polynomials:** smooth and efficient for globally
  regular fields, but less local than B-splines and more exposed to global
  coefficient coupling and endpoint oscillations as the mechanism sharpens.
- **Wavelets:** attractive for multiscale or oscillatory PDEs, but the present
  objective is a localized, first-gradient variational problem rather than a
  time-dependent multiscale residual. They add basis-selection and boundary
  handling decisions without an obvious benefit for this benchmark.

## References

1. Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756. The
   original KAN formulation uses learnable univariate splines, typically
   degree 3, and supports grid extension.
2. Li, *Kolmogorov-Arnold Networks are Radial Basis Function Networks*,
   arXiv:2405.06721. FastKAN replaces the cubic B-spline basis with Gaussian
   RBFs to reduce evaluation cost.
3. Patra et al., *Physics Informed Kolmogorov-Arnold Neural Networks for
   Dynamical Analysis via Efficient-KAN and WAV-KAN*, JMLR 26 (2025). This
   compares B-spline and wavelet KAN variants for physics-informed problems.
4. Howard et al., *Finite basis Kolmogorov-Arnold networks: domain
   decomposition for data-driven and physics-informed problems*, arXiv:2406.19662.
   This supports localized/domain-decomposed KANs when a single grid is not
   enough for fine-scale structure.

## KAN coverage in this folder

Python entry points:

- `LA_PINN_plate_hole_run_histGauss_KAN_torch.py`
- `LA_PINN_hole_Gauss_KAN.py`
- `LA_PINN_hole_Gauss_tf_KAN.py`
- `LA_PINN_hole_3D_KAN.py`
- `Main_UB_PINN_cube_spherical_hole_KAN_torch.py`

MATLAB entry points use the shared `buildKAN.m` helper:

- `Main_UB_PINN_hole_fast_KAN.m`
- `Main_UB_PINN_Hole_adam_LBFGS_KAN.m`
- `Main_UB_PINN_Hole_adam_LBFGS_high_Gauss_KAN.m`
- `LA_PINN_plate_hole_run_adaptive_Gauss_KAN.m`
- `LA_PINN_plate_hole_run_histGauss_KAN.m`
- `LA_PINN_hollow_cylinder_hist_KAN.m`
- `LA_PINN_hollow_cylinder_thin_KAN.m`
- `LA_PINN_VonMises_3D_hole_KAN.m`
- `LA_PINN_plate_hole_run_hist_Activate_KAN.m`

## Paper figure outputs

The paper is image-producing. The KAN entry points write publication-ready
PDF figures rather than relying only on interactive windows. The main mappings
are:

| Paper content | KAN entry point | Representative output |
|---|---|---|
| 2-D load-factor history | `LA_PINN_plate_hole_run_histGauss_KAN_torch.py` | `Hist_gaussp2_80.pdf` |
| 2-D dissipation and sample nodes | same script | `UB_dissipation_gauss_5.pdf`, `Data_gauss_80.pdf` |
| MLP-tanh versus KAN | `LA_PINN_plate_hole_run_hist_Activate_KAN.m` | `activation_comparison_history.pdf`, `activation_comparison_dissipation.pdf` |
| Thick hollow cylinder | `LA_PINN_hollow_cylinder_hist_KAN.m` | `thick_diss_*.pdf`, `thick_history_all_ab_ratios.pdf` |
| 3-D cube dissipation | `Main_UB_PINN_cube_spherical_hole_KAN_torch.py` | `cube_dissipation_3views.pdf` |
| 3-D cube convergence | same script | `cube_history.pdf` |

The Python 3-D KAN scripts now save figures directly, which is necessary for
headless GPU runs. The MATLAB KAN cylinder script is configured for the six
paper ratios `a/b = 0.3, 0.4, 0.5, 0.6, 0.7, 0.8`.
