# GE-GPR vs. standard GPR — all 5 parameters

This directory compares the gradient-enhanced GPR surrogate (GE-GPR) against
standard GPR, one parameter at a time, across every parameter that exists in
the two ROM studies elsewhere in this repo:

| # | Parameter        | Domain    | Source pipeline               |
|---|------------------|-----------|--------------------------------|
| 1 | `D`              | material  | `sklearn_implementation/`      |
| 2 | `Sigma_a`        | material  | `sklearn_implementation/`      |
| 3 | `q`              | material  | `sklearn_implementation/`      |
| 4 | `void_thickness` | geometry  | `void_teflon_geometry/`        |
| 5 | `steepness`      | geometry  | `void_teflon_geometry/`        |

For each parameter, the other four are held fixed at a constant value
(degenerate `(v, v)` bounds — the same convention `GeometryParamSampler` and
`DiffusionParamSampler` already use for "fixed" dimensions), so each study is
a genuine single-parameter surrogate problem: `alpha_k(mu)` is a smooth
1-D curve, and the question is whether the adjoint-gradient information
GE-GPR uses to fit that curve gives it an edge over plain GPR at the same
training-set size.

## What this does *not* duplicate

No modeling code lives in this directory. `run_comparison.py` is a thin
driver that calls, unchanged:

- `common/svd_analysis.py::main()` — Stage 1 (snapshots -> POD) for the 3
  material studies, with `n_bins=1` so the sampled vector is exactly
  `(D, Sigma_a, q)`.
- `sklearn_implementation/ge_gpr.py::main()` — trains GE-GPR and standard
  GPR on the same training subset and evaluates both on a held-out test set.
- `void_teflon_geometry/ge_gpr_geometry.py::main()` — same comparison for
  the 2 geometry studies; it runs its own Stage-1 SVD internally
  (`run_svd_stage=True`).

The **one** change made to existing code: `ge_gpr_geometry.main()` used to
require *both* `void_thickness` and `steepness` to be active (it raised a
`ValueError` otherwise), because it was originally written only for the
2-parameter comparison in `output_graphs_geometry_2param/`. That guard is
relaxed to allow a single active geometry parameter, since the material side
has always supported single-parameter studies the same way.

## Fixed values used

Material studies use `D=1.0, Sigma_a=0.5, q=1.0` as the fixed values for the
two inactive parameters — the same constants `run_pipeline.py`'s
`fixed_mats={1: {...}}` example already uses for "Material 1." Active bounds
(`D in [0.5, 2.0]`, `Sigma_a in [0.05, 1.5]`, `q in [0.0, 2.0]`) are the
existing defaults from `run_pipeline.py` / `common/svd_analysis.py`.

Geometry studies use the existing defaults from `ge_gpr_geometry.py`
(`void_bounds=(0.001, 0.25)`, `steepness_bounds=(100.0, 400.0)` when active,
`steepness` fixed at `200.0` / `void_thickness` fixed at its bounds'
midpoint `0.125` when not the active parameter).

All 5 studies share `m_ge=100` training points for the GE-GPR/std-GPR fit,
`n_restarts=3` hyperparameter restarts, and the same learning-curve sizes
`[10, 20, 30, 50, 75, 100]`, so the results are directly comparable across
parameters.

## Running it

```bash
cd gpr_vs_gegpr_comparison
python3 run_comparison.py
```

## Outputs

- `D/`, `Sigma_a/`, `q/`, `void_thickness/`, `steepness/` — per-parameter
  plots from the underlying `ge_gpr` modules unchanged: error-distribution
  histogram, parity plots, flux reconstruction, learning curve (and, for the
  geometry studies, the `(a, b)` error map).
- `summary_table.csv` — one row per parameter: median/max relative test
  error for GE-GPR and standard GPR, percent improvement, and the POD
  projection-error floor.
- `summary_median_error_bars.png` — grouped bar chart of median error
  (GE-GPR vs std GPR) across all 5 parameters.
- `summary_learning_curves.png` — the 5 learning curves side by side.
