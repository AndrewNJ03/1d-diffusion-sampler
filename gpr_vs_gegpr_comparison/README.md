# `gpr_vs_gegpr_comparison/`: GE-GPR vs. standard GPR, one parameter at a time

This directory compares gradient-enhanced GPR with standard GPR separately for
each of the five parameters that appear in the repo's two ROM problems:

| # | Parameter | Domain | Source pipeline | Active range |
|---|---|---|---|---|
| 1 | $D$ | material | `common/svd_analysis.py` + `sklearn_implementation/ge_gpr.py` | $[0.5, 2.0]$ |
| 2 | $\Sigma_a$ | material | same | $[0.05, 1.5]$ |
| 3 | $q$ | material | same | $[0, 2]$ |
| 4 | `void_thickness` $t_v$ | geometry | `void_teflon_geometry/ge_gpr_geometry.py` | $[0.01, 2.5]$ |
| 5 | `steepness` $a$ | geometry | same | $[10, 40]$ |

In each study the other parameters are fixed with degenerate bounds, so
$\alpha_k(\mu)$ is a smooth 1D curve and $p_\text{act} = 1$. The question is
whether adding $\partial\alpha_k/\partial\mu$ at each training point makes the
surrogate more accurate for the same number of forward solves. The GE-GPR
math is in
[sklearn_implementation/README.md §4](../sklearn_implementation/README.md#4-gradient-enhanced-gpr-ge_gprpy).

## What the driver does

`run_comparison.py` contains no modeling code. For each entry in `STUDIES` it
calls, unchanged:

- material: `common/svd_analysis.py::main()` (Stage 1) and then
  `sklearn_implementation/ge_gpr.py::main()`;
- geometry: `void_teflon_geometry/ge_gpr_geometry.py::main()` with
  `run_svd_stage=True`.

Each study writes its plots and Stage-1 files to `<param>/`. After all five,
the driver writes the summary table and two summary figures.

## Shared baseline

| Setting | Material studies | Geometry studies |
|---|---|---|
| Domain, mesh | $L=10$, `n_bins=1`, `n_cells=200` | $L=10$ (geometry ×10), `n_cells=200` |
| QoI window | $[3, 7]$ | $[3, 7]$ |
| Fixed values | $D=1.0$, $\Sigma_a=0.5$, $q=1.0$ | $a = 20$ when $t_v$ is active; $t_v = 1.25$ when $a$ is active |
| Stage 1 | $M_\text{train}=500$, $M_\text{test}=100$, seed 1 | $M_\text{train}=300$, $M_\text{test}=60$, seed 1 (`ge_gpr_geometry` defaults) |
| POD energy $\eta$ | 0.99999 | 0.99999 |
| GE-GPR / std GPR | `m_ge=250`, `n_restarts=3` | `m_ge=250`, `n_restarts=3` |
| Learning curve $M$ | 10, 20, 30, 50, 75, 100, 150, 200, 250 | same |

With `n_bins=1` the material vector is exactly $(D, \Sigma_a, q)$ on a
homogeneous slab. The geometry is rescaled ×10 (lengths ×10, steepness ÷10,
since it has units of 1/length) so both problems share the domain, mesh, and
window. See
[void_teflon_geometry/README.md §8](../void_teflon_geometry/README.md#8-configurations-and-caveats)
for what the $[3,7]$ window covers in the geometry problem.

## Results (`summary_table.csv`, median test $\varepsilon_y$ at $M = 250$)

| Parameter | Std GPR | GE-GPR | Improvement | POD floor $\varepsilon_\text{proj}$ |
|---|---|---|---|---|
| $D$ | 3.06e-5 | 3.03e-5 | 1.0% | 3.03e-5 |
| $\Sigma_a$ | 1.58e-4 | 1.05e-4 | 33.6% | 1.05e-4 |
| $q$ | 3.00e-5 | 1.38e-6 | 95.4% | 2.72e-7 |
| $t_v$ | 8.84e-4 | 8.84e-4 | 0.0% | 8.84e-4 |
| $a$ | 5.30e-4 | 5.30e-4 | 0.0% | 5.30e-4 |

Improvement is $100\,(\text{std}-\text{GE})/\text{std}$. How to read it:

- **The POD floor bounds the gain.** Since $\varepsilon_y\ge\varepsilon_\text{proj}$
  (see the [top-level README](../README.md#2-shared-notation)), GE-GPR can
  only help as much as standard GPR is above the floor. For $D$, $t_v$, and
  $a$, standard GPR is already at the floor with 250 points, so there is
  nothing left to gain at this $M$. GE-GPR reaches the floor for $\Sigma_a$,
  where standard GPR does not.
- **The $q$ study is exactly linear.** With one homogeneous layer, $A$ doesn't
  depend on $q$ and $s = q\,h$ on interior rows, so
  $\phi(q) = q\,A^{-1}h$ and every snapshot is a multiple of one vector:
  $\alpha_k(q)$ is linear and the snapshot set is rank 1. The floor of about
  $3\times10^{-7}$ is consistent with the 7-significant-digit text files the
  stages exchange. An RBF GP needs many points to represent a straight line
  accurately, while a single exact slope per point pins it down, which is
  why the gain is largest here.
- **Learning curves** (`summary_learning_curves.png`, and
  `<param>/gegpr_learning_curve.png`) show the small-$M$ behavior that the
  single $M=250$ number hides.

The combined 5-parameter case in `../combined_study/` shows a much larger
gain than any single row here (76.8% at $M = 80$ on the same baseline).

## Outputs

- `D/`, `Sigma_a/`, `q/`, `void_thickness/`, `steepness/`: Stage-1 files and
  plots, plus `gegpr_error_comparison.png`, `gegpr_parity.png`,
  `gegpr_flux_reconstruction.png`, `gegpr_learning_curve.png`, and, for the
  geometry studies, `gegpr_error_map_ab.png`.
- `summary_table.csv`: per parameter, median and max $\varepsilon_y$ for both
  models, percent improvement, and median $\varepsilon_\text{proj}$.
- `summary_median_error_bars.png`: grouped bars with the POD floor marked.
- `summary_learning_curves.png`: the five learning curves side by side.
- `run_log.txt`, `geo2param_run.log`: console logs from earlier runs.

## Running

```bash
cd gpr_vs_gegpr_comparison
python3 run_comparison.py
```

`../combined_study/run_combined.py` reads `summary_table.csv` to build its
side-by-side table, so run this first if both are being regenerated.
