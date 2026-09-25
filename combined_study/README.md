# `combined_study/`: all five parameters active at once

This directory runs the fully coupled version of the one-at-a-time studies in
`../gpr_vs_gegpr_comparison/`. All five parameters vary together in one Latin
hypercube design, with one shared forward/adjoint pipeline and one GE-GPR fit
per POD mode:

$$
\mu = (t_v,\ a,\ D_\text{teflon},\ \Sigma_{a,\text{teflon}},\ q_\text{teflon}),\qquad p = p_\text{act} = 5 .
$$

The two geometry parameters are those of
[void_teflon_geometry/](../void_teflon_geometry/README.md). The three material
parameters are the Teflon shell's $(D, \Sigma_a, q)$, which were fixed
constants in that directory and are sampled here.

The subdirectory `cep_sgemp_analysis/` is a separate piece of work on SCEPTRE
data; see [its README](cep_sgemp_analysis/README.md).

| File | Role |
|---|---|
| `params_combined.py` | `CombinedParamSampler` for the 5D vector; `unpack()` returns `(t_v, a, mat_teflon)` |
| `svd_analysis_combined.py` | Stage 1: snapshots via `geometry.solve_void_teflon()` with per-sample `mat_teflon`, POD, standard file outputs, spectrum and projection-error plots |
| `adjoint_combined.py` | Field derivative w.r.t. a Teflon value, `combined_field_derivatives()`, `sensitivity_wrt_combined_param()`; identity and FD checks for all 5 in `main()` |
| `ge_gpr_combined.py` | GE-GPR vs. standard GPR with all 5 adjoint partials per training point, learning curve |
| `run_combined.py` | Driver on the paper baseline; writes `output_graphs_combined/` and `combined_vs_one_at_a_time.csv` |
| `run_combined_seeds.py` | Repeats the driver over 5 independent LHS designs; writes `combined_seed_repeats.{csv,png}` and `output_graphs_combined_seeds/seed*/` |

---

## 1. The one new derivative

The adjoint machinery from `void_teflon_geometry/adjoint_geometry.py` already
takes any per-cell direction vectors $(dD, d\Sigma_a, dq)$
(`sensitivity_from_directions()`, see
[its README §5](../void_teflon_geometry/README.md#5-adjoint-sensitivities)).
Adding the material parameters only needs the field derivative with respect
to $v_\text{teflon}$. The shell blend is

$$
\text{field}_i = v_\text{void} + (v_\text{teflon}-v_\text{void})\,s_i ,
$$

which is linear in $v_\text{teflon}$, so

$$
\frac{\partial\,\text{field}_i}{\partial v_\text{teflon}} = s_i\ \ \text{in the shell},\qquad 0\ \ \text{in Core and Shield}.
$$

This is `field_derivative_wrt_teflon_value()`. The direction vector is the
switch itself: cells deep in the Teflon respond fully, cells deep in the Void
not at all. A given Teflon parameter changes only its own field, so
`combined_field_derivatives()` puts $s$ in that channel and zeros in the other
two. For example, $\partial/\partial D_\text{teflon}$ uses
$(dD, d\Sigma_a, dq) = (s, 0, 0)$. The two geometry parameters use the existing
`geometry_field_derivatives()`.

For the geometry derivatives, $v_\text{teflon}$ is now the sampled value from
$\mu$, so the factor $(v_\text{teflon} - v_\text{void})$ in
$d\,\text{field}/dt_v$ and $d\,\text{field}/da$ changes from sample to sample.
`forward_system_combined()` returns the per-sample `mat_teflon` so the
gradient code uses the right one.

As before, one forward solve plus one adjoint solve per POD mode gives all
five partial derivatives at once:

$$
\frac{\partial\alpha_k}{\partial\mu_j} = \sum_{\text{channels}}\big\langle \text{adjoint sensitivity},\ \partial_j\text{field}\big\rangle,\qquad j = 1,\dots,5 .
$$

`adjoint_combined.main()` verifies all five (identity check, then
central-difference FD with `_fd_eps()` keeping perturbations inside the
bounds). When this study was built, the identity error was about $10^{-11}$
and FD agreement about $10^{-5}$ for all five parameters.

## 2. Configuration (`run_combined.py`)

The geometry is rescaled ×10 and the bounds match the one-at-a-time studies,
so the numbers are comparable with `../gpr_vs_gegpr_comparison/summary_table.csv`:

| Setting | Value |
|---|---|
| Domain, mesh, window | $L=10$, `n_cells=200`, $x\in[3,7]$ |
| $t_v$, $a$ | $[0.01, 2.5]$, $[10, 40]$ |
| $D_\text{teflon}$, $\Sigma_{a,\text{teflon}}$, $q_\text{teflon}$ | $[0.5, 2]$, $[0.05, 1.5]$, $[0, 2]$ |
| Other regions | Core $(1.0, 0.5, 1.0)$, Void $(1.8, 0.01, 0)$, Shield $(0.3, 0.9, 0)$ |
| Stage 1 | $M_\text{train} = 300$, $M_\text{test} = 60$, seed 1, $\eta = 0.99999$ |
| GE-GPR / std GPR | `m_ge=80`, `n_restarts=2`, learning curve at $M = 10, 20, \dots, 80$ |

`m_ge` is smaller than the one-at-a-time studies' 250 because the augmented
kernel has size $M(1+p_\text{act})$ and the fit is cubic in that
([cost note](../sklearn_implementation/README.md#47-cost)). At
$M = 80$, $p_\text{act}=5$ the system is $480\times480$, about the same as
$M = 250$, $p_\text{act}=1$ ($500\times500$).

Note that $q_\text{teflon}\in[0,2]$ adds a source in the shell, where the
geometry-only studies had none.

## 3. Results

**Single design** (`combined_vs_one_at_a_time.csv`, seed 1, $M=80$):

| Study | $p_\text{act}$ | Std GPR | GE-GPR | Improvement |
|---|---|---|---|---|
| combined | 5 | 2.95e-3 | 6.83e-4 | 76.8% |
| (one-at-a-time rows, $M = 250$) | 1 | see `../gpr_vs_gegpr_comparison/` | | 0% to 95.4% |

**Five independent designs** (`combined_seed_repeats.csv`, seeds 1–5; median
over seeds of the median test $\varepsilon_y$, with min–max range):

| $M$ | GE-GPR | Std GPR | Median improvement |
|---|---|---|---|
| 10 | 7.6e-3 (4.6e-3 to 1.1e-2) | 8.1e-2 (4.8e-2 to 1.2e-1) | 88.3% |
| 20 | 3.0e-3 (1.7e-3 to 4.2e-3) | 2.2e-2 (1.7e-2 to 2.5e-2) | 85.5% |
| 40 | 1.3e-3 (8.8e-4 to 1.5e-3) | 7.9e-3 (6.2e-3 to 1.1e-2) | 85.6% |
| 80 | 6.3e-4 (4.9e-4 to 6.8e-4) | 4.2e-3 (2.3e-3 to 5.7e-3) | 85.1% |

The median improvement is between 83.2% and 88.4% at every $M$ in the CSV
(all eight sizes), and the GE-GPR and standard-GPR min–max ranges do not
overlap at any $M$. The single-design 76.8% (seed 1) is below the
across-seed median of 85.1% at $M=80$. Standard GPR's error is still falling
steadily at $M = 80$, so with five active parameters it is nowhere near
converged at these sample sizes and the gradients have room to help. In most
of the one-parameter studies, standard GPR had already reached the POD floor.

`combined_seed_repeats.png` plots these curves with the min–max bands, and
is copied into the paper as `latex_paper/figures/combined_seed_repeats.png`.
`output_graphs_combined/gegpr_learning_curve.png` is the paper's
`combined_learning_curve.png`.

## 4. Running

```bash
cd combined_study
python3 run_combined.py          # about 4 minutes; reads ../gpr_vs_gegpr_comparison/summary_table.csv if present
python3 run_combined_seeds.py    # 5 designs, about 5x as long
python3 adjoint_combined.py      # identity and FD checks for all 5 parameters (module-default L=1 geometry)
```

`svd_analysis_combined.py` is a trimmed copy of the geometry Stage 1: it
keeps the snapshot solve, rank selection, spectrum, and projection-error
plots, and drops the mode-shape and reconstruction plots.

`svd_analysis_combined.py`, `ge_gpr_combined.py`, and `adjoint_combined.py`
share the geometry modules' default QoI window, which has the coordinate
issue described in
[void_teflon_geometry/README.md §8.2](../void_teflon_geometry/README.md#82-the-default-qoi-window-is-not-the-shell).
The two drivers always pass $[3, 7]$, so the results above are not affected.
Running `adjoint_combined.py` directly does use the default window. That
doesn't invalidate the check, since the identity and FD tests hold for any
window, but its window is not the shell.
