# `void_teflon_geometry/`: geometry parameters through a smooth material switch

This directory runs the same ROM workflow as `sklearn_implementation/`, but
the parameters describe **geometry** instead of per-layer material values.
The parameters are the location of the Void/Teflon interface and the
sharpness of that interface. It reuses the mesh, assembly, solve, adjoint,
GPR, and GE-GPR code unchanged. Only two things are new: a differentiable
material switch, and the chain rule that carries adjoint sensitivities through
it. Notation follows the [top-level README](../README.md#2-shared-notation);
the underlying adjoint formulas are in
[common/README.md §5](../common/README.md#5-adjoint-sensitivities-adjoint_solverpy).

Contents: 1. geometry · 2. why the hard assignment can't be differentiated ·
3. the smooth switch · 4. ROM pipeline · 5. adjoint derivation ·
6. verification · 7. GE-GPR · 8. configurations and caveats · 9. file map and running.

---

## 1. The geometry

```
Shield | Dielectric (Teflon) | Void | Core | Core | Void | Dielectric (Teflon) | Shield
        <---------------- half-slab, radial distance r from center -------------->
```

The two central Core regions are the same material, so this is one Core of
half-width `core_radius`. With $r = \lvert x - L/2\rvert$:

| Region | Radial range | Boundary |
|---|---|---|
| Core | $0 \le r < r_c$ | fixed |
| Void | $r_c \le r < r_c + t_v$ | outer edge **moves** |
| Teflon | $r_c + t_v \le r < r_c + w$ | inner edge **moves** |
| Shield | $r_c + w \le r \le L/2$ | fixed |

Here $r_c$ = `core_radius`, $t_v$ = `void_thickness`, and $w$ = `shell_width`.
Across the case sweep, void and dielectric thickness trade off so that
$w = t_v + t_\text{diel}$ stays constant, and $r_c$ and the shield thickness are
constant too. Only the Void/Teflon interface at $r = r_c + t_v$ moves, so
$t_v$ is the one physical geometry parameter.

## 2. Why the hard assignment can't be differentiated

`common/second_order_solver.py::assign_material_properties()` assigns each
cell the value of the layer that contains its center. If the boundary is a
parameter:

- a cell that stays on the same side of the boundary does not change at all,
  so its derivative is exactly 0;
- the one or two cells whose side flips jump between the Void and Teflon
  values, so the derivative is undefined there.

The resulting "gradient" of any QoI is a sum of exact zeros and occasional
jumps. An adjoint method can't use it, and a finite-difference check can't
validate it, because the FD result depends on which cell the perturbed
boundary lands in rather than on the continuum sensitivity.

## 3. The smooth switch

`geometry.py` replaces the hard step, **at the moving interface only**, with a
logistic function:

$$
\sigma(r;\,a,\,b) = \frac{1}{1+e^{-a(r-b)}},\qquad a = \texttt{steepness},\quad b = r_\text{int} = r_c + t_v .
$$

$\sigma \to 0$ well inside the Void, $\sigma \to 1$ well inside the Teflon, and
$\sigma(b) = 1/2$ exactly for any $a$. So $a$ controls only the sharpness and
$b$ only the location. The interface location $b$ is a **linear** function of
$t_v$, which is what makes the chain rule in §5 simple.

Per cell, $s_i = \sigma(r_i;\,a,\,b)\in(0,1)$ reads as "how far past the
interface is cell $i$". It is purely geometric: it depends on the cell
position and on $(a, b)$, not on $D$, $\Sigma_a$, or $q$. The width of the rise
from about 0 to about 1 scales like $1/a$; the 10%–90% width is
$2\ln 9/a \approx 4.39/a$.

`void_teflon_field()` turns $s_i$ into a value of any one property by linear
interpolation between the two endpoint values:

$$
\text{shell}_i = v_\text{void} + (v_\text{teflon} - v_\text{void})\,s_i = (1-s_i)\,v_\text{void} + s_i\,v_\text{teflon},
$$

and the full field keeps hard steps at the two fixed boundaries:

$$
\text{field}(x_i) = \begin{cases}
v_\text{core}, & r_i < r_c,\\
\text{shell}_i, & r_c \le r_i \le r_c + w,\\
v_\text{shield}, & r_i > r_c + w .
\end{cases}
$$

`assign_void_teflon_properties()` calls this three times, once each for
$(D, \Sigma_a, q)$, with the same $s_i$. The three arrays drop straight into the
unmodified `assemble_system()`, and `solve_void_teflon()` chains mesh →
smooth assignment → assemble → `spsolve`.

As $a\to\infty$ the smooth field and the flux converge to the hard-step
reference; `demo_switch.py` shows this (`output_graphs/void_teflon_switch_demo.png`).

> **Modeling note.** At the fixed Core/Void boundary the blend is not exactly
> the Void value: $s(r_c) = \sigma(-a\,t_v)$. When the void is thin compared
> with $1/a$, cells just outside the Core are already a partial Void/Teflon
> mix. In the rescaled studies ($a = 20$ cm$^{-1}$) at the smallest sampled
> void ($t_v = 0.01$ cm), $s(r_c)\approx 0.45$. This is a consequence of
> smoothing an interface that is thinner than the transition, and the
> derivatives in §5 are exact for the field as defined.

## 4. ROM pipeline with geometry parameters

The parameter vector is $\mu = (t_v,\ a)$, $p = 2$. The region material values
$(D, \Sigma_a, q)$ are fixed placeholders, not measured cross sections:
Core $(1.0, 0.5, 1.0)$, Void $(1.8, 0.01, 0)$, Teflon $(0.6, 0.4, 0)$,
Shield $(0.3, 0.9, 0)$. Only the Core has a source.

As everywhere in the repo, either parameter is fixed by giving it degenerate
bounds. `steepness_bounds` defaults to `(200, 200)` (fixed) in the pipeline
driver; pass a real range to make the sharpness an active, sampled,
adjoint-differentiated parameter.

- **`params.py`**: `GeometryParamSampler` draws $(t_v, a)$ by LHS (training)
  or uniformly (test) using `common/lhs_generation.py::latin_hypercube_sample()`.
  `active_names` lists the non-fixed parameters.
- **`svd_analysis_geometry.py`**: the Stage-1 analog of
  `common/svd_analysis.py`. Each snapshot comes from `solve_void_teflon()`.
  It writes the standard five Stage-1 files, so later stages don't need to
  know what the parameters mean.
- **`sklearn_implementation/gpr_pod.py`** is reused as is for standard GPR.
- **`run_pipeline_geometry.py`** chains SVD → GPR → adjoint.

## 5. Adjoint sensitivities

This is the new derivation, in `adjoint_geometry.py`. It is written for
either parameter, $\mu_j\in\{t_v, a\}$.

### 5.1 Starting point

Exactly as for the material problem,

$$
\alpha_k = \psi_k^\top H\phi = g_k^\top\phi,\qquad A^\top\lambda_k = g_k,\qquad
\frac{\partial\alpha_k}{\partial\mu_j} = -\lambda_k^\top\frac{\partial A}{\partial\mu_j}\phi + \lambda_k^\top\frac{\partial s}{\partial\mu_j}.
$$

`build_adjoint_rhs`, `solve_adjoint`, and `alpha_via_adjoint` are reused
unchanged. What changes is that $A$ and $s$ now depend on $\mu_j$ through
**every shell cell**, via the smooth fields, rather than through a few
per-layer scalars.

### 5.2 Chain rule over cells and channels

$A$ and $s$ are assembled from the per-cell fields $D_i$, $\Sigma_{a,i}$, $q_i$,
all of which depend on $\mu_j$. The multivariable chain rule gives

$$
\frac{\partial\alpha_k}{\partial\mu_j}
= \underbrace{\sum_i \frac{\partial\alpha_k}{\partial D_i}\frac{dD_i}{d\mu_j}}_{D\ \text{channel}}
+ \underbrace{\sum_i \frac{\partial\alpha_k}{\partial \Sigma_{a,i}}\frac{d\Sigma_{a,i}}{d\mu_j}}_{\Sigma_a\ \text{channel}}
+ \underbrace{\sum_i \frac{\partial\alpha_k}{\partial q_i}\frac{dq_i}{d\mu_j}}_{q\ \text{channel}} .
$$

This is the ordinary rule $\tfrac{d}{dt}f(x(t),y(t)) = f_x\dot x + f_y\dot y$,
with the fields as $N$-vectors. In code it is literally
`total = dD_term + dSa_term + dq_term` in `sensitivity_from_directions()`.
Two ingredients are needed: (a) the field derivatives $d(\text{field})/d\mu_j$,
which differ by parameter, and (b) the per-cell adjoint sensitivities, which
don't.

### 5.3 (a) Field derivatives

Write the switch argument as $u = a(r - b)$, so $s = (1+e^{-u})^{-1}$.

**Step 1, the logistic derivative.**

$$
\frac{ds}{du} = \frac{e^{-u}}{(1+e^{-u})^2} = \underbrace{\frac{e^{-u}}{1+e^{-u}}}_{1-s}\cdot\underbrace{\frac{1}{1+e^{-u}}}_{s} = s(1-s).
$$

This is the only place the exponential appears; everything else is chain
rule on top of it.

**Step 2, derivatives of $u$.** Cell position $r$ is a fixed mesh coordinate:
$\partial u/\partial a = r - b$ and $\partial u/\partial b = -a$.

**Step 3, one parameter at a time.** $a$ depends only on steepness and $b$
only on void thickness ($db/dt_v = 1$), so each total derivative has a single
term:

$$
\frac{ds}{dt_v} = s(1-s)\cdot(-a),\qquad
\frac{ds}{da} = s(1-s)\cdot(r - r_\text{int}).
$$

The void-thickness derivative is **one-signed**: a positive bump
$s(1-s)$ at the interface times $-a$. The steepness derivative is
**odd about the interface**, because $r - r_\text{int}$ changes sign there:
sharpening pushes cells on the two sides in opposite directions away from the
halfway blend.

**Step 4, back to the field.** The blend is affine in $s$ with constant
endpoints, so

$$
\frac{d\,\text{field}_i}{d\mu_j} = (v_\text{teflon}-v_\text{void})\,\frac{ds_i}{d\mu_j}\quad\text{in the shell},\qquad 0\ \text{in Core and Shield}.
$$

**Step 5, three channels.** $s_i$ and $ds_i/d\mu_j$ don't depend on which
property is being blended, so `geometry_field_derivatives(param_name, ...)`
calls `field_derivative_wrt_void_thickness()` or
`field_derivative_wrt_steepness()` three times, once per
$(v_\text{void}, v_\text{teflon})$ pair, to get the direction vectors
$dD/d\mu_j$, $d\Sigma_a/d\mu_j$, $dq/d\mu_j$.

The fixed boundaries at $r_c$ and $r_c + w$ need no correction term: neither
moves with $t_v$ or $a$, so all dependence lives in the smooth shell term.

### 5.4 (b) Per-cell adjoint sensitivities

The per-layer formulas in `common/adjoint_solver.py` are directional
derivatives along a 0/1 layer indicator. Replacing the indicator with a
general per-cell direction $v$ gives the three channel terms:

$$
\begin{aligned}
q:&\quad \sum_{i\ \text{interior}} (\lambda_k)_i\,h_i\,\frac{dq_i}{d\mu_j},\\
\Sigma_a:&\quad -\sum_{i\ \text{interior}} (\lambda_k)_i\,\phi_i\,h_i\,\frac{d\Sigma_{a,i}}{d\mu_j},\\
D:&\quad -\sum_f\Big[\frac{\partial\beta_f}{\partial D_i}v_i + \frac{\partial\beta_f}{\partial D_{i+1}}v_{i+1}\Big](\phi_i-\phi_{i+1})(\lambda_i-\lambda_{i+1}),\qquad v = \frac{dD}{d\mu_j}.
\end{aligned}
$$

`sensitivity_D_directional()` is a vectorized version of the D formula. It
reduces exactly to the per-layer formula when $v$ is a layer indicator, so it
is a strict generalization. `sensitivity_from_directions()` adds the three
channels and also returns them separately; `sensitivity_wrt_param()` looks up
the right field derivative and calls it. For the placeholder materials here,
the $D$ channel dominates for both parameters, which fits the fact that the
largest Void/Teflon contrast is in $D$ (1.8 vs. 0.6).

One forward solve and one adjoint solve per mode give the derivative for
both parameters at once.

## 6. Verification (`adjoint_geometry.main()`)

Three checks run for every active parameter, mirroring
`sklearn_implementation/adjoint_stage.py`:

1. **Identity check** (`adjoint_identity_geometry.png`): $\psi_k^\top H\phi$ vs.
   $\lambda_k^\top s$ on training samples.
2. **Finite-difference check** (`adjoint_gradient_check_geometry.png`, one
   panel per active parameter): analytic gradient vs. central difference. The
   step is $10^{-5}\max(\lvert\mu_j\rvert, 10^{-3})$, clamped by `_fd_eps()` to at
   most 40% of the distance to the sampling bounds (and, for $t_v$, to
   $[0, w]$).
3. **1D sweeps** (`adjoint_sweep_<param>.png`): $\alpha_k(\mu_j)$ and its adjoint
   gradient over the sampling range, with a tangent line at the midpoint
   (or `expansion_point`). Other parameters sit at the middle of their bounds.

Measured on the original configuration when this module was written
($L = 1$, `n_cells = 3000`): identity error about $10^{-10}$; FD relative error
$10^{-6}$ to $10^{-5}$ with only $t_v$ active (worst case about $10^{-4}$ near
the bounds, where the FD step itself is less accurate); $10^{-6}$ to
$3\times10^{-5}$ for both parameters with `steepness_bounds=(100, 400)`. The
steepness sweep also shows the sensitivity to $a$ shrinking as $a$ grows,
because the field is already close to its hard-step limit.

## 7. GE-GPR on the geometry parameters (`ge_gpr_geometry.py`)

`ge_gpr_geometry.main()` runs Stage 1 (`svd_analysis_geometry.main()`,
unless `run_svd_stage=False`), then fits GE-GPR and standard GPR per POD mode,
exactly as `sklearn_implementation/ge_gpr.py` does
([math in its README §4](../sklearn_implementation/README.md#4-gradient-enhanced-gpr-ge_gprpy)).
The only difference is where the training gradients come from:
`_compute_training_gradients()` calls `sensitivity_wrt_param()` for each
active parameter after one forward and $R$ adjoint solves per sample.
`GEGaussianProcess`, the scaling helper, and the plotting helpers are
imported unchanged.

Plots are labeled in terms of the switch coefficients $(a, b)$ rather than
$(a, t_v)$. Since $b = r_c + t_v$ is a constant shift, $\partial\alpha_k/\partial b =
\partial\alpha_k/\partial t_v$ and the $[0,1]$ scaling is unchanged, so this is a
relabeling only. `gegpr_error_map_ab.png` scatters each test point's error
over the $(b, a)$ plane. `gegpr_flux_reconstruction.png` marks every material
boundary of each plotted sample that falls in the window (via
`plotting.draw_material_boundaries()` / `geometry.material_boundaries()`).

At least one of the two parameters must be active. The module docstring still
says both must be; that guard was relaxed for the one-at-a-time comparison.

## 8. Configurations and caveats

### 8.1 Two baselines

| | Module defaults | Paper baseline (`run_2param_geometry.py`, `../gpr_vs_gegpr_comparison/`, `../combined_study/`) |
|---|---|---|
| $L$, $r_c$, $w$, shield | 1.0, 0.16921, 0.25278, 0.07801 | all ×10: 10, 1.6921, 2.5278, 0.7801 |
| `n_cells` | 3000 (pipeline, SVD, adjoint), 3500 (GE-GPR) | 200 |
| $t_v$ bounds | (0.001, 0.25) | (0.01, 2.5) |
| steepness | 200 fixed, or (100, 400) | 20 fixed, or (10, 40) (units 1/length, so ÷10) |
| QoI window | default (see §8.2) | $x\in[3, 7]$, same as the material studies |
| $\eta$ | 0.9999 (pipeline), 0.99999 (GE-GPR) | 0.99999 |

The rescaling puts the geometry on the same $L=10$ domain, mesh, and QoI window
as the material studies, so errors are directly comparable.
`run_2param_geometry.py` runs the both-active case on that baseline
($M_\text{train}=300$, $M_\text{test}=60$, `m_ge=100`) and writes
`output_graphs_geometry_2param/`.

### 8.2 The default QoI window is not the shell

When `qoi_xmin`/`qoi_xmax` are not given, `svd_analysis_geometry.py`,
`adjoint_geometry.py`, and `ge_gpr_geometry.py` default to
$x\in[r_c,\ r_c + w]$. The comment says this is "the entire Void+Teflon shell on
the right half of the slab", but $r_c$ and $w$ are radii measured from the slab
center, and the window is applied in $x$. With the default $L=1$ geometry the
window is $x\in[0.169, 0.422]$, which is $r\in[0.078, 0.331]$: the outer 0.09 of
the Core plus the inner part of the **left** shell. The right-half shell would
be $x\in[L/2 + r_c,\ L/2 + r_c + w] = [0.669, 0.922]$. Runs that use the
defaults (`run_pipeline_geometry.py`, `output_graphs_geometry/`) therefore
see a window that contains the moving interface only for
$t_v < 0.162$, about 65% of the sampled range. The paper-baseline runs pass
an explicit window and are not affected by this default.

### 8.3 What the paper-baseline window sees

The shared window $x\in[3, 7]$ reaches $r = 2.0$. With $r_c = 1.6921$, the
moving interface is inside the window only when $t_v < 0.308$, which is about
12% of the sampled range $[0.01, 2.5]$. For most samples the window holds the
Core and part of the Void, and the geometry affects the QoI through the flux
solution rather than through material changes inside the window.
`run_comparison.py` notes that this window covers part of the Core; the
limited overlap with the interface is worth keeping in mind when reading the
geometry results.

### 8.4 Mesh resolution of the switch

The rule of thumb for this switch is at least 10–20 cells across $1/a$. On the
paper baseline ($h = 0.05$), $1/a$ is 2 cells at $a = 10$, 1 cell at $a=20$,
and 0.5 cells at $a = 40$; the 10%–90% width is 8.8, 4.4, and 2.2 cells. The
adjoint gradients are still exact for the discrete model, but at the steep end
the discrete field is close to a hard step and the smooth-geometry
interpretation is weak. The module-default mesh ($L=1$, 3000 cells,
$a=200$) has 15 cells across $1/a$.

## 9. File map and running

| File | Role |
|---|---|
| `geometry.py` | `sigmoid_switch`, `void_teflon_field`, `assign_void_teflon_properties`, `material_boundaries`, `solve_void_teflon` |
| `params.py` | `GeometryParamSampler` for $\mu = (t_v, a)$ |
| `svd_analysis_geometry.py` | Stage 1: snapshots, POD, standard file outputs, plots with material boundaries marked |
| `adjoint_geometry.py` | Field derivatives, `sensitivity_D_directional`, `sensitivity_from_directions`, `sensitivity_wrt_param`; identity/FD/sweep verification in `main()` |
| `ge_gpr_geometry.py` | GE-GPR vs. standard GPR over the active geometry parameters, learning curve, $(a,b)$ error map |
| `run_pipeline_geometry.py` | SVD → standard GPR (`sklearn_implementation/gpr_pod.py`) → adjoint verification, on module defaults |
| `run_2param_geometry.py` | Both parameters active on the paper baseline → `output_graphs_geometry_2param/` |
| `demo_switch.py` | Smooth vs. hard field and flux as steepness increases → `output_graphs/void_teflon_switch_demo.png` |
| `plotting.py` | `draw_material_boundaries()`: dashed lines at boundaries inside the current axes |

| Output directory | Produced by |
|---|---|
| `output_graphs/` | `demo_switch.py` |
| `output_graphs_geometry/` | `run_pipeline_geometry.py` (module defaults, $t_v$ only) |
| `output_graphs_geometry_2param/` | `run_2param_geometry.py` (paper baseline). The `gpr_sklearn_*` and `adjoint_*` plots here come from an earlier run; `run_2param_geometry.py` does not regenerate them |
| `output_graphs_geometry_gegpr/` | an earlier `ge_gpr_geometry.main()` run (no driver script) |

```bash
cd void_teflon_geometry
python3 demo_switch.py
python3 run_pipeline_geometry.py        # module defaults
python3 run_2param_geometry.py          # paper baseline, both parameters active
```

To make steepness active in the default pipeline, keep enough cells across
the transition:

```python
from run_pipeline_geometry import main
main(steepness_bounds=(100.0, 400.0), n_cells=4000)
```
