# Void/Teflon Geometry ROM — Smooth Switch and Adjoint Sensitivities

This directory mirrors the ROM workflow in `sklearn_implementation/` (and the underlying sandbox in `common/`), but the active parameter is a geometry parameter, instead of a per-layer material property. It reuses the existing mesh/assembly/solve and GPR infrastructure unmodified, and adds only what's new: a differentiable material switch and the adjoint math needed to differentiate through it.

This document explains, in order:

1. the physical geometry and why it needs its own module,
2. why a hard piecewise-constant assignment can't be differentiated w.r.t.
   its own boundary location,
3. the logistic ("sigmoid") switch used to fix that, and how it's built into
   a full material-property field,
4. how the ROM pipeline (snapshots → POD → GPR) runs with the geometry/switch
   parameter vector `mu = (void_thickness, steepness)`,
5. the full derivation of the adjoint sensitivity `dα_k/d(param)` for
   *either* parameter, channel by channel,
6. how each piece is numerically verified, and
7. the file map and how to reproduce the results.

**Both `void_thickness` and `steepness` are parameterizeable** in the same
sense: each can be made an active, LHS-sampled, adjoint-differentiated
parameter by giving it a real `(lo, hi)` range, or held fixed at a constant
via degenerate `lo == hi` bounds — see §4 and §5.

---

## 1. The geometry

The dataset in `data/constantR_void_10/` (`case_key.txt`, `void.py`)
describes a 1D slab, mirrored about its center, built from four materials in
this order from the outer boundary inward:

```
Shield | Dielectric (Teflon) | Void | Core | Core | Void | Dielectric (Teflon) | Shield
        <---------------- half-slab, radial distance r from center -------------->
```

The two central `Core` regions are the same material, so materially this is
just a single Core region of half-width `core_radius` centered on the slab.
Writing `r = |x − L/2|` (distance from the slab midpoint), the four regions
occupy, moving outward from the center:

| Region  | Radial range                                                   | Fixed or moving? |
|---------|----------------------------------------------------------------|------------------|
| Core    | `0 ≤ r < core_radius`                                          | fixed            |
| Void    | `core_radius ≤ r < core_radius + void_thickness`               | **moving** outer edge |
| Teflon  | `core_radius + void_thickness ≤ r < core_radius + shell_width` | **moving** inner edge |
| Shield  | `core_radius + shell_width ≤ r ≤ L/2`                          | fixed            |

Across the `constantR_void_10` case sweep, `void_thickness` and
`dielectric_thickness` trade off so that their sum

```
shell_width = void_thickness + dielectric_thickness
```

stays fixed, and so do `core_radius` and `shield_thickness`. That means
**only one boundary actually moves**: the Void/Teflon interface at
`r = core_radius + void_thickness`. The Core/Void boundary (`r = core_radius`)
and the Teflon/Shield boundary (`r = core_radius + shell_width`) never move.
`void_thickness` is therefore the single physically meaningful "geometry"
parameter for this study.

---

## 2. Why the hard assignment isn't differentiable

`common/second_order_solver.py::assign_material_properties()` assigns a
material property to cell `i` by testing which fixed layer bin its center
`x_i` falls into (`np.searchsorted`). This is exactly right for a *fixed*
geometry, but breaks down the moment the boundary itself becomes a parameter
we want to differentiate with respect to:

- For a **cell that stays on the same side** of the boundary as
  `void_thickness` changes, the assigned value doesn't change at all — its
  derivative is exactly 0.
- For the (at most one or two) **cells whose side flips** as the boundary
  sweeps past them, the assigned value jumps discontinuously between the
  Void value and the Teflon value — the derivative is undefined there.

So the "gradient" of any QoI with respect to `void_thickness`, computed this
way, is a sum of a great many exact zeros and an occasional undefined jump —
not usable by an adjoint method, and not something a finite-difference check
can validate either (the FD result depends on which cell the perturbed
boundary happens to land in, not on the true continuum sensitivity).

## 3. The smooth switch

`geometry.py` replaces the hard step **only at the moving Void/Teflon
interface** with the logistic activation

```
sigmoid_switch(x; a, b) = 1 / (1 + exp(-a·(x - b)))
```

`sigmoid_switch` → 0 as `a·(x-b) → −∞`, → 1 as `a·(x-b) → +∞`; `sigmoid_switch(b) = 0.5`
exactly, for any `a`. `a` is strictly the steepness of the transition and
`b` is strictly the interface's physical location — no conversion step is
needed, `a` and `b` are passed to `sigmoid_switch()` directly. For the
Void/Teflon interface, `b = r_interface = core_radius + void_thickness`, so

```
b(void_thickness) = core_radius + void_thickness
```

— `b` is a *smooth, linear* function of `void_thickness`, and this is the
single fact that makes the whole chain rule in Section 5 work for that
parameter. `steepness` (`a`) is not just a fixed numerical knob either: it
appears explicitly in `void_teflon_field()`'s signature and in
`solve_void_teflon()`'s, so every forward solve already takes it as a real
argument. What §4–§5 add is treating it as a first-class *parameter* of the
ROM — sampled, learned, and adjoint-differentiated exactly like
`void_thickness` — rather than a constant baked into the pipeline scripts.

### What `s` is

Define, for each cell `i` with radial coordinate `r_i = |x_i - L/2|`,

```
s_i := sigmoid_switch(r_i; a, b) = 1 / (1 + exp(-a·(r_i - b)))   ∈ (0, 1)
```

This is a single scalar per cell, purely geometric — it depends only on the
cell's position `r_i` and the switch's `(a, b)`, and knows nothing about
`D`, `Σₐ`, or `q`. Read it as **"how far past the interface is cell `i`,
on a 0-to-1 scale"**:

- `s_i ≈ 0` for cells deep on the Void side (`r_i ≪ b`),
- `s_i ≈ 1` for cells deep on the Teflon side (`r_i ≫ b`),
- `s_i = 0.5` exactly at the interface (`r_i = b`),
- the width of the transition between "≈0" and "≈1" is set by `1/a` — a
  steeper `a` squeezes the S-shaped rise into a narrower band of `r`.

It's the exact continuous analog of the hard step's 0/1 indicator "which
side of the boundary is cell `i` on" — except it rises smoothly through
every value in between instead of jumping.

### From `s` to a material field

`void_teflon_field()` turns that single geometric quantity `s_i` into a
per-cell value of *any* material property (D, Σₐ, or q) with one line:

```
shell_val_i = val_void + (val_teflon - val_void) * s_i
```

This is nothing more than a **weighted average of the two endpoint values,
weighted by `s_i`** — equivalently `shell_val_i = (1-s_i)·val_void +
s_i·val_teflon`. It's the unique affine (straight-line) blend that
reproduces the two limits a hard step would give: `s_i = 0` returns exactly
`val_void`, `s_i = 1` returns exactly `val_teflon`, and every intermediate
`s_i` gives a proportional mix. The full field, including the fixed hard
boundaries at Core and Shield (§1), is then:

```
r = |x_centers - L/2|
r_interface = core_radius + void_thickness
s(r) = sigmoid_switch(r; steepness, r_interface)
shell_val(r) = val_void + (val_teflon - val_void) * s(r)

field(x) = val_core                              if r < core_radius
         = shell_val(r)                          if core_radius ≤ r ≤ core_radius + shell_width
         = val_shield                            if r > core_radius + shell_width
```

Because `s` doesn't reference `D`, `Σₐ`, or `q` at all, `void_teflon_field()`
is called **three separate times** with three different `(val_void,
val_teflon)` pairs — once for `D`, once for `Σₐ`, once for `q` — to build
the three cell arrays `assign_void_teflon_properties()` returns. The
geometric part of the computation (`s_i`, and later `ds_i/dp`) is identical
across all three calls; only the two endpoint values being blended change.
This is the direct answer to "how do you go from `s` to the field
parameters": the field parameters (`D_i`, `Σₐ,i`, `q_i`) are just three
independent linear interpolations between their own two material constants,
all driven by the same shared geometric weight `s_i`.

Only the **shell interior** uses the sigmoid; the Core/Void and
Teflon/Shield boundaries are still hard steps, because those don't move in
this study (see the table in §1) — there's no need to smooth a boundary that
never changes, and doing so would only cost accuracy for no differentiability
benefit.

`demo_switch.py` verifies this against the exact hard `case_key.txt`-style
geometry: as `steepness → ∞`, the smooth field and the resulting flux
converge pointwise to the hard reference (see
`output_graphs/void_teflon_switch_verification.png`; the earlier chat turn's
plot showed the same convergence with the flux overlay).

`assign_void_teflon_properties()` calls `void_teflon_field()` three times
(once each for D, Σₐ, q) and returns cell arrays that plug directly into the
**unmodified** `assemble_system()` from `common/second_order_solver.py`.
`solve_void_teflon()` chains mesh → smooth assignment → assemble → `spsolve`
into one forward-solve call, reusing every piece of the existing solver
except the material-assignment step.

---

## 4. Running the ROM pipeline with geometry/switch parameters

The parameter vector is

```
mu = (void_thickness, steepness)          p = 2
```

Material property values for the four regions (`D`, `Σₐ`, `q` each for Core,
Void, Teflon, Shield) are held fixed constants for this study — they are
illustrative placeholders (documented in `svd_analysis_geometry.main()`'s
defaults), not measured cross sections, since no physical values were
supplied for these materials.

Following the workflow document's Section 3 design choice — *"the code
should not contain logic to declare parameters uncertain versus fixed"* —
`steepness` is handled exactly like `void_thickness`, or like a material
property in `DiffusionParamSampler`: a parameter is fixed simply by giving
it degenerate `(v, v)` bounds. `steepness_bounds` defaults to
`(200.0, 200.0)` (fixed, matching the original single-value study); pass a
real range, e.g. `steepness_bounds=(100.0, 400.0)`, to make the sigmoid
sharpness itself an active, LHS-sampled, adjoint-differentiated parameter.

- **`params.py`** — `GeometryParamSampler` samples `(void_thickness,
  steepness)` via LHS (training) or uniform random (test), built directly on
  the existing, unmodified `common/lhs_generation.py::latin_hypercube_sample()`.
  `sampler.active_names` reports which of the two are actually varying (the
  same `_fixed_mask` pattern `DiffusionParamSampler` uses for materials).
- **`svd_analysis_geometry.py`** — mirrors `common/svd_analysis.py`
  (Sections 6–8 of the workflow document) exactly, except each snapshot
  comes from `solve_void_teflon()` instead of `solve_diffusion()`, using
  each sample's own `(void_thickness, steepness)` pair. The QoI mask is the
  entire Void+Teflon shell on one side,
  `[core_radius, core_radius + shell_width]` — the one region whose material
  composition actually depends on these parameters. Writes the same file
  formats (`samples_*.txt`, `qoi_values_*.txt`, `pod_basis.txt`,
  `coefficients_train.txt`) as the material-parameter pipeline.
- **`gpr_pod.py`** (in `sklearn_implementation/`, **reused unmodified** other
  than a one-line robustness fix — see §7) — trains one GP per POD mode on
  `mu → α_k`. It only reads those same files and never inspects what the
  columns of `mu` physically mean, so it works here with zero changes to its
  modeling logic, whether `p=1` (only `void_thickness` active) or `p=2`
  (both active).
- **`run_pipeline_geometry.py`** — chains the three stages, mirroring
  `sklearn_implementation/run_pipeline.py`.

With `steepness` fixed (the default), there is only one active parameter, so
the QoI snapshots trace out (to leading order) a 1-D curve in `R^Ny`:
running the pipeline confirmed this — mode 1 alone captures 99.93% of the
SVD energy, and rank `R=2` reaches 99.99%. That's the expected signature of
a genuinely 1-parameter family, and it's why the GPR surrogate's error is
essentially at the POD projection floor (`ε_y ≈ ε_proj`, since a smooth 1-D
map is easy for a GP to fit tightly). Activating `steepness` as a second
parameter (verified separately — see §6) raises this to a genuine 2-D
manifold; the same POD/GPR/adjoint machinery applies unchanged, since none
of it is written in terms of a fixed parameter count.

---

## 5. Adjoint sensitivity: `dα_k / d(param)`

This is the core new derivation, in `adjoint_geometry.py`. It's written
generically over `param ∈ {"void_thickness", "steepness"}` — whichever
parameters `GeometryParamSampler` reports as active (§4) are the ones
`adjoint_geometry.main()` actually FD-checks and sweeps.

### 5.1 What we're differentiating

Following the workflow document's Sections 13–15, each reduced coefficient
is a scalar QoI

```
α_k(mu) = ψ_k^T y(mu) = ψ_k^T H φ(mu) = g_k^T φ(mu),      g_k := H^T ψ_k
```

and its sensitivity to any scalar parameter `p` follows from differentiating
the forward equation `A(mu) φ(mu) = s(mu)` and eliminating `∂φ/∂p` with the
adjoint `A(mu)^T λ_k(mu) = g_k`:

```
∂α_k/∂p = -λ_k^T (∂A/∂p) φ + λ_k^T (∂s/∂p)
```

(eq. 13 in the workflow doc). `build_adjoint_rhs`, `solve_adjoint`, and
`alpha_via_adjoint` — reused unmodified from `common/adjoint_solver.py` —
implement `g_k`, the adjoint solve, and the identity check, exactly as in
the material-parameter pipeline, and don't change at all between the two
parameters. What's new here is `∂A/∂p` and `∂s/∂p` for `p ∈ {void_thickness,
steepness}`, since `A` and `s` no longer depend on either parameter through
a small number of per-layer scalars — they depend on it through *every cell
in the shell*, via the smooth field.

### 5.2 Chain rule through the sigmoid

`A` and `s` are assembled from the per-cell fields `D_i`, `Σₐ,i`, `q_i`, so
for either parameter `p`, by the chain rule:

```
∂α_k/∂p = Σ_i [ ∂α_k/∂D_i · dD_i/dp + ∂α_k/∂Σₐ,i · dΣₐ,i/dp + ∂α_k/∂q_i · dq_i/dp ]
```

So two things are needed: (a) the per-cell field derivatives `dD_i/dp` etc.
(different for each parameter — derived below), and (b) the per-cell
adjoint sensitivities `∂α_k/∂D_i` etc. (§5.2b — identical machinery for
either parameter, since it only cares about the resulting direction
vector).

**(a) Field derivatives.** This is where `s` from §3 gets differentiated.
Write the switch's argument as `u := a·(r - b)`, so `s = 1/(1 + exp(-u))`
(this is just `sigmoid_switch` written with its argument named).

*Step 1 — differentiate the logistic function itself, `ds/du`.* This is
the one piece of calculus every other derivative here is built from, so it's
worth deriving explicitly rather than quoting it. From
`s = (1 + e^{-u})^{-1}`, the chain rule gives

```
ds/du = -(1 + e^{-u})^{-2} · (-e^{-u}) = e^{-u} / (1 + e^{-u})^2
```

Split that as a product of two factors, each of which is a familiar quantity:

```
e^{-u} / (1 + e^{-u})^2  =  [ e^{-u} / (1 + e^{-u}) ]  ·  [ 1 / (1 + e^{-u}) ]
                         =  (1 - s)                    ·  s
```

using `1/(1+e^{-u}) = s` and `e^{-u}/(1+e^{-u}) = 1 - 1/(1+e^{-u}) = 1 - s`.
So

```
ds/du = s(1 - s)
```

— the standard logistic derivative, and the only place `exp()` shows up in
any of what follows; every other step below is plain chain rule on top of
this one fact.

*Step 2 — differentiate `u = a·(r - b)` w.r.t. each parameter.* Cell
position `r` is fixed (it's a mesh coordinate, not something we
differentiate with respect to); `a` and `b` are the two things that can
depend on a parameter `p`:

```
∂u/∂a = (r - b)          ∂u/∂b = -a
```

*Step 3 — plug in how `a` and `b` depend on each parameter.* Recall from §3:
`a = steepness` and `b = r_interface = core_radius + void_thickness` — i.e.
`a` depends only on `steepness`, `b` depends only on `void_thickness`, and
neither depends on the other. That makes each parameter's total derivative
of `u` a single term, not a sum:

- **`p = void_thickness`:** `a` doesn't depend on `vt` (`da/d(vt) = 0`);
  `b` does, with `db/d(vt) = 1`. So
  `du/d(vt) = (∂u/∂b)·(db/d(vt)) = -a · 1 = -a`, and by the chain rule
  through Step 1,

  ```
  ds/d(vt) = (ds/du) · (du/d(vt)) = s(1-s) · (-a)
  ```

- **`p = steepness`:** `b` doesn't depend on steepness (`db/d(a) = 0`); `a`
  *is* steepness, so `da/d(a) = 1`. So
  `du/d(a) = (∂u/∂a)·(da/d(a)) = (r - b) · 1 = r - r_interface`, and

  ```
  ds/d(a) = (ds/du) · (du/d(a)) = s(1-s) · (r - r_interface)
  ```

Two different results from the exact same `ds/du = s(1-s)` fact, because
`void_thickness` and `steepness` enter `u` through different variables
(`b` vs. `a`) with different partial derivatives (`-a` vs. `r - b`).
`void_thickness`'s derivative is **one-signed** (constant sign `-a` times a
positive bump `s(1-s)`); `steepness`'s is **odd-symmetric** about the
interface (`r - r_interface` flips sign there), so increasing `steepness`
pushes Void-side and Teflon-side cells in *opposite* directions relative to
the halfway blend — exactly "sharpening the transition." The measured sweep
(§6) shows this directly: `α_1`'s sensitivity to `steepness` shrinks
monotonically in magnitude as `steepness` grows, since the field is already
close to its hard limit and further sharpening changes less and less of it.

*Step 4 — from `ds/dp` to `dfield/dp` (back to the field, §3).* Since
`shell_val = val_void + (val_teflon - val_void)·s` is affine in `s` with
constants `val_void`, `val_teflon` that don't depend on `p`, differentiating
it is immediate:

```
dfield_i/dp = (val_teflon - val_void) · ds_i/dp,        for cells in the shell
            = 0,                                          for cells in Core or Shield (fixed hard values, §1)
```

which, substituting the two `ds/dp` results from Step 3, gives exactly:

```
dfield_i/d(vt) = (val_teflon - val_void) · s_i(1-s_i) · (-a)
dfield_i/d(a)  = (val_teflon - val_void) · s_i(1-s_i) · (r_i - r_interface)
```

*Step 5 — from one field to three (`dD`, `dΣₐ`, `dq`).* Exactly as in §3,
`s_i` and `ds_i/dp` are purely geometric and don't know which physical
property is being differentiated — only the `(val_void, val_teflon)` prefix
in Step 4 changes per property. So `field_derivative_wrt_void_thickness()`
/ `field_derivative_wrt_steepness()` are each called three times inside
`geometry_field_derivatives(param_name, ...)`, once per `(D_void, D_teflon)`,
`(Σₐ,void, Σₐ,teflon)`, `(q_void, q_teflon)` pair, producing the three
direction vectors `dD_i/dp`, `dΣₐ,i/dp`, `dq_i/dp` that feed into §5.2(b).

Note there is no extra "boundary-motion" correction term at `r = core_radius`
or `r = core_radius + shell_width` for *either* parameter: neither boundary
moves with `void_thickness` or `steepness` (§1), so the hard cutoffs there
contribute zero derivative on their own — the entire dependence on both
parameters lives inside the smooth interior term, which is exactly what's
captured above.

**(b) Per-cell adjoint sensitivities.** The existing per-*layer* formulas in
`common/adjoint_solver.py` are:

```
∂α_k/∂q_m    = Σ_{i in layer m}      (λ_k)_i · Δx_i                          (eq. 14)
∂α_k/∂Σₐ,m   = -Σ_{i in layer m}     (λ_k)_i · φ_i · Δx_i                    (eq. 15)
∂α_k/∂D_m    = -Σ_faces f  [∂β_f/∂D_i · 1_{i∈m} + ∂β_f/∂D_{i+1} · 1_{i+1∈m}] (φ_i-φ_{i+1})(λ_i-λ_{i+1})
```

where `1_{i∈m}` is the 0/1 indicator "cell `i` belongs to layer `m`". These
are already *directional derivatives* of `α_k` in the direction of the
indicator vector for layer `m` — summing per-layer is just dotting with a
0/1 direction. Generalizing to **any** continuous per-cell direction vector
`v` (in particular, `v = dD_i/dp` for `p ∈ {void_thickness, steepness}` — a
smooth bump concentrated near the interface, one-signed or odd-signed
depending on which parameter, rather than a 0/1 step) gives exactly the
chain-rule sum from §5.2:

```
∂α_k/∂p via q      =    Σ_i∈interior (λ_k)_i · Δx_i · dq_i/dp
∂α_k/∂p via Σₐ     = -  Σ_i∈interior (λ_k)_i · φ_i · Δx_i · dΣₐ,i/dp
∂α_k/∂p via D       = -  Σ_faces f  [∂β_f/∂D_i · v_i + ∂β_f/∂D_{i+1} · v_{i+1}] (φ_i-φ_{i+1})(λ_i-λ_{i+1}),   v = dD/dp
```

`sensitivity_D_directional()` implements the D-channel formula vectorized
over faces (reusing the exact `∂β_f/∂D` analytic expressions from
`common/adjoint_solver.py::sensitivity_D`); it reduces exactly to the
original per-layer formula when `v` happens to be a 0/1 layer indicator, so
this is a strict generalization, not a different method — and it's entirely
agnostic to which parameter's direction vector it's handed. `interior` (rows
not overwritten by a Dirichlet BC) comes from the existing
`_interior_mask()` helper, reused unchanged.

`sensitivity_from_directions()` combines all three channels into the total
gradient given any `(dD, dΣₐ, dq)` triple, and also returns the per-channel
breakdown; `sensitivity_wrt_param(param_name, ...)` is the thin wrapper that
looks up the right field-derivative formula from §5.2a and calls it. The
channel breakdown is plotted separately (see §6) — this is useful
diagnostically: for this geometry, the D-channel (diffusion-coefficient
contrast between Void and Teflon) dominates the total sensitivity for both
parameters, with Σₐ and q contributing much less, which matches physical
intuition (the sharpest material contrast at this interface is in `D`, given
the chosen illustrative values).

---

## 6. Verification

Three independent checks are run for every *active* parameter, mirroring
`sklearn_implementation/adjoint_stage.py`:

1. **Identity check** (`adjoint_identity_geometry.png`): for several
   training samples, `α_k` computed directly (`ψ_k^T φ[mask]`) is compared
   against `α_k` computed via the adjoint identity `λ_k^T s`. These are two
   completely different computational paths that must agree exactly (up to
   solver tolerance) if the forward solve, mask, and adjoint solve are all
   self-consistent — this validates the machinery *before* trusting any
   gradient built on top of it, independent of which parameters are active.
   Measured: max relative error `~1e-10`.

2. **Finite-difference gradient check** (`adjoint_gradient_check_geometry.png`,
   one panel per active parameter): for several samples, the analytic
   `∂α_k/∂p` from §5 is compared against a central finite difference of the
   forward solve, `(α_k(p+ε) - α_k(p-ε)) / (2ε)`, with `ε` chosen small
   relative to `p` and clamped so the perturbed value stays inside that
   parameter's sampling bounds (and, for `void_thickness`, inside
   `[0, shell_width]` as well). With only `void_thickness` active: relative
   error `~1e-5`–`1e-6` for most samples (worst case `~1e-4` very close to
   the boundary, where the FD step itself becomes a less accurate
   approximation of the derivative). With **both `void_thickness` and
   `steepness` active** (`steepness_bounds=(100.0, 400.0)`, verified
   separately from the default run): relative error `~1e-6`–`3e-5` across 5
   random samples for both parameters simultaneously — confirming the
   `field_derivative_wrt_steepness()` derivation in §5.2 is correct, not
   just the `void_thickness` one.

3. **1D sweep** (`adjoint_sweep_<param>.png`, one file per active
   parameter — up to 2): `α_k(p)` and its adjoint gradient are evaluated on
   a fine grid over that parameter's full sampling range (all other
   parameters held at the midpoint of their own bounds — which is just their
   fixed value for a degenerate one), together with a first-order Taylor
   expansion about a chosen expansion point — the tangent line's slope is
   the adjoint gradient at that point, and it should (and does) match the
   local slope of the true curve. The `steepness` sweep additionally
   confirms the qualitative behavior predicted in §5.2: sensitivity to
   `steepness` shrinks in magnitude as `steepness` grows, since the field is
   already close to its hard-step limit.

---

## 7. File map

| File | Role |
|---|---|
| `geometry.py` | Sigmoid switch, smooth field builder, forward solve (reuses `common/second_order_solver.py` unchanged); `steepness` is already a plain function argument everywhere here |
| `params.py` | `GeometryParamSampler`: LHS/random sampler for `mu = (void_thickness, steepness)`, either entry fixable via degenerate bounds (reuses `common/lhs_generation.py` unchanged) |
| `svd_analysis_geometry.py` | Snapshot generation + QoI-side POD/SVD over whichever of `(void_thickness, steepness)` are active |
| `adjoint_geometry.py` | Field derivatives for *either* parameter, directional D-sensitivity, full `dα_k/dp`, identity/FD checks, 1D sweep — generic over the active-parameter set |
| `run_pipeline_geometry.py` | Orchestrates SVD → GPR (reuses `sklearn_implementation/gpr_pod.py`) → adjoint stages |
| `demo_switch.py` | Standalone visual check: smooth switch vs. hard `case_key.txt`-style geometry, converging as steepness increases |

`sklearn_implementation/gpr_pod.py` received one small, backward-compatible
fix: `np.loadtxt` collapses a single-column file to 1-D, which broke when
`p=1` (a single geometry parameter). Added `ndim` guards reshape it back to
2-D; behavior for the existing multi-parameter (material) studies is
unchanged.

### Running it

```bash
cd void_teflon_geometry
python3 run_pipeline_geometry.py
```

Outputs land in `output_graphs_geometry/` (override via `output_dir=` on
`main()`). To adjust the study, the parameters most likely to matter are:

- `void_bounds` — the sampled range of `void_thickness` (defaults to the
  same range as `data/constantR_void_10/void.py`, i.e. `(0.001, 0.25)` of
  a `shell_width = 0.25278`).
- `steepness_bounds` — sigmoid sharpness range. Defaults to
  `(200.0, 200.0)` (fixed, single-value study, as in the original run).
  Pass a real range, e.g. `steepness_bounds=(100.0, 400.0)`, to make
  `steepness` a second active, sampled, adjoint-differentiated parameter
  (verified in §6). Larger steepness is closer to the true hard geometry but
  needs more mesh resolution across the transition (rule of thumb: keep
  `n_cells` large enough that `1/steepness` spans at least ~10–20 cells) —
  so widening this range upward should generally go together with
  increasing `n_cells`.
- `mat_core`, `mat_void`, `mat_teflon`, `mat_shield` — the `(D, Σₐ, q)`
  triples per region; currently illustrative placeholders, not measured
  cross sections.

Example — run with `steepness` active as a second parameter:

```python
from run_pipeline_geometry import main
main(steepness_bounds=(100.0, 400.0), n_cells=4000)
```
