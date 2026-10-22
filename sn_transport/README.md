# `sn_transport/`: 1D slab S_N transport and inverse UQ of material properties

This directory is the transport counterpart of `common/` (diffusion forward
model and adjoint) for a different question: **inverse UQ**. A beam of known
direction and uncertain strength enters one face of a layered slab, the
exiting partial currents are measured on both faces, and the goal is to infer
the total cross section and scattering ratio of each material.

```
 | Shield | Dielectric | Core | Dielectric | Shield |
   0.5 cm    1.0 cm     1.0 cm   1.0 cm     0.5 cm      (DEFAULT_REGIONS)
```

There is no internal source. The experimenter controls the beam angle
$\mu_0 = \cos\vartheta$. There are 6 unknowns, plus the beam strength as a
nuisance parameter:

$$
\theta = (\Sigma_{t,S},\ \Sigma_{t,D},\ \Sigma_{t,C},\ c_S,\ c_D,\ c_C),\qquad c = \Sigma_s/\Sigma_t .
$$

$(\Sigma_t, c)$ is used instead of $(\Sigma_t, \Sigma_s)$ because every point of
the box $\Sigma_t>0,\ 0\le c\le 1$ is physical.

| File | Role | Diffusion analogue |
|---|---|---|
| `slab_params.py` | Materials, default geometry and prior bounds, `SlabParamSampler` (LHS / uniform, fixed-by-degenerate-bounds convention) | `common/lhs_generation.py` |
| `sn_solver.py` | Gauss-Legendre S_N, diamond difference, assembled system $A u = s$, sweeps + source iteration, exact uncollided beam, exit-current functionals; `__main__` verifies it | `common/second_order_solver.py` |
| `sn_adjoint.py` | Per-cell adjoint sensitivity formulas for $\Sigma_t$, $\Sigma_s$, $q$ and the uncollided first-collision source | `common/adjoint_solver.py` |
| `slab_experiment.py` | `SlabExperiment`: exiting currents and their full Jacobian for a list of beam angles; `__main__` checks the adjoint (identity + finite differences) | `sklearn_implementation/adjoint_stage.py` |
| `inverse_uq.py` | Driver: angle sensitivities, identifiability, angle-design comparison, MAP, Laplace, MCMC | none |

---

## 1. Forward model (`sn_solver.py`)

### 1.1 Equation and discretization

$$
\mu\frac{\partial\psi}{\partial x} + \Sigma_t(x)\,\psi = \tfrac12\big(\Sigma_s(x)\,\phi(x) + q(x)\big),\qquad
\phi = \int_{-1}^{1}\psi\,d\mu \approx \sum_n w_n\psi_n .
$$

- **Angle:** Gauss-Legendre $S_N$ (`n_angles` even, $\sum w_n = 2$).
- **Space:** diamond difference on a mesh built by `build_slab_mesh()`, with a
  face on every region interface. Material assignment is therefore exact, and
  there is no snapping like `common/`'s `searchsorted` rule.
- **Unknowns:** the outflow edge flux $e_{i,n}$ of every cell and direction.
  This is the right face if $\mu_n>0$ and the left face if $\mu_n<0$.
- **DD balance:** with $a_n=|\mu_n|$ and $b_i=\Sigma_{t,i}h_i/2$, the balance in
  cell $i$, direction $n$ is

$$
(a_n+b_i)\,e_{i,n} + (b_i-a_n)\,e^{\text{in}}_{i,n} - \tfrac{h_i\Sigma_{s,i}}{2}\,\phi_i = \tfrac{h_i q_i}{2},\qquad
\phi_i=\sum_m w_m\,\tfrac{e_{i,m}+e^{\text{in}}_{i,m}}{2}.
$$

  Here $e^{\text{in}}$ is the upwind cell's outflow edge, or the prescribed
  incoming flux `psi_inc[n]` on the boundary (zero means vacuum).

`assemble_system()` stacks every row into one sparse system $A u = s$, with
cell-major ordering $u_{iN_\text{ang}+n} = e_{i,n}$. This is the analogue of
`common`'s `assemble_system()`. The system is block tridiagonal with dense
$N_\text{ang}\times N_\text{ang}$ scattering blocks, so sparse LU is cheap:
$N=200$ and S16 give 3200 unknowns. `sweep()` / `source_iteration()` solve the
same discretization the classical way. They converge at a rate of about
$\max c$, with no acceleration, and are used only as a cross-check.

### 1.2 The beam: exact uncollided flux + S_N for the collided flux

A beam at an arbitrary $\mu_0$ is generally not an S_N direction, and diamond
difference handles a steeply attenuated grazing beam poorly. The flux is
therefore split. The uncollided part is analytic:

$$
\psi_u = \frac{J_0}{\mu_0}e^{-\tau(x)/\mu_0}\,\delta(\mu-\mu_0),\qquad
\bar\phi_{u,i} = \frac{J_0}{\mu_0}e^{-\tau_{i-1/2}/\mu_0}\,E\!\Big(\frac{\Sigma_{t,i}h_i}{\mu_0}\Big),\quad E(a)=\frac{1-e^{-a}}{a}.
$$

This cell average is exact (`uncollided_beam()`). The collided flux solves the
S_N problem above with vacuum inflow and the first-collision source
$q_i = \Sigma_{s,i}\bar\phi_{u,i}$. The measured currents are

$$
J_\text{left} = \sum_{\mu_n<0} w_n|\mu_n|\,\psi_n(0),\qquad
J_\text{right} = \sum_{\mu_n>0} w_n\mu_n\,\psi_n(L) + J_0\,e^{-\tau(L)/\mu_0}
$$

for a beam entering at $x=0$. `exit_current_functionals()` returns the
collided parts as vectors $g$ with $J^c = g^\top u$. A beam on the right face
is handled by mirroring the slab (`SlabExperiment(source_side="right")`).
Outputs are always reported as physical left/right.

### 1.3 Verification (`python3 sn_solver.py`)

| Check | Result |
|---|---|
| Pure absorber, uniform source, exact cell averages at fixed quadrature | error ratio per mesh doubling → 4.00 (second order) |
| Sweeps + source iteration vs. sparse LU, 5-region slab | agree to 1e-13 |
| Global balance $J_0 = J_L + J_R + \sum\Sigma_a h(\phi_c+\bar\phi_u)$ | 1e-16, including $c=1$ (where $J_L+J_R=1$) |
| Uncollided split vs. beam carried in S_N at $\mu_0=\mu_\text{max}$ | difference falls 4× per mesh doubling (1e-4 → 1e-7) |

**Discretization error of the currents** (vs. 200 cells/cm, S64, at the truth
materials):
- At S16 it is 0.2–0.8%, mostly from the angular quadrature, and largest at
  grazing beams.
- At S32 it is about 0.05–0.2%.

The inverse driver uses S16 at 25 cells/cm (about 5 ms for 6 angles) and adds
`sigma_model = 0.003` to the likelihood. For real data, check this against
the actual noise level.

---

## 2. Adjoint gradients (`sn_adjoint.py`, `slab_experiment.py`)

For each measured functional $Q = g^\top u$, solve $A^\top\lambda = g$. Then
$Q=\lambda^\top s$ and

$$
\frac{\partial Q}{\partial p} = \lambda^\top\Big(\frac{\partial s}{\partial p}-\frac{\partial A}{\partial p}u\Big).
$$

$\Sigma_t$, $\Sigma_s$ and $q$ enter row $(i,n)$ only through cell $i$, so
with $\Lambda_i=\sum_n\lambda_{i,n}$:

$$
\frac{\partial Q}{\partial\Sigma_{t,i}} = -h_i\sum_n\lambda_{i,n}\bar\psi_{i,n},\qquad
\frac{\partial Q}{\partial\Sigma_{s,i}} = \tfrac{h_i}{2}\Lambda_i\phi_i,\qquad
\frac{\partial Q}{\partial q_i} = \tfrac{h_i}{2}\Lambda_i .
$$

A per-material derivative is the sum over that material's cells, as in
`common/adjoint_solver.py`. Block $n$ of $A^\top\lambda=g$ divided by $w_n$
shows that $\lambda_{:,n}/w_n$ is the discrete adjoint angular flux, swept in
the $-\mu_n$ direction.

The beam adds two chain-rule terms:

- **First-collision source** $q=\Sigma_s\bar\phi_u$. Its derivative adds
  $\bar\phi_u$ to the $\Sigma_s$ term. $\bar\phi_u$ also depends on $\Sigma_t$
  through the attenuation (`dlog_phi_u_dsigma_t()`):

$$
\frac{\partial\log\bar\phi_{u,i}}{\partial\Sigma_{t,m}} = -\frac{\ell_m(i)}{\mu_0} + \mathbb 1_{i\in m}\frac{h_i}{\mu_0}\Big(\frac{1}{e^{a_i}-1}-\frac1{a_i}\Big),
$$

  where $\ell_m(i)$ is the path length in material $m$ upstream of cell $i$.
- **Uncollided transmission:** $\partial J_u/\partial\Sigma_{t,m} = -J_u T_m/\mu_0$,
  with $T_m$ the total thickness of material $m$.

Finally, $(\Sigma_t,\Sigma_s)\to(\Sigma_t,c)$:
$\partial_{\Sigma_t}|_c = \partial_{\Sigma_t}|_{\Sigma_s} + c\,\partial_{\Sigma_s}$ and
$\partial_c = \Sigma_t\,\partial_{\Sigma_s}$.

**Why this is cheap.** $A$ and both functionals $g$ are independent of
$\mu_0$. One LU factorization of $A(\theta)$ serves every beam angle. The two
adjoint solves (left and right face) reuse it via `lu.solve(g, trans='T')`
and give the gradient of **every** measurement at **every** angle. One call
to `currents_and_jacobian()` costs 1 LU, $K$ forward solves and 2 adjoint
solves. Timing it showed the gradients add almost nothing on top of the
currents.

**Verification** (`python3 slab_experiment.py`, both source faces, 3 random
$\theta$, 3 angles):
- The identity check holds to about 1e-15.
- Central finite differences agree to 1e-9–1e-6 for all 6 parameters.

---

## 3. Inverse UQ (`inverse_uq.py`)

### 3.1 Statistical model

$$
\log d_{k,f} = \log J_0 + \log J_f(\theta;\mu_{0,k}) + \varepsilon,\qquad \varepsilon\sim\mathcal N(0,\ \sigma_\text{meas}^2+\sigma_\text{model}^2),
$$

- **Prior:** $\theta$ uniform on the `SlabParamSampler` box, and
  $\log J_0\sim\mathcal N(\log J_0^\text{nom}, \sigma_{J_0}^2)$.
- **Effect of $J_0$:** it is additive in log space, so an uncertain beam
  strength removes exactly one direction (the overall scale) from what the
  data can determine.
- **Synthetic data:** made with a finer model (100 cells/cm, S48) than the
  inference model, so the test does not use the same discretization for data
  and inference.

Steps:
1. **Angle scan** (`angle_sensitivity.png`): $\partial\log J_f/\partial\log\theta_j$ against beam angle.
2. **Identifiability and design** (`design_comparison.png`):
   - singular values of $\partial\log J/\partial\log(\theta,J_0)$;
   - Laplace (Gauss-Newton) posterior width for several angle sets. For the
     Laplace approximation, the uniform box is replaced by a Gaussian of equal
     variance, so uninformed directions stay at their prior width.
   - This step needs no data and is the tool for choosing angles.
3. **Synthetic data.**
4. **MAP:** bounded Gauss-Newton (`scipy.optimize.least_squares`, adjoint
   Jacobian), multi-start from LHS.
5. **Laplace approximation** at the MAP.
6. **Adaptive Metropolis:** several chains, split-R-hat, samples written to
   `posterior_samples.txt`.

### 3.2 What the default problem shows (1% noise, 5% beam-strength uncertainty, 6 angles 0–80°)

**The data can't separate the dielectric from the core.** Singular values of
the log-sensitivity matrix:

```
8.7   3.0   0.82   0.66   0.16   9.0e-3   1.3e-4
```

- About five combinations are informed.
- The weakest direction is mostly $(-0.72\,\Sigma_{t,D},\ +0.61\,\Sigma_{t,C})$,
  plus some of $c_D$ and $c_C$. It moves optical thickness between dielectric
  and core while keeping the total about fixed.
- Going from 6 to 30 angles scales every singular value by about
  $\sqrt{\text{angles}}$ but does not remove the weak direction. **Only a
  different kind of measurement will break it, not more angles.**
- The reflected current is nearly blind to the core: its sensitivity to
  $\Sigma_{t,C}$ is about 0 at every angle.
- Because the slab is symmetric, moving the beam to the other face gives the
  same information.

**Effect of the angle design** (Laplace std / truth; prior-only in the last
row):

| design | $\Sigma_{t,S}$ | $\Sigma_{t,D}$ | $\Sigma_{t,C}$ | $c_S$ | $c_D$ | $c_C$ | $\log J_0$ |
|---|---|---|---|---|---|---|---|
| normal only | 0.21 | 0.35 | 0.31 | 0.22 | 0.28 | 0.51 | 0.050 |
| 0°, 60° | 0.16 | 0.28 | 0.28 | 0.040 | 0.19 | 0.35 | 0.045 |
| 6 angles 0–80° | 0.11 | 0.28 | 0.26 | 0.012 | 0.19 | 0.31 | 0.012 |
| 12 angles 0–82° | 0.094 | 0.28 | 0.25 | 0.009 | 0.17 | 0.28 | 0.009 |
| prior only | 0.29 | 0.39 | 0.36 | 0.87 | 0.29 | 0.52 | 0.050 |

- Steering the beam pins down $c_\text{shield}$ and the beam strength. The
  angle dependence is what separates $J_0$ from the material properties.
- $\Sigma_{t,D}$, $\Sigma_{t,C}$ and $c_C$ stay close to their prior width.

**MCMC has not converged on this posterior.** The posterior is a long, thin,
curved ridge that reaches the $c_D=0.95$ prior bound, and the MAP sits on
that bound. None of these samplers converged:

| Sampler | Run | R-hat |
|---|---|---|
| Default adaptive Metropolis | 4 × 10k steps | up to about 3 |
| Logit-space adaptive Metropolis | 4 × 40k steps | 1.2–1.8 |
| Affine-invariant ensemble | 32 walkers | 1.5–2.3 |

The driver prints the R-hat values and a warning. Don't use the MCMC credible
intervals until R-hat is below about 1.05. The MAP, Laplace and
identifiability results do not depend on the sampler.

---

## 4. Running

```bash
cd sn_transport
python3 sn_solver.py          # forward verification, ~4 s
python3 slab_experiment.py    # adjoint verification, ~2 s
python3 inverse_uq.py         # full inverse study, ~2.5 min (mostly MCMC)
```

Outputs:
- `output_graphs/`: `flux_truth_angles.png`, `adjoint_gradient_check.png`.
- `output_graphs_inverse/`: `angle_sensitivity.png`, `design_comparison.png`,
  `posterior_marginals.png`, `posterior_correlation.png`, `posterior_samples.txt`.

For real data, replace the synthetic-data block in `inverse_uq.main()` with
the measured currents. The data are a `(K, 2)` array of left/right exiting
currents, one row per beam angle. Pass them to `LogPosterior` as logs.
Geometry, bounds, the fixed parameters (degenerate bounds) and the S_N
resolution are all arguments of `SlabExperiment` / `SlabParamSampler`.
