# `common/`: forward model, sampling, POD, and adjoint

Everything the other directories build on lives here: the finite-volume
diffusion solver, the Latin hypercube sampler, the QoI mask, the SVD/POD
stage, and the adjoint solver with its per-layer sensitivities. No file here
depends on scikit-learn or on any surrogate. Notation follows the
[top-level README](../README.md#2-shared-notation).

| File | Role |
|---|---|
| `second_order_solver.py` | Mesh, piecewise-constant material assignment, FV assembly, forward solve |
| `lhs_generation.py` | `latin_hypercube_sample()` and `DiffusionParamSampler` for the layered-material vector |
| `masking_function.py` | `mask_solution()` / `plot_masked_solution()`: restrict a solution to a window (used by `examples/`) |
| `svd_analysis.py` | Stage 1: snapshots → QoI mask → thin SVD → rank selection → file outputs and diagnostic plots |
| `adjoint_solver.py` | Adjoint right-hand side, adjoint solve, identity check, and $\partial\alpha_k/\partial\{D_m,\Sigma_{a,m},q_m\}$ |

---

## 1. Forward model (`second_order_solver.py`)

### 1.1 The PDE

$$
-\frac{d}{dx}\!\left[D(x)\,\frac{d\phi}{dx}\right] + \Sigma_a(x)\,\phi(x) = q(x),\qquad x\in(0,L),
$$

with $D$ the diffusion coefficient, $\Sigma_a$ the absorption cross section,
and $q$ a volumetric source. In the layered problem each of the three is
piecewise constant on `n_bins` equal-width layers.

### 1.2 Mesh and material assignment

`build_mesh(L, N)` makes a uniform cell-centered mesh: faces
$x_{i\pm1/2}$, centers $x_i$, widths $h_i = L/N$.
`assign_material_properties()` gives cell $i$ the properties of whichever layer
contains its center (`np.searchsorted` on the interior layer bounds). Layer
boundaries that fall between centers are therefore snapped to the nearest
face. That is harmless for fixed geometry, but it is why moving a boundary
is not differentiable (see `void_teflon_geometry/`).

### 1.3 Finite-volume discretization

Integrate the PDE over cell $i$:

$$
-\Big[D\phi'\Big]_{x_{i-1/2}}^{x_{i+1/2}} + \Sigma_{a,i}\,\phi_i\,h_i = q_i\,h_i .
$$

The face flux between cells $i$ and $i+1$ comes from requiring the flux to be
continuous across the face, with a linear profile in each half-cell:

$$
D_i\,\frac{\phi_f-\phi_i}{h_i/2} = D_{i+1}\,\frac{\phi_{i+1}-\phi_f}{h_{i+1}/2}
\;\;\Longrightarrow\;\;
D\phi'\big|_{i+1/2} \approx \beta_{i+1/2}\,(\phi_{i+1}-\phi_i),
$$

$$
\beta_{i+1/2} = \left(\frac{h_i}{2D_i}+\frac{h_{i+1}}{2D_{i+1}}\right)^{-1}
= \frac{2\,d_i\,d_{i+1}}{d_i+d_{i+1}},\qquad d_i := \frac{D_i}{h_i}.
$$

This is the harmonic mean of the two half-cell conductances (two resistors in
series). It is what keeps the flux exact across a material interface that
sits on a face. `_interface_beta()` returns all $N-1$ interior $\beta$'s.

Interior row $i$ of $A\phi = s$ (`assemble_system()`):

$$
-\beta_{i-1/2}\,\phi_{i-1} + \big(\beta_{i-1/2}+\beta_{i+1/2}+\Sigma_{a,i}h_i\big)\phi_i - \beta_{i+1/2}\,\phi_{i+1} = q_i\,h_i .
$$

The matrix is tridiagonal; it is assembled as `lil_matrix` and solved with
`scipy.sparse.linalg.spsolve`.

### 1.4 Boundary conditions

**Dirichlet** `('dirichlet', v)`: row 0 (or row $N-1$) is overwritten with
$\phi_0 = v$. Two consequences matter later:

1. Those rows no longer contain $D$, $\Sigma_a$, or $q$, so every adjoint
   sensitivity must skip them (`_interior_mask()` in `adjoint_solver.py`).
2. $A$ is no longer symmetric. Row 0 is $[1, 0, \dots]$ while row 1 still has
   $-\beta_{1/2}$ in column 0. The adjoint must therefore solve with $A^\top$,
   not $A$.

> **Accuracy note (measured).** The value is imposed at the first/last cell
> *center*, not at the wall, so the effective domain is $[h/2,\,L-h/2]$. For
> the homogeneous test case in `__main__` ($D=\Sigma_a=q=1$, $L=10$), the RMS
> error against the exact solution on $[0,L]$ converges at **first order**
> (N = 50 → 800: error ratios 2.05, 2.03, 2.01, 2.01 per doubling). Against the exact solution
> on the shifted domain $[h/2, L-h/2]$ it converges at exactly **second order**
> (ratios 4.00). So the interior stencil is second order, as the module name
> says, and the $O(h)$ error is entirely the boundary location. This does not
> affect the ROM studies: the surrogates reproduce the discrete model, and the
> adjoint is exact for the discrete model.

**Robin** `('robin', a, b)`: the boundary flux is written as
$(\phi_\text{face}-b)/a$ with $\phi_\text{face}\approx\phi_0$, which adds $1/a$
to the diagonal and $b/a$ to the right-hand side. With the same sign on both
ends, this is $\phi + a\,D\,\partial\phi/\partial n = b$ with $n$ the
**outward** normal (so $a=2$, $b=0$ is the Marshak vacuum condition). The
docstring writes the left condition as $\phi(0)+a_L D\phi'(0)=b_L$, which has
the opposite sign on the left face; the code, not the docstring, is the
outward-normal form. Approximating $\phi_\text{face}$ by $\phi_0$ is also first
order. No study in this repository uses Robin conditions.

---

## 2. Sampling (`lhs_generation.py`)

### 2.1 Latin hypercube

`latin_hypercube_sample(M, bounds, seed)` splits each dimension into $M$
equal strata, draws one uniform point per stratum, and shuffles the strata
independently per dimension:

$$
X_{ij} = \ell_j + \frac{\pi_j(i) + U_{ij}}{M}\,(u_j-\ell_j),\qquad U_{ij}\sim\mathcal U[0,1),\quad \pi_j \text{ a random permutation of } \{0,\dots,M-1\}.
$$

Each 1D projection hits every stratum exactly once. The `__main__` block
checks this marginal-coverage property.

### 2.2 `DiffusionParamSampler`

Parameter vector ordering, used by every material-layer file in the repo:

$$
\mu = (D_1,\dots,D_{n},\;\Sigma_{a,1},\dots,\Sigma_{a,n},\;q_1,\dots,q_{n}),\qquad p = 3n .
$$

- Bounds can be one `(lo, hi)` pair for all layers or one pair per layer.
- **Fixed parameters** use degenerate bounds `(v, v)`. `sample()` widens those
  dimensions by 1 so the LHS routine accepts them, then overwrites the column
  with `v`; `_fixed_mask` records which ones they are.
- `sample()` is used for training designs; `sample_random()` (i.i.d. uniform)
  for test sets.
- `unpack()` / `unpack_batch()` split $\mu$ back into the three per-layer
  arrays `solve_diffusion()` expects.

---

## 3. QoI restriction $H$

The QoI is the flux on a spatial window $[x_\text{min}, x_\text{max}]$. Cell
$i$ is included if its interval overlaps the window:

$$
H:\; i \in \text{QoI} \iff x_{i-1/2} < x_\text{max}\ \text{and}\ x_{i+1/2} > x_\text{min}.
$$

`svd_analysis.py` (and every stage that must match it) builds this as a
boolean array `qoi_mask` from the face coordinates. $H$ is never formed as a
matrix: $H\phi$ is `phi[qoi_mask]` and $H^\top v$ is "scatter $v$ into a zero
vector at the masked cells" (`build_adjoint_rhs()`).
`masking_function.mask_solution()` applies the same rule but infers $h$ from
the spacing of the first two centers, so it assumes a uniform mesh.

---

## 4. Snapshot POD (`svd_analysis.py`)

`main()` is Stage 1 of every material-layer pipeline.

1. **Snapshots.** Draw $M_\text{train}$ LHS and $M_\text{test}$ random
   parameter vectors, solve each, and stack the masked flux as columns:
   $Y_\text{tr}\in\mathbb R^{N_y\times M_\text{train}}$ and
   $Y_\text{te}$.
2. **Thin SVD.** $Y_\text{tr} = U\,\Sigma\,V^\top$. The snapshots are **not**
   mean-centered, so mode 1 is essentially the mean flux shape.
3. **Rank selection** by the energy criterion

   $$
   R = \min\Big\{r : \frac{\sum_{k\le r}\sigma_k^2}{\sum_k\sigma_k^2}\ge\eta\Big\},
   $$

   implemented as `searchsorted(cum_energy, eta) + 1`. The default
   $\eta$ is `0.999`; the comparison and combined studies pass `0.99999`.
4. **Basis and coefficients.** $\Psi = U_{:,1:R}$ and
   $A_\alpha = \Psi^\top Y_\text{tr}$. By Eckart–Young, $\Psi\Psi^\top Y$ is
   the best rank-$R$ approximation of $Y$, with training error

   $$
   \frac{\lVert Y-\Psi\Psi^\top Y\rVert_F}{\lVert Y\rVert_F} = \frac{\sqrt{\sum_{k>R}\sigma_k^2}}{\lVert Y\rVert_F}.
   $$

5. **Held-out check.** $\varepsilon_\text{proj}$ on the test snapshots for
   several ranks (`svd_proj_error.png`). This is the floor for every
   surrogate built on $\Psi$.

Outputs (in `output_dir`): the five files of the
[Stage-1 contract](../README.md#3-conventions-shared-by-all-directories) plus
`qoi_snapshots.png`, `svd_spectrum.png`, `svd_modes.png`,
`svd_reconstruction.png`, `svd_proj_error.png`.

---

## 5. Adjoint sensitivities (`adjoint_solver.py`)

### 5.1 Each POD coefficient is a linear functional of $\phi$

$$
\alpha_k(\mu) = \psi_k^\top H\phi(\mu) = g_k^\top\phi(\mu),\qquad g_k := H^\top\psi_k \in\mathbb R^N .
$$

`build_adjoint_rhs(psi_k, qoi_mask)` builds $g_k$. Define the adjoint
$\lambda_k$ by

$$
A(\mu)^\top\lambda_k = g_k \qquad(\texttt{solve\_adjoint}).
$$

Then $\alpha_k = g_k^\top\phi = \lambda_k^\top A\phi = \lambda_k^\top s$.
`alpha_via_adjoint()` returns $\lambda_k^\top s$; comparing it with
$\psi_k^\top\phi[\text{mask}]$ is the **identity check** every adjoint stage
runs before trusting any gradient.

### 5.2 General sensitivity formula

Differentiate $A\phi = s$ with respect to a parameter $\mu_j$:
$A\,\partial_j\phi = \partial_j s - (\partial_j A)\phi$. Then

$$
\frac{\partial\alpha_k}{\partial\mu_j} = g_k^\top\partial_j\phi = \lambda_k^\top A\,\partial_j\phi
= \lambda_k^\top\frac{\partial s}{\partial\mu_j} - \lambda_k^\top\frac{\partial A}{\partial\mu_j}\,\phi .
$$

One forward solve plus one adjoint solve per mode gives the derivative with
respect to **all** $p$ parameters. Central finite differences would need
$2p$ extra forward solves. Only $\partial_j A$ and $\partial_j s$ depend on
which parameter $\mu_j$ is, and each is sparse:

**Source, layer $m$** (`sensitivity_q`, "eq. 14"). $s_i = q_i h_i$ on interior
rows, so

$$
\frac{\partial\alpha_k}{\partial q_m} = \sum_{i\in m,\ i\ \text{interior}} (\lambda_k)_i\,h_i .
$$

**Absorption, layer $m$** (`sensitivity_sigma_a`, "eq. 15"). $\Sigma_{a,i}h_i$
sits on the diagonal of interior rows, so

$$
\frac{\partial\alpha_k}{\partial \Sigma_{a,m}} = -\sum_{i\in m,\ i\ \text{interior}} (\lambda_k)_i\,\phi_i\,h_i .
$$

**Diffusion coefficient, layer $m$** (`sensitivity_D`). $D$ enters only
through the face conductances. Face $f$ between cells $i$ and $i+1$ adds
$\beta_f$ to $A_{ii}$ and $A_{i+1,i+1}$ and $-\beta_f$ to both off-diagonals, so

$$
\lambda^\top\frac{\partial A}{\partial\beta_f}\phi = (\lambda_i-\lambda_{i+1})(\phi_i-\phi_{i+1}),
$$

with $\lambda$ set to 0 on Dirichlet rows (those rows do not contain
$\beta_f$). Differentiating the harmonic mean,

$$
\frac{\partial\beta_f}{\partial D_i} = \frac{2\,d_{i+1}^2}{(d_i+d_{i+1})^2}\cdot\frac{1}{h_i},\qquad
\frac{\partial\beta_f}{\partial D_{i+1}} = \frac{2\,d_{i}^2}{(d_i+d_{i+1})^2}\cdot\frac{1}{h_{i+1}},
$$

and summing over faces:

$$
\frac{\partial\alpha_k}{\partial D_m} = -\sum_{f}\Big[\tfrac{\partial\beta_f}{\partial D_i}\,\mathbb 1_{i\in m} + \tfrac{\partial\beta_f}{\partial D_{i+1}}\,\mathbb 1_{i+1\in m}\Big](\lambda_i-\lambda_{i+1})(\phi_i-\phi_{i+1}) .
$$

$\partial s/\partial D = 0$, so there is no source term. Each face contributes
to the layer of each of its two neighbors, which is how a face on a layer
boundary splits its sensitivity between the two layers.

`compute_full_gradient()` concatenates the three into one length-$3n$ vector
in the same `[D | Sigma_a | q]` order as `DiffusionParamSampler`.
`get_layer_assignment()` reproduces the solver's `searchsorted` rule so the
layer indices match the forward assembly exactly.

### 5.3 How this generalizes

The three formulas are dot products of per-cell quantities with a 0/1 layer
indicator. Replacing the indicator by any per-cell direction vector
$v = \partial(\text{field})/\partial\mu_j$ gives the sensitivity to any
parameter that changes the fields smoothly. `void_teflon_geometry/` and
`combined_study/` use exactly that generalization.

---

## Running the modules directly

```bash
cd common
mkdir -p output_graphs
python3 second_order_solver.py   # homogeneous-slab check + 3-layer example
python3 lhs_generation.py        # LHS marginal-coverage check + scatter/histograms
python3 svd_analysis.py          # Stage 1 with the 5-layer defaults
```

`svd_analysis.main()` is normally called from a pipeline driver
(`sklearn_implementation/run_pipeline.py`, `gpr_vs_gegpr_comparison/run_comparison.py`)
rather than run by hand.
