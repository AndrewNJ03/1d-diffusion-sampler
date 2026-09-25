# `sklearn_implementation/`: GPR, adjoint verification, and GE-GPR

This is the current pipeline for the layered-material problem, and
`ge_gpr.py` holds the gradient-enhanced GP that every later study
(`void_teflon_geometry/`, `gpr_vs_gegpr_comparison/`, `combined_study/`,
`examples/sceptre_sgemp_charge/`) imports unchanged. The forward model, POD
stage, and adjoint formulas are documented in
[common/README.md](../common/README.md); notation follows the
[top-level README](../README.md#2-shared-notation).

| File | Stage | Role |
|---|---|---|
| `run_pipeline.py` | driver | Runs Stages 1–4 with one shared configuration |
| `gpr_pod.py` | 2 | One scikit-learn GP per POD coefficient; test-set evaluation and plots |
| `adjoint_stage.py` | 3 | Adjoint identity check, finite-difference gradient check, 1D sensitivity sweeps |
| `ge_gpr.py` | 4 | `GEGaussianProcess` class, adjoint training gradients, GE-GPR vs. standard GPR comparison and learning curve |

Stage 1 is `common/svd_analysis.py`.

---

## 1. Default problem (`run_pipeline.py`)

- $L = 10$, `n_bins = 3` equal layers, `n_cells = 200`, Dirichlet
  $\phi = 0$ at both ends, QoI window $[3, 7]$.
- Bounds: $D\in[0.5, 2]$, $\Sigma_a\in[0.05, 1.5]$, $q\in[1,1]$ (so the
  source is **fixed** in every layer by default).
- `fixed_mats={1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}}` also fixes Material 1.
  The active parameters are therefore $D_2, D_3, \Sigma_{a,2}, \Sigma_{a,3}$
  ($p = 9$, $p_\text{act} = 4$).
- $M_\text{train} = 500$ (LHS, seed 1), $M_\text{test} = 100$ (random, seed 2),
  $\eta = 0.999$.

`_per_layer_bounds()` turns the scalar bounds plus `fixed_mats` into per-layer
`[lo, hi]` lists; the same helper is duplicated in `adjoint_stage.py` and
`ge_gpr.py` so each stage can be called on its own.

---

## 2. Standard GPR on POD coefficients (`gpr_pod.py`)

### 2.1 Model

The $R$ coefficients are modeled by $R$ **independent** GPs, one per mode:
$\alpha_k(\mu) \sim \mathcal{GP}(0, k_\theta)$ after standardization. Inputs are
standardized with `StandardScaler` (zero mean, unit variance, fit on training
data), and each coefficient is z-scored,
$z_k = (\alpha_k - \bar\alpha_k)/s_k$.

Kernel (`ConstantKernel * RBF + WhiteKernel`), an ARD squared exponential with
a noise term:

$$
k(x,x') = \sigma_f^2 \exp\!\Big(-\tfrac12\sum_{j=1}^{p}\frac{(x_j-x'_j)^2}{\ell_j^2}\Big) + \sigma_n^2\,\delta_{xx'} .
$$

One length scale $\ell_j$ per input lets the GP learn which parameters matter
(automatic relevance determination). Fixed parameters are constant columns,
so their length scales are irrelevant. Bounds: $\sigma_f^2\in[10^{-3},10^3]$,
$\ell_j\in[10^{-3},10^3]$, $\sigma_n^2\in[10^{-6},10]$.

### 2.2 Training and prediction

Hyperparameters $\theta$ maximize the log marginal likelihood

$$
\log p(z\mid X,\theta) = -\tfrac12 z^\top K_\theta^{-1}z - \tfrac12\log\lvert K_\theta\rvert - \tfrac n2\log 2\pi ,
$$

by L-BFGS-B with `n_restarts` random restarts (scikit-learn's optimizer). The
posterior at a test point $x_\ast$ is

$$
\hat z_k(x_\ast) = k_\ast^\top K^{-1} z,\qquad
\sigma_k^2(x_\ast) = k(x_\ast,x_\ast) - k_\ast^\top K^{-1}k_\ast ,
$$

where $k_\ast = [k(x_\ast, x^{(i)})]_i$. scikit-learn includes the fitted
white-noise level in $k(x_\ast,x_\ast)$. Predictions are un-standardized,
$\hat\alpha_k = \bar\alpha_k + s_k\hat z_k$.

### 2.3 Back to the field

$$
\hat y = \Psi\hat\alpha,\qquad
\operatorname{Var}[\hat y_i] = \sum_{k=1}^R \Psi_{ik}^2\,\sigma_k^2 .
$$

The variance formula assumes the $R$ GPs are independent, so it ignores any
correlation between mode errors. The script reports median and max of
$\varepsilon_y$, $\varepsilon_\text{proj}$, and $\varepsilon_\alpha$ on the test set.

Outputs: `gpr_sklearn_lml.png` (fitted log-MLL per mode),
`gpr_sklearn_coefficient_parity.png` (with $\pm2\sigma$ bars),
`gpr_sklearn_error_comparison.png` ($\varepsilon_y$ vs. $\varepsilon_\text{proj}$),
`gpr_sklearn_flux_reconstruction.png` (with $\pm2\sigma$ band).

`main()` also guards the $p=1$ / $R=1$ case, where `np.loadtxt` returns 1-D
arrays; that is what lets `void_teflon_geometry/` reuse this file unchanged.

---

## 3. Adjoint verification (`adjoint_stage.py`)

This stage checks the formulas from
[common/README.md §5](../common/README.md#5-adjoint-sensitivities-adjoint_solverpy)
before any gradient is used for training. It writes three kinds of plot.

1. **Identity check** (`adjoint_identity.png`). For `n_verify` training
   samples and all $R$ modes, compare $\alpha_k = \psi_k^\top H\phi$ with
   $\lambda_k^\top s$. The two are computed by different solves, so agreement
   to solver precision validates the forward system, the mask, and the
   adjoint solve together.
2. **Finite-difference check** (`adjoint_gradient_check.png`). For
   `n_fd_check` samples and mode `mode_indices[0]`, compare the analytic
   gradient with the central difference
   $\big(\alpha_k(\mu+\epsilon e_j)-\alpha_k(\mu-\epsilon e_j)\big)/2\epsilon$,
   $\epsilon = 10^{-5}\max(\lvert\mu_j\rvert, 10^{-3})$, for every active $j$.
3. **1D sweeps** (`adjoint_sweep_<param>.png`, up to two active parameters).
   Hold the other parameters at the middle of their bounds and sweep $\mu_j$
   over its range. The top panel overlays $\alpha_k(\mu_j)$ with the tangent
   line $\alpha_k(\mu_0) + g_0(\mu_j-\mu_0)$, whose slope $g_0$ is the adjoint
   gradient at `expansion_point`. The bottom panel shows the adjoint gradient
   along the sweep with its own linear Taylor line; the second derivative for
   that line is a central difference of the sampled gradients.

---

## 4. Gradient-enhanced GPR (`ge_gpr.py`)

### 4.1 Why derivatives can be added to a GP

If $f\sim\mathcal{GP}(0,k)$ and $k$ is twice differentiable, then $f$ and its
partial derivatives are jointly Gaussian, because differentiation is a linear
operator. Their covariances are derivatives of the kernel:

$$
\operatorname{Cov}\!\Big(f(x),\,\frac{\partial f}{\partial x'_n}(x')\Big) = \frac{\partial k}{\partial x'_n},\qquad
\operatorname{Cov}\!\Big(\frac{\partial f}{\partial x_m}(x),\,\frac{\partial f}{\partial x'_n}(x')\Big) = \frac{\partial^2 k}{\partial x_m\,\partial x'_n}.
$$

For the ARD squared exponential
$k(x,x') = \sigma_f^2\exp\!\big(-\tfrac12\sum_j r_j^2/\ell_j^2\big)$ with
$r = x - x'$:

$$
\frac{\partial k}{\partial x'_n} = k\,\frac{r_n}{\ell_n^2},\qquad
\frac{\partial^2 k}{\partial x_m\,\partial x'_n} = k\left[\frac{\delta_{mn}}{\ell_m^2} - \frac{r_m\,r_n}{\ell_m^2\,\ell_n^2}\right].
$$

(The first follows from $\partial(-\tfrac12 r_n^2/\ell_n^2)/\partial x'_n = r_n/\ell_n^2$;
differentiating it again in $x_m$ gives the product rule on $k$ and $r_n$.)

### 4.2 The augmented system

For one POD mode, with $M$ training points and $p_\text{act}$ active inputs,
the targets are the $M$ values and the $M p_\text{act}$ gradient entries:

$$
y_\text{aug} = \big[\,z(x^1),\dots,z(x^M),\;\;\partial_1 z(x^1),\dots,\partial_{p}z(x^1),\;\;\dots,\;\;\partial_1 z(x^M),\dots,\partial_{p}z(x^M)\,\big]^\top \in\mathbb R^{M(1+p_\text{act})} .
$$

Gradients are stored point-major: entry $M + i\,p_\text{act} + m$ is
$\partial_m z(x^i)$, which is `G_active.ravel(order='C')`. The joint covariance
(`_build_aug_K()`) is

$$
K_\text{aug} = \begin{bmatrix} K_{ff} + \sigma_{nf}^2 I & K_{fd} \\ K_{fd}^\top & K_{dd} + \sigma_{nd}^2 I \end{bmatrix} + 10^{-6} I ,
$$

$$
[K_{ff}]_{ij} = k(x^i,x^j),\quad
[K_{fd}]_{i,\,jp+n} = k(x^i,x^j)\frac{x^i_n-x^j_n}{\ell_n^2},\quad
[K_{dd}]_{ip+m,\,jp+n} = k(x^i,x^j)\left[\frac{\delta_{mn}}{\ell_m^2}-\frac{r_m r_n}{\ell_m^2\ell_n^2}\right].
$$

Values and derivatives get separate noise levels $\sigma_{nf}$ and
$\sigma_{nd}$, and a fixed jitter keeps the Cholesky factorization stable. The
code builds all blocks with broadcasting over an $(M, M, p_\text{act})$ array of
pairwise differences, then reshapes.

### 4.3 Hyperparameters

$$
\theta = \big[\log\sigma_f,\ \log\ell_1,\dots,\log\ell_{p_\text{act}},\ \log\sigma_{nf},\ \log\sigma_{nd}\big].
$$

$\theta$ holds logs of standard deviations (`_parse_theta` squares them).
`fit()` minimizes the negative log marginal likelihood of $y_\text{aug}$
(same formula as §2.2 with $K_\text{aug}$) with SciPy L-BFGS-B:

- bounds $[-5, 5]$ on $\log\sigma_f$ and $\log\ell$, and $[-10, 0]$ on the two
  log-noises (so noise standard deviations stay in $[4.5\times10^{-5}, 1]$);
- restart 0 starts from $\sigma_f=\ell=1$ with the noises at `noise_f_init`,
  `noise_d_init`; the other `n_restarts` start from uniform draws
  (fixed RNG seed 0, so fits are reproducible);
- no analytic gradient is supplied, so L-BFGS-B uses finite differences of
  the NLML, and each evaluation is a Cholesky factorization of size
  $M(1+p_\text{act})$. A failed factorization returns $10^{10}$.

The class docstring calls the noise inits "initial (and lower-bound)" levels.
The code only uses them as starting values; the lower bound is $e^{-10}$.

### 4.4 Prediction

Only function values are predicted. The cross-covariance between $f(x_\ast)$
and the training targets is

$$
k_\ast = \Big[\,k(x_\ast,x^j)\ \Big|\ k(x_\ast,x^j)\,\frac{x_{\ast,n}-x^j_n}{\ell_n^2}\,\Big],
$$

and

$$
\hat z(x_\ast) = k_\ast^\top K_\text{aug}^{-1}y_\text{aug},\qquad
\sigma^2(x_\ast) = \sigma_f^2 - k_\ast^\top K_\text{aug}^{-1}k_\ast .
$$

This variance is for the latent function; unlike scikit-learn's, it does not
include observation noise.

### 4.5 Getting gradients into the scaled space

`_compute_training_gradients()` does one forward solve and $R$ adjoint solves
per training point and calls `compute_full_gradient()`, giving
$G_{ikj} = \partial\alpha_k(\mu^i)/\partial\mu_j$ in physical units. The GP
works in scaled coordinates:

$$
x_j = \frac{\mu_j - \mu_j^{\min}}{\mu_j^{\max} - \mu_j^{\min}}\in[0,1],\qquad z_k = \frac{\alpha_k-\bar\alpha_k}{s_k}
\quad\Longrightarrow\quad
\frac{\partial z_k}{\partial x_j} = \frac{\partial\alpha_k}{\partial\mu_j}\cdot\frac{\mu_j^{\max}-\mu_j^{\min}}{s_k}
$$

(`_gradients_to_standardized()`). The gradients must be scaled consistently
with the values; otherwise the GP sees inconsistent data and the fit
degrades.

Note that `ge_gpr.py` scales inputs to $[0,1]$ using the sampler bounds,
while `gpr_pod.py` uses `StandardScaler`. The standard GPR used *inside*
`ge_gpr.py` for comparison uses the same $[0,1]$ inputs, the same kernel
family, and the same training subset as GE-GPR, so the two are compared on
equal terms.

### 4.6 What `main()` does

1. Load the Stage-1 files and rebuild the sampler, so `active_idx` is known.
2. Take the **first** `m_ge` training rows as the GE-GPR set. Coefficient mean
   and standard deviation come from the full training set.
3. Compute adjoint gradients for those rows and scale them (§4.5).
4. For each mode: fit `GEGaussianProcess` on values plus gradients, and a
   standard `GaussianProcessRegressor` on values only, both on the active
   columns.
5. Predict on the test set and report $\varepsilon_y$ for both, plus
   $\varepsilon_\text{proj}$.
6. **Learning curve.** Refit both models on nested prefixes of the GE-GPR
   set (default sizes 10, 15, 20, 30, … up to `m_ge`) and record the median
   and 75th percentile of $\varepsilon_y$. The gradients are computed once
   and reused. A prefix of an LHS design is not itself a Latin hypercube, so
   small-$M$ points are closer to a random design.

Outputs: `gegpr_error_comparison.png`, `gegpr_parity.png`,
`gegpr_flux_reconstruction.png` (the $x$ axis is an evenly spaced stand-in
for the QoI cell centers, for plotting only), `gegpr_learning_curve.png`.
The plotting helpers are module-level functions so other directories can
reuse them.

### 4.7 Cost

Each NLML evaluation factorizes an $n\times n$ matrix with
$n = M(1+p_\text{act})$, which costs $O(n^3)$. Doubling $p_\text{act}$ at fixed $M$
roughly multiplies the cost by $\big((1+2p)/(1+p)\big)^3$. That is why
`combined_study/` ($p_\text{act}=5$) uses a smaller `m_ge` than the
one-parameter studies.

---

## Running

```bash
cd sklearn_implementation
python3 run_pipeline.py                 # all four stages, output in output_graphs_sklearn/
```

Stages can be skipped with `skip_svd`, `skip_gpr`, `skip_adjoint`,
`skip_ge_gpr` (for example, rerun only GE-GPR on existing Stage-1 files). Each
stage module also has its own `__main__` with the same defaults.
