# 1D Diffusion ROM Sandbox: POD + GPR + Adjoint-Enhanced GPR

This repository is a small, fully verifiable test bed for building parametric
reduced-order models (ROMs) of a 1D one-group neutron diffusion problem. The
model uses Proper Orthogonal Decomposition (POD) for the spatial field and
Gaussian Process Regression (GPR) for the map from parameters to POD
coefficients. Its main research content is **gradient-enhanced GPR
(GE-GPR)**: the GP is trained on POD coefficients *and* on their exact
parameter derivatives, which come from one adjoint solve per mode.

The diffusion problem is cheap enough to sweep exhaustively and simple enough
that every adjoint gradient can be checked by finite differences. That makes it
a verification and validation (V&V) platform for the method before it moves to
a production code (SCEPTRE, see `examples/sceptre_sgemp_charge/` and
`combined_study/cep_sgemp_analysis/`).

Every directory has its own README that covers the math and the code for that
directory. This file has the shared notation, the overall pipeline, and a map.

---

## 1. The pipeline in one picture

```
 parameters mu            forward model                QoI restriction          POD                      surrogate
 (LHS design)   ──►  A(mu) phi(mu) = s(mu)   ──►   y(mu) = H phi(mu)   ──►   Y = U S V^T         ──►   mu ↦ alpha(mu)
                     (finite volume, N cells)      (Ny cells in window)      Psi = U[:, :R]            (one GP per mode)
                                                                             alpha = Psi^T y
                             │                                                                              ▲
                             └── adjoint:  A^T lambda_k = H^T psi_k  ──►  d alpha_k / d mu  ──(GE-GPR only)─┘
```

Prediction at a new parameter point is $\hat y(\mu^\ast) = \Psi\,\hat\alpha(\mu^\ast)$,
where $\hat\alpha$ is the GP posterior mean.

## 2. Shared notation

| Symbol | Shape | Meaning | Where in code |
|---|---|---|---|
| $\mu$ | $p$ | parameter vector (material values, geometry, or both) | rows of `samples_*.txt` |
| $N$ | | number of finite-volume cells | `n_cells` |
| $\phi(\mu)$ | $N$ | cell-averaged scalar flux | `phi` |
| $A(\mu)$, $s(\mu)$ | $N\times N$, $N$ | FV system matrix and right-hand side, $A\phi = s$ | `assemble_system()` returns `(A, rhs)` |
| $H$ | $N_y\times N$ | QoI restriction (a boolean cell mask, never formed as a matrix) | `qoi_mask` |
| $y(\mu)=H\phi(\mu)$ | $N_y$ | flux restricted to the QoI window | columns of `Y_tr`, rows of `qoi_values_*.txt` |
| $Y$ | $N_y\times M$ | snapshot matrix, one column per training sample | `Y_tr` |
| $\Psi$ | $N_y\times R$ | rank-$R$ POD basis | `pod_basis.txt` |
| $\alpha(\mu)=\Psi^\top y(\mu)$ | $R$ | POD (reduced) coefficients | `coefficients_train.txt` |
| $\lambda_k$ | $N$ | adjoint solution for mode $k$ | `lambda_k` |
| $p_\text{act}$ | | number of parameters that actually vary | `active_idx` |

**Error metrics** used everywhere (all relative $\ell_2$, per test sample):

$$
\varepsilon_y(\mu)=\frac{\lVert y-\Psi\hat\alpha\rVert_2}{\lVert y\rVert_2},\qquad
\varepsilon_\text{proj}(\mu)=\frac{\lVert y-\Psi\Psi^\top y\rVert_2}{\lVert y\rVert_2},\qquad
\varepsilon_\alpha(\mu)=\frac{\lVert \alpha-\hat\alpha\rVert_2}{\lVert \alpha\rVert_2}.
$$

Because $\Psi$ has orthonormal columns, $\varepsilon_y^2 = \varepsilon_\text{proj}^2 + \lVert\alpha-\hat\alpha\rVert^2/\lVert y\rVert^2$.
So $\varepsilon_\text{proj}$ is a hard **floor**: no surrogate for $\alpha$ can
beat it. A surrogate whose median $\varepsilon_y$ equals the median
$\varepsilon_\text{proj}$ is limited by the POD basis, not by the GP.

## 3. Conventions shared by all directories

- **Fixed vs. active parameters.** No code path has an "is this parameter
  fixed?" flag. A parameter is fixed by giving it degenerate bounds
  `(v, v)`; every sampler records this in `_fixed_mask` and returns the exact
  value `v` in that column. Downstream code builds `active_idx` from the mask.
- **Stage-1 file contract.** Every SVD/POD stage writes the same five text
  files into its `output_dir`: `samples_train.txt` ($M\times p$),
  `samples_test.txt`, `qoi_values_train.txt` / `qoi_values_test.txt`
  ($M\times N_y$), `pod_basis.txt` ($N_y\times R$), and
  `coefficients_train.txt` ($M\times R$). The GPR and GE-GPR stages only read
  these files, so they work for any forward model that writes them.
- **Training vs. test designs.** Training points come from a Latin hypercube
  (`sample()`, seed `seed`); test points are uniform random
  (`sample_random()`, seed `seed + 1`). The POD basis is built from training
  snapshots only.
- **Boundary conditions.** All studies use homogeneous Dirichlet conditions,
  $\phi = 0$ at both ends.
- **Section references** such as "Section 15" or "eq. 13" in docstrings refer
  to the project's workflow document, which is not part of this repository.

## 4. Directory map

| Directory | What it is | README |
|---|---|---|
| `common/` | Forward solver, LHS sampler, QoI mask, SVD/POD stage, adjoint solver and per-layer sensitivities | [common/README.md](common/README.md) |
| `sklearn_implementation/` | Current pipeline for the layered-material problem: GPR on POD coefficients, adjoint verification, and the GE-GPR implementation used everywhere else | [sklearn_implementation/README.md](sklearn_implementation/README.md) |
| `pytorch_implementation/` | **Legacy** GPyTorch version of the GPR stage. Superseded by `sklearn_implementation/` | [pytorch_implementation/README.md](pytorch_implementation/README.md) |
| `void_teflon_geometry/` | Geometry parameters: a smooth sigmoid material switch so the Void/Teflon interface location and sharpness become differentiable | [void_teflon_geometry/README.md](void_teflon_geometry/README.md) |
| `gpr_vs_gegpr_comparison/` | GE-GPR vs. standard GPR, one parameter active at a time (5 studies) | [gpr_vs_gegpr_comparison/README.md](gpr_vs_gegpr_comparison/README.md) |
| `combined_study/` | All 5 parameters active at once, plus a 5-seed repeatability run | [combined_study/README.md](combined_study/README.md) |
| `combined_study/cep_sgemp_analysis/` | Standard-GPR baselines on the pre-reduced SCEPTRE SGEMP sweeps | [combined_study/cep_sgemp_analysis/README.md](combined_study/cep_sgemp_analysis/README.md) |
| `examples/` | Early exploratory scripts (single-layer sweeps, POD sensitivity study) and two SCEPTRE sub-projects | [examples/README.md](examples/README.md) |
| `latex_paper/` | ANS M&C 2027 paper source | [latex_paper/README.md](latex_paper/README.md) |
| `output_graphs_compare/` | Output of `compare.py` (legacy PyTorch vs. scikit-learn GPR comparison); no source | (this file, §6) |

## 5. Environment and running

The scripts need `numpy`, `scipy`, `scikit-learn`, and `matplotlib`. The
virtual environment one level up, `../rom_sandbox/`, has these (Python 3.12).
It does **not** have `torch`/`gpytorch`, so the legacy PyTorch code and
`compare.py` do not run in it.

Most scripts put `common/` (and any sibling directories they need) on
`sys.path` themselves and can be run from their own directory:

```bash
cd sklearn_implementation && python3 run_pipeline.py
cd gpr_vs_gegpr_comparison && python3 run_comparison.py      # ~5 studies, longest run here
cd combined_study && python3 run_combined.py                 # ~4 minutes
```

The exceptions are the loose scripts in `examples/` and everything in
`pytorch_implementation/`, which import `common/` modules by bare name. Run
those with `PYTHONPATH=../common` and from a directory that has an
`output_graphs/` subdirectory (see [examples/README.md](examples/README.md)).

## 6. `compare.py` (repository root)

`compare.py` runs one Stage-1 SVD, then fits both the legacy GPyTorch GPR
(`pytorch_implementation/gpr_pod.py`) and the scikit-learn GPR
(`sklearn_implementation/gpr_pod.py`) on the same POD artifacts and overlays
their errors, parity plots, and reconstructions in `output_graphs_compare/`.
It loads the two same-named `gpr_pod.py` files under different module names
with `importlib`, which is why it doesn't use plain imports. It was the check
that the scikit-learn port matches the GPyTorch original before the latter
was retired.
