# `pytorch_implementation/` (legacy, superseded)

This directory holds the original GPyTorch version of the GPR stage. It has
been **replaced by `sklearn_implementation/`** and is kept only for history
and for `compare.py`. Do not use it for new work or cite it as the current
pipeline.

| File | Role |
|---|---|
| `gpr_pod.py` | One GPyTorch `ExactGP` per POD coefficient, trained with Adam |
| `run_pipeline.py` | Stage 1 (`svd_analysis.main()`) followed by this GPR stage |
| `output_graphs*/` | Saved outputs from earlier runs (`M100`, `M500`, `pipeline`), plus plots from the old `examples/` scripts in `output_graphs/` |

## The math, and how it differs from the scikit-learn version

The model is the same as in
[sklearn_implementation/README.md §2](../sklearn_implementation/README.md#2-standard-gpr-on-pod-coefficients-gpr_podpy):
$R$ independent GPs, each with an ARD RBF kernel scaled by $\sigma_f^2$
(`ScaleKernel(RBFKernel(ard_num_dims=p))`) and Gaussian observation noise
(`GaussianLikelihood`), trained by maximizing the exact log marginal
likelihood. Three details differ:

| | GPyTorch (here) | scikit-learn (current) |
|---|---|---|
| Mean function | learned constant (`ConstantMean`) | zero, after z-scoring |
| Input scaling | min-max to $[0,1]$ using the sampler bounds | `StandardScaler` |
| Optimizer | Adam for `n_epochs` steps at rate `lr` (default 3000 / 0.05; `run_pipeline.py` uses 10000) | L-BFGS-B with `n_restarts` random restarts |

Everything is float64. Predictions are un-standardized and reconstructed as
$\hat y = \Psi\hat\alpha$, with the same error metrics as the current version.
`compare.py` at the repository root fits both versions on one set of Stage-1
files and plots them side by side in `output_graphs_compare/`.

## Running (only if you need to reproduce old results)

This code needs `torch` and `gpytorch`, which the `../rom_sandbox` environment
does not have. It also imports `svd_analysis` and `second_order_solver` by
bare name, so `common/` has to be on the path:

```bash
cd pytorch_implementation
PYTHONPATH=../common python3 run_pipeline.py     # output in output_graphs_pipeline/
```
