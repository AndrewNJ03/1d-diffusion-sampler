# `combined_study/cep_sgemp_analysis/`: standard-GPR baselines on SCEPTRE SGEMP sweeps

This directory fits standard GPR surrogates to the scalar SGEMP net charge
$q_\text{net}$ from three SCEPTRE parameter sweeps supplied by Sandia. It is
the "production code" counterpart to the diffusion sandbox, but it uses
**value-only** GPR. The SCEPTRE runs do produce adjoint output wiring it into GE-GPR
is deliberately deferred, so these results are the baseline a later GE-GPR
fit will be compared against.

| File | Role |
|---|---|
| `extract_qoi.py` | Reads each sweep's `summary.csv` and writes `output/qoi_<source>.txt` with columns `case_id <params> q_net` |
| `gpr_scalar_void_sweep.py` | Learning curve on the 999-case monoenergetic $t_v$ sweep |
| `gpr_scalar_pithon_sweep.py` | Learning curve on the 100-case Pithon-spectrum $t_v$ sweep |
| `gpr_scalar_geometry_grid.py` | Learning curve on the 1000-case void/dielectric/shield thickness grid |

## 1. The data and the QoI

| Source key | CSV | Cases | Parameters | Design |
|---|---|---|---|---|
| `void_sweep` | `constantR_void_1000_monoenergy/summary(1).csv` | 999 | $t_v$ | evenly spaced, 0.001 to 0.4995 |
| `void_sweep_pithon` | `constantR_void_100_pithon/summary.csv` | 100 | $t_v$ | evenly spaced; Pithon x-ray spectrum |
| `void_dielectric_shield_grid` | `constantR_voidDielectricShield_1000_monoenergy/summary(2).csv` | 1000 | $t_v$, $t_\text{diel}$, $t_\text{shield}$ | full $10\times10\times10$ factorial |

SCEPTRE's toolchain has already reduced each case to scalars. The column
`response_1d` is the net SGEMP direct-drive charge $q_\text{net}$
(SAND2014-15576, Eq. 4.8), the same quantity that
`examples/sceptre_sgemp_charge/extract_qoi.py` computes from raw angular flux
(the definition is written out in
[that README](../../examples/sceptre_sgemp_charge/README.md#1-the-qoi)).
`response_2d` is a different quantity and is not used. `extract_qoi.py` only
selects columns; it does no physics.

## 2. The surrogate and the error metric

Each script fits one scikit-learn GP with the repo's standard kernel,

$$
k(x,x') = \sigma_f^2\exp\!\Big(-\tfrac12\sum_j\frac{(x_j-x'_j)^2}{\ell_j^2}\Big) + \sigma_n^2\delta_{xx'},
$$

with the noise lower bound lowered to $10^{-8}$ (from $10^{-6}$ in the diffusion
code) because the SCEPTRE responses are smooth and nearly noise-free. Inputs
are standardized with `StandardScaler` and outputs are z-scored, both using
the training pool only. Hyperparameters maximize the log marginal likelihood
with `n_restarts=5` restarts. Math details:
[sklearn_implementation/README.md §2](../../sklearn_implementation/README.md#2-standard-gpr-on-pod-coefficients-gpr_podpy).

**Split.** One random permutation (seed 1) holds out 20% of the cases as a
test set. The learning curve fits on nested prefixes of the remaining pool.

**Metric.** One number per fit, the relative $\ell_2$ error over the whole
test set:

$$
\varepsilon_q = \frac{\lVert \hat q - q\rVert_2}{\lVert q\rVert_2}\quad(\text{vectors over test cases}).
$$

This is not the per-sample median $\varepsilon_y$ used for the diffusion
studies, so the two are not directly comparable.

## 3. Results (`output_*/learning_curve.csv`)

| Sweep | $p$ | $\varepsilon_q$ at smallest $M$ | $\varepsilon_q$ at $M\approx200$ | $\varepsilon_q$ at largest $M$ |
|---|---|---|---|---|
| void, monoenergetic | 1 | 1.65e-3 ($M$=10) | 1.15e-3 | 1.15e-3 ($M$=799) |
| void, Pithon | 1 | 1.36e-3 ($M$=5) | n/a | 3.69e-4 ($M$=79) |
| void/dielectric/shield grid | 3 | 3.32e-3 ($M$=10) | 1.53e-3 | 1.38e-3 ($M$=800) |

- The monoenergetic void sweep plateaus by $M\approx200$. The map
  $q_\text{net}(t_v)$ is close to linear over this range (see
  `output_void_sweep/gpr_fit.png`), and additional points no longer reduce the
  error. The error at $M = 20$–30 is slightly higher than at $M = 10$, which is
  the kind of noise a single split produces.
- The Pithon sweep reaches about $4\times10^{-4}$ by $M = 10$ and stays there.
- The three-parameter grid starts about twice as high as the one-parameter
  sweep and is still decreasing at $M = 800$. It is the harder map, and the
  natural test case for GE-GPR once the adjoint is connected, by analogy with
  `combined_study/`.

## 4. Relation to `examples/sceptre_sgemp_charge/`

`gpr_scalar_void_sweep.py` and `gpr_scalar_geometry_grid.py` also exist in
`examples/sceptre_sgemp_charge/`. Those copies read the `summary.csv` files
directly; the copies here read the `output/qoi_*.txt` tables from
`extract_qoi.py`. The fitting code is otherwise the same, and the learning
curves agree to about six significant digits. The Pithon sweep exists only
here.

## 5. Running

Run from this directory, because the GPR scripts import `extract_qoi` by bare
name:

```bash
cd combined_study/cep_sgemp_analysis
python3 extract_qoi.py                 # writes output/qoi_*.txt
python3 gpr_scalar_void_sweep.py       # -> output_void_sweep/
python3 gpr_scalar_pithon_sweep.py     # -> output_void_sweep_pithon/
python3 gpr_scalar_geometry_grid.py    # -> output_void_dielectric_shield_grid/
```

Each output directory holds `learning_curve.csv`, `gpr_learning_curve.png`,
`gpr_parity.png`, and, for the one-parameter sweeps, `gpr_fit.png` (mean,
$\pm2\sigma$ band, training and test points).
