# `examples/`: exploratory scripts and SCEPTRE sub-projects

The loose scripts here are the early experiments that shaped the ROM design:
what the 5-layer solutions look like, how far a perturbation in one layer
reaches, and how the POD rank depends on the number of snapshots and the QoI
window. The two subdirectories are separate projects on SCEPTRE data.

| Path | What it is | README |
|---|---|---|
| `constant_example.py` | Homogeneous slab, interior plateau check | this file, §1 |
| `1d_diffusion_example.py` | 1000 LHS solves of the 5-layer problem, 5 profiles plotted | §1 |
| `varying_*.py` (6 scripts) | One property in one layer varies, the rest fixed; how far the effect reaches | §2 |
| `pod_sensitivity_study.py` | POD rank and projection error vs. $M_\text{train}$ and vs. QoI window | §3 |
| `sceptre_sgemp_charge/` | SGEMP net charge from SCEPTRE angular flux; scalar GPR / GE-GPR | [README](sceptre_sgemp_charge/README.md) |
| `charge_deposition/` | Coarse GPR with leave-one-out CV on an 11-case coax delamination sweep | [README](charge_deposition/README.md) |

Forward-model math is in [common/README.md](../common/README.md).

---

## 1. Basic solves

**`constant_example.py`**: $D = 1/3$, $\Sigma_a = 1$, $q = 1$ everywhere, $L=10$,
$N = 100$, $\phi=0$ at both ends. Away from the walls $\phi''\approx 0$, so the
PDE reduces to $\Sigma_a\phi = q$ and the flux is flat at $q/\Sigma_a = 1$. It
drops to zero in boundary layers whose width is set by the diffusion length
$L_d = \sqrt{D/\Sigma_a}\approx 0.58$ cm. Output: `constant_solution.png`.

**`1d_diffusion_example.py`**: 5 equal layers, $p = 15$ parameters with
$D\in[0.2, 2]$, $\Sigma_a\in[0.05, 1]$, $q\in[0, 2]$ per layer, $M = 1000$ LHS
samples, $N = 100$. Plots 5 random profiles with the layer boundaries
marked. Output: `5layer_diffusion.png`.

## 2. One-layer perturbation sweeps (`varying_*.py`)

All six scripts share the same setup: 5 layers of 2 cm on $[0, 10]$,
$N = 200$, 40 LHS samples, and exactly one property in one layer active (the
other 14 are fixed with degenerate bounds). Each solution is plotted only on
$x\in[1, 5]$ (layers 1–3), colored by the sampled value, and the script
prints the spread of $\phi$ at the two ends of that window.

The values **in the code** (some docstrings list different ranges):

| Script | Active | Range (code) | Fixed values | Output |
|---|---|---|---|---|
| `varying_D5.py` | $D_5$ | $[0, 3]$ | $D=1$, $\Sigma_a=0.5$, $q=1$ | `varying_D_layer5.png` |
| `varying_xs5.py` | $\Sigma_{a,5}$ | $[0, 3]$ | $D=1$, $\Sigma_a=0.5$, $q=1$ | `varying_xs_layer5.png` |
| `varying_source5.py` | $q_5$ | $[0, 3]$ | $D=1$, $\Sigma_a=0.5$, $q=1$ | `varying_source_layer5.png` |
| `varying_source4.py` | $q_4$ | $[0, 3]$ | $D=1$, $\Sigma_a=0.5$, $q=1$ | `varying_source_layer4.png` |
| `varying_constant_xs5.py` | $\Sigma_{a,5}$ | $[0.5, 1.5]$ | $D=1$, $\Sigma_a=1$, $q=1$ | `varying_constant_xs_layer5.png` |
| `varying_constant_source5.py` | $q_5$ | $[0.5, 2]$ | $D=1$, $\Sigma_a=1$, $q=1$ | `varying_constant_source_layer5.png` |

**What they show.** In a uniform region, a perturbation's effect on the flux
decays roughly like $e^{-\lvert\Delta x\rvert/L_d}$ with $L_d = \sqrt{D/\Sigma_a}$.
With $D=1$, $\Sigma_a=0.5$ ($L_d\approx1.41$ cm), a change in layer 5
($x\ge 8$) is damped by roughly $e^{-3/1.41}\approx0.12$ at $x=5$ and
$e^{-7/1.41}\approx0.007$ at $x=1$. The "constant" variants use
$\Sigma_a = 1$ ($L_d = 1$ cm), so the decay is faster. This locality is why
the QoI window choice matters for the POD rank (§3), and why a window far from
the varied region sees an almost one-dimensional family. (These decay factors
are an estimate from the uniform-medium Green's function, not output of the
scripts.)

## 3. POD sensitivity study (`pod_sensitivity_study.py`)

Same 5-layer, $p = 15$ problem ($D\in[0.2, 2]$, $\Sigma_a\in[0.05, 1]$,
$q\in[0,2]$), $N = 200$, one fixed test set of 100 samples (an LHS design,
seed 2, unlike the random test sets in the pipelines).
For each configuration it computes the SVD of the training snapshots and
records the first 10 singular values, the rank for $\eta = 99.9\%$ and
$99.99\%$, and the median and max test $\varepsilon_\text{proj}$ (the POD math is in
[common/README.md §4](../common/README.md#4-snapshot-pod-svd_analysispy)).

- **Study 1, snapshot convergence**: $M_\text{train}\in\{50, 100, 200, 400\}$,
  each an independent LHS design, window $[L/3, 2L/3]$. If the rank and the test
  projection error stop changing as $M$ grows, the snapshot set has
  captured the solution manifold. Output: `study1_convergence.png`.
- **Study 2, window comparison**: $M_\text{train} = 400$ and windows: left,
  middle, and right thirds, the middle half, and the full domain. A window
  that sees fewer layers needs fewer modes. Output: `study2_qoi_windows.png`.

## Running the loose scripts

These scripts import `common/` modules by bare name and save to
`output_graphs/` relative to the working directory, so:

```bash
cd examples
mkdir -p output_graphs
PYTHONPATH=../common python3 varying_D5.py
```

Earlier outputs of all of them are in `../pytorch_implementation/output_graphs/`,
where they were originally run.
