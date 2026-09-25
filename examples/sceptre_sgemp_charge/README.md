# `examples/sceptre_sgemp_charge/`: SGEMP net charge from SCEPTRE output

This directory builds scalar surrogates for the SGEMP direct-drive net charge
$q_\text{net}$ on the coaxial Core/Void/Teflon/Shield geometry that
`void_teflon_geometry/` models with diffusion. It has two parts:

1. a **pilot** that computes $q_\text{net}$ from raw SCEPTRE angular-flux dumps
   (2 cases) and fits GPR and a finite-difference GE-GPR to them;
2. **learning curves** on the pre-reduced 999-case and 1000-case sweeps, which
   are duplicated in `combined_study/cep_sgemp_analysis/` (see §4).

| File | Role |
|---|---|
| `extract_qoi.py` | Raw angular flux → electron current → $q_\text{net}$ for each `case_*_forward_1d.csv`; writes `output/qoi_scalar.txt` |
| `gpr_scalar.py` | Standard GPR on the pilot table |
| `ge_gpr_scalar.py` | GE-GPR on the pilot table with a finite-difference gradient |
| `gpr_scalar_void_sweep.py` | Learning curve, 999-case $t_v$ sweep, reading `summary(1).csv` directly |
| `gpr_scalar_geometry_grid.py` | Learning curve, 1000-case $10^3$ thickness grid, reading `summary(2).csv` directly |

---

## 1. The QoI

Reference: SAND2014-15576, Eq. 4.8.

$$
q_\text{net} = \int_\text{dielectric} E(x)\,J(x)\,dx ,
$$

where $E$ is the electrostatic field with the center conductor at unit
potential and the shield grounded, and $J$ is the net electron current. The
field is the one produced by a unit potential on the conductor whose charge
is sought, which is the weighting-field (Shockley–Ramo) form: electron
current through a region where that field is nonzero induces charge on the
conductor. (This reading is offered for orientation; the report is the
authority.) The code integrates over the two Teflon blocks only.

### 1.1 Geometry of the dumps

The SCEPTRE 1D dumps use a mirror-symmetric slab normalized to $x\in[0,1]$
with block ids (confirmed from the $x$ ranges against `case_key.txt`):

```
1 Shield_L | 2 Dielectric_L | 3 Void_L | 4 Core | 5 Void_R | 6 Dielectric_R | 7 Shield_R
```

### 1.2 Field: series capacitor, closed form

In the Void and Teflon there is no free charge, so
$\frac{d}{dx}\big(\varepsilon\,\frac{dV}{dx}\big)=0$ and the displacement
$D_e = -\varepsilon\,dV/dx$ is constant on each side of the Core. The potential
drop from Core ($V=1$) to Shield ($V=0$) is shared by the two layers in series:

$$
1 = D_e\Big(\frac{t_v}{\varepsilon_\text{void}}+\frac{t_d}{\varepsilon_\text{teflon}}\Big)
\;\Longrightarrow\;
D_e = \Big(\frac{t_v}{\varepsilon_\text{void}}+\frac{t_d}{\varepsilon_\text{teflon}}\Big)^{-1},\qquad
E = \pm\frac{D_e}{\varepsilon}\ \text{in each layer}.
$$

$E$ points from the Core toward the Shield: negative on the left half,
positive on the right half. `compute_field_by_block()` returns one constant
per block; there is no discretization. $\varepsilon_\text{void} = 1$ and
$\varepsilon_\text{teflon} = 2.1$ (the standard PTFE value, which is not in the
dataset).

### 1.3 Current: first angular moment of the electron flux

$$
J(x) = \sum_{g=20}^{38}\ \sum_{a=0}^{7} w_a\,\mu_a\,\psi_g(x,\mu_a),
$$

summed over the 19 electron groups only (groups 1–19 are photons). Per
element, $\psi$ is the average of the left and right nodal values.

The angular quadrature is 8-point Gauss–Lobatto on $[-1, 1]$
(`gauss_lobatto()`): nodes $\pm1$ plus the roots of $P_7'$, weights
$w_i = 2/\big(n(n-1)P_{n-1}(\mu_i)^2\big)$ with $n = 8$.

### 1.4 Discrete form

`compute_net_charge()`:

$$
q_\text{net} = \sum_{e\,\in\,\text{blocks }2,6} E_{\text{block}(e)}\,J_e\,(x_{e,\text{right}}-x_{e,\text{left}}).
$$

### 1.5 Data quirks and assumptions

- In the raw CSVs the `angle` and `group` headers are **swapped**. Column 8
  (`angle`, values 1–38) is the energy group and column 9 (`group`, 0–7) is
  the angle index. This was confirmed with the data's author, and the code
  reads the columns by position with the corrected meaning.
- Angle index $0\dots7$ is assumed to be ascending in $\mu$ ($\mu=-1$ at index 0).
- $\varepsilon_\text{teflon} = 2.1$ is assumed.

The later `summary.csv` files provide `response_1d`, which is this same
$q_\text{net}$ computed on the Sandia side.

---

## 2. Pilot surrogates (2 cases)

**`gpr_scalar.py`**: one GP on $(t_v, q_\text{net})$ with the standard kernel
([math](../../sklearn_implementation/README.md#2-standard-gpr-on-pod-coefficients-gpr_podpy)),
`StandardScaler` inputs, z-scored outputs. With two points this only checks
that the pipeline runs.

**`ge_gpr_scalar.py`**: `GEGaussianProcess` from
`sklearn_implementation/ge_gpr.py` with one active input. The gradient
$dq_\text{net}/dt_v$ comes from `np.gradient` on the sorted cases (central
differences inside, one-sided at the ends). With two cases it is the same
secant slope at both points. It is scaled like the adjoint gradients in the
diffusion code:

$$
\frac{\partial z}{\partial x_\text{norm}} = \frac{dq_\text{net}}{dt_v}\cdot\frac{s_x}{s_q},
$$

with $s_x$ the `StandardScaler` scale and $s_q$ the output standard deviation.
The script reports the log marginal likelihood of both models. At $M = 2$
that measures fit to the prior and the two points, not accuracy, and should
not be read as an accuracy comparison.

Outputs: `output/qoi_scalar.txt`, `output/gpr_scalar_prediction.png`,
`output/ge_gpr_scalar_prediction.png`.

## 3. Learning curves on the large sweeps

`gpr_scalar_void_sweep.py` (999 cases, $p=1$) and
`gpr_scalar_geometry_grid.py` ($10\times10\times10$ grid over void, dielectric,
and shield thickness, $p = 3$) fit standard GPR on nested prefixes of a
random 80% training pool and report
$\varepsilon_q = \lVert\hat q - q\rVert_2/\lVert q\rVert_2$ on the fixed 20% test
set. Kernel, split, metric, and results are described in
[combined_study/cep_sgemp_analysis/README.md](../../combined_study/cep_sgemp_analysis/README.md#2-the-surrogate-and-the-error-metric).
In brief: $\varepsilon_q$ goes from 1.65e-3 to a plateau of 1.15e-3 by
$M\approx200$ for the void sweep, and from 3.32e-3 to 1.38e-3 at $M=800$
(still falling) for the grid.

Outputs: `output_void1000_monoenergy/` and `output_voidDielShield1000_monoenergy/`,
each with `learning_curve.csv`, `gpr_learning_curve.png`, `gpr_parity.png`,
plus `gpr_fit.png` for the one-parameter sweep.

## 4. Relation to `combined_study/cep_sgemp_analysis/`

That directory has copies of the two learning-curve scripts that read
`extract_qoi.py`'s `output/qoi_*.txt` tables instead of the CSVs, plus a
third sweep (Pithon spectrum). Its `extract_qoi.py` only selects columns from
`summary.csv`; the flux integration exists only here. The two sets of
learning curves agree to about six significant digits.

## 5. Running

`gpr_scalar.py` and `ge_gpr_scalar.py` import `extract_qoi` and `gpr_scalar`
by bare name, so run them from this directory:

```bash
cd examples/sceptre_sgemp_charge
python3 extract_qoi.py
python3 gpr_scalar.py
python3 ge_gpr_scalar.py
python3 gpr_scalar_void_sweep.py
python3 gpr_scalar_geometry_grid.py
```

Data paths are resolved relative to the files, as `../../../data/...`
(the `data/` directory next to this repository).
