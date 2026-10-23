# `examples/sceptre_sgemp_charge/`: SGEMP net charge from SCEPTRE output

This directory builds scalar surrogates for the SGEMP direct-drive net charge
$q_\text{net}$ on the coaxial Core/Void/Teflon/Shield geometry that
`void_teflon_geometry/` models with diffusion. It has two parts:

1. a **pilot** that computes $q_\text{net}$ from raw SCEPTRE angular-flux dumps
   (2 cases) and fits GPR and GE-GPR to them, with adjoint gradients from §5
   (or a finite-difference placeholder);
2. **learning curves** on the pre-reduced 999-case and 1000-case sweeps, which
   are duplicated in `combined_study/cep_sgemp_analysis/` (see §4).

| File | Role |
|---|---|
| `exodus_to_csv.py` | SCEPTRE Exodus `psi_g_0_a_0_n` element variables → the per-element CSVs below (forward, adjoint flux, optionally adjoint source); single-file, `--list` and `--batch` modes |
| `extract_qoi.py` | Raw angular flux → electron current → $q_\text{net}$ for each `case_*_forward_1d.csv`; writes `output/qoi_scalar.txt` |
| `gpr_scalar.py` | Standard GPR on the pilot table |
| `ge_gpr_scalar.py` | GE-GPR on $q=\langle Q^\dagger,\psi\rangle$ with adjoint $dq/dt_v$ from `output/qoi_void_adjoint.txt` (default), or on $q_\text{net}$ with a finite-difference gradient (`fd`) |
| `gpr_scalar_void_sweep.py` | Learning curve, 999-case $t_v$ sweep, reading `summary(1).csv` directly |
| `gpr_scalar_geometry_grid.py` | Learning curve, 1000-case $10^3$ thickness grid, reading `summary(2).csv` directly |
| `void_sensitivity.py` | Adjoint derivative $dq/dt_v$ from forward + adjoint flux dumps of `constantR_void_10`; writes `output/qoi_void_adjoint.txt` |
| `diagnose_identity.py` | Per-case checks behind the boundary identity (incoming current, adjoint vacuum faces, units, $Q^\dagger\psi$ by block, all four angle conventions) |
| `verify_void_sensitivity.py` | Checks the $dq/dt_v$ formula against finite differences on a synthetic S_N slab (`sn_transport/`) |

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
`sklearn_implementation/ge_gpr.py` with one active input. By default
(`python3 ge_gpr_scalar.py`, or `adjoint`) the values and gradients both come
from `output/qoi_void_adjoint.txt` (§5): $q=\langle Q^\dagger,\psi\rangle$ and
the adjoint-form $dq/dt_v$, so both describe the same functional. Cases with no
adjoint flux dump, or whose boundary identity failed (status 2, §5), are
dropped and listed; cases the identity cannot check (status 1) are kept and
listed. `python3 ge_gpr_scalar.py fd` is the old placeholder: $q_\text{net}$
from `output/qoi_scalar.txt` with `np.gradient` on the sorted cases. Either
gradient is scaled like the adjoint gradients in the diffusion code:

$$
\frac{\partial z}{\partial x_\text{norm}} = \frac{dq}{dt_v}\cdot\frac{s_x}{s_q},
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

## 5. Adjoint derivative with respect to void thickness

`void_sensitivity.py` computes $dq/dt_v$ at fixed outer radius from each case's
forward flux, adjoint source (`case_*_adjoint_1d.csv`) and adjoint flux
(`case_*_adjoint_flux_1d.csv`, same column layout as the forward dump; rename
`ADJOINT_FLUX_PATTERN` if the files are named differently).

**The functional is the adjoint source's, not §1's.** In cases 0000 and 0001
the adjoint source is $c\,\mu_a$ on electron groups with $c=\mp0.443896$
(left/right half), uniform over void *and* dielectric, independent of $t_v$,
plus one-element spikes in the shield and core elements bordering the gap. So
$q=\langle Q^\dagger,\psi\rangle$ is 4.49e-8 and 2.12e-8 for those cases, against
3.18e-8 and 2.52e-8 from `extract_qoi.py`. GE-GPR values and gradients must
both use $q=\langle Q^\dagger,\psi\rangle$.

**Formula.** Growing $t_v$ turns a thin dielectric layer at each void/dielectric
interface $x_i$ into void. With $L=\mu\,\partial_x+H$, $H$ local in $x$, and no
forward source near $x_i$, the adjoint equation removes all cross sections:

$$
\frac{dq}{dt_v}=\sum_i\sum_{g,a}w_a\,\psi_{g,a}\,\mu_a\Big(\partial_x\psi^\dagger_{g,a}\big|_\text{diel}-\partial_x\psi^\dagger_{g,a}\big|_\text{void}\Big)\Big|_{x_i}.
$$

The derivatives are the streaming integrals of the elements touching $x_i$,
including the upwind face jump. A forward-flux form is also reported as a
diagnostic.

**Angle convention.** The code assumed angle index 0 is $\mu=-1$ (ascending).
SCEPTRE's dumps can be descending, and the adjoint dump can be relabelled
relative to the forward one. `void_sensitivity.py` scores all four
combinations (ascending/descending for all three dumps × adjoint same/reversed)
and picks **one for all cases**:
1. the adjoint must vanish on the faces where the forward flux leaves (this
   fixes the adjoint's orientation, leaving two candidates);
2. of those, the one under which the boundary identity holds in the most cases;
3. if it holds in none, the one whose forward flux has a vacuum face (a wrong
   labelling swaps entering and leaving current, so a vacuum face looks like a
   source).

Force it with `python3 void_sensitivity.py DATA_DIR auto descending` (or
`main(angle_order=..., adjoint_angles_reversed=...)`). $q=\langle Q^\dagger,\psi\rangle$
does not depend on the convention, but $dq/dt_v$ does: on synthetic descending
dumps, treating them as ascending with a reversed adjoint gave the gradient the
wrong sign (−1.05e-2 against a true +1.20e-2) while $q$ was unchanged, and the
identity failed on every case by 85%. With detection, all four labellings give
$q$ exact and $dq/dt_v$ within 3e-8 of finite differences. `extract_qoi.py`'s
$q_\text{net}$ still assumes ascending order; under descending dumps its sign
flips (it is not used by the GE-GPR path).

**Units.** $s$ is the dump's void width divided by `Void_Thickness`
(column `dump_per_key_length`). The interface formula gives $dq/dx_i$ per dump
length and the void moves $s$ dump lengths per unit $t_v$, so
$dq/dt_v = s\,dq/dx_i$ is per unit `Void_Thickness` **whatever units the solve
used**. Only $q=\int Q^\dagger\psi\,dx$ depends on them: it must be integrated in
the solve's length unit. With `q_mesh="auto"` (default) the boundary identity
picks the dump mesh or the case_key mesh when it applies, and otherwise the dump
mesh, which is right whenever the CSVs carry SCEPTRE's own coordinates (e.g.
from `exodus_to_csv.py`). Force it with `python3 void_sensitivity.py DATA_DIR
dump` or `... case_key`. The earlier version always used the case_key mesh,
which puts $q$ off by $1/s$ (and fails the identity on every case) when the dumps
are in SCEPTRE's units and case_key is not.

**Electron-only adjoint.** SCEPTRE's forward problem is a photon beam on the
left face; $Q^\dagger$ is on electron groups only. If the adjoint dump carries
no photon importance (max $|\psi^\dagger|$ over photon groups ≈ 0) while photons
carry the entering current, the adjoint is the adjoint of the electron problem
with photons acting as a **fixed electron source** $S_e$, which is nonzero in
the dielectric and zero in the void. Then:
- the boundary identity is $\approx 0$ (no importance on the incoming photons)
  and misses $q$ by ~100% — it cannot hold, because the real identity is
  $q=\langle\psi^\dagger,S_e\rangle$ and $S_e$ is not dumped;
- the **adjoint form** misses a term $\psi^\dagger\cdot(S_\text{void}-S_\text{diel})$ at
  each interface and is wrong by ~100% (case C below: it does not converge);
- the **forward form** stays exact, because that source jump cancels in it.

`void_sensitivity.py` prints the photon/electron split for every case and, when
it finds this situation, marks the case status 1 and uses the forward form
(column `dq_dtv_used`). Neither form includes how the photon flux itself
changes with $t_v$ (attenuation in the thinner dielectric), since that needs
photon importance; a finite-difference check in $t_v$ measures what it adds.

**Forward form accuracy.** $\mu\,\partial_x\psi$ is extrapolated to the
interface from the two nearest elements of each block and the adjoint is taken
as its face trace. That makes the forward form second order (5e-3 at 25
cells/cm to 2e-5 at 400 in cases A–C); with element averages it was first
order (5% to 0.3%).

**$Q^\dagger$ depending on $t_v$.** $Q^\dagger=-\mu\cdot E$ and the gap field of
the series capacitor depends on $t_v$, so $Q^\dagger$'s level $c_b$ in each gap
block changes between cases. The script fits $1/c_b$ linearly in $t_v$ (exact for
a series capacitor; it falls back to finite differences of $\ln c_b$ if the fit
is poor) and adds $\langle\partial Q^\dagger/\partial t_v,\psi\rangle=\sum_b
(d\ln c_b/dt_v)\,\langle Q^\dagger|_b,\psi\rangle$ to both forms. With it the
adjoint form matches finite differences to 1e-8 on synthetic data with
$c_b\propto E_b(t_v)$.

**Boundary identity**, $q=\sum_\text{incoming}w|\mu|\psi^\dagger\psi_\text{in}$:
a check, not an input. Status per case in column 8:

| status | meaning | used by GE-GPR |
|---|---|---|
| 0 | forward flux enters through the faces and the identity matches $q$ to 5% | yes |
| 1 | identity not applicable: no forward flux enters (internal source), or an electron-only adjoint (forward form used) | yes, listed |
| 2 | flux enters but the identity misses on both meshes: forward volume source as well, an adjoint of a different problem, or angle mismatch | no |

The $q$ mesh is chosen by the identity only when it holds; otherwise the dump
mesh is used.

On synthetic dumps, the gradient and $q$ are exact (to 3e-8 against finite
differences) for: $x$ normalised after the solve (identity picks the case_key
mesh), dumps in solve units with case_key in other units (picks the dump mesh),
all four angle labellings, and an internal source away from the gap (status 1).
For an electron-only adjoint shaped like the SCEPTRE dumps (38 groups, photon
beam, descending angles, $Q^\dagger\propto E(t_v)$) the forward form used matches
finite differences to 2–6e-4 at 80 cells/cm. A perturbed adjoint is caught as
status 2. `diagnose_identity.py [DATA_DIR]` prints the separate checks.

**Verification** (`verify_void_sensitivity.py`, one-group DD S16): A and B (flux
entering both faces) — the adjoint form matches central finite differences in
$t_v$ to about 5e-8 at 25–400 cells/cm and the forward form converges at second
order. C (fixed source in the dielectric, vacuum faces: the electron-only
analogue) — the identity misses by 100%, the adjoint form is off by ~100%, the
forward form converges at second order. This checks the formulas and the code,
not the transport solve's discretization: its dumps are discontinuous, so a
finite-difference check on real runs at $t_v\pm\delta$ is still required.

## 6. From Exodus to CSV

`exodus_to_csv.py` reads element variables named `psi_g_0_a_0_n` (group $g$,
1-based in SCEPTRE; angle $a$, 0–7; local node $n$ of the 2-node element) on
any number of blocks. It maps node 0/1 to left/right by coordinate, orders
elements by $x$, and writes the 11-column CSV the loaders read: `elem_id,
block_id, exodus_elem, time, x_left, x_right, x_center, energy_group,
angle_idx, left_value, right_value`. Columns 7–8 are where the hand-made dumps
had them (with the header now labelled correctly). By default it dumps the last
time step (`--step`).

It refuses rather than guesses when: a `psi` variable is missing, the 2nd/4th
indices are nonzero, the elements are not 2-node, the elements do not tile one
interval, or a variable is absent from a block in the truth table (unless
`--missing-as-zero`, which batch mode uses for the adjoint source only).

```bash
python3 exodus_to_csv.py --list some_case.exo        # see the variable families
python3 exodus_to_csv.py --batch ../../../data/constantR_void_10 \
    --forward "<forward name>_{:04d}.exo" --adjoint "<adjoint name>_{:04d}.exo" \
    [--adjoint-source-prefix <Q+ prefix>]
```

The patterns are Python format strings of the case id from `case_key.txt`.
Batch mode writes `case_XXXX_forward_1d.csv`, `case_XXXX_adjoint_flux_1d.csv`
and, with `--adjoint-source-prefix`, `case_XXXX_adjoint_1d.csv`. The reader is
`netCDF4` if installed, else `scipy.io.netcdf_file` (classic and 64-bit-offset
files only).

**Tested** on synthetic Exodus files (38 groups, 8 angles, blocks stored out of
$x$ order, shuffled node numbering, 40% of elements with reversed connectivity,
two time steps, adjoint source only on the gap blocks): the CSVs reproduce the
written fields to 5e-12, and through `void_sensitivity.py` the boundary
identity holds to 1e-11 and $dq/dt_v$ matches central finite differences to
3e-8 on all 10 cases.

## 7. Running

`gpr_scalar.py` and `ge_gpr_scalar.py` import `extract_qoi` and `gpr_scalar`
by bare name, so run them from this directory:

```bash
cd examples/sceptre_sgemp_charge
python3 exodus_to_csv.py --batch ...   # §6, once per data set
python3 extract_qoi.py
python3 gpr_scalar.py
python3 void_sensitivity.py      # writes output/qoi_void_adjoint.txt ([DATA_DIR] [auto|dump|case_key] [auto|ascending|descending])
python3 diagnose_identity.py     # only if cases show status 2
python3 ge_gpr_scalar.py         # adjoint values + gradients (default)
python3 ge_gpr_scalar.py fd      # q_net + finite-difference placeholder
python3 gpr_scalar_void_sweep.py
python3 gpr_scalar_geometry_grid.py
python3 verify_void_sensitivity.py
```

Data paths are resolved relative to the files, as `../../../data/...`
(the `data/` directory next to this repository).
