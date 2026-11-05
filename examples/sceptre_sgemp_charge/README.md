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
| `collect_exodus_results.py` | Case directories of `forward_1d.exo`, `adjoint_transport_1d.exo` and `response_1d.exo` → the per-case CSVs in one data directory, with mesh and case id checks; resumable ([§8.1](#81-workflow)) |
| `potential_to_source.py` | SCEPTRE's adjoint-source potential $\phi$ → angular source $Q^\dagger=\mu\,d\phi/dx$, in place (run once before `void_sensitivity.py`) |
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

**Q⁺ dump vs. the source the adjoint was solved with.** With a coupled
adjoint (photon importance present) and a forward problem driven only by flux
entering the faces, the identity must hold. If it fails on every case, the
likeliest cause is that `case_*_adjoint_1d.csv` is not the source SCEPTRE's
adjoint solve used. For each failing case the script prints the boundary term
divided by $\langle Q^\dagger,\psi\rangle$ restricted to: all blocks, the gap only,
all but the core, all but the shields. The variant with ratio ≈ 1 is the source
the adjoint belongs to. (On synthetic data with a spurious core spike added to
the dump, "gap only" gives 1.0000.)

Either fix the $Q^\dagger$ dumps, or run with `--q-source boundary`: $q$ is then
the boundary term, which is the functional the adjoint flux belongs to, so values
and adjoint-form gradients are consistent. The $\langle\partial Q^\dagger/\partial
t_v,\psi\rangle$ term is then off by default (`--source-term auto`): the dumps are
not the adjoint's source, so their $t_v$ dependence says nothing about it. Cases
are marked status 1. On the synthetic spike data this recovers $q$ exactly and
the gradient to 1e-7.

**SCEPTRE's adjoint-source files are a potential, not Q⁺.** The field tool's
output (what SCEPTRE's adjoint mode and `q_directdrive` read) is a nodal scalar
field $\phi$: continuous, identical for every angle and electron group, zero for
photons. In the SGEMP runs it is linear through void and dielectric (slope
0.443896) and takes up the rest of the 1 V drop in the conductor element next
to each interface ($\phi=1$ in the core, 0 in the shields). SCEPTRE's source is

$$Q^\dagger_{e,g,a} = -\mu_a E_e = \mu_a\,\frac{d\phi}{dx}\Big|_e,$$

constant per element: $c\,\mu$ with $c=0.443896$ over void and dielectric plus
one-element spikes (~45) in the shield and core elements bordering the gap, the
structure of the original adjoint-source dumps. Reading $\phi$ itself as $Q^\dagger$
gave $\langle Q^\dagger,\psi\rangle$ ≈ 18× too large, 94% of it from $\phi=1$ across the
core. `potential_to_source.py DATA_DIR --angle-order descending` converts every
case in place (the potential is kept as `case_XXXX_adjoint_potential_1d.csv`;
already-converted files are skipped), and `void_sensitivity.py` refuses a
source that is the same for every angle.

**Angular weights.** SCEPTRE's `q_directdrive` for cases 0 and 1 (2.2433e-8,
1.0612e-8) is half of $\langle Q^\dagger,\psi\rangle$ with the original source dumps
(4.49e-8, 2.12e-8): SCEPTRE's weights sum to 1, this script's to 2. Use
`--weight-sum 1` to match; $q$ and $dq/dt_v$ scale together. The source is
$t_v$-independent here (fixed gap, uniform $c$), so the
$\langle\partial Q^\dagger/\partial t_v,\psi\rangle$ term vanishes.

**Run for SCEPTRE data:**

```bash
python3 potential_to_source.py DATA_DIR --angle-order descending   # once
python3 void_sensitivity.py DATA_DIR --weight-sum 1 [--reference-q qdd.txt]
python3 ge_gpr_scalar.py
```

`--reference-q` takes a two-column file `case_id q` (e.g. SCEPTRE's
`q_directdrive` per case) and prints the comparison. On synthetic data built
like SCEPTRE's (potential file, 38 groups, photon groups empty, descending
angles, conductor spikes), this gives the boundary identity in 10/10 cases,
$q$ equal to the reference to 5e-13, and $dq/dt_v$ to 4e-8. `--q-source
boundary` remains as a fallback when no usable source file exists: $q$ is then
the boundary term (consistent with the adjoint flux) and the source term is off
by default; the cross-case check flags a wrong source-term setting (median 120%
against 1% on synthetic data).

**Cross-case check.** At the end, `void_sensitivity.py` compares each case's
$dq/dt_v$ with finite differences of $q$ between the cases themselves
(second-order `np.gradient` over sorted $t_v$). It needs no extra transport
runs. Its own error is the curvature of $q(t_v)$ over the case spacing, which the
script reports: for a nearly linear response the comparison is a strict test of
the gradients; for a curved one, agreement to a few percent is what to expect
(about 2% on the synthetic sets, whose adjoint gradients are exact to 1e-7).

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
python3 potential_to_source.py DATA_DIR --angle-order descending   # once, SCEPTRE potential -> Q+
python3 extract_qoi.py
python3 gpr_scalar.py
python3 void_sensitivity.py      # writes output/qoi_void_adjoint.txt; --help for options
python3 diagnose_identity.py     # only if cases show status 2
python3 ge_gpr_scalar.py         # adjoint values + gradients (default)
python3 ge_gpr_scalar.py fd      # q_net + finite-difference placeholder
python3 gpr_scalar_void_sweep.py
python3 gpr_scalar_geometry_grid.py
python3 verify_void_sensitivity.py
```

Data paths are resolved relative to the files, as `../../../data/...`
(the `data/` directory next to this repository).

## 8. GE-GPR vs standard GP on void, dielectric and shield thickness

This section compares GE-GPR with a standard GP on one data set from the coupled
electron-photon deterministic transport solve. `t_v`, `t_d` and `t_s` vary
independently with the core fixed, and one mirrored `shield_thickness` applies to both
shield blocks.

### 8.1 Workflow

```bash
# 1. Design the cases and per-case mesh settings (see --help for bounds, sizes, mesh, FD step).
python3 make_geometry_cases.py --out-dir DATA_DIR

# 2. On the server: run every case in DATA_DIR/case_key.txt and dump into DATA_DIR
#      every case:  case_NNNN_forward_1d.csv and case_NNNN_adjoint_1d.csv (Q+)
#      train, test: also case_NNNN_adjoint_flux_1d.csv
#    Roles are in DATA_DIR/case_manifest.csv. Sections 5 and 6 cover the conversions;
#    collect_exodus_results.py does them all for runs laid out as case_*/1d/ (below).
#    exodus_to_csv.py takes Q+ from the adjoint Exodus file, so fd cases need one too,
#    unless the source can be written without an adjoint solve. Their adjoint flux is
#    not used.

# 3. Compute q and all three gradients (writes qoi_geometry_adjoint.csv).
python3 void_sensitivity.py DATA_DIR --parameterization independent --out-dir RESULTS --weight-sum 1

# 4. Check the gradients against the perturbed runs (exit status 1 on failure).
python3 check_gradients_fd.py --table RESULTS/qoi_geometry_adjoint.csv \
    --manifest DATA_DIR/case_manifest.csv --out-dir RESULTS

# 5. Compare the models.
python3 compare_ge_vs_gp.py --table RESULTS/qoi_geometry_adjoint.csv \
    --manifest DATA_DIR/case_manifest.csv --gradient-check RESULTS/gradient_check.json \
    --out-dir RESULTS/comparison --n-jobs 8
```

The default design is 350 training and test cases plus 120 perturbed cases: 940 solves
with an adjoint run for every case, or 820 if the perturbed cases can skip the adjoint
solve. Use the same `--weight-sum` you use for the rest of your data.

**Collecting Exodus runs.** For runs in `CASES_ROOT/case_*/1d/`, run this from the
repository root, with `case_key.txt` in the output directory or `--design-dir`:

```bash
python3 examples/sceptre_sgemp_charge/collect_exodus_results.py --cases-root CASES_ROOT \
    --out-dir DATA_DIR --inspect    # check the variables first; writes nothing
python3 examples/sceptre_sgemp_charge/collect_exodus_results.py --cases-root CASES_ROOT \
    --out-dir DATA_DIR --jobs 8     # rerun after fixes; finished cases are skipped
```

| Exodus file | CSV |
|---|---|
| `forward_1d.exo` | `case_NNNN_forward_1d.csv` |
| `adjoint_transport_1d.exo` | `case_NNNN_adjoint_flux_1d.csv` (not needed for fd cases) |
| `response_1d.exo` | `case_NNNN_adjoint_1d.csv`, converted to $Q^\dagger$ if it holds the potential |

That `response_1d.exo` holds the adjoint source is an assumption until `--inspect` and the
boundary identity (status 0) confirm it on real files. To read the source from another file,
use `--response-name` and `--response-prefix`. A case's CSVs are written only after it
passes the checks, and each layer's width is compared with `case_key.txt`, which catches a
wrong case id. The script refuses an output directory inside the repository. Results for
every case are in `collection_report.csv`, and the exit status is 0 only if all are clean.

**Meshing.** Use `linear` meshing in every region, with the max sizes that
`make_geometry_cases.py` writes per case to `mesh_settings.csv`. This relies on `linear`
giving `ceil(thickness / max size)` equal elements, which holds for all 8 regions checked
in the local dumps. Compare the first few dumps with the `n_*` columns to confirm it.

| Region | Rule | Default | Why |
|---|---|---|---|
| Void | fixed count | 5 elements | Flux is nearly flat; the width scales smoothly |
| Dielectric | fixed count | 600 elements (about 5e-4) | At h = 1e-3 electron flux still jumps up to 36% between elements |
| Shield | fixed size | 1e-3, `t_s` snapped to multiples | Holds the Q+ spike element's width constant |
| Core | fixed size | 1e-3, 100 elements at thickness 0.1 | Thickness never changes; 1e-3 at the spike elements |

The spike elements decide the conductor rule. Q+ is a one-element spike with a fixed
integral in each conductor element bordering the gap, so that element's width sets which
flux average enters q. In cases 0 and 1 of the local dumps the four spike elements carry
terms from 38% to 199% of |q| each, and the flux changes by up to 40% across them. If a
shield element stretched with `t_s`, q would gain a mesh term that the adjoint gradient
does not contain. On the synthetic slab with spikes, stretching gives a `t_s` error of 1.4%
at the coarsest mesh, and a fixed size with whole-element steps gives at most 7e-5.

Log meshing is not used in the shield: its element count would change with thickness and
cannot be set directly. Avoid fixed max sizes in void and dielectric too. Those make the
count jump with thickness, and fixed sizes may contribute to the jagged `q_net` in the
1000-case grid, whose successive differences along `shield_thickness` run from 0.0038e-8
to 0.0247e-8. Element size explains only part of that jaggedness (about 16% to 26% of the
residual), so the refinement test below is the real check.

**Refinement test, before the full study.** These sizes are starting values from two local
cases. Run one nominal case, forward and adjoint, with the sizes as above, all halved, and
all doubled, and compare q and the three gradients. Refine if the first two differ by more
than about 1%. Then keep one mesh for the whole study, since q depends on the spike
element's width.

### 8.2 The gradients

Thickening a layer inserts material in it while everything outside translates rigidly,
so only the inserted slab contributes. Per mirrored side:

$$\frac{dq}{dt_{\mathrm{layer}}} = -\psi\cdot\mu\frac{d\psi^\dagger}{dx}
\quad\text{(adjoint form)},\qquad
\frac{dq}{dt_{\mathrm{layer}}} = Q^\dagger\cdot\psi + \psi^\dagger\cdot\mu\frac{d\psi}{dx}
\quad\text{(forward form)},$$

each evaluated inside the layer, plus $\langle\partial Q^\dagger/\partial t,\psi\rangle$
when the source level depends on thickness. The fixed-radius derivative of section 5 is
the difference $L_v - L_d$, which the driver prints against the interface formula.

These forms are constant through a uniform layer, so a Q+ spike in a neighbouring
conductor does not enter. The driver evaluates them on elements at the block's typical
Q+ level, away from spikes and block faces. In cases 0 and 1 of the local dumps the
conductor spike height is about 100 times the gap level, but its integral equals the
gap level (0.4439), so the source-slope fit tracks that integral in conductors.

`verify_geometry_sensitivity.py` compares these against central differences on a
synthetic S_N slab:

| Case | Worst error at the finest mesh |
|---|---|
| Uniform Q+, Q+ jump, source depending on thickness, single-element void | about 2e-8 |
| Dielectric volume source (forward form; the adjoint form is wrong here) | about 3e-8 |
| Q+ spikes, shield widths scaling with `t_s` | 8e-4 on `t_s`, 1.4% at the coarsest mesh; about 2e-8 on `t_v`, `t_d` |
| Q+ spikes, fixed shield element size (the prescribed mesh) | 2e-7 on `t_s`, 7e-5 at the coarsest mesh; about 2e-8 on `t_v`, `t_d` |

**Limits:**

- The synthetic slab has one group and no photons. An electron-only adjoint treats
  photons as a fixed electron source and misses changes in photon attenuation with
  thickness, so its forward-form gradients need the perturbed-run check most.
- A large `max_layer_spread` in the table means the value varied through a layer, which
  flags a volume source or a non-uniform Q+.
- `check_gradients_fd.py` on real data is the deciding test. None of this has been run on
  real adjoint flux yet.

### 8.3 What the comparison holds equal

For every study, seed and M, both models use the same training and held-out cases,
values `q`, kernel family, hyperparameter ranges, diagonal jitter (1e-6) and restart
count, and every fit starts cold. Scaling comes from the training subset only.

The jitter matters because it sets an error floor. With GE-GPR's 1e-6 against sklearn's
default 1e-10, the synthetic test showed GE-GPR losing at large M purely for that reason.

Results to read:

- `comparison_summary.csv`: `median_improvement_same_M` is the paired per-seed gain.
  `gp_eps_q_at_same_solves` is the GP at 2M cases, since a GE-GPR case costs 2 solves and
  a GP case costs 1. Compare it with `ge_gpr_eps_q_median` for the cost-based answer.
- `*_calibration.png`: the share of held-out cases inside ±2σ. A narrow band only means
  something if this stays near 95%.
- `run_info.json`: settings and the gradient-check verdict. `at_bound` in
  `comparison_results.csv` lists hyperparameters at a bound for each fit.

`python3 compare_ge_vs_gp.py --synthetic --out-dir DIR` runs the whole comparison on an
analytic response, and `--synthetic-gradient-error wrong` shuffles the gradients to show
that the comparison notices. `python3 verify_geometry_sensitivity.py --end-to-end` runs
from CSV dumps through all of the steps above. Synthetic results test the tooling and
say nothing about the real response.

### 8.4 Reading the uncertainty bands

`ge_gpr_scalar.py` and the comparison use `band_plots.py`, which has three panels: means
with ±2σ bands, deviations from the GE-GPR mean, and σ on a log axis.

GE-GPR bands are often too thin to see on the mean panel. In the 2-case
finite-difference pilot the full ±2σ band is 0.17% of the y-range. With two cases, both
finite-difference slopes equal the same secant, so the data look exactly linear and the
band collapses; that is not evidence of accuracy.

Both models now report both σ definitions, because sklearn's `return_std` includes the
fitted noise and `GEGaussianProcess.predict` does not unless `include_noise=True`. Bands
include the noise for both. No σ is inflated to make a band visible. Each run prints the
band width as a share of the y-range and any hyperparameters at a bound, and
`ge_gpr_scalar.py` saves σ to `ge_gpr_scalar_prediction.npz`. Noise at its lower bound is
expected for a deterministic solve; other hyperparameters at a bound mean the data do not
determine them, as with the GP's length scale in the 2-case pilot.
