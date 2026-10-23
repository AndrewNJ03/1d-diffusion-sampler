"""
void_sensitivity.py: adjoint derivative of the SGEMP charge functional with
respect to void thickness, dq/dt_v, from forward and adjoint flux dumps of the
coupled electron-photon deterministic transport solve.

The functional
--------------
The transport solve's adjoint source Q+ (case_*_adjoint_1d.csv) defines the
functional its adjoint solution belongs to:

    q = <Q+, psi> = sum_g sum_a w_a int Q+_{g,a}(x) psi_{g,a}(x) dx .

This is NOT the same number as extract_qoi.py's q_net (series-capacitor field
with eps_teflon = 2.1, dielectric blocks only). In the dumps checked so far
(cases 0000, 0001) Q+ is c*mu_a on electron groups, with c = -/+0.443896 on
the left/right half, uniform over void AND dielectric and independent of t_v,
plus one-element spikes in the conductor elements bordering the gap. The
derivative below is the derivative of THIS functional, so the GE-GPR values
must be q from charge_functional(), not extract_qoi.py's q_net.

The derivative
--------------
Write the transport operator as L = mu d/dx + H(x), where H (collision,
scattering, photon-electron coupling, slowing down) is local in x and depends
only on the material. With L psi = S and L+ psi+ = Q+, where L+ = -mu d/dx + H+,

    dq/dt_v = <dQ+/dt_v, psi> - <psi+, (dL/dt_v) psi> + <psi+, dS/dt_v> .

At fixed outer radius, increasing t_v moves each void/dielectric interface x_i
into the dielectric, turning a thin dielectric shell into void:
dL/dt_v = sum_i (H_void - H_diel) delta(x - x_i). If no forward source sits at
the interface, H psi = -mu dpsi/dx on each side, and the adjoint equation gives
H+ psi+ = Q+ + mu dpsi+/dx on each side. That removes every cross section:

  forward form:  dq/dt_v = sum_i [ (Q+_void - Q+_diel) . psi
                                   + psi+ . mu (dpsi/dx|_void - dpsi/dx|_diel) ]_{x_i}
  adjoint form:  dq/dt_v = sum_i [ psi . mu (dpsi+/dx|_diel - dpsi+/dx|_void) ]_{x_i}

where a . b = sum_g sum_a w_a a_{g,a} b_{g,a}. The Q+ jump term of the forward
form cancels in the adjoint form. The two forms agree in the continuum; their
difference on real dumps is a discretization diagnostic. Use the adjoint form:
on the synthetic diamond-difference slab in verify_void_sensitivity.py it
matches central finite differences to ~5e-8 at every mesh, while the forward
form converges at second order with face-extrapolated derivatives (it was
first order, 5% to 0.3%, with element averages).

Discretely, mu dpsi/dx in the element next to the interface is taken as the
element's streaming integral divided by its width, including the jump from the
upwind neighbour (the upwind trace), because the dumps are discontinuous and
the electron flux jumps by up to ~85% at material faces. The forward flux is
upwinded along +mu, the adjoint along -mu.

Assumptions (check before trusting a number)
--------------------------------------------
  - The forward source is on the outer boundary, not in the volume near x_i
    (boundary_functional() then reproduces q; a mismatch flags a volume source).
  - Q+ does not depend on t_v away from the interfaces (true for the dumps
    seen so far; main() warns if the gap source differs between cases).
  - Outer radius fixed: the dielectric shrinks as the void grows.
  - The adjoint dump's angle index means the same mu as the forward dump.
    main() tests both conventions against the boundary identity.
  - Every term is first order in the element width at x_i. Verify against
    finite differences in t_v before using the gradients
    (verify_void_sensitivity.py checks the method on a synthetic S_N slab).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from extract_qoi import (
    COL_X_LEFT, COL_X_RIGHT, COL_ELEM_ID, COL_BLOCK_ID, COL_ENERGY_GROUP,
    COL_ANGLE_IDX, COL_LEFT_FLUX, COL_RIGHT_FLUX,
    BLOCK_DIELECTRIC_L, BLOCK_VOID_L, BLOCK_VOID_R, BLOCK_DIELECTRIC_R,
    DATA_DIR, OUT_DIR, gauss_lobatto, load_case_key, ELECTRON_GROUP_MIN,
)

DIELECTRIC_BLOCKS = (BLOCK_DIELECTRIC_L, BLOCK_DIELECTRIC_R)
VOID_BLOCKS = (BLOCK_VOID_L, BLOCK_VOID_R)
GAP_BLOCKS = DIELECTRIC_BLOCKS + VOID_BLOCKS

FORWARD_PATTERN = "case_{:04d}_forward_1d.csv"
ADJOINT_SOURCE_PATTERN = "case_{:04d}_adjoint_1d.csv"
# Adjoint solution dumps: assumed to share the forward dump's column layout,
# with the adjoint flux in the left/right value columns. Rename to match the
# files when they arrive.
ADJOINT_FLUX_PATTERN = "case_{:04d}_adjoint_flux_1d.csv"
# Relative tolerance on q from the boundary identity vs. q = <Q+, psi>.
IDENTITY_TOL = 0.05


# --------------------------------------------------------------------------- #
# Nodal fields                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class NodalField:
    """Discontinuous linear field: left/right nodal values per element, (E, G, A)."""
    x_left: np.ndarray
    x_right: np.ndarray
    block: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @property
    def h(self) -> np.ndarray:
        return self.x_right - self.x_left

    def average(self) -> np.ndarray:
        return 0.5 * (self.left + self.right)

    def with_values(self, left, right) -> "NodalField":
        return NodalField(self.x_left, self.x_right, self.block, left, right)


def load_nodal_field(path: Path) -> NodalField:
    """Read one per-element dump (forward flux, adjoint source or adjoint flux)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    e = data[:, COL_ELEM_ID].astype(int) - 1
    g = data[:, COL_ENERGY_GROUP].astype(int) - 1
    a = data[:, COL_ANGLE_IDX].astype(int)
    shape = (e.max() + 1, g.max() + 1, a.max() + 1)
    if len(data) != np.prod(shape):
        raise ValueError(f"{path.name}: {len(data)} rows, expected {np.prod(shape)} for {shape}")

    left, right = np.zeros(shape), np.zeros(shape)
    left[e, g, a] = data[:, COL_LEFT_FLUX]
    right[e, g, a] = data[:, COL_RIGHT_FLUX]

    x_left, x_right, block = np.zeros(shape[0]), np.zeros(shape[0]), np.zeros(shape[0], dtype=int)
    x_left[e], x_right[e] = data[:, COL_X_LEFT], data[:, COL_X_RIGHT]
    block[e] = data[:, COL_BLOCK_ID].astype(int)
    if np.any(np.diff(x_left) <= 0):
        raise ValueError(f"{path.name}: elements are not ordered by x")
    return NodalField(x_left, x_right, block, left, right)


def same_mesh(a: NodalField, b: NodalField) -> bool:
    return (a.left.shape == b.left.shape and np.allclose(a.x_left, b.x_left)
            and np.allclose(a.x_right, b.x_right))


def reverse_angles(field: NodalField, mu: np.ndarray) -> NodalField:
    """Relabel angle index a -> A-1-a, i.e. mu -> -mu (quadrature must be symmetric)."""
    if not np.allclose(mu, -mu[::-1]):
        raise ValueError("angle reversal needs a quadrature symmetric about mu = 0")
    return field.with_values(field.left[..., ::-1], field.right[..., ::-1])


# --------------------------------------------------------------------------- #
# Inner products                                                              #
# --------------------------------------------------------------------------- #

def _dot(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """a . b = sum_g sum_a w_a a_{g,a} b_{g,a} at one point, (G, A) arrays."""
    return float(np.einsum("ga,ga,a->", a, b, w))


def volume_inner_product(f: NodalField, u: NodalField, w: np.ndarray, elem_mask=None) -> float:
    """sum_g sum_a w_a int f u dx, exact for linear f and u on each element."""
    integrand = (2 * f.left * u.left + f.left * u.right
                 + f.right * u.left + 2 * f.right * u.right) / 6.0
    per_elem = f.h * np.einsum("ega,a->e", integrand, w)
    if elem_mask is not None:
        per_elem = per_elem[elem_mask]
    return float(per_elem.sum())


def charge_functional(fwd: NodalField, src: NodalField, w: np.ndarray) -> float:
    """q = <Q+, psi>, the functional the adjoint solution belongs to."""
    return volume_inner_product(src, fwd, w)


def boundary_functional(fwd: NodalField, adj: NodalField, mu: np.ndarray, w: np.ndarray) -> float:
    """
    q from the adjoint side for a boundary-only forward source:
        q = sum over incoming directions on both faces of w_a |mu_a| psi+ psi_in.
    Equal to charge_functional() only if the forward problem has no volume source
    and the adjoint is the true adjoint (to discretization error).
    """
    inc_l, inc_r = mu > 0, mu < 0
    wl, wr = w * np.abs(mu) * inc_l, w * np.abs(mu) * inc_r
    return (_dot(adj.left[0], fwd.left[0], wl) + _dot(adj.right[-1], fwd.right[-1], wr))


# --------------------------------------------------------------------------- #
# Interface sensitivity                                                       #
# --------------------------------------------------------------------------- #

def find_void_dielectric_interfaces(block: np.ndarray, dielectric_blocks=DIELECTRIC_BLOCKS,
                                    void_blocks=VOID_BLOCKS):
    """(dielectric element, void element) index pairs that share a face."""
    pairs = []
    for e in range(len(block) - 1):
        b0, b1 = block[e], block[e + 1]
        if b0 in dielectric_blocks and b1 in void_blocks:
            pairs.append((e, e + 1))
        elif b0 in void_blocks and b1 in dielectric_blocks:
            pairs.append((e + 1, e))
    return pairs


def element_streaming(field: NodalField, mu: np.ndarray, e: int, flow_sign: int = +1) -> np.ndarray:
    """
    int_e mu dpsi/dx dx for element e, (G, A), with the inflow face taken from
    the upwind neighbour's trace (so the face jump counts as streaming inside e).
    flow_sign = +1 for a forward flux (flows along +mu), -1 for an adjoint flux.
    """
    n_elem = len(field.h)
    d = np.sign(mu) * flow_sign
    upwind_left = field.right[e - 1] if e > 0 else field.left[e]
    upwind_right = field.left[e + 1] if e < n_elem - 1 else field.right[e]
    psi_xl = np.where(d > 0, upwind_left, field.left[e])
    psi_xr = np.where(d > 0, field.right[e], upwind_right)
    return mu * (psi_xr - psi_xl)


def void_thickness_sensitivity(fwd: NodalField, adj: NodalField, src: NodalField,
                               mu: np.ndarray, w: np.ndarray, pairs=None) -> dict:
    """
    dq/dt_v at fixed outer radius, in the forward and adjoint forms of the
    module docstring, with the per-interface terms for diagnostics.
    """
    if pairs is None:
        pairs = find_void_dielectric_interfaces(fwd.block)
    if not pairs:
        raise ValueError("no void/dielectric interface found")
    psi_avg, adj_avg, src_avg = fwd.average(), adj.average(), src.average()

    def mu_dpsi_at_interface(e, away):
        """
        mu dpsi/dx of the forward flux at the interface face of element e: the
        element's streaming average, linearly extrapolated to the face using the
        next element of the same block (index e + away). Falls back to the
        element average if that element is in another block.
        """
        d1 = element_streaming(fwd, mu, e, +1) / fwd.h[e]
        e2 = e + away
        if e2 < 0 or e2 >= len(fwd.h) or fwd.block[e2] != fwd.block[e]:
            return d1
        d2 = element_streaming(fwd, mu, e2, +1) / fwd.h[e2]
        return d1 + (d1 - d2) * fwd.h[e] / (fwd.h[e] + fwd.h[e2])

    total_fwd, total_adj, per_interface = 0.0, 0.0, []
    for t, v in pairs:
        ht, hv = fwd.h[t], fwd.h[v]
        mu_dadj_t = element_streaming(adj, mu, t, -1) / ht
        mu_dadj_v = element_streaming(adj, mu, v, -1) / hv

        # Traces at x_i: forward and adjoint as the mean of the two sides.
        if t < v:
            psi_face = 0.5 * (fwd.right[t] + fwd.left[v])
            adj_face = 0.5 * (adj.right[t] + adj.left[v])
            away_t, away_v = -1, +1
        else:
            psi_face = 0.5 * (fwd.left[t] + fwd.right[v])
            adj_face = 0.5 * (adj.left[t] + adj.right[v])
            away_t, away_v = +1, -1

        # Forward form: second order with face-extrapolated derivatives and the
        # adjoint's face trace (first order with element averages).
        mu_dpsi_t = mu_dpsi_at_interface(t, away_t)
        mu_dpsi_v = mu_dpsi_at_interface(v, away_v)
        source_term = _dot(src_avg[v] - src_avg[t], psi_face, w)
        operator_fwd = _dot(adj_face, mu_dpsi_v, w) - _dot(adj_face, mu_dpsi_t, w)
        operator_adj = _dot(mu_dadj_t, psi_avg[t], w) - _dot(mu_dadj_v, psi_avg[v], w)

        total_fwd += source_term + operator_fwd
        total_adj += operator_adj
        per_interface.append(dict(x=fwd.x_right[min(t, v)], source=source_term,
                                  operator_forward=operator_fwd, operator_adjoint=operator_adj))

    return dict(forward_form=total_fwd, adjoint_form=total_adj, interfaces=per_interface)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def _source_levels(src: NodalField, mu: np.ndarray) -> dict:
    """Mean |Q+ / mu| over the nonzero entries of each block, to compare cases."""
    vals = np.abs(src.average() / mu[None, None, :])
    out = {}
    for b in np.unique(src.block):
        v = vals[src.block == b]
        out[int(b)] = float(v[v > 0].mean()) if np.any(v > 0) else 0.0
    return out


def _reverse(field: NodalField) -> NodalField:
    """Relabel angle index a -> A-1-a (no quadrature check; mu stays ascending)."""
    return field.with_values(field.left[..., ::-1], field.right[..., ::-1])


def dump_length_per_key_length(field: NodalField, t_v_key: float, cid: int = -1) -> float:
    """
    s = (void width in the dump's x) / (Void_Thickness in case_key.txt), per side.

    The interface formula gives dq/dx_i per unit dump length, and the void
    width moves by s dump lengths per unit Void_Thickness, so
    dq/dt_v[case_key units] = s * dq/dx_i, whatever units the solve used.
    s = 1 if the units already agree. Raises if the two voids differ in width.
    """
    w_l = float(field.h[field.block == BLOCK_VOID_L].sum())
    w_r = float(field.h[field.block == BLOCK_VOID_R].sum())
    if not np.isclose(w_l, w_r, rtol=1e-6):
        raise ValueError(f"case {cid}: left/right void widths differ in the dump "
                         f"({w_l:.8g} vs {w_r:.8g}); the formula assumes both grow together")
    return 0.5 * (w_l + w_r) / t_v_key


STATUS_OK, STATUS_UNCHECKED, STATUS_FAILED = 0, 1, 2


def _vacuum_leak(adj: NodalField, mu: np.ndarray) -> float:
    """
    max |psi+| on the directions where psi+ must vanish (forward-outgoing:
    mu < 0 on the left face, mu > 0 on the right face), relative to max |psi+|.
    Near zero for the right angle convention, independent of units or sources.
    """
    scale = max(np.abs(adj.left).max(), np.abs(adj.right).max(), 1e-300)
    leak = max(np.abs(adj.left[0][:, mu < 0]).max(), np.abs(adj.right[-1][:, mu > 0]).max())
    return float(leak / scale)


def _incoming_fraction(fwd: NodalField, mu: np.ndarray, w: np.ndarray) -> float:
    """Forward partial current entering through both faces / (entering + leaving)."""
    wa = w * np.abs(mu)
    j_in = (np.einsum("ga,a->", fwd.left[0][:, mu > 0], wa[mu > 0])
            + np.einsum("ga,a->", fwd.right[-1][:, mu < 0], wa[mu < 0]))
    j_out = (np.einsum("ga,a->", fwd.left[0][:, mu < 0], wa[mu < 0])
             + np.einsum("ga,a->", fwd.right[-1][:, mu > 0], wa[mu > 0]))
    return float(abs(j_in) / max(abs(j_in) + abs(j_out), 1e-300))


def _forward_vacuum_score(fwd: NodalField, mu: np.ndarray, w: np.ndarray) -> float:
    """
    min over the two faces of (current entering) / (entering + leaving). Small
    when at least one face is vacuum under this angle labelling; a wrong
    labelling swaps entering and leaving, so the vacuum face looks like a source.
    """
    wa = w * np.abs(mu)
    left_in = np.einsum("ga,a->", fwd.left[0][:, mu > 0], wa[mu > 0])
    left_out = np.einsum("ga,a->", fwd.left[0][:, mu < 0], wa[mu < 0])
    right_in = np.einsum("ga,a->", fwd.right[-1][:, mu < 0], wa[mu < 0])
    right_out = np.einsum("ga,a->", fwd.right[-1][:, mu > 0], wa[mu > 0])
    fl = abs(left_in) / max(abs(left_in) + abs(left_out), 1e-300)
    fr = abs(right_in) / max(abs(right_in) + abs(right_out), 1e-300)
    return float(min(fl, fr))


def _particle_split(fwd: NodalField, adj: NodalField, mu: np.ndarray, w: np.ndarray) -> dict:
    """
    Photon (groups 1..ELECTRON_GROUP_MIN-1) vs electron share of: the adjoint's
    magnitude, the boundary term, and the forward current entering the faces.
    """
    n_ph = ELECTRON_GROUP_MIN - 1
    a_all = max(np.abs(adj.left).max(), np.abs(adj.right).max(), 1e-300)
    a_ph = max(np.abs(adj.left[:, :n_ph]).max(), np.abs(adj.right[:, :n_ph]).max()) if n_ph else 0.0
    wl = w * np.abs(mu) * (mu > 0)
    wr = w * np.abs(mu) * (mu < 0)
    bt = (np.einsum("ga,ga,a->g", adj.left[0], fwd.left[0], wl)
          + np.einsum("ga,ga,a->g", adj.right[-1], fwd.right[-1], wr))
    jin = np.einsum("ga,a->g", fwd.left[0], wl) + np.einsum("ga,a->g", fwd.right[-1], wr)
    return dict(adj_photon_rel=float(a_ph / a_all), bt_photon=float(bt[:n_ph].sum()),
                bt_electron=float(bt[n_ph:].sum()), jin_photon=float(jin[:n_ph].sum()),
                jin_electron=float(jin[n_ph:].sum()))


def _source_log_slopes(t_vals: np.ndarray, levels: list) -> tuple:
    """
    d ln c_b / dt_v per block and case, where c_b(t_v) is the Q+ level of block b.
    Q+ = -mu.E and the gap field of a series capacitor goes as 1/(a + b t_v), so
    1/c_b is fitted linearly in t_v; if that fit is poor (max relative residual
    > 1e-3) the slope comes from np.gradient of ln c_b instead.
    Returns ({block: slopes (n_cases,)}, {block: description}).
    """
    slopes, how = {}, {}
    if len(t_vals) < 2:
        return slopes, how
    for b in sorted(set().union(*levels)):
        c = np.array([lv.get(b, 0.0) for lv in levels])
        if c.max() == 0 or np.ptp(c) <= 1e-6 * c.max():
            continue
        if np.any(c <= 0):
            how[b] = "Q+ vanishes in some cases; term skipped"
            continue
        A = np.column_stack([np.ones_like(t_vals), t_vals])
        coef, *_ = np.linalg.lstsq(A, 1.0 / c, rcond=None)
        resid = np.max(np.abs(A @ coef - 1.0 / c) * c)
        if resid <= 1e-3:
            slopes[b] = -coef[1] * c                      # d ln c/dt = -b / (a + b t) = -b c
            how[b] = f"1/c linear in t_v (max rel. residual {resid:.1e})"
        else:
            order = np.argsort(t_vals)
            g = np.empty_like(c)
            g[order] = np.gradient(np.log(c[order]), t_vals[order])
            slopes[b] = g
            how[b] = f"finite differences of ln c (1/c not linear in t_v, residual {resid:.1e})"
    return slopes, how


def cross_case_check(rows) -> None:
    """
    Compare the adjoint dq/dt_v with finite differences of q across the cases
    themselves (np.gradient over sorted t_v: second-order central inside,
    second-order one-sided at the ends). No extra transport runs; accurate when q(t_v) is smooth on the case
    spacing, exact for a linear response. Interior cases are the reliable ones.
    """
    r = np.array([row for row in rows if np.isfinite(row[8])], dtype=float) if rows else None
    if r is None or len(r) < 3:
        return
    order = np.argsort(r[:, 1])
    t, q, g = r[order, 1], r[order, 2], r[order, 8]
    if len(np.unique(t)) < len(t):
        return
    fd = np.gradient(q, t, edge_order=2) if len(t) >= 3 else np.gradient(q, t)
    curv = np.polyfit(t, q, 2)[0] * np.ptp(t) ** 2 / max(np.ptp(q), 1e-300)
    print("\nCross-case check: adjoint dq/dt_v vs finite differences of q between cases")
    print(f"  q(t_v) curvature over the range: {curv:+.2e} of its span "
          "(small = nearly linear, so the finite differences are accurate)")
    print("    case    t_v          q              dq/dt_v used   FD across cases   rel. diff")
    for k in range(len(t)):
        tag = "" if 0 < k < len(t) - 1 else "  (one-sided)"
        print(f"    {int(r[order[k], 0]):4d}  {t[k]:10.5g}  {q[k]:+.6e}  {g[k]:+.6e}  {fd[k]:+.6e}"
              f"     {(g[k] - fd[k]) / max(abs(fd[k]), 1e-300):+.2e}{tag}")
    inner = slice(1, len(t) - 1)
    rel = np.abs(g[inner] - fd[inner]) / np.maximum(np.abs(fd[inner]), 1e-300)
    print(f"  interior cases: median |rel. diff| {np.median(rel):.2e}, max {rel.max():.2e}")


def compare_reference_q(rows, path) -> None:
    """
    Compare q with reference values, e.g. SCEPTRE's q_directdrive, from a text
    file with two columns per line: case_id q (commas or whitespace, '#' comments).
    """
    ref = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].replace(",", " ").split()
        if len(line) >= 2:
            try:
                ref[int(float(line[0]))] = float(line[1])
            except ValueError:
                continue                       # header line
    print(f"\nComparison with reference q from {Path(path).name}:")
    print("    case   q (this script)    reference q       rel. diff")
    diffs = []
    for row in rows:
        cid, q = int(row[0]), row[2]
        if cid in ref and np.isfinite(q):
            d = (q - ref[cid]) / abs(ref[cid])
            diffs.append(abs(d))
            print(f"    {cid:4d}  {q:+.6e}    {ref[cid]:+.6e}    {d:+.2e}")
    if diffs:
        print(f"  max |rel. diff| {max(diffs):.2e} over {len(diffs)} case(s)")
    else:
        print("  no case ids in common")


def _looks_like_potential(src: NodalField) -> bool:
    """True if the 'source' is the same for every angle: a scalar field, not Q+ = -mu.E."""
    scale = max(np.abs(src.left).max(), np.abs(src.right).max(), 1e-300)
    return bool(np.ptp(src.left, axis=2).max() <= 1e-12 * scale
                and np.ptp(src.right, axis=2).max() <= 1e-12 * scale)


def c_label(conv) -> str:
    return f"{conv[0]}, adjoint {'reversed' if conv[1] else 'same'}"


CONVENTIONS = [  # (dump angle order, adjoint relabelled relative to forward)
    ("ascending", False), ("ascending", True), ("descending", False), ("descending", True),
]


def main(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR, adjoint_angles_reversed=None,
         q_mesh: str = "auto", angle_order: str = "auto", q_source: str = "volume",
         weight_sum: float = 2.0, source_term: str = "auto", reference_q=None):
    """
    weight_sum: normalisation of the angular quadrature weights. 2.0 (default)
        is sum w = 2 over mu in [-1, 1]; SCEPTRE's q_directdrive uses 1.0, i.e.
        half the default. q, the boundary term and dq/dt_v all scale with it.
    source_term: whether to add <dQ+/dt_v, psi> from the change of the Q+ dumps
        between cases. "auto" adds it with q_source="volume" and skips it with
        q_source="boundary" (then the Q+ dumps are not the adjoint's source, so
        their t_v dependence says nothing about it). "on"/"off" force it.
    q_source: "volume" uses q = <Q+, psi> from the adjoint-source dump (default).
        "boundary" uses q = sum_incoming w|mu| psi+ psi_in, the functional the
        adjoint flux itself belongs to, for a forward problem driven only by
        flux entering the faces. Use it when the Q+ dump is not the source the
        adjoint was solved with (the identity then fails on every case, with
        a coupled adjoint). Q+ is still used for <dQ+/dt_v, psi>.
    angle_order: "ascending" (index 0 is mu = -1), "descending" (index 0 is
        mu = +1), or "auto". Applies to all three dumps (forward, Q+, adjoint).
    adjoint_angles_reversed: True if the adjoint dump's index is reversed
        relative to the forward dump's; None ("auto") to determine it.
    q_mesh: which length unit q = <Q+, psi> is integrated in.
        "dump"     the dumps' own x (right whenever the CSVs carry the SCEPTRE
                   mesh coordinates, e.g. from exodus_to_csv.py),
        "case_key" x rescaled so the void width equals Void_Thickness (right only
                   if the dumps' x was normalised after the solve),
        "auto"     let the boundary identity choose when it applies, else "dump".
    dq/dt_v does not depend on q_mesh: it is always per unit Void_Thickness.

    Angle convention ("auto"), chosen once for all cases since the dumps come
    from one code: the adjoint must vanish on the faces where the forward flux
    leaves; that fixes the adjoint's orientation but not whether the dumps are
    ascending with a reversed adjoint, or descending with the same order
    throughout. Between those, the convention under which the boundary identity
    holds in the most cases wins; if it holds in none, the labelling whose
    forward flux has a vacuum face. q does not depend on the convention;
    dq/dt_v does.
    """
    if q_mesh not in ("auto", "dump", "case_key"):
        raise ValueError("q_mesh must be 'auto', 'dump' or 'case_key'")
    if source_term not in ("auto", "on", "off"):
        raise ValueError("source_term must be 'auto', 'on' or 'off'")
    use_source_term = source_term == "on" or (source_term == "auto" and q_source == "volume")
    if q_source not in ("volume", "boundary"):
        raise ValueError("q_source must be 'volume' or 'boundary'")
    if angle_order not in ("auto", "ascending", "descending"):
        raise ValueError("angle_order must be 'auto', 'ascending' or 'descending'")
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mu, w = gauss_lobatto(8)
    w = w * (weight_sum / w.sum())
    case_key = load_case_key(data_dir / "case_key.txt")

    print("=" * 72)
    print("dq/dt_v from forward + adjoint dumps (fixed outer radius)")
    print("=" * 72)
    print(f"angular weights sum to {w.sum():g}; q from the {q_source} form; "
          f"<dQ+/dt_v, psi> term {'on' if use_source_term else 'off'}")

    def load_case(cid):
        paths = [data_dir / pat.format(cid) for pat in
                 (FORWARD_PATTERN, ADJOINT_SOURCE_PATTERN, ADJOINT_FLUX_PATTERN)]
        if not (paths[0].exists() and paths[1].exists()):
            return None
        fwd_raw, src_raw = load_nodal_field(paths[0]), load_nodal_field(paths[1])
        if _looks_like_potential(src_raw):
            raise ValueError(
                f"case {cid}: {paths[1].name} is the same for every angle, so it is SCEPTRE's "
                "adjoint-source potential, not the angular source Q+ = -mu.E. Convert it first: "
                "python3 potential_to_source.py DATA_DIR --angle-order descending")
        if not same_mesh(fwd_raw, src_raw):
            raise ValueError(f"case {cid}: forward and adjoint-source meshes differ")
        adj_raw = load_nodal_field(paths[2]) if paths[2].exists() else None
        if adj_raw is not None and not same_mesh(fwd_raw, adj_raw):
            raise ValueError(f"case {cid}: forward and adjoint meshes differ")
        return fwd_raw, src_raw, adj_raw, paths[2]

    def relabel(fwd_raw, src_raw, adj_raw, order, rel_rev):
        desc = order == "descending"
        f = _reverse(fwd_raw) if desc else fwd_raw
        sr = _reverse(src_raw) if desc else src_raw
        a = _reverse(adj_raw) if desc else adj_raw
        return f, sr, (_reverse(a) if rel_rev else a)

    convs = [c for c in CONVENTIONS
             if (angle_order == "auto" or c[0] == angle_order)
             and (adjoint_angles_reversed is None or c[1] == bool(adjoint_angles_reversed))]
    if not np.allclose(mu, -mu[::-1]):
        raise ValueError("angle relabelling needs a quadrature symmetric about mu = 0")

    # ---- Pass 1: score every convention on every case (scores only). ----
    scores, level_t, level_list, level_ids = {}, [], [], []
    for cid in sorted(case_key):
        loaded = load_case(cid)
        if loaded is None:
            continue
        level_ids.append(cid)
        level_t.append(case_key[cid]["Void_Thickness"])
        level_list.append(_source_levels(loaded[1], mu))
        if loaded[2] is None:
            continue
        fwd_raw, src_raw, adj_raw, _ = loaded
        s = dump_length_per_key_length(fwd_raw, case_key[cid]["Void_Thickness"], cid)
        q_dump = charge_functional(fwd_raw, src_raw, w)   # same under any common relabelling
        q_key = q_dump / s
        per = {}
        for conv in convs:
            f, _, a = relabel(fwd_raw, src_raw, adj_raw, *conv)
            qb = boundary_functional(f, a, mu, w)
            applicable = _incoming_fraction(f, mu, w) >= 1e-8 and qb != 0.0
            e_dump = abs(qb - q_dump) / abs(q_dump) if applicable else np.inf
            e_key = abs(qb - q_key) / abs(q_key) if applicable else np.inf
            per[conv] = dict(q_b=qb, leak=_vacuum_leak(a, mu), fvac=_forward_vacuum_score(f, mu, w),
                             applicable=applicable, e_dump=e_dump, e_key=e_key,
                             err=min(e_dump, e_key) if q_mesh == "auto"
                             else (e_dump if q_mesh == "dump" else e_key))
        scores[cid] = per

    # ---- One convention for all cases: the dumps come from one code. ----
    conv_use, how = None, ""
    if scores:
        med_leak = {c: float(np.median([scores[k][c]["leak"] for k in scores])) for c in convs}
        best = min(med_leak.values())
        ok = [c for c in convs if med_leak[c] <= max(10 * best, 1e-6)]
        votes = {c: sum(scores[k][c]["applicable"] and scores[k][c]["err"] <= IDENTITY_TOL
                        for k in scores) for c in ok}
        if max(votes.values()) > 0:
            conv_use = max(ok, key=lambda c: (votes[c], -c[1]))
            how = f"boundary identity holds in {votes[conv_use]}/{len(scores)} case(s)"
        else:
            fv = {c: float(np.median([scores[k][c]["fvac"] for k in scores])) for c in ok}
            conv_use = min(ok, key=lambda c: (fv[c], c[1]))
            how = "forward vacuum face (boundary identity holds in no case)"
        print("\nAngle convention, medians over cases:")
        print("  convention               adj vacuum leak  fwd vacuum face  identity OK in")
        for c in convs:
            fvm = float(np.median([scores[k][c]["fvac"] for k in scores]))
            n_ok = sum(scores[k][c]["applicable"] and scores[k][c]["err"] <= IDENTITY_TOL for k in scores)
            print(f"    {c[0]:10s} adj {'reversed' if c[1] else 'same    '}   {med_leak[c]:.2e}"
                  f"         {fvm:.2e}         {n_ok}/{len(scores)}")
        print(f"  -> using {c_label(conv_use)}  (chosen by {how})")
        if best > 1e-3:
            print("  WARNING: no convention gives a vacuum adjoint on the outer faces; "
                  "check the adjoint dump's boundary conditions and angle order.")

    # ---- Q+ dependence on t_v (adds <dQ+/dt_v, psi> to dq/dt_v). ----
    slopes, slope_how = _source_log_slopes(np.array(level_t, dtype=float), level_list)
    if slope_how:
        print("\nQ+ dumps change with t_v in block(s):")
        for b, h in slope_how.items():
            print(f"  block {b}: {h}")
        if use_source_term:
            print("  -> adding <dQ+/dt_v, psi> = sum_b (d ln c_b/dt_v) <Q+|_b, psi> to dq/dt_v")
        else:
            print("  -> not added (source_term off): the Q+ dumps are not taken as the adjoint's "
                  "source. If the source the adjoint was solved with depends on t_v, dq/dt_v "
                  "misses that term; the cross-case check shows it.")
    if not use_source_term:
        slopes = {}
    slope_at = {cid: {b: float(v[level_ids.index(cid)]) for b, v in slopes.items()}
                for cid in level_ids}

    # ---- Pass 2: q and dq/dt_v under that convention. ----
    rows, src_levels, scales, meshes, identity_failures = [], [], [], [], []
    for cid in sorted(case_key):
        t_v = case_key[cid]["Void_Thickness"]
        loaded = load_case(cid)
        if loaded is None:
            print(f"\nCase {cid:04d}: [skip] forward or adjoint-source dump missing")
            continue
        fwd_raw, src_raw, adj_raw, a_path = loaded
        s = dump_length_per_key_length(fwd_raw, t_v, cid)
        scales.append(s)
        src_levels.append(_source_levels(src_raw, mu))
        q_dump = charge_functional(fwd_raw, src_raw, w)
        q_key = q_dump / s
        print(f"\nCase {cid:04d}  t_v = {t_v:.6g}")
        print(f"  dump void width / t_v       = {s:.6g}  (dump x per case_key length unit)")
        print(f"  q = <Q+, psi>, dump mesh    = {q_dump:.6e}")
        if not np.isclose(s, 1.0):
            print(f"  q = <Q+, psi>, case_key mesh= {q_key:.6e}")

        q = q_dump if q_mesh != "case_key" else q_key
        len_factor = 1.0 if q_mesh != "case_key" else 1.0 / s   # length unit of volume integrals
        q_b = dq_f = dq_a = dq_use = np.nan
        status = STATUS_UNCHECKED
        use_form = "adjoint"
        if adj_raw is None:
            print(f"  [no adjoint solution: {a_path.name}]")
        else:
            sc = scores[cid][conv_use]
            fwd, src, adj = relabel(fwd_raw, src_raw, adj_raw, *conv_use)
            q_b = sc["q_b"]
            print(f"  adjoint vacuum-face leak    = {sc['leak']:.2e}")
            ps = _particle_split(fwd, adj, mu, w)
            print(f"  photon share: max|psi+| photon/all = {ps['adj_photon_rel']:.2e};  entering current "
                  f"photon {ps['jin_photon']:.3e}, electron {ps['jin_electron']:.3e};  boundary term "
                  f"photon {ps['bt_photon']:+.3e}, electron {ps['bt_electron']:+.3e}")
            electron_only = (ps["adj_photon_rel"] <= 1e-6
                             and abs(ps["jin_photon"]) > 10 * abs(ps["jin_electron"]))
            if electron_only:
                use_form = "forward"
                print("  The adjoint has no photon importance but photons carry the entering current, "
                      "so it is an electron-only adjoint: photons act as a fixed electron source. "
                      "The boundary identity cannot hold and is skipped; the adjoint form misses the "
                      "source jump at the interface, so the forward form is used. Neither includes "
                      "the change of photon attenuation with t_v; check with finite differences.")
            elif not sc["applicable"]:
                print("  boundary identity not applicable: no forward flux enters through the "
                      "faces, so the source is internal. Rely on a finite-difference check in t_v.")
            else:
                print(f"  boundary term sum w|mu|psi+psi_in = {q_b:.6e}   identity miss: "
                      f"{sc['e_dump']:.2e} on dump mesh, {sc['e_key']:.2e} on case_key mesh")
                if q_mesh == "auto" and sc["err"] <= IDENTITY_TOL:
                    mesh = ("dump" if sc["e_dump"] <= sc["e_key"] or np.isclose(s, 1.0, rtol=1e-6)
                            else "case_key")
                    q = q_dump if mesh == "dump" else q_key
                    len_factor = 1.0 if mesh == "dump" else 1.0 / s
                    meshes.append(mesh)
                if sc["err"] <= IDENTITY_TOL:
                    status = STATUS_OK
                elif q_source == "boundary":
                    print(f"  (the Q+ dump misses the boundary term by {sc['err']:.1%}; expected "
                          "with --q-source boundary, which does not use the dump for q)")
                else:
                    status = STATUS_FAILED
                    identity_failures.append(cid)
                    print(f"  WARNING: boundary identity misses q by {sc['err']:.1%}.")
                    # Which part of the Q+ dump does the adjoint actually belong to?
                    print("  Q+ consistency: boundary term / <Q+, psi> restricted to")
                    variants = [("all blocks", None), ("gap only (2,3,5,6)", GAP_BLOCKS),
                                ("all but core (4)", (1, 2, 3, 5, 6, 7)),
                                ("all but shields (1,7)", (2, 3, 4, 5, 6))]
                    for name, blocks in variants:
                        mask = None if blocks is None else np.isin(src_raw.block, blocks)
                        qv = volume_inner_product(src_raw, fwd_raw, w, elem_mask=mask)
                        ratio = q_b / qv if qv != 0 else np.inf
                        print(f"    {name:22s} <Q+,psi> = {qv:+.6e}   ratio {ratio:+.4f}")
                if q_source == "boundary":
                    q = q_b
                    status = STATUS_UNCHECKED
                    print("  q taken from the boundary term (q_source='boundary'): consistent with "
                          "the adjoint flux, not checked against the Q+ dump.")

            # Interface formula on the dump mesh gives dq/dx_i per dump length; the void
            # width moves by s dump lengths per unit Void_Thickness.
            sens = void_thickness_sensitivity(fwd, adj, src, mu, w)
            dq_f, dq_a = s * sens["forward_form"], s * sens["adjoint_form"]
            # <dQ+/dt_v, psi>, in the same length unit as q.
            dsrc = 0.0
            for b, slope in slope_at.get(cid, {}).items():
                dsrc += slope * volume_inner_product(src_raw, fwd_raw, w, elem_mask=src_raw.block == b)
            dsrc *= len_factor
            if slope_at.get(cid):
                print(f"  <dQ+/dt_v, psi>             = {dsrc:.6e}   (added to both forms)")
            dq_f, dq_a = dq_f + dsrc, dq_a + dsrc
            dq_use = dq_f if use_form == "forward" else dq_a
            print(f"  dq/dt_v forward form        = {dq_f:.6e}   (per case_key unit)")
            print(f"  dq/dt_v adjoint form        = {dq_a:.6e}   "
                  f"(forms differ by {abs(dq_f - dq_a) / max(abs(dq_a), 1e-300):.1%})")
            print(f"  dq/dt_v used                = {dq_use:.6e}   ({use_form} form)")
            for k, it in enumerate(sens["interfaces"]):
                print(f"    interface {k} at x={it['x']:.5f}: source {it['source']:+.3e}, "
                      f"operator fwd {it['operator_forward']:+.3e}, adj {it['operator_adjoint']:+.3e}")
        print(f"  q used                      = {q:.6e}   status: "
              f"{['identity OK', 'unchecked', 'identity FAILED'][status]}")
        rows.append((cid, t_v, q, q_b, dq_f, dq_a, s, status, dq_use))

    if len(src_levels) > 1:
        changed = []
        for b in sorted(set().union(*src_levels)):
            lv = np.array([d.get(b, 0.0) for d in src_levels])
            if lv.max() > 0 and np.ptp(lv) > 1e-6 * lv.max():
                changed.append(f"{b} ({lv.min():.4g}..{lv.max():.4g})")
        if changed and not slopes and use_source_term:
            print("\nWARNING: the adjoint source Q+ changes between cases in block(s) "
                  + ", ".join(changed) + " but no slope could be fitted, so dq/dt_v is missing "
                  "a <dQ+/dt_v, psi> term.")
    if len(scales) > 1 and np.ptp(scales) > 1e-3 * np.mean(scales):
        print("\nWARNING: dump void width / t_v varies between cases "
              f"({min(scales):.6g} to {max(scales):.6g}). Either case_key.txt does not match "
              "the dumps, or each case is normalised by a different total length, in which "
              "case q itself is not comparable across cases.")
    if len(set(meshes)) > 1:
        print("\nWARNING: the boundary identity picked different q meshes for different cases "
              f"({meshes}); q values are on mixed units. Force q_mesh='dump' or 'case_key'.")
    cross_case_check(rows)
    if reference_q is not None:
        compare_reference_q(rows, reference_q)
    if identity_failures and q_source == "volume" and len(identity_failures) == len(scores):
        print("\nThe boundary identity failed on every case with a coupled adjoint and flux "
              "entering the faces. The likeliest cause is that the adjoint-source dump is not the "
              "source SCEPTRE's adjoint was solved with (see the Q+ consistency ratios: the variant "
              "with ratio ~1 is the one it used). Either fix the Q+ dumps, or take q from the "
              "boundary term: python3 void_sensitivity.py DATA_DIR --q-source boundary")
    if conv_use is not None:
        print(f"\nAngle convention used for every case: {c_label(conv_use)}")

    table = np.array(rows, dtype=float)
    out_path = out_dir / "qoi_void_adjoint.txt"
    np.savetxt(out_path, table,
               header="case_id void_thickness q q_boundary dq_dtv_forward_form "
                      "dq_dtv_adjoint_form dump_per_key_length status dq_dtv_used\n"
                      "dq_dtv_* are per unit of case_key Void_Thickness and include <dQ+/dt_v, psi>; "
                      "status 0 = identity OK, 1 = unchecked (internal source or electron-only "
                      "adjoint), 2 = identity failed; dq_dtv_used is the form GE-GPR should use",
               fmt=["%d"] + ["%.8e"] * 6 + ["%d", "%.8e"])
    counts = np.bincount(table[:, 7].astype(int), minlength=3) if len(rows) else [0, 0, 0]
    print(f"\nWrote {out_path}  ({len(rows)} case(s): {counts[0]} OK, {counts[1]} unchecked, "
          f"{counts[2]} failed)")
    return table


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="dq/dt_v from forward + adjoint dumps")
    ap.add_argument("data_dir", nargs="?", default=str(DATA_DIR))
    ap.add_argument("--q-mesh", default="auto", choices=["auto", "dump", "case_key"])
    ap.add_argument("--angle-order", default="auto", choices=["auto", "ascending", "descending"])
    ap.add_argument("--q-source", default="volume", choices=["volume", "boundary"])
    ap.add_argument("--weight-sum", type=float, default=2.0,
                    help="angular weight normalisation; 1 matches SCEPTRE's q_directdrive")
    ap.add_argument("--source-term", default="auto", choices=["auto", "on", "off"],
                    help="add <dQ+/dt_v, psi> from the Q+ dumps (auto: only with --q-source volume)")
    ap.add_argument("--reference-q", metavar="FILE",
                    help="two-column file 'case_id q' to compare q with (e.g. SCEPTRE q_directdrive)")
    a = ap.parse_args()
    main(data_dir=a.data_dir, q_mesh=a.q_mesh, angle_order=a.angle_order, q_source=a.q_source,
         weight_sum=a.weight_sum, source_term=a.source_term, reference_q=a.reference_q)
