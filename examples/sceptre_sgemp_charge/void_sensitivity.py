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
form converges at first order (5% at 25 cells/cm, 0.3% at 400).

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from extract_qoi import (
    COL_X_LEFT, COL_X_RIGHT, COL_ELEM_ID, COL_BLOCK_ID, COL_ENERGY_GROUP,
    COL_ANGLE_IDX, COL_LEFT_FLUX, COL_RIGHT_FLUX,
    BLOCK_DIELECTRIC_L, BLOCK_VOID_L, BLOCK_VOID_R, BLOCK_DIELECTRIC_R,
    DATA_DIR, OUT_DIR, gauss_lobatto, load_case_key,
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

    total_fwd, total_adj, per_interface = 0.0, 0.0, []
    for t, v in pairs:
        ht, hv = fwd.h[t], fwd.h[v]
        mu_dpsi_t = element_streaming(fwd, mu, t, +1) / ht
        mu_dpsi_v = element_streaming(fwd, mu, v, +1) / hv
        mu_dadj_t = element_streaming(adj, mu, t, -1) / ht
        mu_dadj_v = element_streaming(adj, mu, v, -1) / hv

        # Forward flux at x_i: mean of the two traces.
        if t < v:
            psi_face = 0.5 * (fwd.right[t] + fwd.left[v])
        else:
            psi_face = 0.5 * (fwd.left[t] + fwd.right[v])

        source_term = _dot(src_avg[v] - src_avg[t], psi_face, w)
        operator_fwd = _dot(adj_avg[v], mu_dpsi_v, w) - _dot(adj_avg[t], mu_dpsi_t, w)
        operator_adj = _dot(mu_dadj_t, psi_avg[t], w) - _dot(mu_dadj_v, psi_avg[v], w)

        total_fwd += source_term + operator_fwd
        total_adj += operator_adj
        per_interface.append(dict(x=fwd.x_right[min(t, v)], source=source_term,
                                  operator_forward=operator_fwd, operator_adjoint=operator_adj))

    return dict(forward_form=total_fwd, adjoint_form=total_adj, interfaces=per_interface)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def _gap_source_level(src: NodalField, mu: np.ndarray) -> float:
    """Mean |Q+ / mu| over gap elements and nonzero entries, to compare cases."""
    gap = np.isin(src.block, GAP_BLOCKS)
    vals = np.abs(src.average()[gap] / mu[None, None, :])
    return float(vals[vals > 0].mean())


def main(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR, adjoint_angles_reversed=None):
    """
    adjoint_angles_reversed: None picks the convention whose boundary identity
    matches q best; True/False forces it.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mu, w = gauss_lobatto(8)
    case_key = load_case_key(data_dir / "case_key.txt")

    print("=" * 72)
    print("dq/dt_v from forward + adjoint dumps (fixed outer radius)")
    print("=" * 72)

    rows, gap_levels = [], []
    for cid in sorted(case_key):
        f_path = data_dir / FORWARD_PATTERN.format(cid)
        s_path = data_dir / ADJOINT_SOURCE_PATTERN.format(cid)
        a_path = data_dir / ADJOINT_FLUX_PATTERN.format(cid)
        t_v = case_key[cid]["Void_Thickness"]
        if not (f_path.exists() and s_path.exists()):
            print(f"\nCase {cid:04d}: [skip] forward or adjoint-source dump missing")
            continue

        fwd, src = load_nodal_field(f_path), load_nodal_field(s_path)
        if not same_mesh(fwd, src):
            raise ValueError(f"case {cid}: forward and adjoint-source meshes differ")
        q = charge_functional(fwd, src, w)
        gap_levels.append(_gap_source_level(src, mu))
        print(f"\nCase {cid:04d}  t_v = {t_v:.6g}")
        print(f"  q = <Q+, psi>               = {q:.6e}")

        q_adj = dq_f = dq_a = np.nan
        if a_path.exists():
            adj = load_nodal_field(a_path)
            if not same_mesh(fwd, adj):
                raise ValueError(f"case {cid}: forward and adjoint meshes differ")
            q_same = boundary_functional(fwd, adj, mu, w)
            q_rev = boundary_functional(fwd, reverse_angles(adj, mu), mu, w)
            print(f"  q from adjoint (same mu)    = {q_same:.6e}   rel. diff {abs(q_same - q) / abs(q):.2e}")
            print(f"  q from adjoint (reversed mu)= {q_rev:.6e}   rel. diff {abs(q_rev - q) / abs(q):.2e}")
            reverse = (abs(q_rev - q) < abs(q_same - q)) if adjoint_angles_reversed is None \
                else adjoint_angles_reversed
            if reverse:
                adj = reverse_angles(adj, mu)
            q_adj = q_rev if reverse else q_same

            sens = void_thickness_sensitivity(fwd, adj, src, mu, w)
            dq_f, dq_a = sens["forward_form"], sens["adjoint_form"]
            print(f"  dq/dt_v forward form        = {dq_f:.6e}")
            print(f"  dq/dt_v adjoint form        = {dq_a:.6e}   "
                  f"(forms differ by {abs(dq_f - dq_a) / max(abs(dq_a), 1e-300):.1%})")
            for k, it in enumerate(sens["interfaces"]):
                print(f"    interface {k} at x={it['x']:.5f}: source {it['source']:+.3e}, "
                      f"operator fwd {it['operator_forward']:+.3e}, adj {it['operator_adjoint']:+.3e}")
        else:
            print(f"  [no adjoint solution: {a_path.name}]")
        rows.append((cid, t_v, q, q_adj, dq_f, dq_a))

    if len(gap_levels) > 1 and np.ptp(gap_levels) > 1e-8 * max(gap_levels):
        print("\nWARNING: the adjoint source in the gap changes between cases, so Q+ depends "
              "on t_v and the derivative is missing a <dQ+/dt_v, psi> term.")

    table = np.array(rows, dtype=float)
    out_path = out_dir / "qoi_void_adjoint.txt"
    np.savetxt(out_path, table,
               header="case_id void_thickness q q_adjoint_boundary dq_dtv_forward_form dq_dtv_adjoint_form",
               fmt=["%d"] + ["%.8e"] * 5)
    print(f"\nWrote {out_path}  ({len(rows)} case(s))")
    return table


if __name__ == "__main__":
    main()
