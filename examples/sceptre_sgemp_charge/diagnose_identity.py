"""
diagnose_identity.py: why does void_sensitivity.py's boundary identity fail?

The identity  q = <Q+, psi> = sum_incoming w |mu| psi+ psi_in  needs four things:
  1. the forward problem is driven ONLY by incoming flux on the two faces
     (no volume source),
  2. the adjoint dump obeys vacuum conditions where the forward flux enters
     (psi+ = 0 on the faces' forward-outgoing directions),
  3. the adjoint dump's angle index means the same mu as the forward dump's,
  4. the mesh is in the units the transport solve used (q scales with length,
     the boundary term does not).
This script checks each one separately on every case and prints a summary.

    python3 diagnose_identity.py [DATA_DIR]      (default: same as void_sensitivity.py)

Paste the output back. Nothing is written.
"""

import sys
from pathlib import Path

import numpy as np

from extract_qoi import DATA_DIR, gauss_lobatto, load_case_key, ELECTRON_GROUP_MIN
from void_sensitivity import (
    FORWARD_PATTERN, ADJOINT_SOURCE_PATTERN, ADJOINT_FLUX_PATTERN,
    load_nodal_field, charge_functional, reverse_angles, dump_length_per_key_length,
)

BLOCK_NAMES = {1: "Shield_L", 2: "Diel_L", 3: "Void_L", 4: "Core",
               5: "Void_R", 6: "Diel_R", 7: "Shield_R"}


def partial_currents(field, mu, w):
    """Partial currents sum w|mu| psi on each face, split into entering/leaving for +mu flow."""
    jl_in = float(np.einsum("ga,a->", field.left[0][:, mu > 0], (w * np.abs(mu))[mu > 0]))
    jl_out = float(np.einsum("ga,a->", field.left[0][:, mu < 0], (w * np.abs(mu))[mu < 0]))
    jr_in = float(np.einsum("ga,a->", field.right[-1][:, mu < 0], (w * np.abs(mu))[mu < 0]))
    jr_out = float(np.einsum("ga,a->", field.right[-1][:, mu > 0], (w * np.abs(mu))[mu > 0]))
    return jl_in, jl_out, jr_in, jr_out


def boundary_terms(fwd, adj, mu, w):
    """Left-face and right-face pieces of sum_incoming w|mu| psi+ psi_in."""
    wl = w * np.abs(mu) * (mu > 0)
    wr = w * np.abs(mu) * (mu < 0)
    return (float(np.einsum("ga,ga,a->", adj.left[0], fwd.left[0], wl)),
            float(np.einsum("ga,ga,a->", adj.right[-1], fwd.right[-1], wr)))


def q_by(fwd, src, w, labels):
    """Contribution to <Q+, psi> per block and photon/electron."""
    integ = (2 * src.left * fwd.left + src.left * fwd.right
             + src.right * fwd.left + 2 * src.right * fwd.right) / 6.0
    per = fwd.h[:, None] * np.einsum("ega,a->eg", integ, w)          # (E, G)
    by_block = {b: float(per[fwd.block == b].sum()) for b in labels}
    n_ph = ELECTRON_GROUP_MIN - 1
    return by_block, float(per[:, :n_ph].sum()), float(per[:, n_ph:].sum())


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def main(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    mu, w = gauss_lobatto(8)
    key = load_case_key(data_dir / "case_key.txt")
    print(f"Data: {data_dir}")
    print(f"Quadrature (Gauss-Lobatto 8): mu = {np.round(mu, 4).tolist()}, sum w = {w.sum():.3f}\n")

    rows = []
    for cid in sorted(key):
        paths = [data_dir / p.format(cid) for p in
                 (FORWARD_PATTERN, ADJOINT_SOURCE_PATTERN, ADJOINT_FLUX_PATTERN)]
        if not all(p.exists() for p in paths):
            print(f"Case {cid:04d}: missing {[p.name for p in paths if not p.exists()]}\n")
            continue
        fwd, src, adj = (load_nodal_field(p) for p in paths)
        t_v = key[cid]["Void_Thickness"]
        s = dump_length_per_key_length(fwd, t_v, cid)
        L = fwd.x_right[-1] - fwd.x_left[0]
        print(f"Case {cid:04d}  t_v = {t_v:.6g}   dump x in [{fwd.x_left[0]:.6g}, {fwd.x_right[-1]:.6g}]"
              f"   void width / t_v = s = {s:.6g}")
        print(f"  shapes: fwd {fwd.left.shape}  Q+ {src.left.shape}  adj {adj.left.shape}  (E, G, A)")

        # 1. Is the forward problem boundary-driven?
        jl_in, jl_out, jr_in, jr_out = partial_currents(fwd, mu, w)
        print(f"  [1] forward partial currents: left face in {jl_in:.3e} / out {jl_out:.3e};"
              f"  right face in {jr_in:.3e} / out {jr_out:.3e}")

        # 2./3. Adjoint vacuum faces: psi+ should vanish where the forward flux LEAVES.
        for name, a in (("same mu", adj), ("reversed mu", reverse_angles(adj, mu))):
            scale = max(np.abs(a.left).max(), np.abs(a.right).max(), 1e-300)
            leak = max(np.abs(a.left[0][:, mu < 0]).max(), np.abs(a.right[-1][:, mu > 0]).max())
            print(f"  [2] adjoint ({name:11s}): max |psi+| on its vacuum faces / max |psi+| = {leak / scale:.2e}")

        # Q+ structure.
        blocks = sorted(set(fwd.block.tolist()))
        by_block, q_ph, q_el = q_by(fwd, src, w, blocks)
        q_dump = charge_functional(fwd, src, w)
        parts = ", ".join(f"{BLOCK_NAMES.get(b, b)} {v / q_dump:+.2f}" for b, v in by_block.items()
                          if abs(v) > 1e-6 * abs(q_dump))
        print(f"  Q+ . psi share by block: {parts}")
        print(f"  Q+ . psi share: photon groups {q_ph / q_dump:+.2f}, electron groups {q_el / q_dump:+.2f}")

        # 4. Units: q on the dump mesh vs. on the case_key mesh, against the boundary term.
        bl_s, br_s = boundary_terms(fwd, adj, mu, w)
        bl_r, br_r = boundary_terms(fwd, reverse_angles(adj, mu), mu, w)
        qb_s, qb_r = bl_s + br_s, bl_r + br_r
        print(f"  q = <Q+,psi> on dump mesh      = {q_dump:+.6e}")
        print(f"  q = <Q+,psi> on case_key mesh  = {q_dump / s:+.6e}   (what void_sensitivity.py now uses)")
        print(f"  boundary term, same mu         = {qb_s:+.6e}   (left {bl_s:+.3e}, right {br_s:+.3e})")
        print(f"  boundary term, reversed mu     = {qb_r:+.6e}   (left {bl_r:+.3e}, right {br_r:+.3e})")
        for name, qb in (("same", qb_s), ("reversed", qb_r)):
            print(f"  [4] {name:8s}: boundary / q_dump = {qb / q_dump:+.5g}   "
                  f"boundary / q_case_key = {qb / (q_dump / s):+.5g}")
        rows.append((cid, s, L, q_dump, qb_s, qb_r, jl_in + jr_in, jl_out + jr_out))
        print()

    if not rows:
        return
    r = np.array(rows)
    print("=" * 78)
    print("Summary across cases")
    print("=" * 78)
    s, L, q, qb_s, qb_r, j_in, j_out = r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6], r[:, 7]
    print(f"  s (dump void width / t_v): {s.min():.6g} .. {s.max():.6g}")
    print(f"  dump domain length       : {L.min():.6g} .. {L.max():.6g}")
    for name, qb in (("same mu", qb_s), ("reversed mu", qb_r)):
        ratio = qb / q
        print(f"  boundary/q_dump, {name:11s}: mean {ratio.mean():+.5g}, spread {np.ptp(ratio) / max(abs(ratio.mean()), 1e-300):.1%}")
    print(f"  total incoming forward current / outgoing: {(j_in / np.maximum(j_out, 1e-300)).min():.3e} .. "
          f"{(j_in / np.maximum(j_out, 1e-300)).max():.3e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
