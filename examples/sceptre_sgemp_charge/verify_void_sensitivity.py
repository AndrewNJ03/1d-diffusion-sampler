"""
verify_void_sensitivity.py: checks void_sensitivity.py's interface formula
for dq/dt_v against finite differences on a synthetic problem with a known
answer, using the one-group diamond-difference S_N solver in sn_transport/.

Synthetic slab (mirror symmetric, outer radius fixed, so t_d = G - t_v):

    | shield | dielectric | void | core | void | dielectric | shield |

The forward problem has an isotropic incoming flux on both faces and no volume
source. The adjoint source mimics the real dumps: Q+ = c(x) mu on the gap,
sign -1 on the left half and +1 on the right. The adjoint flux is obtained
by solving the forward problem with mirrored angles (H+ = H for one group with
isotropic scattering) and relabelling mu -> -mu.

Cases:
  A. vacuum void, Q+ uniform over void and dielectric (the structure of the
     real adjoint-source dumps)
  B. absorbing "void", Q+ jumps at the interface (exercises the source term
     and the void-side derivative)

The finite difference perturbs t_v with a fixed number of cells per region
(widths scale), so the mesh moves smoothly and the FD has no remeshing noise.
Both sides discretize the same continuum derivative; their gap should shrink
roughly linearly with the cell width.
"""

import os
import sys

import numpy as np
from scipy.sparse.linalg import splu

from void_sensitivity import (NodalField, charge_functional, boundary_functional,
                              void_thickness_sensitivity, reverse_angles)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sn_transport"))
from sn_solver import gauss_legendre, assemble_system, inflow_edges  # noqa: E402

# Block ids as in the real dumps.
SHIELD_L, DIEL_L, VOID_L, CORE, VOID_R, DIEL_R, SHIELD_R = 1, 2, 3, 4, 5, 6, 7

T_SHIELD, T_CORE, GAP = 0.5, 1.0, 1.0       # cm; t_v + t_d = GAP
MATERIALS = {   # (Sigma_t, Sigma_s) [1/cm]
    "shield": (2.0, 1.0),
    "dielectric": (4.0, 3.0),
    "core": (3.0, 1.5),
}
CASES = {
    "A: vacuum void, uniform Q+": dict(void=(0.0, 0.0), c_diel=1.0, c_void=1.0),
    "B: absorbing void, Q+ jump": dict(void=(0.3, 0.1), c_diel=1.0 / 2.1, c_void=1.0),
}


def build_problem(t_v, cells_per_cm, void_xs, c_diel, c_void):
    """Fixed cell counts per region (set at the reference t_v = 0.3) so widths scale with t_v."""
    t_d = GAP - t_v
    regions = [  # (block, material, thickness, reference thickness)
        (SHIELD_L, "shield", T_SHIELD, T_SHIELD), (DIEL_L, "dielectric", t_d, GAP - 0.3),
        (VOID_L, "void", t_v, 0.3), (CORE, "core", T_CORE, T_CORE),
        (VOID_R, "void", t_v, 0.3), (DIEL_R, "dielectric", t_d, GAP - 0.3),
        (SHIELD_R, "shield", T_SHIELD, T_SHIELD),
    ]
    faces, block, sig_t, sig_s, c = [0.0], [], [], [], []
    for blk, mat, t, t_ref in regions:
        n = max(1, int(round(t_ref * cells_per_cm)))
        faces.extend(faces[-1] + t * np.arange(1, n + 1) / n)
        st, ss = void_xs if mat == "void" else MATERIALS[mat]
        side = -1.0 if blk < CORE else 1.0
        cq = {"dielectric": c_diel, "void": c_void}.get(mat, 0.0) * side
        block += [blk] * n; sig_t += [st] * n; sig_s += [ss] * n; c += [cq] * n
    faces = np.asarray(faces)
    return faces[:-1], faces[1:], np.array(block), np.array(sig_t), np.array(sig_s), np.array(c)


def _solve(h, sig_t, sig_s, mu, w, psi_inc=None, angular_source=None):
    """DD S_N solve with an optional angular volume source S(i, n); returns node values (N, 1, A)."""
    A, rhs = assemble_system(h, sig_t, sig_s, np.zeros_like(h), mu, w, psi_inc)
    if angular_source is not None:
        rhs = rhs + (h[:, None] * angular_source).ravel()
    e = splu(A).solve(rhs).reshape(len(h), len(mu))
    e_in = inflow_edges(e, mu, psi_inc)
    left = np.where(mu > 0, e_in, e)
    right = np.where(mu > 0, e, e_in)
    return left[:, None, :], right[:, None, :]


def solve_case(t_v, cells_per_cm, n_angles, case):
    mu, w = gauss_legendre(n_angles)
    xl, xr, block, sig_t, sig_s, c = build_problem(t_v, cells_per_cm, case["void"],
                                                   case["c_diel"], case["c_void"])
    h = xr - xl
    psi_inc = np.ones(n_angles)          # isotropic incoming flux on both faces

    fl, fr = _solve(h, sig_t, sig_s, mu, w, psi_inc=psi_inc)
    fwd = NodalField(xl, xr, block, fl, fr)

    q_src = c[:, None] * mu[None, :]     # Q+(i, a), constant per element
    src = NodalField(xl, xr, block, q_src[:, None, :], q_src[:, None, :])

    # Adjoint: forward solve with source Q+(x, -mu), vacuum inflow, then mu -> -mu.
    al, ar = _solve(h, sig_t, sig_s, mu, w, angular_source=q_src[:, ::-1])
    adj = reverse_angles(NodalField(xl, xr, block, al, ar), mu)
    return fwd, adj, src, mu, w


def main(t_v=0.3, delta=1e-4, n_angles=16, densities=(25, 50, 100, 200, 400)):
    print("=" * 78)
    print(f"Interface sensitivity vs. central FD in t_v  (t_v={t_v}, S{n_angles}, FD step {delta})")
    print("=" * 78)
    for name, case in CASES.items():
        print(f"\n{name}")
        print(f"  {'cells/cm':>8}  {'q':>11}  {'q adj/q-1':>9}  {'FD':>11}  "
              f"{'fwd form':>11}  {'err':>8}  {'adj form':>11}  {'err':>8}")
        for n in densities:
            fwd, adj, src, mu, w = solve_case(t_v, n, n_angles, case)
            q = charge_functional(fwd, src, w)
            q_bnd = boundary_functional(fwd, adj, mu, w)
            q_p = charge_functional(*_fwd_src(t_v + delta, n, n_angles, case))
            q_m = charge_functional(*_fwd_src(t_v - delta, n, n_angles, case))
            fd = (q_p - q_m) / (2 * delta)
            s = void_thickness_sensitivity(fwd, adj, src, mu, w)
            ef = s["forward_form"] / fd - 1
            ea = s["adjoint_form"] / fd - 1
            print(f"  {n:8d}  {q:11.4e}  {q_bnd / q - 1:9.1e}  {fd:11.4e}  "
                  f"{s['forward_form']:11.4e}  {ef:8.1e}  {s['adjoint_form']:11.4e}  {ea:8.1e}")


def _fwd_src(t_v, n, n_angles, case):
    fwd, _, src, _, w = solve_case(t_v, n, n_angles, case)
    return fwd, src, w


if __name__ == "__main__":
    main()
