"""
verify_geometry_sensitivity.py: checks the independent-thickness gradients in
void_sensitivity.py against finite differences on a synthetic problem with a known
answer, using the one-group diamond-difference S_N solver in sn_transport/.

The slab is the one in verify_void_sensitivity.py, except that t_v, t_d and t_s vary
independently with the core fixed, so the outer faces move. Cell counts per region are
fixed, so widths scale with thickness and the finite difference has no remeshing noise.

Cases:
  A. vacuum void, Q+ uniform over void and dielectric
  B. absorbing "void", Q+ jumps at the void/dielectric interface
  C. fixed source in the dielectric and vacuum faces. Only the forward form holds in
     that layer, which is why void_sensitivity.py selects the form per case.
  D. as B, plus one-element Q+ spikes in the conductor elements bordering the gap,
     with fixed integral, as in the real adjoint-source dumps
  E. as B, with a series-capacitor Q+ level that depends on t_v and t_d, which brings
     in the <dQ+/dt, psi> term from _source_log_slopes_multi
  F. as A, with a single-element void, as in the thinnest real cases
  G. as D, with the meshing make_geometry_cases.py prescribes: a fixed shield element
     size, t_s on whole elements and FD steps of one element. The spike element keeps
     its width, so this removes the error D shows on t_s.

--end-to-end also writes these problems as per-element CSV dumps for a small design
from make_geometry_cases.py and runs void_sensitivity.py, check_gradients_fd.py and
compare_ge_vs_gp.py on them.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from extract_qoi import ELECTRON_GROUP_MIN, gauss_lobatto
from verify_void_sensitivity import _solve, gauss_legendre
from void_sensitivity import (GEOMETRY_PARAMS, NodalField, _source_log_slopes_multi, _source_weights,
                              charge_functional, geometry_sensitivity, reverse_angles,
                              void_thickness_sensitivity, volume_inner_product)
from void_sensitivity import main as sensitivity_main

SHIELD_L, DIEL_L, VOID_L, CORE, VOID_R, DIEL_R, SHIELD_R = 1, 2, 3, 4, 5, 6, 7
EPS = 2.1
T_CORE = 1.0
REFERENCE = np.array([0.3, 0.7, 0.5])     # t_v, t_d, t_s that set the cell counts
MATERIALS = {"shield": (2.0, 1.0), "dielectric": (4.0, 3.0), "core": (3.0, 1.5)}
ABSORBING_VOID = (0.3, 0.1)
CASES = {
    "A: vacuum void, uniform Q+": dict(void=(0.0, 0.0), c_diel=1.0, c_void=1.0),
    "B: absorbing void, Q+ jump": dict(void=ABSORBING_VOID, c_diel=1 / EPS, c_void=1.0),
    "C: dielectric source, vacuum faces": dict(void=ABSORBING_VOID, c_diel=1 / EPS, c_void=1.0, diel_source=1.0),
    "D: Q+ spikes in conductors": dict(void=ABSORBING_VOID, c_diel=1 / EPS, c_void=1.0, spike=0.05),
    "E: Q+ level depends on thickness": dict(void=ABSORBING_VOID, series=True),
    "F: single-element void": dict(void=(0.0, 0.0), c_diel=1.0, c_void=1.0, void_cells=1),
    "G: Q+ spikes, fixed shield element size": dict(void=ABSORBING_VOID, c_diel=1 / EPS, c_void=1.0,
                                                    spike=0.05, shield_h=True),
}


def shield_element(cells_per_cm, case):
    """Fixed shield element size, or None if shield widths scale with t_s."""
    h = case.get("shield_h")
    if h is True:
        return REFERENCE[2] / max(2, int(round(REFERENCE[2] * cells_per_cm)))
    return h


def source_levels(t, case):
    """Q+ level (c_void, c_diel) for thicknesses t."""
    if case.get("series"):
        c_void = 1.0 / (t[0] + t[1] / EPS)
        return c_void, c_void / EPS
    return case["c_void"], case["c_diel"]


def build_problem(t, cells_per_cm, case):
    t_v, t_d, t_s = t
    regions = [(SHIELD_L, "shield", t_s, REFERENCE[2]), (DIEL_L, "dielectric", t_d, REFERENCE[1]),
               (VOID_L, "void", t_v, REFERENCE[0]), (CORE, "core", T_CORE, T_CORE),
               (VOID_R, "void", t_v, REFERENCE[0]), (DIEL_R, "dielectric", t_d, REFERENCE[1]),
               (SHIELD_R, "shield", t_s, REFERENCE[2])]
    c_void, c_diel = source_levels(t, case)
    faces, block, sig_t, sig_s, c = [0.0], [], [], [], []
    for blk, mat, width, width_ref in regions:
        n = max(2, int(round(width_ref * cells_per_cm)))
        if mat == "void" and "void_cells" in case:
            n = case["void_cells"]
        if mat == "shield" and shield_element(cells_per_cm, case):
            n = int(round(width / shield_element(cells_per_cm, case)))
        faces.extend(faces[-1] + width * np.arange(1, n + 1) / n)
        st, ss = case["void"] if mat == "void" else MATERIALS[mat]
        side = -1.0 if blk < CORE else 1.0
        block += [blk] * n
        sig_t += [st] * n
        sig_s += [ss] * n
        c += [{"dielectric": c_diel, "void": c_void}.get(mat, 0.0) * side] * n
    faces = np.asarray(faces)
    xl, xr, block, c = faces[:-1], faces[1:], np.array(block), np.array(c)
    if case.get("spike"):
        h = xr - xl
        for e in range(len(block) - 1):
            for conductor, other, k, sign in (
                    (SHIELD_L, DIEL_L, e, -1.0), (CORE, VOID_L, e + 1, -1.0),
                    (CORE, VOID_R, e, +1.0), (SHIELD_R, DIEL_R, e + 1, +1.0)):
                pair = (block[e], block[e + 1])
                if pair == ((conductor, other) if k == e else (other, conductor)):
                    c[k] = sign * case["spike"] / h[k]
    return xl, xr, block, np.array(sig_t), np.array(sig_s), c


def solve_case(t, cells_per_cm, mu, w, case, adjoint=True):
    xl, xr, block, sig_t, sig_s, c = build_problem(t, cells_per_cm, case)
    h = xr - xl
    if case.get("diel_source"):
        S = np.where(np.isin(block, (DIEL_L, DIEL_R))[:, None], case["diel_source"], 0.0) * np.ones((1, len(mu)))
        fl, fr = _solve(h, sig_t, sig_s, mu, w, angular_source=S)
    else:
        fl, fr = _solve(h, sig_t, sig_s, mu, w, psi_inc=np.ones(len(mu)))
    fwd = NodalField(xl, xr, block, fl, fr)
    q_src = c[:, None] * mu[None, :]
    src = NodalField(xl, xr, block, q_src[:, None, :], q_src[:, None, :])
    if not adjoint:
        return fwd, src
    al, ar = _solve(h, sig_t, sig_s, mu, w, angular_source=q_src[:, ::-1])
    return fwd, reverse_angles(NodalField(xl, xr, block, al, ar), mu), src


def fd_gradient(t, cells_per_cm, mu, w, case, delta=1e-4):
    out = []
    for j in range(3):
        step = np.zeros(3)
        # With a fixed shield element size, t_s can only move by whole elements.
        step[j] = shield_element(cells_per_cm, case) if j == 2 and case.get("shield_h") else delta
        qp = charge_functional(*solve_case(t + step, cells_per_cm, mu, w, case, adjoint=False), w)
        qm = charge_functional(*solve_case(t - step, cells_per_cm, mu, w, case, adjoint=False), w)
        out.append((qp - qm) / (2 * step[j]))
    return np.array(out)


def source_term(t, cells_per_cm, mu, w, case, fwd, src):
    """<dQ+/dt, psi> per thickness, from the same fit void_sensitivity.main uses."""
    if not case.get("series"):
        return np.zeros(3)
    rng = np.random.default_rng(0)
    T = np.vstack([t, t * (1 + 0.2 * rng.uniform(-1, 1, size=(12, 3)))])
    levels = [_source_weights(solve_case(ti, cells_per_cm, mu, w, case, adjoint=False)[1], mu) for ti in T]
    slopes, _ = _source_log_slopes_multi(T, levels)
    return np.array([sum(v[0, j] * volume_inner_product(src, fwd, w, elem_mask=src.block == b)
                         for b, v in slopes.items()) for j in range(3)])


def verify_slab(t=REFERENCE, n_angles=16, densities=(25, 100, 400)) -> bool:
    print("=" * 96)
    print(f"Layer sensitivity vs. central FD  (t_v, t_d, t_s = {t.tolist()}, S{n_angles})")
    print("=" * 96)
    mu, w = gauss_legendre(n_angles)
    all_ok = True
    for name, case in CASES.items():
        print(f"\n{name}")
        print(f"  {'cells/cm':>8}  {'thickness':22s}{'FD':>12s}{'adj form err':>14s}{'fwd form err':>14s}"
              f"{'spread adj':>12s}")
        form = "forward" if case.get("diel_source") else "adjoint"
        last_errs = []
        for n in densities:
            fwd, adj, src = solve_case(t, n, mu, w, case)
            geom = geometry_sensitivity(fwd, adj, src, mu, w)
            fd = fd_gradient(t, n, mu, w, case)
            term = source_term(t, n, mu, w, case, fwd, src)
            last_errs = []
            for j, p in enumerate(GEOMETRY_PARAMS):
                ea = (geom[p]["adjoint_form"] + term[j]) / fd[j] - 1
                ef = (geom[p]["forward_form"] + term[j]) / fd[j] - 1
                last_errs.append(ef if form == "forward" else ea)
                print(f"  {n if j == 0 else '':>8}  {p:22s}{fd[j]:12.4e}{ea:14.1e}{ef:14.1e}"
                      f"{geom[p]['spread_adjoint']:12.1e}")
            # The fixed-radius result is the difference of two layers.
            if not case.get("series"):
                s = void_thickness_sensitivity(fwd, adj, src, mu, w)[f"{form}_form"]
                d = geom["void_thickness"][f"{form}_form"] - geom["dielectric_thickness"][f"{form}_form"]
                print(f"  {'':>8}  L_v - L_d vs interface formula at fixed radius: {d / s - 1:+.1e}")
        ok = max(abs(e) for e in last_errs) < 2e-3
        all_ok = all_ok and ok
        print(f"  {'OK' if ok else 'FAIL'}: largest error at the finest mesh, {form} form, "
              f"{max(abs(e) for e in last_errs):.1e}")
    return all_ok


# --------------------------------------------------------------------------- #
# End to end, through the real driver and the dump format                     #
# --------------------------------------------------------------------------- #

def write_dump(path, field: NodalField, group: int, n_groups: int):
    """One per-element CSV in the transport dumps' layout, with all flux in one group."""
    E, _, A = field.left.shape
    rows = np.zeros((E * n_groups * A, 11))
    e = np.repeat(np.arange(E), n_groups * A)
    g = np.tile(np.repeat(np.arange(n_groups), A), E)
    a = np.tile(np.arange(A), E * n_groups)
    in_group = g == group
    rows[:, 0], rows[:, 1] = e + 1, field.block[e]
    rows[:, 2], rows[:, 3] = e + 1, e + 2
    rows[:, 4], rows[:, 5] = field.x_left[e], field.x_right[e]
    rows[:, 6] = 0.5 * (rows[:, 4] + rows[:, 5])
    rows[:, 7], rows[:, 8] = g + 1, a
    rows[in_group, 9] = field.left[e[in_group], 0, a[in_group]]
    rows[in_group, 10] = field.right[e[in_group], 0, a[in_group]]
    np.savetxt(path, rows, delimiter=",", fmt=["%d"] * 4 + ["%.16e"] * 3 + ["%d", "%d"] + ["%.16e"] * 2,
               header="elem_id,block_id,node1,node2,x_left,x_right,x_center,angle,group,left_flux,right_flux",
               comments="")


def end_to_end(work_dir, cells_per_cm=30) -> bool:
    from check_gradients_fd import check, read_table
    from compare_ge_vs_gp import main as compare_main
    from make_geometry_cases import build_design, read_manifest, write_design

    work_dir = Path(work_dir)
    bounds = np.array([[0.2, 0.4], [0.5, 0.9], [0.4, 0.6]])
    h_shield = 0.01
    X, manifest = build_design(bounds, bounds.mean(axis=1), n_train_single=8, n_test_single=6,
                               n_train_all=24, n_test_all=12, fd_parents=3, shield_element=h_shield)
    write_design(work_dir, X, manifest, core_thickness=T_CORE)

    # Everything the real dumps have: thickness-dependent Q+, spikes and a fixed shield element.
    case = dict(CASES["E: Q+ level depends on thickness"], spike=0.05, shield_h=h_shield)
    mu, w = gauss_lobatto(8)
    w = w * (2.0 / w.sum())
    group, n_groups = ELECTRON_GROUP_MIN - 1, ELECTRON_GROUP_MIN
    print(f"\nWriting dumps for {len(X)} cases to {work_dir}")
    for m, x in zip(manifest, X):
        cid = m["case_id"]
        if m["role"] == "fd":
            fwd, src = solve_case(x, cells_per_cm, mu, w, case, adjoint=False)
            adj = None
        else:
            fwd, adj, src = solve_case(x, cells_per_cm, mu, w, case)
        write_dump(work_dir / f"case_{cid:04d}_forward_1d.csv", fwd, group, n_groups)
        write_dump(work_dir / f"case_{cid:04d}_adjoint_1d.csv", src, group, n_groups)
        if adj is not None:
            write_dump(work_dir / f"case_{cid:04d}_adjoint_flux_1d.csv", adj, group, n_groups)

    print("Running void_sensitivity.py --parameterization independent")
    log = work_dir / "void_sensitivity.log"
    stdout = sys.stdout
    with open(log, "w") as sys.stdout:
        sensitivity_main(data_dir=work_dir, out_dir=work_dir, parameterization="independent")
    sys.stdout = stdout

    table_path, manifest_path = work_dir / "qoi_geometry_adjoint.csv", work_dir / "case_manifest.csv"
    table, man = read_table(table_path), read_manifest(manifest_path)
    with_grad = [r for r in table.values() if np.isfinite(r["dq_dtv"])]
    status = np.bincount([r["status"] for r in with_grad], minlength=3)
    print(f"  {len(table)} cases in the table, {len(with_grad)} with gradients, identity status "
          f"OK/unchecked/failed = {status.tolist()}")
    rows, verdicts = check(table, man, tol=0.02)
    for p in GEOMETRY_PARAMS:
        v = verdicts[p]
        print(f"  {p:22s} n={v['n']:3d} median FD error {v['median_rel_error']:.2e}, "
              f"max {v['max_rel_error']:.2e}: {'pass' if v['passed'] else 'FAIL'}")
    ok = all(verdicts[p]["passed"] for p in GEOMETRY_PARAMS) and status[2] == 0

    print("Running compare_ge_vs_gp.py")
    compare_main(["--table", str(table_path), "--manifest", str(manifest_path),
                  "--out-dir", str(work_dir / "comparison"), "--seeds", "2", "--restarts", "2",
                  "--sizes-single", "3", "5", "8", "--sizes-all", "6", "12", "24"])
    produced = {p.name for p in (work_dir / "comparison").iterdir()}
    needed = {"comparison_results.csv", "comparison_summary.csv", "predictions.npz", "run_info.json",
              "all_learning_curves.png", "void_bands_void_thickness.png"}
    if needed - produced:
        print(f"Missing outputs: {needed - produced}")
        ok = False
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent-thickness gradients vs finite differences")
    ap.add_argument("--end-to-end", action="store_true", help="also run the dump-to-comparison pipeline")
    ap.add_argument("--work-dir", help="directory for the end-to-end files (default: a temporary one)")
    a = ap.parse_args(argv)
    ok = verify_slab()
    if a.end_to_end:
        if a.work_dir:
            ok = end_to_end(a.work_dir) and ok
        else:
            with tempfile.TemporaryDirectory() as tmp:
                ok = end_to_end(tmp) and ok
    print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
