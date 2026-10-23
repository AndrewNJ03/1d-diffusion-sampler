"""
potential_to_source.py: turn SCEPTRE's adjoint-source field files into the
angular adjoint source Q+ that void_sensitivity.py integrates.

What SCEPTRE's field tool writes (case_XXXX_adjoint_1d.csv from exodus_to_csv.py)
is a nodal scalar field phi: continuous, the same for every angle and every
electron group, zero for photon groups. In the SGEMP runs it is linear through
the void and dielectric (slope 0.443896 in case 1) and takes up the rest of the
1 V drop in the conductor element next to each interface (phi = 1 in the core,
0 in the shields). SCEPTRE's adjoint mode and q_directdrive use the source

    Q+_{e,g,a} = -mu_a E_e = mu_a dphi/dx |_e      (E = -dphi/dx)

constant over each element (phi is linear in it), on the groups where phi is
nonzero. That reproduces the original adjoint-source dumps: c * mu with
c = 0.443896 over void and dielectric, plus one-element spikes in the shield and
core elements bordering the gap.

Usage (from this directory):
    python3 potential_to_source.py DATA_DIR --angle-order descending

For every case in DATA_DIR/case_key.txt whose case_XXXX_adjoint_1d.csv holds a
potential (identical across angles), the file is moved to
case_XXXX_adjoint_potential_1d.csv and a new case_XXXX_adjoint_1d.csv with Q+ is
written. Files that already hold an angular source are left alone, so the script
can be rerun. --angle-order is the dumps' angle labelling: SCEPTRE's is
descending (index 0 is mu = +1). void_sensitivity.py's boundary identity checks
both the sign and the order: with the right ones it holds (status 0).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from extract_qoi import DATA_DIR, gauss_lobatto, load_case_key

COLS = ("elem_id,block_id,exodus_elem,time,x_left,x_right,x_center,"
        "energy_group,angle_idx,left_value,right_value")


def is_potential(data: np.ndarray, tol: float = 1e-12) -> bool:
    """True if every element/group has the same left and right value for all angles."""
    key = data[:, 0] * 1000 + data[:, 7]
    order = np.lexsort((data[:, 8], key))
    d = data[order]
    n_ang = int(d[:, 8].max()) + 1
    vals = d[:, 9:11].reshape(-1, n_ang, 2)
    scale = max(np.abs(vals).max(), 1e-300)
    return bool(np.all(np.ptp(vals, axis=1) <= tol * scale))


def source_from_potential(data: np.ndarray, angle_order: str) -> np.ndarray:
    """Rows of the same layout with left/right values replaced by mu_a dphi/dx."""
    n_ang = int(data[:, 8].max()) + 1
    mu_asc, _ = gauss_lobatto(n_ang)
    mu_label = mu_asc if angle_order == "ascending" else mu_asc[::-1]   # mu of each index
    h = data[:, 5] - data[:, 4]
    dphi = (data[:, 10] - data[:, 9]) / h
    out = data.copy()
    q = mu_label[data[:, 8].astype(int)] * dphi
    out[:, 9] = q
    out[:, 10] = q
    return out


def convert_case(path: Path, angle_order: str) -> str:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if not is_potential(data):
        return "already an angular source, left alone"
    backup = path.with_name(path.name.replace("_adjoint_1d.csv", "_adjoint_potential_1d.csv"))
    if backup.exists():
        raise FileExistsError(f"{backup.name} exists; not overwriting it")
    src = source_from_potential(data, angle_order)
    shutil.move(path, backup)
    fmt = ["%d", "%d", "%d", "%.10e", "%.12e", "%.12e", "%.12e", "%d", "%d", "%.10e", "%.10e"]
    np.savetxt(path, src, delimiter=",", header=COLS, comments="", fmt=fmt)

    # Summary: gap slope and conductor spikes, for comparison with expectations.
    blk, g = data[:, 1].astype(int), data[:, 7].astype(int)
    h = data[:, 5] - data[:, 4]
    grad = (data[:, 10] - data[:, 9]) / h
    sel = g == g.max()
    parts = []
    for b, name in ((2, "diel"), (3, "void"), (1, "shield"), (4, "core")):
        m = sel & (blk == b) & (np.abs(grad) > 0)
        if m.any():
            parts.append(f"{name} |dphi/dx| {np.abs(grad[m]).min():.6g}..{np.abs(grad[m]).max():.6g}"
                         f" ({int(m.sum() // (int(data[:, 8].max()) + 1))} elem)")
    return "converted; " + ", ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SCEPTRE adjoint-source potential -> Q+ = mu dphi/dx")
    ap.add_argument("data_dir", nargs="?", default=str(DATA_DIR))
    ap.add_argument("--angle-order", required=True, choices=["ascending", "descending"],
                    help="angle labelling of the dumps (SCEPTRE: descending)")
    a = ap.parse_args(argv)
    data_dir = Path(a.data_dir)
    for cid in sorted(load_case_key(data_dir / "case_key.txt")):
        path = data_dir / f"case_{cid:04d}_adjoint_1d.csv"
        if not path.exists():
            print(f"Case {cid:04d}: {path.name} not found")
            continue
        print(f"Case {cid:04d}: {convert_case(path, a.angle_order)}")


if __name__ == "__main__":
    main()
