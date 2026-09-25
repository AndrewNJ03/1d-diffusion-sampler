"""
extract_qoi.py — Scalar QoI extraction for the SCEPTRE SGEMP dataset.

Computes the net SGEMP direct-drive charge

    q_net = Integral over the dielectric region of  E(r) . J(r) dr

No pandas dependency, matching the rest of this repo's plain numpy/csv IO
conventions (see sklearn_implementation/gpr_pod.py, examples/charge_deposition
/coarse_gpr_delam.py).

Column-naming note: in the raw CSVs the 'angle' and 'group' columns are
swapped from their labels (confirmed with the data's author):
    csv column 8  ('angle', 1-38) -> actually the 38 energy groups
                                       (1-19 photon, 20-38 electron)
    csv column 9  ('group', 0-7)  -> actually the 8-point Gauss-Lobatto
                                       quadrature angle index
Column indices below (0-based, after the header row) are read with these
corrected semantics rather than the (mislabeled) header names.

Geometry (mirror-symmetric slab, domain normalised to x in [0,1]), block_id
-> region confirmed empirically from x_center ranges against case_key.txt:
    1 = Shield_L | 2 = Dielectric_L (Teflon) | 3 = Void_L | 4 = Core
    5 = Void_R   | 6 = Dielectric_R (Teflon) | 7 = Shield_R
The potential V solves d/dx(eps(x) dV/dx) = 0 with V=1 on the Core
(center conductor) and V=0 on the Shield (grounded outer shield); with no
free charge in Void/Dielectric this reduces to a closed-form series
(voltage-divider) solve, no discretisation needed.

Open assumptions (flagged, not directly verifiable from the data alone):
  - angle-index 0..7 assumed ascending in mu (mu=-1 at index 0, +1 at 7).
  - Teflon relative permittivity taken as the standard PTFE value, 2.1.
"""

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "constantR_void_10"
OUT_DIR = Path(__file__).resolve().parent / "output"

# CSV column indices (0-based), with 'angle'/'group' semantics corrected.
COL_ELEM_ID, COL_BLOCK_ID = 0, 1
COL_X_LEFT, COL_X_RIGHT, COL_X_CENTER = 4, 5, 6
COL_ENERGY_GROUP = 7   # mislabeled 'angle' in the header
COL_ANGLE_IDX = 8      # mislabeled 'group' in the header
COL_LEFT_FLUX, COL_RIGHT_FLUX = 9, 10

N_PHOTON_GROUPS = 19
N_ELECTRON_GROUPS = 19
ELECTRON_GROUP_MIN = N_PHOTON_GROUPS + 1                        # 20
ELECTRON_GROUP_MAX = N_PHOTON_GROUPS + N_ELECTRON_GROUPS         # 38

EPS_VOID = 1.0      # vacuum relative permittivity
EPS_TEFLON = 2.1    # standard PTFE relative permittivity (not given in the dataset)

BLOCK_SHIELD_L, BLOCK_DIELECTRIC_L, BLOCK_VOID_L = 1, 2, 3
BLOCK_CORE = 4
BLOCK_VOID_R, BLOCK_DIELECTRIC_R, BLOCK_SHIELD_R = 5, 6, 7
DIELECTRIC_BLOCKS = (BLOCK_DIELECTRIC_L, BLOCK_DIELECTRIC_R)


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #

def load_case_key(path: Path = None) -> dict:
    """Parse case_key.txt -> {case_id: {Core_Thickness, Void_Thickness, ...}}."""
    path = Path(path) if path is not None else DATA_DIR / "case_key.txt"
    cases = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["Case"])
            cases[cid] = {k: float(v) for k, v in row.items() if k != "Case"}
    return cases


def load_case_csv(path: Path) -> np.ndarray:
    """Load one case_*_forward_1d.csv as a plain (N, 11) float array."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


# --------------------------------------------------------------------------- #
# Angular quadrature                                                         #
# --------------------------------------------------------------------------- #

def gauss_lobatto(n: int):
    """
    n-point Gauss-Lobatto nodes/weights on [-1, 1] (includes both endpoints).

    Nodes: -1, 1, and the roots of P'_{n-1} (Legendre). Weights:
        w_i = 2 / (n(n-1) [P_{n-1}(x_i)]^2)
    """
    from numpy.polynomial import legendre as L

    c = np.zeros(n)
    c[-1] = 1.0
    dc = L.legder(c)
    interior = np.sort(np.real(L.legroots(dc)))
    nodes = np.concatenate(([-1.0], interior, [1.0]))
    Pn1 = L.legval(nodes, c)
    weights = 2.0 / (n * (n - 1) * Pn1 ** 2)
    return nodes, weights


def angle_quadrature_map(n_angles: int = 8):
    """angle_idx (0..n_angles-1, ascending) -> (mu, weight) arrays."""
    return gauss_lobatto(n_angles)


# --------------------------------------------------------------------------- #
# Electrostatic potential / field (closed-form series-dielectric solve)      #
# --------------------------------------------------------------------------- #

def compute_field_by_block(case_params: dict, eps_teflon: float = EPS_TEFLON) -> dict:
    """
    Closed-form E-field magnitude (signed) in each Void/Dielectric block.

    V=1 on the Core, V=0 on the Shield. No free charge in Void/Dielectric,
    so the displacement D = eps * dV/dx is constant along each half-domain
    (series voltage divider): D = 1 / (t_void/eps_void + t_diel/eps_teflon).
    E = -dV/dx points from the Core toward the Shield on each side, i.e.
    negative (-x) on the left half and positive (+x) on the right half.
    """
    t_void = float(case_params["Void_Thickness"])
    t_diel = float(case_params["Dielectric_Thickness"])

    D = 1.0 / (t_void / EPS_VOID + t_diel / eps_teflon)

    return {
        BLOCK_VOID_L: -D / EPS_VOID,
        BLOCK_DIELECTRIC_L: -D / eps_teflon,
        BLOCK_CORE: 0.0,
        BLOCK_VOID_R: +D / EPS_VOID,
        BLOCK_DIELECTRIC_R: +D / eps_teflon,
        BLOCK_SHIELD_L: 0.0,
        BLOCK_SHIELD_R: 0.0,
    }


# --------------------------------------------------------------------------- #
# Electron current                                                           #
# --------------------------------------------------------------------------- #

def compute_electron_current(data: np.ndarray, mu: np.ndarray, w: np.ndarray):
    """
    Per-element electron current  J(elem) = sum_g sum_a w_a * mu_a * psi(elem,g,a),
    summed over electron energy groups only (photon groups excluded), using
    psi = (left_flux + right_flux) / 2 as the element-average angular flux.

    Returns (elem_ids, x_left, x_right, block_id, J), each length = n_elements,
    ordered ascending by elem_id.
    """
    elem_id = data[:, COL_ELEM_ID].astype(int)
    energy_group = data[:, COL_ENERGY_GROUP].astype(int)
    angle_idx = data[:, COL_ANGLE_IDX].astype(int)

    electron_mask = (energy_group >= ELECTRON_GROUP_MIN) & (energy_group <= ELECTRON_GROUP_MAX)

    psi = 0.5 * (data[:, COL_LEFT_FLUX] + data[:, COL_RIGHT_FLUX])
    weight = w[angle_idx] * mu[angle_idx]
    contrib = np.where(electron_mask, psi * weight, 0.0)

    n_elem = elem_id.max()
    J_by_elem = np.bincount(elem_id, weights=contrib, minlength=n_elem + 1)[1:]  # index 1..n_elem

    # One representative row per element (geometry/block are constant per element).
    first_group, first_angle = energy_group.min(), angle_idx.min()
    rep_mask = (energy_group == first_group) & (angle_idx == first_angle)
    rep = data[rep_mask]
    rep = rep[np.argsort(rep[:, COL_ELEM_ID])]

    elem_ids = rep[:, COL_ELEM_ID].astype(int)
    x_left = rep[:, COL_X_LEFT]
    x_right = rep[:, COL_X_RIGHT]
    block_id = rep[:, COL_BLOCK_ID].astype(int)
    J = J_by_elem[elem_ids - 1]

    return elem_ids, x_left, x_right, block_id, J


# --------------------------------------------------------------------------- #
# Net charge                                                                  #
# --------------------------------------------------------------------------- #

def compute_net_charge(data: np.ndarray, case_params: dict, verbose: bool = False) -> float:
    """q_net = sum over Dielectric elements of E(block) * J(elem) * (x_right - x_left)."""
    mu, w = angle_quadrature_map()
    elem_ids, x_left, x_right, block_id, J = compute_electron_current(data, mu, w)
    E_by_block = compute_field_by_block(case_params)

    diel_mask = np.isin(block_id, DIELECTRIC_BLOCKS)
    E = np.array([E_by_block[b] for b in block_id[diel_mask]])
    dx = (x_right - x_left)[diel_mask]
    q_net = float(np.sum(E * J[diel_mask] * dx))

    if verbose:
        e_left = E_by_block[BLOCK_DIELECTRIC_L]
        e_right = E_by_block[BLOCK_DIELECTRIC_R]
        print(f"    E_dielectric: left={e_left:.4e}  right={e_right:.4e}  "
              f"(should be antisymmetric: left ~= -right)")
        print(f"    q_net = {q_net:.6e}")

    return q_net


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("SCEPTRE SGEMP scalar QoI extraction")
    print("=" * 64)
    print(f"Data directory  : {data_dir}")
    print(f"Output directory: {out_dir}")

    case_key = load_case_key(data_dir / "case_key.txt")

    rows = []
    for path in sorted(data_dir.glob("case_*_forward_1d.csv")):
        case_id = int(path.stem.split("_")[1])
        if case_id not in case_key:
            print(f"  [skip] case {case_id}: not in case_key.txt")
            continue

        params = case_key[case_id]
        print(f"\nCase {case_id:04d}  (void_thickness={params['Void_Thickness']:.6g})")
        data = load_case_csv(path)
        q_net = compute_net_charge(data, params, verbose=True)
        rows.append((case_id, params["Void_Thickness"], q_net))

    table = np.array(rows, dtype=float)
    out_path = out_dir / "qoi_scalar.txt"
    np.savetxt(
        out_path, table,
        header="case_id void_thickness q_net",
        fmt=["%d", "%.8e", "%.8e"],
    )
    print(f"\nWrote {out_path}  ({len(rows)} case(s))")
    print("-" * 64)

    return table


if __name__ == "__main__":
    main()
