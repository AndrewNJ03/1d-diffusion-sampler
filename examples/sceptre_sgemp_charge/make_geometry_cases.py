"""
make_geometry_cases.py: case design for the GE-GPR vs standard GP thickness study
with the coupled electron-photon deterministic transport solve.

Writes, to --out-dir:
  case_key.txt       Case,Core_Thickness,Void_Thickness,Dielectric_Thickness,Shield_Thickness
  case_manifest.csv  case_id,study,role,fd_parent,fd_param,fd_sign,fd_step
  mesh_settings.csv  case_id, the max element size to give each linear region, and the
                     element count it should produce

Studies: void, dielectric and shield each vary one thickness with the others at
nominal, and all varies the three together. Every study has its own Latin hypercube
training and held-out test sets. Role fd marks the +/- perturbed runs of a few
training cases, used only by check_gradients_fd.py. They need a forward dump and an
adjoint-source dump, not the adjoint flux. exodus_to_csv.py takes the source from the
adjoint Exodus file, so they cost an adjoint run unless the source can be written
without an adjoint solve.

Input decks are not written, since their format is specific to the transport code.
Meshing assumes a linear region gets ceil(thickness / max size) equal elements, which
holds for every region of the local dumps. Void and dielectric keep a FIXED ELEMENT COUNT,
so their widths scale smoothly with thickness. Shield and core keep a FIXED ELEMENT SIZE,
because the Q+ spike sits in their elements bordering the gap and its width changes q.
To make that exact, shield thicknesses are snapped to whole multiples of the element
size, and shield FD steps are whole elements. See the README, section 8.1.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from lhs_generation import latin_hypercube_sample  # noqa: E402

PARAMS = ["void_thickness", "dielectric_thickness", "shield_thickness"]
CASE_KEY_COLUMNS = ["Void_Thickness", "Dielectric_Thickness", "Shield_Thickness"]
STUDY_PARAMS = {"void": [0], "dielectric": [1], "shield": [2], "all": [0, 1, 2]}
MANIFEST_COLUMNS = ["case_id", "study", "role", "fd_parent", "fd_param", "fd_sign", "fd_step"]
DEFAULT_BOUNDS = dict(void=(1e-4, 0.1), dielectric=(0.25, 0.35), shield=(0.15, 0.25))
DEFAULT_MESH = dict(void_elements=5, dielectric_elements=600, shield_element=1e-3, core_element=1e-3)
# Max sizes are this much above the target, so rounding cannot make ceil() add an element.
MAX_SIZE_MARGIN = 1.0 + 1e-9


def _snap_shield(value: float, element: float, lo: float, hi: float, used=None) -> float:
    """Nearest whole number of elements inside [lo, hi], avoiding levels in used if given."""
    k_lo, k_hi = int(np.ceil(lo / element - 1e-9)), int(np.floor(hi / element + 1e-9))
    if k_hi < k_lo:
        raise ValueError(f"no whole multiple of the shield element {element} in [{lo}, {hi}]")
    levels = sorted(range(k_lo, k_hi + 1), key=lambda k: (abs(k - value / element), k))
    for k in levels:
        if used is None or k not in used:
            if used is not None:
                used.add(k)
            return k * element
    raise ValueError("more shield-only cases than shield levels; use a smaller element or fewer cases")


def build_design(bounds: np.ndarray, nominal: np.ndarray, studies=("void", "dielectric", "shield", "all"),
                 n_train_single=30, n_test_single=20, n_train_all=150, n_test_all=50,
                 fd_parents=10, fd_step_frac=0.01, seed=1, shield_element=None):
    """
    Returns (X, manifest): thicknesses (n_cases, 3) and one dict per case. Perturbed
    runs must stay inside the bounds, so parents are drawn from training cases at
    least one step from every bound of the thicknesses that study varies. With
    shield_element, every t_s and shield FD step is a whole number of elements.
    """
    bounds, nominal = np.asarray(bounds, float), np.asarray(nominal, float)
    if np.any(bounds[:, 0] <= 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each thickness needs 0 < lower bound < upper bound")
    if np.any(nominal < bounds[:, 0]) or np.any(nominal > bounds[:, 1]):
        raise ValueError("nominal thicknesses must lie inside the bounds")
    rng = np.random.default_rng(seed)
    steps = fd_step_frac * (bounds[:, 1] - bounds[:, 0])
    if shield_element:
        nominal = nominal.copy()
        nominal[2] = _snap_shield(nominal[2], shield_element, *bounds[2])
        steps[2] = max(1, round(steps[2] / shield_element)) * shield_element
    X, manifest = [], []

    def add(x, study, role, parent="", param="", sign="", step=""):
        manifest.append(dict(case_id=len(X), study=study, role=role, fd_parent=parent,
                             fd_param=param, fd_sign=sign, fd_step=step))
        X.append(np.array(x, float))
        return manifest[-1]["case_id"]

    for study in studies:
        active = STUDY_PARAMS[study]
        n_train, n_test = (n_train_all, n_test_all) if study == "all" else (n_train_single, n_test_single)
        train_ids = []
        # Only the shield study can repeat a case, since nothing else varies in it.
        used_levels = set() if study == "shield" else None
        for role, n in (("train", n_train), ("test", n_test)):
            if n == 0:
                continue
            sample = latin_hypercube_sample(n, bounds[active], random_state=int(rng.integers(2**31)))
            for row in sample:
                x = nominal.copy()
                x[active] = row
                if shield_element and 2 in active:
                    x[2] = _snap_shield(x[2], shield_element, *bounds[2], used=used_levels)
                cid = add(x, study, role)
                if role == "train":
                    train_ids.append(cid)
        eligible = [c for c in train_ids
                    if np.all(X[c][active] - steps[active] >= bounds[active, 0])
                    and np.all(X[c][active] + steps[active] <= bounds[active, 1])]
        if len(eligible) < min(fd_parents, len(train_ids)):
            print(f"NOTE: study {study}: only {len(eligible)} training cases can take a +/- step")
        n_parents = min(fd_parents, len(eligible))
        for parent in rng.choice(eligible, size=n_parents, replace=False) if n_parents else []:
            parent = int(parent)
            for j in active:
                for sign in (+1, -1):
                    x = X[parent].copy()
                    x[j] += sign * steps[j]
                    add(x, study, "fd", parent=parent, param=PARAMS[j], sign=sign, step=f"{steps[j]:.10e}")
    return np.array(X), manifest


def write_design(out_dir, X, manifest, core_thickness: float) -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path, manifest_path = out_dir / "case_key.txt", out_dir / "case_manifest.csv"
    with open(key_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "Core_Thickness"] + CASE_KEY_COLUMNS)
        for m, x in zip(manifest, X):
            writer.writerow([f"{m['case_id']:04d}", repr(float(core_thickness))]
                            + [repr(float(v)) for v in x])
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)
    return key_path, manifest_path


def write_mesh_settings(out_dir, X, manifest, core_thickness: float, void_elements: int,
                        dielectric_elements: int, shield_element: float, core_element: float):
    """Per-case max element size for each linear region, and the count it should give."""
    path = Path(out_dir) / "mesh_settings.csv"
    n_core = int(np.ceil(core_thickness / core_element - 1e-9))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "max_size_void", "max_size_dielectric", "max_size_shield",
                         "max_size_core", "n_void", "n_dielectric", "n_shield", "n_core"])
        for m, (t_v, t_d, t_s) in zip(manifest, X):
            n_shield = int(round(t_s / shield_element))
            sizes = [t_v / void_elements, t_d / dielectric_elements, t_s / n_shield, core_thickness / n_core]
            writer.writerow([f"{m['case_id']:04d}"] + [repr(float(h * MAX_SIZE_MARGIN)) for h in sizes]
                            + [void_elements, dielectric_elements, n_shield, n_core])
    return path


def read_manifest(path) -> dict:
    """{case_id: row}, with fd_parent, fd_sign and fd_step parsed where present."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["case_id"])
            row["case_id"] = cid
            if row["role"] == "fd":
                row["fd_parent"] = int(row["fd_parent"])
                row["fd_sign"] = int(row["fd_sign"])
                row["fd_step"] = float(row["fd_step"])
            out[cid] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Case design for the GE-GPR vs GP thickness study")
    ap.add_argument("--out-dir", required=True)
    for name, b in DEFAULT_BOUNDS.items():
        ap.add_argument(f"--{name}-bounds", nargs=2, type=float, default=b, metavar=("LO", "HI"))
    ap.add_argument("--nominal", nargs=3, type=float, metavar=("VOID", "DIELECTRIC", "SHIELD"),
                    help="thicknesses held fixed in the single-thickness studies (default: midpoints)")
    ap.add_argument("--core-thickness", type=float, default=0.1)
    ap.add_argument("--studies", nargs="+", default=list(STUDY_PARAMS), choices=list(STUDY_PARAMS))
    ap.add_argument("--train-single", type=int, default=30)
    ap.add_argument("--test-single", type=int, default=20)
    ap.add_argument("--train-all", type=int, default=150)
    ap.add_argument("--test-all", type=int, default=50)
    ap.add_argument("--fd-parents", type=int, default=10, help="perturbed parents per study")
    ap.add_argument("--fd-step-frac", type=float, default=0.01, help="FD step / thickness range")
    ap.add_argument("--void-elements", type=int, default=DEFAULT_MESH["void_elements"],
                    help="fixed element count per void region")
    ap.add_argument("--dielectric-elements", type=int, default=DEFAULT_MESH["dielectric_elements"],
                    help="fixed element count per dielectric region")
    ap.add_argument("--shield-element", type=float, default=DEFAULT_MESH["shield_element"],
                    help="fixed shield element size; shield thicknesses snap to multiples of it")
    ap.add_argument("--core-element", type=float, default=DEFAULT_MESH["core_element"],
                    help="core element size (the core does not change between cases)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="overwrite an existing design")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    if not a.force and ((out_dir / "case_key.txt").exists() or (out_dir / "case_manifest.csv").exists()):
        ap.error(f"{out_dir} already has a design; pass --force to overwrite it")
    bounds = np.array([a.void_bounds, a.dielectric_bounds, a.shield_bounds], float)
    nominal = np.array(a.nominal, float) if a.nominal else bounds.mean(axis=1)
    X, manifest = build_design(bounds, nominal, a.studies, a.train_single, a.test_single,
                               a.train_all, a.test_all, a.fd_parents, a.fd_step_frac, a.seed,
                               shield_element=a.shield_element)
    key_path, manifest_path = write_design(out_dir, X, manifest, a.core_thickness)
    mesh_path = write_mesh_settings(out_dir, X, manifest, a.core_thickness, a.void_elements,
                                    a.dielectric_elements, a.shield_element, a.core_element)

    n_base = sum(m["role"] != "fd" for m in manifest)
    n_fd = len(manifest) - n_base
    print(f"Wrote {key_path}, {manifest_path} and {mesh_path}")
    for study in a.studies:
        roles = [m["role"] for m in manifest if m["study"] == study]
        print(f"  {study:10s} train {roles.count('train'):4d}  test {roles.count('test'):4d}  "
              f"fd {roles.count('fd'):4d}")
    print(f"Solves: {n_base} forward + {n_base} adjoint for the train and test cases, and {n_fd} "
          f"forward for the fd cases, which also need an adjoint-source dump: {2 * n_base + 2 * n_fd} "
          f"in total, or {2 * n_base + n_fd} if that dump needs no adjoint solve.")
    print("Use linear meshing with the max sizes in mesh_settings.csv, and check the element counts "
          "in the first few dumps against its n_* columns.")


if __name__ == "__main__":
    main()
