"""
make_fd_cases.py: add perturbed cases to an existing design, so check_gradients_fd.py
can test the adjoint gradients directly.

Reads case_key.txt and case_manifest.csv from --data-dir. Picks parents among the training
cases that can take a +/- step in every thickness their study varies, without leaving the
range of thicknesses already run. Appends the new cases to both files with new ids and
role fd, and writes fd_cases.csv: the thicknesses to run and, when collection_report.csv
gives the existing element counts, the max element sizes that reproduce the parent's mesh.

Void and dielectric keep the parent's element count. Shield steps are whole elements when
the shield element size is one fixed value, so the Q+ spike element keeps its width.

After running the new cases, collect them WITHOUT --make-case-key, which would rebuild the
design from the runs and lose the fd rows:

    python3 collect_exodus_results.py --jobs 8

Then rerun void_sensitivity.py and run check_gradients_fd.py. Default: 3 parents, each
stepped +/- in 3 thicknesses, so 18 cases.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from check_gradients_fd import read_table
from extract_qoi import load_case_key
from make_geometry_cases import (CASE_KEY_COLUMNS, MANIFEST_COLUMNS, MAX_SIZE_MARGIN, PARAMS, STUDY_PARAMS,
                                 read_manifest)

ELEMENT_COLUMNS = dict(zip(PARAMS, ("n_void", "n_dielectric", "n_shield")))
FIXED_SIZE_TOL = 1e-6


def read_counts(path) -> dict:
    """{case_id: {column: count}} from collection_report.csv, for cases that have counts."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["case_id"]): {k: int(r[k]) for k in (*ELEMENT_COLUMNS.values(), "n_core")}
                for r in csv.DictReader(f) if r["n_shield"]}


def fixed_shield_element(key: dict, counts: dict):
    """The shield element size if every case with counts shares one, else None."""
    sizes = [key[c]["Shield_Thickness"] / n["n_shield"] for c, n in counts.items() if c in key]
    if not sizes or np.ptp(sizes) > FIXED_SIZE_TOL * np.mean(sizes):
        return None
    return float(np.mean(sizes))


def design_fd_cases(key: dict, manifest: dict, table: dict, counts: dict, n_parents: int,
                    step_frac: float, seed: int, shield_element=None) -> tuple:
    """Returns (new manifest rows, {new id: thicknesses}, parents). Raises if none can be picked."""
    ids = sorted(key)
    T = {c: np.array([key[c][col] for col in CASE_KEY_COLUMNS]) for c in ids}
    lo, hi = np.min(list(T.values()), axis=0), np.max(list(T.values()), axis=0)
    steps = step_frac * (hi - lo)
    if shield_element:
        steps[2] = max(1, round(steps[2] / shield_element)) * shield_element

    usable = []
    for c in ids:
        m = manifest[c]
        active = STUDY_PARAMS[m["study"]]
        fits = np.all(T[c][active] - steps[active] >= lo[active]) and np.all(T[c][active] + steps[active] <= hi[active])
        ok_identity = c in table and table[c]["status"] != 2 if table else True
        if m["role"] == "train" and fits and ok_identity:
            usable.append(c)
    if not usable:
        raise SystemExit("No training case can take a step inside the thickness ranges.")
    if len(usable) < n_parents:
        print(f"NOTE: only {len(usable)} usable parent case(s), fewer than the {n_parents} requested.")
    parents = sorted(np.random.default_rng(seed).choice(usable, size=min(n_parents, len(usable)), replace=False).tolist())

    new_id = max(ids) + 1
    rows, thicknesses = [], {}
    for parent in parents:
        study = manifest[parent]["study"]
        for j in STUDY_PARAMS[study]:
            for sign in (+1, -1):
                x = T[parent].copy()
                x[j] += sign * steps[j]
                rows.append(dict(case_id=new_id, study=study, role="fd", fd_parent=parent, fd_param=PARAMS[j],
                                 fd_sign=sign, fd_step=f"{steps[j]:.10e}"))
                thicknesses[new_id] = x
                new_id += 1
    return rows, thicknesses, parents


def mesh_row(x, parent_counts, core, shield_element):
    """Max element sizes reproducing the parent's mesh, and the counts they should give."""
    n_void, n_diel = parent_counts["n_void"], parent_counts["n_dielectric"]
    n_core = parent_counts["n_core"]
    n_shield = int(round(x[2] / shield_element)) if shield_element else parent_counts["n_shield"]
    sizes = (x[0] / n_void, x[1] / n_diel, x[2] / n_shield, core / n_core)
    return dict(zip(("max_size_void", "max_size_dielectric", "max_size_shield", "max_size_core"),
                    (repr(float(h * MAX_SIZE_MARGIN)) for h in sizes)),
                n_void=n_void, n_dielectric=n_diel, n_shield=n_shield, n_core=n_core)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Add perturbed cases to an existing design")
    ap.add_argument("--data-dir", required=True, help="holds case_key.txt and case_manifest.csv")
    ap.add_argument("--table", help="qoi_geometry_adjoint.csv, to avoid parents with a failed identity")
    ap.add_argument("--parents", type=int, default=3, help="parent cases (default: %(default)s)")
    ap.add_argument("--step-frac", type=float, default=0.01, help="step / thickness range (default: %(default)s)")
    ap.add_argument("--shield-element", type=float, help="default: from collection_report.csv, if it is one size")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    data_dir = Path(a.data_dir)
    key_path, manifest_path = data_dir / "case_key.txt", data_dir / "case_manifest.csv"
    key, manifest = load_case_key(key_path), read_manifest(manifest_path)
    if any(m["role"] == "fd" for m in manifest.values()):
        raise SystemExit(f"{manifest_path} already has fd cases; not adding more.")
    counts = read_counts(data_dir / "collection_report.csv")
    shield_element = a.shield_element or fixed_shield_element(key, counts)
    if shield_element is None:
        print("WARNING: no single shield element size found, so shield steps are not whole elements. "
              "q may gain a mesh term in the shield check. Pass --shield-element if there is one size.")
    table = read_table(a.table) if a.table else {}
    rows, thicknesses, parents = design_fd_cases(key, manifest, table, counts, a.parents, a.step_frac, a.seed,
                                                 shield_element)

    core = next(iter(key.values()))["Core_Thickness"]
    with open(key_path, "a", newline="") as f:
        writer = csv.writer(f)
        for cid, x in thicknesses.items():
            writer.writerow([f"{cid:04d}", repr(float(core))] + [repr(float(v)) for v in x])
    with open(manifest_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS).writerows(rows)

    fields = ["case_id", "fd_parent", "fd_param", "fd_sign", "Core_Thickness", *CASE_KEY_COLUMNS]
    out_rows = []
    for r in rows:
        x = thicknesses[r["case_id"]]
        out = dict(case_id=f"{r['case_id']:04d}", fd_parent=r["fd_parent"], fd_param=r["fd_param"], fd_sign=r["fd_sign"],
                   Core_Thickness=repr(float(core)), **{c: repr(float(v)) for c, v in zip(CASE_KEY_COLUMNS, x)})
        if r["fd_parent"] in counts:
            out.update(mesh_row(x, counts[r["fd_parent"]], core, shield_element))
        out_rows.append(out)
    fields += [k for k in out_rows[0] if k not in fields]
    path = data_dir / "fd_cases.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Added {len(rows)} fd cases ({rows[0]['case_id']} to {rows[-1]['case_id']}) from parents {parents}.")
    print(f"Thicknesses to run: {path}")
    print("Each needs forward_1d.exo and an adjoint file holding the source: up to "
          f"{2 * len(rows)} solves.")
    print("Then collect WITHOUT --make-case-key, rerun void_sensitivity.py and run check_gradients_fd.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
