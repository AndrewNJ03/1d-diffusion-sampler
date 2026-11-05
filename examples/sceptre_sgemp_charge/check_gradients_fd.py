"""
check_gradients_fd.py: compares adjoint thickness gradients with central finite
differences of q from the perturbed cases in case_manifest.csv.

This is the deciding test of dq/dt_v, dq/dt_d and dq/dt_s on real data from the
coupled electron-photon deterministic transport solve. Reads the table written by
void_sensitivity.py --parameterization independent, where perturbed cases have q but
no gradient. Writes per-case errors to gradient_check.csv and a verdict for each
thickness to gradient_check.json. Exit status 1 if any thickness fails.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from make_geometry_cases import PARAMS, read_manifest

GRADIENT_COLUMNS = dict(zip(PARAMS, ["dq_dtv", "dq_dtd", "dq_dts"]))


def read_table(path) -> dict:
    """{case_id: row}, with numeric columns as floats."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {k: (v if k == "form" else float(v)) for k, v in row.items()}
            parsed["case_id"] = int(parsed["case_id"])
            parsed["status"] = int(parsed["status"])
            out[parsed["case_id"]] = parsed
    return out


def check(table: dict, manifest: dict, tol: float = 0.05) -> tuple:
    """Returns (per-case rows, {param: verdict}). A thickness passes on its median error."""
    pairs = {}
    for cid, m in manifest.items():
        if m["role"] == "fd":
            pairs.setdefault((m["study"], m["fd_parent"], m["fd_param"]), {})[m["fd_sign"]] = cid

    rows, missing = [], 0
    for (study, parent, param), by_sign in sorted(pairs.items()):
        ids = [parent, by_sign.get(+1), by_sign.get(-1)]
        if any(i not in table for i in ids):
            missing += 1
            continue
        grad = table[parent][GRADIENT_COLUMNS[param]]
        dt = table[ids[1]][param] - table[ids[2]][param]
        fd = (table[ids[1]]["q"] - table[ids[2]]["q"]) / dt
        if not (np.isfinite(grad) and np.isfinite(fd)):
            missing += 1
            continue
        # One-sided slopes against the base case show whether q is smooth at this step.
        slope_p = (table[ids[1]]["q"] - table[parent]["q"]) / (table[ids[1]][param] - table[parent][param])
        slope_m = (table[parent]["q"] - table[ids[2]]["q"]) / (table[parent][param] - table[ids[2]][param])
        rows.append(dict(study=study, parent=parent, param=param, adjoint=grad, fd=fd,
                         rel_error=abs(grad - fd) / max(abs(fd), 1e-300),
                         fd_asymmetry=abs(slope_p - slope_m) / max(abs(fd), 1e-300),
                         status=table[parent]["status"], form=table[parent]["form"]))

    verdicts = {}
    for param in PARAMS:
        errs = [r["rel_error"] for r in rows if r["param"] == param]
        asym = [r["fd_asymmetry"] for r in rows if r["param"] == param]
        if not errs:
            verdicts[param] = dict(n=0, passed=None)
            continue
        verdicts[param] = dict(n=len(errs), median_rel_error=float(np.median(errs)),
                               max_rel_error=float(np.max(errs)),
                               median_fd_asymmetry=float(np.median(asym)),
                               passed=bool(np.median(errs) <= tol))
    verdicts["missing_pairs"] = missing
    verdicts["tolerance"] = tol
    return rows, verdicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adjoint thickness gradients vs perturbed runs")
    ap.add_argument("--table", required=True, help="qoi_geometry_adjoint.csv")
    ap.add_argument("--manifest", required=True, help="case_manifest.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tol", type=float, default=0.05, help="pass if median relative error is at most this")
    a = ap.parse_args(argv)

    rows, verdicts = check(read_table(a.table), read_manifest(a.manifest), a.tol)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_dir / "gradient_check.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "gradient_check.json").write_text(json.dumps(verdicts, indent=2))

    print(f"Adjoint gradients vs central differences ({len(rows)} pairs, "
          f"{verdicts['missing_pairs']} missing), pass if median rel. error <= {a.tol:g}")
    print(f"  {'thickness':22s} {'n':>4s} {'median err':>11s} {'max err':>10s} {'FD asym.':>10s}  verdict")
    failed = False
    for param in PARAMS:
        v = verdicts[param]
        if not v["n"]:
            print(f"  {param:22s} {0:4d}  no pairs: NOT CHECKED")
            failed = True
            continue
        failed = failed or not v["passed"]
        print(f"  {param:22s} {v['n']:4d} {v['median_rel_error']:11.2e} {v['max_rel_error']:10.2e} "
              f"{v['median_fd_asymmetry']:10.2e}  {'pass' if v['passed'] else 'FAIL'}")
    print("  FD asym. compares the one-sided slopes. If it is as large as the error, the check is "
          "limited by jitter or curvature in q, not by the adjoint gradient.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
