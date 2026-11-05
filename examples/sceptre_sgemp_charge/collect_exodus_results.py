"""
collect_exodus_results.py: gather one Exodus run per case directory into the flat
per-case CSVs that void_sensitivity.py reads.

Input, for every case directory CASES_ROOT/case_<id>/1d/:
    forward_1d.exo            forward angular flux  -> case_NNNN_forward_1d.csv
    adjoint_transport_1d.exo  adjoint angular flux  -> case_NNNN_adjoint_flux_1d.csv
    response_1d.exo           adjoint source        -> case_NNNN_adjoint_1d.csv

If the response file holds SCEPTRE's scalar potential (the same for every angle), it is
converted to Q+ = mu dphi/dx and kept as case_NNNN_adjoint_potential_1d.csv. Cases with
role fd in case_manifest.csv do not need the adjoint flux.

Run from the repository root:
    # 1. Check what the files hold before converting anything.
    python3 examples/sceptre_sgemp_charge/collect_exodus_results.py --inspect
    # 2. Convert. Rerunning skips cases whose CSVs are newer than their Exodus files.
    python3 examples/sceptre_sgemp_charge/collect_exodus_results.py --jobs 8

Defaults are relative to the repository's parent directory: output goes to
data/50_case_multistudy, and CASES_ROOT is gradient_enchanced_gpr/cases, looked for in
the current directory, the repository's parent and that directory's parent.

case_key.txt must be in the output directory, or in --design-dir. The ids come from the
directory names, so case_7 and case_0007 are both case 7.

A case's CSVs appear only after all of its files converted and passed these checks:
  failed   a file does not read; meshes, groups or angles differ between files; a field is
           not finite; layer widths disagree with case_key.txt (wrong case id, usually)
  warning  mirrored layers or element counts differ from mesh_settings.csv; the CSVs are
           kept, but the mesh is not the designed one
Every case gets a row in collection_report.csv. Exit status is 0 only if every case in
case_key.txt is ok.
"""

import argparse
import contextlib
import csv
import filecmp
import io
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from exodus_to_csv import ExodusFile, _parse_cases, list_variables, read_field, write_csv
from extract_qoi import load_case_key
from make_geometry_cases import read_manifest
from potential_to_source import convert_case

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT.parent / "data" / "50_case_multistudy"
CASES_SUBDIR = Path("gradient_enchanced_gpr") / "cases"
DESIGN_FILES = ("case_key.txt", "case_manifest.csv", "mesh_settings.csv")
REPORT_NAME = "collection_report.csv"
REPORT_COLUMNS = ("case_id", "role", "status", "n_void", "n_dielectric", "n_shield", "n_core",
                  "thickness_scale", "source_was_potential", "message")

# Block ids of the left and right copy of each layer, and the case_key.txt column.
LAYERS = {"void": ((3, 5), "Void_Thickness"), "dielectric": ((2, 6), "Dielectric_Thickness"),
          "shield": ((1, 7), "Shield_Thickness"), "core": ((4, 4), "Core_Thickness")}
THICKNESS_TOL = 1e-5     # relative; the smallest design FD step is several times 1e-3
FIVE_INDEX = re.compile(r"^(.*?)_\d+_\d+_\d+_\d+_\d+$")


def output_paths(out_dir, cid):
    return {"forward": out_dir / f"case_{cid:04d}_forward_1d.csv",
            "adjoint": out_dir / f"case_{cid:04d}_adjoint_flux_1d.csv",
            "response": out_dir / f"case_{cid:04d}_adjoint_1d.csv",
            "potential": out_dir / f"case_{cid:04d}_adjoint_potential_1d.csv"}


def find_case_dirs(cases_root):
    """{case id: case directory}. Refuses two directories with the same id."""
    found = {}
    for path in sorted(Path(cases_root).glob("case_*")):
        match = re.fullmatch(r"case_(\d+)", path.name)
        if not (match and path.is_dir()):
            continue
        cid = int(match.group(1))
        if cid in found:
            raise SystemExit(f"{found[cid].name} and {path.name} are both case {cid}")
        found[cid] = path
    return found


def resolve_cases_root(given):
    if given:
        return Path(given).resolve()
    tried = [base / CASES_SUBDIR for base in (Path.cwd(), REPO_ROOT.parent, REPO_ROOT.parent.parent)]
    for path in tried:
        if path.is_dir():
            return path.resolve()
    raise SystemExit("Could not find the case directories. Tried:\n  "
                     + "\n  ".join(str(p) for p in tried) + "\nPass --cases-root.")


def source_prefix(exo_path, requested=None):
    """The response file's variable prefix: as requested, or its only 5-index family."""
    if requested:
        return requested
    with ExodusFile(exo_path) as exo:
        names = exo.names("name_elem_var")
    families = sorted({m.group(1) for m in map(FIVE_INDEX.match, names) if m})
    if len(families) != 1:
        raise ValueError(f"{Path(exo_path).name} has element variable families {families}; expected "
                         "exactly one with 5 indices. Run --inspect, then pass --response-prefix.")
    return families[0]


def _same_structure(a, b):
    return a["left"].shape == b["left"].shape and np.array_equal(a["block"], b["block"]) \
        and np.allclose(a["x_left"], b["x_left"], rtol=1e-10, atol=1e-12) \
        and np.allclose(a["x_right"], b["x_right"], rtol=1e-10, atol=1e-12)


def check_geometry(field, key_row, mesh_row):
    """Element counts and thickness scale; raises if the layers disagree with case_key.txt."""
    counts, widths, warnings = {}, {}, []
    blocks = set(np.unique(field["block"]).tolist())
    expected = {b for ids, _ in LAYERS.values() for b in ids}
    if blocks != expected:
        raise ValueError(f"block ids {sorted(blocks)}, expected {sorted(expected)}")
    for name, (ids, _) in LAYERS.items():
        side = []
        for b in ids:
            m = field["block"] == b
            side.append((int(m.sum()), float(field["x_right"][m].max() - field["x_left"][m].min())))
        counts[name], widths[name] = side[0]
        if side[0][0] != side[1][0] or abs(side[0][1] - side[1][1]) > THICKNESS_TOL * side[0][1]:
            warnings.append(f"{name} differs between mirrored sides: {side}")
    ratios = {name: widths[name] / key_row[col] for name, (_, col) in LAYERS.items()}
    scale = float(np.median(list(ratios.values())))
    bad = {n: r for n, r in ratios.items() if abs(r / scale - 1) > THICKNESS_TOL}
    if bad:
        raise ValueError("layer widths disagree with case_key.txt: "
                         + ", ".join(f"{n} {widths[n]:.8g} vs {key_row[LAYERS[n][1]]:.8g}" for n in bad))
    if mesh_row:
        for name, n in counts.items():
            want = int(mesh_row[f"n_{name}"])
            if n != want:
                warnings.append(f"{name} has {n} elements, mesh_settings.csv says {want}")
    return counts, scale, warnings


def is_current(paths, sources):
    """True if every output exists and is at least as new as every source file."""
    newest = max(p.stat().st_mtime for p in sources)
    return all(p.exists() and p.stat().st_mtime >= newest for p in paths)


def collect_case(cid, case_dir, out_dir, opts, key_row, mesh_row, role):
    """Convert one case. Returns (report row, log text); never raises for a bad case."""
    row = dict.fromkeys(REPORT_COLUMNS, "")
    row.update(case_id=cid, role=role, status="failed")
    run_dir = Path(case_dir) / opts["subdir"]
    sources = {"forward": run_dir / opts["forward_name"], "response": run_dir / opts["response_name"]}
    if role != "fd":
        sources["adjoint"] = run_dir / opts["adjoint_name"]
    absent = [p.name for p in sources.values() if not p.exists()]
    if absent:
        row.update(status="missing", message="not found in " + str(run_dir) + ": " + ", ".join(absent))
        return row, ""

    out = output_paths(Path(out_dir), cid)
    if not opts["force"] and is_current([out[k] for k in sources], list(sources.values())):
        row.update(status="current", message="CSVs are newer than the Exodus files")
        return row, ""

    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log), \
                tempfile.TemporaryDirectory(dir=out_dir, prefix=f".collect_{cid:04d}_") as tmp:
            tmp_out = output_paths(Path(tmp), cid)
            fwd = read_field(sources["forward"], opts["flux_prefix"], opts["step"])
            counts, scale, warnings = check_geometry(fwd, key_row, mesh_row)
            fields = {"forward": fwd}
            if "adjoint" in sources:
                fields["adjoint"] = read_field(sources["adjoint"], opts["flux_prefix"], opts["step"])
            prefix = source_prefix(sources["response"], opts["response_prefix"])
            fields["response"] = read_field(sources["response"], prefix, opts["step"], missing_as_zero=True)

            for name, field in fields.items():
                if not (np.isfinite(field["left"]).all() and np.isfinite(field["right"]).all()):
                    raise ValueError(f"{sources[name].name} has non-finite values")
                if not _same_structure(fwd, field):
                    raise ValueError(f"{sources[name].name}: mesh, groups or angles differ from the "
                                     f"forward file ({field['left'].shape} vs {fwd['left'].shape})")
                write_csv(field, tmp_out[name])
            if not np.any(fields["response"]["left"]) and not np.any(fields["response"]["right"]):
                raise ValueError(f"{sources['response'].name}: variable family '{prefix}' is all zero")

            converted = convert_case(tmp_out["response"], opts["angle_order"])
            was_potential = converted.startswith("converted")
            print(f"  {sources['response'].name} [{prefix}]: {converted}")

            # All good: publish the whole case at once.
            out["potential"].unlink(missing_ok=True)
            for name in ("forward", "adjoint", "potential", "response"):
                if tmp_out[name].exists():
                    tmp_out[name].replace(out[name])
    except Exception as err:   # one bad case must not stop the others
        row["message"] = f"{type(err).__name__}: {err}"
        return row, log.getvalue()

    row.update(status="warning" if warnings else "ok", thickness_scale=f"{scale:.10g}",
               source_was_potential=int(was_potential), message="; ".join(warnings),
               **{f"n_{name}": n for name, n in counts.items()})
    return row, log.getvalue()


def read_csv_rows(path, key):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {int(r[key]): r for r in csv.DictReader(f)}


def prepare_design(out_dir, design_dir, force):
    """Copy the design files into the output directory; refuses to replace different ones."""
    if design_dir is None or Path(design_dir).resolve() == out_dir:
        return
    for name in DESIGN_FILES:
        src, dst = Path(design_dir) / name, out_dir / name
        if not src.exists():
            continue
        if dst.exists() and not force and not filecmp.cmp(src, dst, shallow=False):
            raise SystemExit(f"{dst} differs from {src}. A new design invalidates the existing CSVs; "
                             "use a new output directory, or --force to replace it.")
        shutil.copy2(src, dst)


def inspect(case_dirs, key, opts):
    print(f"\n{len(case_dirs)} case directories, {len(key)} cases in case_key.txt")
    no_dir, no_key = sorted(set(key) - set(case_dirs)), sorted(set(case_dirs) - set(key))
    if no_dir:
        print(f"  in case_key.txt, no directory: {no_dir}")
    if no_key:
        print(f"  directory, not in case_key.txt: {no_key}")
    if not case_dirs:
        return
    cid = min(case_dirs)
    run_dir = case_dirs[cid] / opts["subdir"]
    print(f"\nVariables in case {cid} ({run_dir}):")
    for name in (opts["forward_name"], opts["adjoint_name"], opts["response_name"]):
        if (run_dir / name).exists():
            list_variables(run_dir / name)
        else:
            print(f"{name}: not found")
    response = run_dir / opts["response_name"]
    if response.exists():
        try:
            print(f"\nAdjoint source: family '{source_prefix(response, opts['response_prefix'])}' "
                  f"in {response.name}")
        except ValueError as err:
            print(f"\nAdjoint source: {err}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-root", help=f"directory holding case_* (default: find {CASES_SUBDIR})")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where the CSVs go (default: %(default)s)")
    ap.add_argument("--design-dir", help="copy case_key.txt, case_manifest.csv, mesh_settings.csv from here")
    ap.add_argument("--subdir", default="1d", help="run directory inside each case (default: %(default)s)")
    ap.add_argument("--forward-name", default="forward_1d.exo")
    ap.add_argument("--adjoint-name", default="adjoint_transport_1d.exo")
    ap.add_argument("--response-name", default="response_1d.exo")
    ap.add_argument("--flux-prefix", default="psi", help="angular flux variable prefix (default: %(default)s)")
    ap.add_argument("--response-prefix", help="adjoint source variable prefix (default: the file's only family)")
    ap.add_argument("--angle-order", default="descending", choices=["ascending", "descending"],
                    help="angle labelling, for potential -> Q+ (SCEPTRE: %(default)s)")
    ap.add_argument("--step", type=int, default=-1, help="time step index (default: last)")
    ap.add_argument("--cases", help="case ids, e.g. '0-9' or '0,3,5' (default: all in case_key.txt)")
    ap.add_argument("--jobs", type=int, default=1, help="cases converted at once (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="reconvert cases whose CSVs are already current")
    ap.add_argument("--inspect", action="store_true", help="list the variables of one case and write nothing")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir).resolve()
    if out_dir == REPO_ROOT or REPO_ROOT in out_dir.parents:
        raise SystemExit(f"{out_dir} is inside the repository. Data must stay out of the repository.")
    cases_root = resolve_cases_root(a.cases_root)
    case_dirs = find_case_dirs(cases_root)
    print(f"Case directories: {cases_root}\nOutput:           {out_dir}")

    opts = dict(subdir=a.subdir, forward_name=a.forward_name, adjoint_name=a.adjoint_name,
                response_name=a.response_name, flux_prefix=a.flux_prefix,
                response_prefix=a.response_prefix, angle_order=a.angle_order, step=a.step, force=a.force)

    key_dir = Path(a.design_dir) if a.inspect and a.design_dir else out_dir
    if not a.inspect:
        out_dir.mkdir(parents=True, exist_ok=True)
        prepare_design(out_dir, a.design_dir, a.force)
    if not (key_dir / "case_key.txt").exists():
        message = (f"No case_key.txt in {key_dir}. Generate the design with make_geometry_cases.py "
                   "--out-dir, or pass --design-dir.")
        if not a.inspect:
            raise SystemExit(message)
        print(f"\nNOTE: {message}")
        return inspect(case_dirs, {}, opts) or 0
    key = load_case_key(key_dir / "case_key.txt")
    if a.inspect:
        return inspect(case_dirs, key, opts) or 0

    manifest = read_manifest(out_dir / "case_manifest.csv") if (out_dir / "case_manifest.csv").exists() else {}
    mesh = read_csv_rows(out_dir / "mesh_settings.csv", "case_id")
    report_path = out_dir / REPORT_NAME
    previous = read_csv_rows(report_path, "case_id")
    wanted = sorted(_parse_cases(a.cases)) if a.cases else sorted(key)
    unknown = sorted(set(a.cases and wanted or []) - set(key))
    if unknown:
        raise SystemExit(f"--cases not in case_key.txt: {unknown}")
    if not manifest:
        print("NOTE: no case_manifest.csv, so every case needs an adjoint flux file.")
    if not mesh:
        print("NOTE: no mesh_settings.csv, so element counts are not checked.")

    rows = dict(previous)
    for cid in wanted:
        if cid not in case_dirs:
            rows[cid] = dict.fromkeys(REPORT_COLUMNS, "")
            rows[cid].update(case_id=cid, role=manifest.get(cid, {}).get("role", ""), status="missing",
                             message=f"no directory for this case in {cases_root}")

    def record(row, log):
        cid = int(row["case_id"])
        if row["status"] == "current" and cid in previous and previous[cid]["status"] in ("ok", "warning"):
            row = previous[cid]     # keep the checks from when the CSVs were written
        rows[cid] = row
        print(f"Case {cid:04d}: {row['status']}" + (f": {row['message']}" if row["message"] else ""))
        if log and row["status"] in ("failed", "warning"):
            print(log, end="")

    tasks = [(cid, case_dirs[cid], out_dir, opts, key[cid], mesh.get(cid), manifest.get(cid, {}).get("role", ""))
             for cid in wanted if cid in case_dirs]
    try:
        if a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as pool:
                for future in as_completed([pool.submit(collect_case, *t) for t in tasks]):
                    record(*future.result())
        else:
            for t in tasks:
                record(*collect_case(*t))
    finally:
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows[c] for c in sorted(rows))

    selected = [rows[c] for c in wanted if c in rows]
    tally = {s: sum(r["status"] == s for r in selected) for s in ("ok", "current", "warning", "missing", "failed")}
    scales = sorted({r["thickness_scale"] for r in selected if r["thickness_scale"]})
    size = sum(p.stat().st_size for p in out_dir.glob("case_*_1d.csv"))
    extra = sorted(set(case_dirs) - set(key))
    print(f"\n{len(selected)} case(s): " + ", ".join(f"{v} {k}" for k, v in tally.items() if v))
    print(f"{size / 1e9:.2f} GB of CSVs in {out_dir}")
    print(f"Report: {report_path}")
    if extra:
        print(f"WARNING: directories not in case_key.txt were ignored: {extra}")
    if len(scales) > 1:
        print(f"WARNING: mesh length / case_key.txt length is not the same in every case: {scales}")
    elif scales and abs(float(scales[0]) - 1) > THICKNESS_TOL:
        print(f"NOTE: mesh lengths are {scales[0]} times the case_key.txt lengths.")

    if tally["failed"] or tally["missing"] or tally["warning"]:
        print("\nFix the cases above and rerun; finished cases are skipped.")
        return 1
    results = out_dir / "results"
    print("\nNext:\n"
          f"  python3 examples/sceptre_sgemp_charge/void_sensitivity.py {out_dir} "
          f"--parameterization independent --weight-sum 1 --out-dir {results}\n"
          f"  python3 examples/sceptre_sgemp_charge/check_gradients_fd.py --table {results}/qoi_geometry_adjoint.csv "
          f"--manifest {out_dir}/case_manifest.csv --out-dir {results}\n"
          f"  python3 examples/sceptre_sgemp_charge/compare_ge_vs_gp.py --table {results}/qoi_geometry_adjoint.csv "
          f"--manifest {out_dir}/case_manifest.csv --gradient-check {results}/gradient_check.json "
          f"--out-dir {results}/comparison --n-jobs 8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
