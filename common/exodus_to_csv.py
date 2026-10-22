"""
exodus_to_csv.py: dump SCEPTRE element-wise angular flux from an Exodus file
into the per-element CSV that extract_qoi.py and void_sensitivity.py read.

Input
-----
Element variables named  <prefix>_<g>_<i2>_<a>_<i4>_<n>  (default prefix psi):
    g   energy group (1..38 in SCEPTRE; 0-based files are shifted to 1-based)
    a   angle index (0..7)
    n   local node of the 1D element (0 or 1), mapped to left/right by x
    i2, i4 must be 0 (they are in every file seen so far)
on 2-node 1D elements, in any number of blocks.

Output (one row per element, group, angle; 11 columns, header row first)
------
  0 elem_id      1..E, ordered by x (not the Exodus numbering)
  1 block_id     Exodus block id (eb_prop1), e.g. 1..7 Shield_L..Shield_R
  2 exodus_elem  Exodus global element id (elem_num_map, else 1-based order)
  3 time         time value of the dumped step
  4 x_left       5 x_right      6 x_center
  7 energy_group 1-based
  8 angle_idx    0-based
  9 left_value   10 right_value
Columns 7 and 8 sit where the hand-made dumps had them; this header labels
them correctly. The loaders read columns by position, not by name.

Usage
-----
  # what element variables does a file have?
  python3 exodus_to_csv.py --list run.exo

  # one file
  python3 exodus_to_csv.py run.exo case_0000_forward_1d.csv [--prefix psi] [--step -1]

  # every case in case_key.txt (file names are Python format strings of the case id)
  python3 exodus_to_csv.py --batch ../../../data/constantR_void_10 \\
      --forward "forward_{:04d}.exo" --adjoint "adjoint_{:04d}.exo" \\
      [--adjoint-source-prefix Qadj]

Batch mode writes case_XXXX_forward_1d.csv and case_XXXX_adjoint_flux_1d.csv
into the data directory, and case_XXXX_adjoint_1d.csv (the adjoint source Q+)
if --adjoint-source-prefix names the Q+ variables in the adjoint file.

Reader: netCDF4 if installed, otherwise scipy.io.netcdf_file, which handles
classic and 64-bit-offset Exodus files but not netCDF-4/HDF5 ones.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Reading                                                                     #
# --------------------------------------------------------------------------- #

class ExodusFile:
    """Minimal read-only view of an Exodus file: dims, variables, names."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            import netCDF4
            self._ds = netCDF4.Dataset(self.path, "r")
            self._ds.set_auto_mask(False)
            self._dims = {k: len(v) for k, v in self._ds.dimensions.items()}
        except ImportError:
            from scipy.io import netcdf_file
            try:
                self._ds = netcdf_file(self.path, "r", mmap=False)
            except TypeError as err:
                raise RuntimeError(
                    f"{self.path.name} is not a classic netCDF file; install netCDF4 "
                    "(pip install netCDF4) to read netCDF-4/HDF5 Exodus files") from err
            self._dims = {k: int(v) if v is not None else len(self._ds.variables["time_whole"].data)
                          for k, v in self._ds.dimensions.items()}

    def close(self):
        self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def has(self, name):
        return name in self._ds.variables

    def get(self, name):
        return np.array(self._ds.variables[name][:])

    def dim(self, name, default=None):
        return self._dims.get(name, default)

    def names(self, key):
        """Decode a char array of names (name_elem_var etc.) into strings."""
        if not self.has(key):
            return []
        raw = self.get(key)
        out = []
        for row in np.atleast_2d(raw):
            s = "".join(c.decode("ascii", "ignore") if isinstance(c, bytes) else str(c)
                        for c in row.tolist())
            out.append(s.strip("\x00 ").strip())
        return out


# --------------------------------------------------------------------------- #
# Mesh                                                                        #
# --------------------------------------------------------------------------- #

def read_elements(exo: ExodusFile):
    """
    Per element, in file order (block by block): block index (1-based, file
    order), block id, Exodus global id, x of local node 0 and node 1.
    """
    if exo.has("coordx"):
        x = exo.get("coordx").astype(float)
    elif exo.has("coord"):
        x = exo.get("coord")[0].astype(float)
    else:
        raise ValueError(f"{exo.path.name}: no coordx/coord variable")

    n_blk = exo.dim("num_el_blk")
    blk_ids = exo.get("eb_prop1").astype(int) if exo.has("eb_prop1") else np.arange(1, n_blk + 1)
    b_idx, b_id, x0, x1 = [], [], [], []
    for b in range(1, n_blk + 1):
        conn = exo.get(f"connect{b}").astype(int)
        if conn.ndim != 2 or conn.shape[1] != 2:
            raise ValueError(f"{exo.path.name}: block {b} has {conn.shape[-1]}-node elements; "
                             "expected 2-node 1D elements")
        b_idx += [b] * len(conn)
        b_id += [blk_ids[b - 1]] * len(conn)
        x0.append(x[conn[:, 0] - 1])
        x1.append(x[conn[:, 1] - 1])
    n_elem = len(b_idx)
    gid = exo.get("elem_num_map").astype(int) if exo.has("elem_num_map") else np.arange(1, n_elem + 1)
    return dict(b_idx=np.array(b_idx), b_id=np.array(b_id), gid=gid,
                x0=np.concatenate(x0), x1=np.concatenate(x1))


# --------------------------------------------------------------------------- #
# Variables                                                                   #
# --------------------------------------------------------------------------- #

def find_fields(exo: ExodusFile, prefix: str):
    """{(g, a, n): variable index (1-based)} for <prefix>_g_0_a_0_n element variables."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")
    fields = {}
    for i, name in enumerate(exo.names("name_elem_var")):
        m = pat.match(name)
        if not m:
            continue
        g, i2, a, i4, n = map(int, m.groups())
        if i2 != 0 or i4 != 0:
            raise ValueError(f"{exo.path.name}: {name} has a nonzero 2nd/4th index; "
                             "this script assumes they are always 0")
        if n not in (0, 1):
            raise ValueError(f"{exo.path.name}: {name} has node index {n}; expected 0 or 1")
        fields[(g, a, n)] = i + 1
    if not fields:
        raise ValueError(f"{exo.path.name}: no element variables named {prefix}_g_0_a_0_n "
                         f"(run with --list to see what is there)")
    return fields


def list_variables(path):
    """Print the element/nodal variables, collapsing <prefix>_i_j_k_l_m families."""
    with ExodusFile(path) as exo:
        print(f"{Path(path).name}: {exo.dim('num_el_blk')} block(s), "
              f"{exo.dim('num_elem')} element(s), {exo.dim('time_step', 0)} time step(s)")
        for key, label in (("name_elem_var", "element"), ("name_nod_var", "nodal"),
                           ("name_glo_var", "global")):
            names = exo.names(key)
            fams = {}
            for n in names:
                m = re.match(r"^(.*?)((?:_\d+)+)$", n)
                stem, idx = (m.group(1), tuple(int(t) for t in m.group(2)[1:].split("_"))) \
                    if m else (n, ())
                fams.setdefault((stem, len(idx)), []).append(idx)
            print(f"  {label} variables: {len(names)}")
            for (stem, k), idxs in sorted(fams.items()):
                if k == 0:
                    print(f"    {stem}")
                else:
                    rng = [f"{min(t[j] for t in idxs)}..{max(t[j] for t in idxs)}" for j in range(k)]
                    print(f"    {stem}_" + "_".join(rng) + f"   ({len(idxs)} variables)")


def read_field(path, prefix="psi", step=-1, missing_as_zero=False):
    """
    Read one angular field. Returns a dict with the mesh sorted by x and
    left/right values of shape (E, G, A), plus group labels (1-based) and time.
    """
    with ExodusFile(path) as exo:
        el = read_elements(exo)
        fields = find_fields(exo, prefix)
        groups = sorted({k[0] for k in fields})
        angles = sorted({k[1] for k in fields})
        if angles != list(range(len(angles))):
            raise ValueError(f"{Path(path).name}: angle indices {angles} are not 0..A-1")
        expected = {(g, a, n) for g in groups for a in angles for n in (0, 1)}
        missing = expected - set(fields)
        if missing:
            raise ValueError(f"{Path(path).name}: {len(missing)} {prefix} variable(s) missing, "
                             f"e.g. {sorted(missing)[:4]}")
        if groups != list(range(groups[0], groups[0] + len(groups))):
            raise ValueError(f"{Path(path).name}: group indices are not contiguous: {groups}")

        n_t = exo.dim("time_step", 0)
        times = exo.get("time_whole") if exo.has("time_whole") else np.zeros(max(n_t, 1))
        t_val = float(times[step]) if len(times) else 0.0

        n_blk = exo.dim("num_el_blk")
        E, G, A = len(el["b_idx"]), len(groups), len(angles)
        node_vals = np.full((E, G, A, 2), np.nan)
        start = 0
        absent_blocks = set()
        for b in range(1, n_blk + 1):
            nb = int((el["b_idx"] == b).sum())
            sl = slice(start, start + nb)
            for (g, a, n), vi in fields.items():
                key = f"vals_elem_var{vi}eb{b}"
                if exo.has(key):
                    node_vals[sl, g - groups[0], a, n] = exo.get(key)[step]
                else:
                    absent_blocks.add(int(el["b_id"][sl][0]) if nb else b)
                    if missing_as_zero:
                        node_vals[sl, g - groups[0], a, n] = 0.0
            start += nb
        if absent_blocks and not missing_as_zero:
            raise ValueError(f"{Path(path).name}: {prefix} is not stored on block id(s) "
                             f"{sorted(absent_blocks)} (truth table); pass --missing-as-zero "
                             "if it is identically zero there")
        if absent_blocks:
            print(f"  note: {prefix} absent on block id(s) {sorted(absent_blocks)}, written as 0")

    # Local node 0/1 -> left/right by coordinate.
    flip = el["x0"] > el["x1"]
    left = np.where(flip[:, None, None], node_vals[..., 1], node_vals[..., 0])
    right = np.where(flip[:, None, None], node_vals[..., 0], node_vals[..., 1])
    xl = np.minimum(el["x0"], el["x1"])
    xr = np.maximum(el["x0"], el["x1"])

    order = np.argsort(xl, kind="stable")
    xl, xr = xl[order], xr[order]
    if np.any(xr - xl <= 0):
        raise ValueError(f"{Path(path).name}: zero-width element")
    gaps = np.abs(xl[1:] - xr[:-1])
    if np.any(gaps > 1e-9 * max(1.0, abs(xr[-1] - xl[0]))):
        raise ValueError(f"{Path(path).name}: elements do not tile a single interval "
                         f"(largest gap {gaps.max():.3e})")

    g0 = groups[0]
    group_labels = np.array(groups) + (1 if g0 == 0 else 0)
    if g0 == 0:
        print(f"  note: {Path(path).name} groups start at 0; written 1-based")
    return dict(x_left=xl, x_right=xr, block=el["b_id"][order], gid=el["gid"][order],
                left=left[order], right=right[order], groups=group_labels,
                n_angles=A, time=t_val)


# --------------------------------------------------------------------------- #
# Writing                                                                     #
# --------------------------------------------------------------------------- #

HEADER = ("elem_id,block_id,exodus_elem,time,x_left,x_right,x_center,"
          "energy_group,angle_idx,left_value,right_value")


def write_csv(field: dict, out_path):
    E, G, A = field["left"].shape
    e, g, a = np.meshgrid(np.arange(E), np.arange(G), np.arange(A), indexing="ij")
    e, g, a = e.ravel(), g.ravel(), a.ravel()
    xl, xr = field["x_left"][e], field["x_right"][e]
    table = np.column_stack([
        e + 1, field["block"][e], field["gid"][e], np.full(len(e), field["time"]),
        xl, xr, 0.5 * (xl + xr), field["groups"][g], a,
        field["left"][e, g, a], field["right"][e, g, a],
    ])
    fmt = ["%d", "%d", "%d", "%.10e", "%.12e", "%.12e", "%.12e", "%d", "%d", "%.10e", "%.10e"]
    np.savetxt(out_path, table, delimiter=",", header=HEADER, comments="", fmt=fmt)
    return table.shape


def convert(exo_path, out_path, prefix="psi", step=-1, missing_as_zero=False):
    field = read_field(exo_path, prefix, step, missing_as_zero)
    shape = write_csv(field, out_path)
    E, G, A = field["left"].shape
    print(f"  {Path(exo_path).name} [{prefix}] -> {Path(out_path).name}: "
          f"{E} elements x {G} groups x {A} angles = {shape[0]} rows, t = {field['time']:.6g}")
    return field


# --------------------------------------------------------------------------- #
# Batch                                                                       #
# --------------------------------------------------------------------------- #

def batch(data_dir, forward_fmt, adjoint_fmt, prefix="psi", adjoint_source_prefix=None,
          step=-1, missing_as_zero=False, case_ids=None):
    data_dir = Path(data_dir)
    if case_ids is None:
        from extract_qoi import load_case_key
        case_ids = sorted(load_case_key(data_dir / "case_key.txt"))
    for cid in case_ids:
        print(f"Case {cid:04d}")
        f_exo, a_exo = data_dir / forward_fmt.format(cid), data_dir / adjoint_fmt.format(cid)
        for p in (f_exo, a_exo):
            if not p.exists():
                print(f"  [skip] {p.name} not found")
        if f_exo.exists():
            fwd = convert(f_exo, data_dir / f"case_{cid:04d}_forward_1d.csv", prefix, step,
                          missing_as_zero)
        if a_exo.exists():
            adj = convert(a_exo, data_dir / f"case_{cid:04d}_adjoint_flux_1d.csv", prefix, step,
                          missing_as_zero)
            if f_exo.exists() and not (np.allclose(fwd["x_left"], adj["x_left"])
                                       and fwd["left"].shape == adj["left"].shape):
                print("  WARNING: forward and adjoint meshes or shapes differ")
            if adjoint_source_prefix:
                convert(a_exo, data_dir / f"case_{cid:04d}_adjoint_1d.csv",
                        adjoint_source_prefix, step, missing_as_zero=True)


def _parse_cases(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="input.exo output.csv  (single-file mode)")
    ap.add_argument("--list", action="store_true", help="list the variables of each input file")
    ap.add_argument("--batch", metavar="DATA_DIR", help="convert every case in DATA_DIR/case_key.txt")
    ap.add_argument("--forward", help="forward Exodus file name pattern, e.g. 'fwd_{:04d}.exo'")
    ap.add_argument("--adjoint", help="adjoint Exodus file name pattern")
    ap.add_argument("--cases", help="case ids, e.g. '0-9' or '0,3,5' (default: case_key.txt)")
    ap.add_argument("--prefix", default="psi", help="angular flux variable prefix (default psi)")
    ap.add_argument("--adjoint-source-prefix", help="prefix of Q+ variables in the adjoint file")
    ap.add_argument("--step", type=int, default=-1, help="time step index (default: last)")
    ap.add_argument("--missing-as-zero", action="store_true",
                    help="write 0 where a variable is not stored on a block")
    args = ap.parse_args(argv)

    if args.list:
        for f in args.files:
            list_variables(f)
        return
    if args.batch:
        if not (args.forward and args.adjoint):
            ap.error("--batch needs --forward and --adjoint patterns")
        batch(args.batch, args.forward, args.adjoint, args.prefix, args.adjoint_source_prefix,
              args.step, args.missing_as_zero,
              _parse_cases(args.cases) if args.cases else None)
        return
    if len(args.files) != 2:
        ap.error("give input.exo output.csv, or use --list / --batch")
    convert(args.files[0], args.files[1], args.prefix, args.step, args.missing_as_zero)


if __name__ == "__main__":
    sys.exit(main())