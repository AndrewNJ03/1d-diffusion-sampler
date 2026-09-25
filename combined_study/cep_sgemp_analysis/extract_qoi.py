"""
extract_qoi.py — Scalar QoI extraction for the SCEPTRE SGEMP dataset.

The scalar QoI is q_net, the net SGEMP direct-drive charge (SAND2014-15576
Eq. 4.8). Unlike examples/sceptre_sgemp_charge/extract_qoi.py, this does NOT
integrate raw per-element angular flux -- the Sandia-side SCEPTRE toolchain
already reduces each case down to a scalar and hands it back as
response_1d in a per-sweep summary.csv 

response_1d == q_net (confirmed in examples/sceptre_sgemp_charge/
gpr_scalar_void_sweep.py against the raw-flux integration on the original
2-case pilot); response_2d is a different, currently-unused quantity.

This module writes output/qoi_<source>.txt tables per sweep
(case_id, <param columns>, q_net) for downstream GPR / GE-GPR fitting.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
OUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class QoISource:
    name: str
    csv_path: Path
    param_cols: tuple


SOURCES = {
    "void_sweep": QoISource(
        name="void_sweep",
        csv_path=DATA_ROOT / "constantR_void_1000_monoenergy" / "summary(1).csv",
        param_cols=("void_thickness",),
    ),
    "void_dielectric_shield_grid": QoISource(
        name="void_dielectric_shield_grid",
        csv_path=DATA_ROOT / "constantR_voidDielectricShield_1000_monoenergy" / "summary(2).csv",
        param_cols=("void_thickness", "dielectric_thickness", "shield_thickness"),
    ),
    "void_sweep_pithon": QoISource(
        name="void_sweep_pithon",
        csv_path=DATA_ROOT / "constantR_void_100_pithon" / "summary.csv",
        param_cols=("void_thickness",),
    ),
}


def load_qoi_table(source: QoISource):
    """Load (case_ids, X[params], y=q_net) from a Sandia summary.csv."""
    table = np.genfromtxt(source.csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    case_ids = table["case_id"].astype(int)
    X = np.column_stack([table[c].astype(float) for c in source.param_cols])
    y = table["response_1d"].astype(float)
    return case_ids, X, y


def main(out_dir: Path = OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("SCEPTRE SGEMP scalar QoI extraction (response_1d = q_net)")
    print("=" * 64)

    results = {}
    for key, source in SOURCES.items():
        if not source.csv_path.exists():
            print(f"\n[skip] {key}: {source.csv_path} not found")
            continue

        case_ids, X, y = load_qoi_table(source)
        print(f"\n{key}  ({source.csv_path})")
        print(f"  M = {len(y)} cases, p = {len(source.param_cols)} params")
        for j, name in enumerate(source.param_cols):
            print(f"    {name:20s}: [{X[:, j].min():.6g}, {X[:, j].max():.6g}], "
                  f"{len(np.unique(X[:, j]))} unique levels")
        print(f"    q_net             : [{y.min():.6e}, {y.max():.6e}]")

        table = np.column_stack([case_ids, X, y])
        out_path = out_dir / f"qoi_{key}.txt"
        header = "case_id " + " ".join(source.param_cols) + " q_net"
        fmt = ["%d"] + ["%.8e"] * (X.shape[1] + 1)
        np.savetxt(out_path, table, header=header, fmt=fmt)
        print(f"  Wrote {out_path}")

        results[key] = (case_ids, X, y)

    return results


if __name__ == "__main__":
    main()
