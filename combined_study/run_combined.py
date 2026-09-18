"""
run_combined.py — driver for the fully-coupled 5-parameter study: all of
void_thickness, steepness, D_teflon, Sigma_a_teflon, q_teflon varying
simultaneously in one LHS design, one shared forward/adjoint pipeline, one
GE-GPR fit -- filling in the paper's "Combined case" placeholder
(latex_paper/main.tex, Sec. 5) with real numbers instead of the 5
one-at-a-time studies in ../gpr_vs_gegpr_comparison/.

Reuses the same rescaled geometry and parameter bounds
gpr_vs_gegpr_comparison/run_comparison.py already uses for its
void_thickness/steepness/D/Sigma_a/q studies (L=10 domain, QoI window
[3, 7], geometry scaled x10 so the shell spans the same domain as the
material studies), so error metrics here are on the same footing as that
file's summary_table.csv -- just with all 5 active at once instead of one
at a time. The only intentional difference is a smaller m_ge / n_restarts
budget (this is a first combined-study pass; the one-at-a-time comparison
used m_ge=250, n_restarts=3 per parameter -- here p_act=5 instead of
p_act<=2, so the augmented kernel is already p+1 = 6x wider at the same M,
and cost grows cubically in that).

Usage
-----
    python3 run_combined.py
"""

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_COMPARISON_DIR = os.path.join(_ROOT, 'gpr_vs_gegpr_comparison')

sys.path.insert(0, _HERE)
import ge_gpr_combined  # noqa: E402

# Same x10 rescaling as gpr_vs_gegpr_comparison/run_comparison.py, so the
# void/Teflon slab spans the same L=10 domain and [3, 7] QoI window as the
# material studies there.
_GEO_SCALE = 10.0
_GEO_L = 10.0
_GEO_CORE_RADIUS = 0.16921 * _GEO_SCALE
_GEO_SHIELD_THICKNESS = 0.07801 * _GEO_SCALE
_GEO_SHELL_WIDTH = 0.25278 * _GEO_SCALE
_GEO_QOI_XMIN, _GEO_QOI_XMAX = 3.0, 7.0

# Same bound ranges as the 5 one-at-a-time studies, all active together
# instead of one at a time.
COMBINED_KWARGS = dict(
    L=_GEO_L, core_radius=_GEO_CORE_RADIUS, shield_thickness=_GEO_SHIELD_THICKNESS,
    shell_width=_GEO_SHELL_WIDTH, qoi_xmin=_GEO_QOI_XMIN, qoi_xmax=_GEO_QOI_XMAX,
    void_bounds=(0.001 * _GEO_SCALE, 0.25 * _GEO_SCALE),
    steepness_bounds=(100.0 / _GEO_SCALE, 400.0 / _GEO_SCALE),
    d_teflon_bounds=(0.5, 2.0),
    sigma_a_teflon_bounds=(0.05, 1.5),
    q_teflon_bounds=(0.0, 2.0),
    n_cells=200, m_train=300, m_test=60, seed=1, energy_threshold=0.99999,
    m_ge=80, n_restarts=2,
    learning_curve_sizes=[10, 20, 30, 40, 50, 60, 70, 80],
    output_dir=os.path.join(_HERE, "output_graphs_combined"),
)


def _load_one_at_a_time_summary():
    """Read gpr_vs_gegpr_comparison/summary_table.csv if it exists, for
    a side-by-side comparison against the combined result. Returns None
    (with a warning) if that comparison hasn't been run yet."""
    path = os.path.join(_COMPARISON_DIR, "summary_table.csv")
    if not os.path.exists(path):
        print(f"  (No one-at-a-time baseline found at {path} -- run "
              f"gpr_vs_gegpr_comparison/run_comparison.py first for a "
              f"side-by-side comparison.)")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_comparison(result, one_at_a_time_rows):
    path = os.path.join(_HERE, "combined_vs_one_at_a_time.csv")
    rows = []
    if one_at_a_time_rows:
        for row in one_at_a_time_rows:
            rows.append(dict(
                study=row["parameter"], p_act=1,
                median_eps_std_gpr=row["median_eps_std_gpr"],
                median_eps_ge_gpr=row["median_eps_ge_gpr"],
                improvement_pct=row["improvement_pct"],
            ))
    med_std = float(np.median(result["eps_std"]))
    med_ge = float(np.median(result["eps_ge"]))
    rows.append(dict(
        study="combined (5 active)", p_act=result["p_act"],
        median_eps_std_gpr=f"{med_std:.4e}", median_eps_ge_gpr=f"{med_ge:.4e}",
        improvement_pct=f"{100.0 * (med_std - med_ge) / med_std:.1f}",
    ))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 78}")
    print(f"{'Study':<22}{'p_act':>7}{'median std':>14}{'median GE':>14}{'improve %':>12}")
    print(f"{'=' * 78}")
    for row in rows:
        print(f"{row['study']:<22}{row['p_act']!s:>7}"
              f"{row['median_eps_std_gpr']:>14}{row['median_eps_ge_gpr']:>14}"
              f"{row['improvement_pct']:>11}%")
    print(f"{'=' * 78}")
    print(f"Saved: {path}")


def main():
    print("#" * 70)
    print("# Fully-coupled 5-parameter study (all active simultaneously)")
    print("#" * 70)

    os.makedirs(COMBINED_KWARGS["output_dir"], exist_ok=True)
    result = ge_gpr_combined.main(**COMBINED_KWARGS)

    one_at_a_time_rows = _load_one_at_a_time_summary()
    _write_comparison(result, one_at_a_time_rows)

    print(f"\n{'=' * 70}")
    print(f"  Combined study complete. Plots in {COMBINED_KWARGS['output_dir']}/, "
          f"comparison in {_HERE}/combined_vs_one_at_a_time.csv")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
