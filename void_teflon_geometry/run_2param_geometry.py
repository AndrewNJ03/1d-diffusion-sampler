"""
run_2param_geometry.py — GE-GPR vs standard GPR for the void/Teflon
geometry with *both* void_thickness and steepness active simultaneously.

Previously this case (behind the paper's `output_graphs_geometry_2param`
figure) was produced by an undocumented one-off call to
ge_gpr_geometry.main(output_dir='output_graphs_geometry_2param',
run_svd_stage=False) using that function's bare defaults: the original
un-rescaled geometry (L=1.0, core_radius=0.16921, n_cells=3500). That does
not match the shared baseline established in
../gpr_vs_gegpr_comparison/run_comparison.py and ../combined_study/
run_combined.py (L=10 domain, geometry scaled x10, n_cells=200,
energy_threshold=0.99999). This script reruns the 2-param case on that
same shared baseline so all sandbox examples in the paper are on
consistent footing, using the same void_bounds/steepness_bounds ranges as
the two single-parameter studies in run_comparison.py (both active here
instead of one at a time).

Usage
-----
    python3 run_2param_geometry.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import ge_gpr_geometry  # noqa: E402

_GEO_SCALE = 10.0
_GEO_L = 10.0
_GEO_CORE_RADIUS = 0.16921 * _GEO_SCALE
_GEO_SHIELD_THICKNESS = 0.07801 * _GEO_SCALE
_GEO_SHELL_WIDTH = 0.25278 * _GEO_SCALE
_GEO_QOI_XMIN, _GEO_QOI_XMAX = 3.0, 7.0

KWARGS = dict(
    L=_GEO_L, core_radius=_GEO_CORE_RADIUS, shield_thickness=_GEO_SHIELD_THICKNESS,
    shell_width=_GEO_SHELL_WIDTH, qoi_xmin=_GEO_QOI_XMIN, qoi_xmax=_GEO_QOI_XMAX,
    n_cells=200, energy_threshold=0.99999,
    void_bounds=(0.001 * _GEO_SCALE, 0.25 * _GEO_SCALE),
    steepness_bounds=(100.0 / _GEO_SCALE, 400.0 / _GEO_SCALE),
    m_train=300, m_test=60, seed=1,
    m_ge=100, n_restarts=3,
    output_dir=os.path.join(_HERE, "output_graphs_geometry_2param"),
    run_svd_stage=True,
)


def main():
    os.makedirs(KWARGS["output_dir"], exist_ok=True)
    ge_gpr_geometry.main(**KWARGS)


if __name__ == "__main__":
    main()
