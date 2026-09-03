"""
run_pipeline_geometry.py — End-to-end ROM pipeline, geometry parameter.

Mirrors sklearn_implementation/run_pipeline.py, but the active parameter
vector mu = (void_thickness, steepness) holds *geometry / switch*
parameters (the Void/Teflon interface location within the fixed
Void+Teflon shell, and the sigmoid steepness) rather than the per-layer
material vector (D, Sigma_a, q). Either entry of mu can be made active by
passing a real (lo, hi) range, or held fixed via degenerate (lo == hi)
bounds — steepness_bounds defaults to fixed, matching the original
single-steepness study; pass e.g. steepness_bounds=(100.0, 400.0) to make
the switch sharpness itself an active, adjoint-differentiated parameter.

Stages
------
1. svd_analysis_geometry.main()  — snapshots over the active parameter(s), POD basis
2. gpr_pod.main()  (reused unmodified from sklearn_implementation/)
                    — GP surrogate mu -> alpha(mu); this module only reads
                      samples/coefficients/POD-basis files and does not care
                      what the parameter physically represents.
3. adjoint_geometry.main()  — adjoint sensitivities d(alpha_k)/d(param) for
                    every active parameter, identity check, FD gradient
                    verification, 1D sweep.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sklearn_implementation'))

from svd_analysis_geometry import main as run_svd   # noqa: E402
from gpr_pod import main as run_gpr                 # noqa: E402
from adjoint_geometry import main as run_adjoint    # noqa: E402


def main(
    L=1.0,
    core_radius=0.16921,
    shield_thickness=0.07801,
    shell_width=0.25278,
    mat_core=(1.0, 0.5, 1.0),
    mat_void=(1.8, 0.01, 0.0),
    mat_teflon=(0.6, 0.4, 0.0),
    mat_shield=(0.3, 0.9, 0.0),
    n_cells=3000,
    void_bounds=(0.001, 0.25),
    steepness_bounds=(200.0, 200.0),
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs_geometry",
    # Stage 1 — SVD / POD
    m_train=300,
    m_test=60,
    seed=1,
    energy_threshold=0.9999,
    # Stage 2 — GPR (sklearn, reused)
    n_restarts=5,
    # Stage 3 — Adjoint sensitivities
    n_verify=10,
    n_fd_check=5,
    n_sweep=60,
    mode_indices=(0, 1),
    expansion_point=None,
    # Pipeline control
    skip_svd=False,
    skip_gpr=False,
    skip_adjoint=False,
):
    shared_geom = dict(
        L=L, core_radius=core_radius, shield_thickness=shield_thickness, shell_width=shell_width,
        mat_core=mat_core, mat_void=mat_void, mat_teflon=mat_teflon, mat_shield=mat_shield,
        n_cells=n_cells, void_bounds=void_bounds, steepness_bounds=steepness_bounds,
        qoi_xmin=qoi_xmin, qoi_xmax=qoi_xmax, output_dir=output_dir,
    )

    pipeline_start = time.perf_counter()

    if not skip_svd:
        print(f"\n{'=' * 60}")
        print("Stage 1 — SVD / POD analysis (geometry parameter)")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()
        run_svd(**shared_geom, m_train=m_train, m_test=m_test, seed=seed,
                energy_threshold=energy_threshold)
        print(f"\n  [Stage 1]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 1 — SVD / POD analysis")

    if not skip_gpr:
        print(f"\n{'=' * 60}")
        print("  Stage 2 — GPR (sklearn, reused) on POD coefficients")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()
        run_gpr(L=L, n_cells=n_cells, qoi_xmin=shared_geom['qoi_xmin'] or core_radius,
                qoi_xmax=shared_geom['qoi_xmax'] or (core_radius + shell_width),
                output_dir=output_dir, n_restarts=n_restarts)
        print(f"\n  [Stage 2]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 2 — GPR on POD coefficients")

    if not skip_adjoint:
        print(f"\n{'=' * 60}")
        print("  Stage 3 — Adjoint sensitivities (geometry / switch parameters)")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()
        run_adjoint(
            **shared_geom,
            n_verify=n_verify, n_fd_check=n_fd_check, n_sweep=n_sweep,
            mode_indices=mode_indices, expansion_point=expansion_point,
        )
        print(f"\n  [Stage 3]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 3 — Adjoint sensitivities")

    total = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete  ({total:.1f}s total)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
