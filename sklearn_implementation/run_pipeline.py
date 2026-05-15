"""
run_pipeline.py — End-to-end ROM pipeline runner (scikit-learn version)

Mirrors pytorch_implementation/run_pipeline.py exactly, substituting
the GPyTorch GPR stage with the sklearn GaussianProcessRegressor stage.

Stages
------
1. svd_analysis.main()  — Generate snapshots, compute POD basis, write output_dir/
2. gpr_pod.main()       — Train sklearn GP regressors on POD coefficients, evaluate, plot

Key parameter difference vs PyTorch version
--------------------------------------------
PyTorch  :  n_epochs (Adam iterations per GP)  +  lr (learning rate)
sklearn  :  n_restarts (L-BFGS-B random restarts per GP)
"""

import os
import sys
import time

# Common utilities live one level up
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from svd_analysis import main as run_svd
from gpr_pod import main as run_gpr
from adjoint_stage import main as run_adjoint
from ge_gpr import main as run_ge_gpr


def _per_layer_bounds(scalar_bounds, n_layers, fixed_mats, key):
    """
    Build a list of [lo, hi] pairs for each layer.

    scalar_bounds : (lo, hi) applied to all non-fixed layers
    fixed_mats    : dict mapping 1-based material index to a dict of fixed
                    values, e.g. {1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}}
    key           : "D", "sigma_a", or "q"
    """
    lo, hi = scalar_bounds
    bounds = [[lo, hi] for _ in range(n_layers)]
    if fixed_mats:
        for mat_idx, vals in fixed_mats.items():
            if key in vals:
                v = vals[key]
                bounds[mat_idx - 1] = [v, v]
    return bounds


def main(
    # Shared physical / mesh parameters
    L=10.0,
    n_bins=3,
    n_cells=200,
    d_bounds=(0.5, 2.0),
    sigma_a_bounds=(0.05, 1.5),
    q_bounds=(1.0, 1.0),
    qoi_xmin=3.0,
    qoi_xmax=7.0,
    output_dir="output_graphs_sklearn",
    # Example: fix Material 1 at D=1.0, sigma_a=0.5, q=1.0
    fixed_mats={1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}},
    # Stage 1 — SVD / POD
    m_train=500,
    m_test=100,
    seed=1,
    energy_threshold=0.999,
    # Stage 2 — GPR (sklearn)
    n_restarts=5,
    # Stage 3 — Adjoint sensitivities
    n_verify=10,
    n_fd_check=3,
    n_sweep=60,
    mode_indices=(0,),
    expansion_point=1.2,
    # Stage 4 — GE-GPR
    m_ge=100,
    ge_n_restarts=3,
    ge_noise_f=1e-3,
    ge_noise_d=1e-3,
    ge_run_lc=True,
    ge_lc_sizes=None,
    # Pipeline control
    skip_svd=False,
    skip_gpr=False,
    skip_adjoint=False,
    skip_ge_gpr=False,
):
    # Build per-layer bounds, fixing any materials listed in fixed_mats
    d_bounds_layers       = _per_layer_bounds(d_bounds,       n_bins, fixed_mats, "D")
    sigma_a_bounds_layers = _per_layer_bounds(sigma_a_bounds, n_bins, fixed_mats, "sigma_a")
    q_bounds_layers       = _per_layer_bounds(q_bounds,       n_bins, fixed_mats, "q")

    if fixed_mats:
        fixed_str = ", ".join(f"Mat {k}" for k in sorted(fixed_mats))
        print(f"  Fixed materials: {fixed_str}")

    shared = dict(
        L=L,
        n_bins=n_bins,
        n_cells=n_cells,
        d_bounds=d_bounds_layers,
        sigma_a_bounds=sigma_a_bounds_layers,
        q_bounds=q_bounds_layers,
        qoi_xmin=qoi_xmin,
        qoi_xmax=qoi_xmax,
        output_dir=output_dir,
    )

    pipeline_start = time.perf_counter()

    if not skip_svd:
        print(f"\n{'='*60}")
        print("Stage 1 — SVD / POD analysis")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        run_svd(**shared, m_train=m_train, m_test=m_test, seed=seed,
                energy_threshold=energy_threshold)
        print(f"\n  [Stage 1]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 1 — SVD / POD analysis")

    if not skip_gpr:
        print(f"\n{'='*60}")
        print("  Stage 2 — GPR (sklearn) on POD coefficients")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        run_gpr(**shared, n_restarts=n_restarts)
        print(f"\n  [Stage 2]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 2 — GPR on POD coefficients")

    if not skip_adjoint:
        print(f"\n{'='*60}")
        print("  Stage 3 — Adjoint sensitivities")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        run_adjoint(
            **shared,
            fixed_mats=fixed_mats,
            n_verify=n_verify,
            n_fd_check=n_fd_check,
            n_sweep=n_sweep,
            mode_indices=mode_indices,
            expansion_point=expansion_point,
        )
        print(f"\n  [Stage 3]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 3 — Adjoint sensitivities")

    if not skip_ge_gpr:
        print(f"\n{'='*60}")
        print("  Stage 4 — GE-GPR (gradient-enhanced surrogate)")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        run_ge_gpr(
            **shared,
            fixed_mats=fixed_mats,
            m_ge=m_ge,
            n_restarts=ge_n_restarts,
            noise_f_init=ge_noise_f,
            noise_d_init=ge_noise_d,
            run_learning_curve=ge_run_lc,
            learning_curve_sizes=ge_lc_sizes,
        )
        print(f"\n  [Stage 4]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 4 — GE-GPR")

    total = time.perf_counter() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete  ({total:.1f}s total)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example: fix Material 1 at mid-range values; Mat 2 and Mat 3 vary freely.
    main(
        fixed_mats={1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}},
    )
