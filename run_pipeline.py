"""
run_pipeline.py — End-to-end ROM pipeline runner

Stages
------
1. svd_analysis.main()  — Generate snapshots, compute POD basis, write output_graphs/
2. gpr_pod.main()       — Train GP regressors on POD coefficients, evaluate, plot
"""

import time

from svd_analysis import main as run_svd
from gpr_pod import main as run_gpr


def main(
    # Shared physical / mesh parameters
    L=10.0,
    n_bins=3,
    n_cells=200,
    d_bounds=(0.2, 2.0),
    sigma_a_bounds=(0.05, 1.0),
    q_bounds=(0.0, 2.0),
    qoi_xmin=10.0 / 3,
    qoi_xmax=10.0 / 3 * 2,
    output_dir="output_graphs_M500",
    # Stage 1 — SVD / POD
    m_train=500,
    m_test=100,
    seed=1,
    energy_threshold=0.999,
    # Stage 2 — GPR
    n_epochs=10000,
    lr=0.05,
    # Pipeline control
    skip_svd=False,
    skip_gpr=False,
):
    shared = dict(
        L=L,
        n_bins=n_bins,
        n_cells=n_cells,
        d_bounds=d_bounds,
        sigma_a_bounds=sigma_a_bounds,
        q_bounds=q_bounds,
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
        print("  Stage 2 — GPR on POD coefficients")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        run_gpr(**shared, n_epochs=n_epochs, lr=lr)
        print(f"\n  [Stage 2]  DONE  ({time.perf_counter() - t0:.1f}s)")
    else:
        print("\n  Skipping: Stage 2 — GPR on POD coefficients")

    total = time.perf_counter() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete  ({total:.1f}s total)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
