"""
svd_analysis_combined.py — QoI-side Snapshot SVD / POD for the fully-coupled
5-parameter study, mu = (void_thickness, steepness, D_teflon, Sigma_a_teflon,
q_teflon).

Mirrors void_teflon_geometry/svd_analysis_geometry.py, substituting
CombinedParamSampler for GeometryParamSampler so the Teflon region's
material triple (previously fixed mat_teflon constants) is drawn from the
same 5D LHS design as void_thickness/steepness instead of held fixed.
Forward solves still go through geometry.solve_void_teflon() unchanged --
only mat_teflon is now read per-sample from mu[2:5] rather than passed in
as one constant tuple. Output files are written in the same format as
svd_analysis_geometry.py, so sklearn_implementation/ge_gpr.py's
_load_pod_artifacts() and GEGaussianProcess can be reused unmodified.

Trimmed relative to svd_analysis_geometry.py: keeps the snapshot solve,
SVD/rank selection, and the spectrum + projection-error sanity plots;
drops the per-mode POD-mode-shape and reconstruction-quadrant plots, which
add little for a first combined-study pass.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'void_teflon_geometry'))
from second_order_solver import build_mesh    # noqa: E402
from geometry import solve_void_teflon        # noqa: E402
from params_combined import CombinedParamSampler  # noqa: E402


def main(
    L=1.0,
    core_radius=0.16921,
    shield_thickness=0.07801,
    shell_width=0.25278,
    mat_core=(1.0, 0.5, 1.0),
    mat_void=(1.8, 0.01, 0.0),
    mat_shield=(0.3, 0.9, 0.0),
    n_cells=3000,
    void_bounds=(0.001, 0.25),
    steepness_bounds=(200.0, 200.0),
    d_teflon_bounds=(0.5, 2.0),
    sigma_a_teflon_bounds=(0.05, 1.5),
    q_teflon_bounds=(0.0, 2.0),
    m_train=200,
    m_test=60,
    seed=1,
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs_combined",
    energy_threshold=0.9999,
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
):
    QOI_XMIN = qoi_xmin if qoi_xmin is not None else core_radius
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else core_radius + shell_width

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Combined 5-parameter ROM — SVD / POD analysis")
    print("=" * 60)

    sampler = CombinedParamSampler(
        void_bounds, steepness_bounds, d_teflon_bounds, sigma_a_teflon_bounds, q_teflon_bounds,
    )
    print(f"  Active parameter(s): {sampler.active_names or '(none — all fixed)'}")
    X_tr = sampler.sample(m_train, random_state=seed)
    X_te = sampler.sample_random(m_test, random_state=seed + 1)

    x_centers, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi = x_centers[qoi_mask]
    Ny = qoi_mask.sum()
    print(f"  QoI interval: [{QOI_XMIN:.4f}, {QOI_XMAX:.4f}] cm  (Ny={Ny} of {n_cells} cells)")

    def solve_qoi_batch(X_params, label):
        M = len(X_params)
        Y_out = np.empty((Ny, M))
        print(f"  Solving {M} {label} snapshots")
        for i, mu in enumerate(X_params):
            vt, steepness, mat_teflon = sampler.unpack(mu)
            _, phi_i, _ = solve_void_teflon(
                L, n_cells, core_radius, vt, shell_width,
                mat_core, mat_void, mat_teflon, mat_shield, steepness,
                bc_left=bc_left, bc_right=bc_right,
            )
            Y_out[:, i] = phi_i[qoi_mask]
        return Y_out

    Y_tr = solve_qoi_batch(X_tr, "training")
    Y_te = solve_qoi_batch(X_te, "test")
    print(f"  Y_tr: {Y_tr.shape},  Y_te: {Y_te.shape}")

    # ------------------------------------------------------------------ #
    # Save samples and QoI values                                        #
    # ------------------------------------------------------------------ #
    param_header = "  ".join(sampler.param_names)
    x_header = "  ".join([f"x={xi:.4f}" for xi in x_qoi])
    for tag, X_out, Y_out in [("train", X_tr, Y_tr), ("test", X_te, Y_te)]:
        np.savetxt(f"{output_dir}/samples_{tag}.txt", X_out, header=param_header, fmt="%.6e", comments="# ")
        np.savetxt(f"{output_dir}/qoi_values_{tag}.txt", Y_out.T, header=x_header, fmt="%.6e", comments="# ")
        print(f"  Saved: samples_{tag}.txt ({X_out.shape}),  qoi_values_{tag}.txt ({Y_out.T.shape})")

    # ------------------------------------------------------------------ #
    # Thin SVD, rank selection, coefficient matrix                       #
    # ------------------------------------------------------------------ #
    print("\nComputing thin SVD of Y_tr (training snapshots only)")
    U, sigma, Vh = np.linalg.svd(Y_tr, full_matrices=False)
    r_max = len(sigma)

    energy = sigma ** 2
    rel_energy = energy / energy.sum()
    cum_energy = np.cumsum(rel_energy)
    R = int(np.searchsorted(cum_energy, energy_threshold)) + 1
    Psi = U[:, :R]
    print(f"  Using r = {R}  (eta = {energy_threshold}),  energy in mode 1 = {rel_energy[0]*100:.4f}%")

    A_alpha = Psi.T @ Y_tr
    np.savetxt(f"{output_dir}/pod_basis.txt", Psi, fmt="%.6e",
               header=f"POD basis Psi  shape ({Ny} x {R})", comments="# ")
    np.savetxt(f"{output_dir}/coefficients_train.txt", A_alpha.T, fmt="%.6e",
               header=f"Reduced coefficients A_alpha.T  shape ({m_train} x {R})", comments="# ")
    print(f"  Saved: pod_basis.txt ({Ny} x {R}),  coefficients_train.txt ({m_train} x {R})")

    # ------------------------------------------------------------------ #
    # Sanity plots: spectrum + test-set projection error vs rank         #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    modes_plot = min(30, r_max)
    axes[0].semilogy(np.arange(1, modes_plot + 1), sigma[:modes_plot], 'o-', ms=4, lw=1.5, color='steelblue')
    axes[0].set_xlabel("Mode index  $k$"); axes[0].set_ylabel(r"Singular value  $\sigma_k$")
    axes[0].set_title("Singular value decay"); axes[0].grid(True, which='both', ls='--', alpha=0.5)

    axes[1].plot(np.arange(1, modes_plot + 1), cum_energy[:modes_plot] * 100, 's-', ms=4, lw=1.5, color='tomato')
    axes[1].axvline(R, color='gray', ls='--', lw=1.2, label=f'R={R}')
    axes[1].set_xlabel("Number of modes  $r$"); axes[1].set_ylabel("Cumulative energy [%]")
    axes[1].set_title("Cumulative energy"); axes[1].legend(); axes[1].grid(True, ls='--', alpha=0.5)
    fig.suptitle(f"SVD spectrum — combined 5-parameter study (Ny={Ny}, M_train={m_train})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_spectrum.png", dpi=150)
    print(f"  Saved: {output_dir}/svd_spectrum.png")
    plt.close(fig)

    RANK_TEST = sorted(set([1, 2, 3, min(5, r_max), R]))
    eps_by_rank = {}
    for r_val in RANK_TEST:
        Psi_r = U[:, :r_val]
        proj = Psi_r @ (Psi_r.T @ Y_te)
        eps_by_rank[r_val] = np.linalg.norm(Y_te - proj, axis=0) / np.linalg.norm(Y_te, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([eps_by_rank[r] for r in RANK_TEST], tick_labels=[str(r) for r in RANK_TEST],
                     patch_artist=True, medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], plt.get_cmap('cool')(np.linspace(0, 1, len(RANK_TEST)))):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_xlabel("Rank  $r$"); ax.set_ylabel(r"$\varepsilon_{\mathrm{proj}}(\mu)$")
    ax.set_title(f"Projection error on test set  (M_test={m_test})")
    ax.set_yscale('log'); ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_proj_error.png", dpi=150)
    print(f"  Saved: {output_dir}/svd_proj_error.png")
    plt.close(fig)

    print("\n=== SVD summary (combined 5-parameter) ===")
    print(f"  Train/test snapshots  : {m_train} / {m_test}")
    print(f"  Chosen rank R         : {R}")
    print(f"  Test eps_proj (r={R}) — median: {np.median(eps_by_rank[R]):.2e}, max: {np.max(eps_by_rank[R]):.2e}")

    return dict(Psi=Psi, A_alpha=A_alpha, X_tr=X_tr, X_te=X_te, Y_tr=Y_tr, Y_te=Y_te,
                x_qoi=x_qoi, qoi_mask=qoi_mask, R=R, sampler=sampler)


if __name__ == "__main__":
    main()
