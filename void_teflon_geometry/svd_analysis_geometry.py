"""
QoI-side Snapshot SVD / POD — geometry-parameterized Void/Teflon slab.

Mirrors common/svd_analysis.py (Sections 6-8 of the workflow document),
substituting:
  - parameter        : mu = (void_thickness, steepness)  instead of (D, Sigma_a, q) per layer
  - forward solver    : geometry.solve_void_teflon()  instead of solve_diffusion()
  - sampler           : params.GeometryParamSampler   instead of DiffusionParamSampler

Material property values (D, Sigma_a, q per region) are held fixed constants
for this study. void_thickness and steepness are both routed through
GeometryParamSampler: pass a real (lo, hi) range to make either one an
active, LHS-sampled parameter, or lo == hi (the default for steepness) to
hold it fixed, exactly as DiffusionParamSampler does for materials — see
params.py. Output files are written in the same format as
common/svd_analysis.py so the existing sklearn_implementation/gpr_pod.py can
be reused unmodified.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh          # noqa: E402
from geometry import solve_void_teflon               # noqa: E402
from params import GeometryParamSampler              # noqa: E402


def _mu_label(mu, sampler, fmt="{name}={val:.4g}"):
    """
    Human-readable label for one parameter vector, showing only the *active*
    parameters (falls back to all of them if none are active) — so plot
    titles/legends automatically reflect whichever of void_thickness /
    steepness is actually varying, instead of hardcoding one of them.
    """
    names = sampler.active_names or sampler.param_names
    idx = [sampler.param_names.index(n) for n in names]
    return ", ".join(fmt.format(name=sampler.param_names[j], val=mu[j]) for j in idx)


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
    m_train=300,
    m_test=60,
    seed=1,
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs_geometry",
    energy_threshold=0.9999,
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
):
    # Default QoI window: the entire (fixed-width) Void+Teflon shell on the
    # right half of the slab — this is exactly the region whose material
    # composition depends on the active geometry parameter.
    QOI_XMIN = qoi_xmin if qoi_xmin is not None else core_radius
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else core_radius + shell_width

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Geometry-parameterized ROM — SVD / POD analysis")
    print("=" * 60)
    print(f"  Geometry: L={L}, core_radius={core_radius}, shield_thickness={shield_thickness}, "
          f"shell_width={shell_width}")
    print(f"  void_thickness bounds: {void_bounds}   steepness bounds: {steepness_bounds}")

    # ------------------------------------------------------------------ #
    # Step 1 - Build snapshot matrices for training and test sets        #
    # ------------------------------------------------------------------ #
    sampler = GeometryParamSampler(void_bounds, steepness_bounds)
    print(f"  Active parameter(s): {sampler.active_names or '(none — all fixed)'}")
    X_tr = sampler.sample(m_train, random_state=seed)            # (m_train, 2)
    X_te = sampler.sample_random(m_test, random_state=seed + 1)  # (m_test, 2)

    x_centers, x_faces, _ = build_mesh(L, n_cells)

    # ------------------------------------------------------------------ #
    # Step 2 - QoI masking operator H                                    #
    # ------------------------------------------------------------------ #
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi = x_centers[qoi_mask]
    Ny = qoi_mask.sum()

    print(f"QoI interval  : [{QOI_XMIN:.4f}, {QOI_XMAX:.4f}] cm  (Ny={Ny} of {n_cells} cells)")

    def solve_qoi_batch(X_params, label):
        M = len(X_params)
        Y_out = np.empty((Ny, M))
        print(f"  Solving {M} {label} snapshots")
        for i, mu in enumerate(X_params):
            vt, steepness = sampler.unpack(mu)
            _, phi_i, _ = solve_void_teflon(
                L, n_cells, core_radius, vt, shell_width,
                mat_core, mat_void, mat_teflon, mat_shield, steepness,
                bc_left=bc_left, bc_right=bc_right,
            )
            Y_out[:, i] = phi_i[qoi_mask]
        return Y_out

    Y_tr = solve_qoi_batch(X_tr, "training")
    Y_te = solve_qoi_batch(X_te, "test")

    print(f"Y_tr : {Y_tr.shape},  Y_te : {Y_te.shape}")

    # ------------------------------------------------------------------ #
    # Save samples and QoI values                                        #
    # ------------------------------------------------------------------ #
    param_header = "  ".join(sampler.param_names)
    x_header = "  ".join([f"x={xi:.4f}" for xi in x_qoi])

    for tag, X_out, Y_out in [("train", X_tr, Y_tr), ("test", X_te, Y_te)]:
        np.savetxt(f"{output_dir}/samples_{tag}.txt", X_out, header=param_header, fmt="%.6e", comments="# ")
        np.savetxt(f"{output_dir}/qoi_values_{tag}.txt", Y_out.T, header=x_header, fmt="%.6e", comments="# ")
        print(f"Saved: samples_{tag}.txt  ({X_out.shape[0]} x {X_out.shape[1]}),  "
              f"qoi_values_{tag}.txt  ({Y_out.shape[1]} x {Y_out.shape[0]})")

    # ------------------------------------------------------------------ #
    # Step 3 - Plot representative masked snapshots                      #
    # ------------------------------------------------------------------ #
    N_EXAMPLES = 8
    rng_plot = np.random.default_rng(0)
    ex_idx = rng_plot.choice(m_train, size=N_EXAMPLES, replace=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, N_EXAMPLES))
    # Sort by the first active parameter for a visually ordered legend
    # (falls back to void_thickness if nothing is active).
    sort_col = sampler.param_names.index(sampler.active_names[0]) if sampler.active_names else 0
    order = np.argsort(X_tr[ex_idx, sort_col])
    for k, idx in zip(range(N_EXAMPLES), ex_idx[order]):
        ax.plot(x_qoi, Y_tr[:, idx], color=colors[k], lw=1.4, label=_mu_label(X_tr[idx], sampler))

    ax.axvline(QOI_XMIN, color='k', ls='--', lw=1.0, label='QoI boundary')
    ax.axvline(QOI_XMAX, color='k', ls='--', lw=1.0)
    ax.axvline(core_radius, color='gray', ls=':', lw=0.8, label='Core radius (fixed)')

    ax.set_xlabel("x  [cm]", fontsize=12)
    ax.set_ylabel(r"$y(\mu) = H\phi(\mu)$", fontsize=12)
    active_desc = " & ".join(sampler.active_names) if sampler.active_names else "no active parameters"
    ax.set_title(
        f"Representative QoI snapshots vs. {active_desc}  "
        f"(mask: [{QOI_XMIN:.3f}, {QOI_XMAX:.3f}] cm, Ny={Ny})",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qoi_snapshots.png", dpi=150)
    print(f"Saved: {output_dir}/qoi_snapshots.png")

    # ------------------------------------------------------------------ #
    # Step 4 - Thin SVD of the QoI snapshot matrix Y                     #
    # ------------------------------------------------------------------ #
    print("\nComputing thin SVD of Y_tr  (training snapshots only)")
    U, sigma, Vh = np.linalg.svd(Y_tr, full_matrices=False)
    r_max = len(sigma)
    print(f"Singular values (first 10): {sigma[:10].round(4)}")

    # ------------------------------------------------------------------ #
    # Step 5 - Energy criterion and rank selection                       #
    # ------------------------------------------------------------------ #
    energy = sigma ** 2
    total_energy = energy.sum()
    rel_energy = energy / total_energy
    cum_energy = np.cumsum(rel_energy)

    r_999 = int(np.searchsorted(cum_energy, 0.999)) + 1
    r_9999 = int(np.searchsorted(cum_energy, 0.9999)) + 1
    print(f"\nModes for 99.9  % energy : {r_999}")
    print(f"Modes for 99.99 % energy : {r_9999}")

    R = int(np.searchsorted(cum_energy, energy_threshold)) + 1
    Psi = U[:, :R]
    print(f"Using r = {R}  (eta = {energy_threshold})")

    # ------------------------------------------------------------------ #
    # Coefficient matrix                                                  #
    # ------------------------------------------------------------------ #
    A_alpha = Psi.T @ Y_tr
    print(f"Coefficient matrix A_alpha: {A_alpha.shape}  (R x M_train)")

    np.savetxt(f"{output_dir}/pod_basis.txt", Psi, fmt="%.6e",
               header=f"POD basis Psi  shape ({Ny} x {R})  -- columns are spatial modes", comments="# ")
    np.savetxt(f"{output_dir}/coefficients_train.txt", A_alpha.T, fmt="%.6e",
               header=f"Reduced coefficients A_alpha.T  shape ({m_train} x {R})  -- rows are alpha^(i) = Psi^T y^(i)",
               comments="# ")
    print(f"Saved: {output_dir}/pod_basis.txt            ({Ny} x {R})")
    print(f"Saved: {output_dir}/coefficients_train.txt   ({m_train} x {R})")

    # ------------------------------------------------------------------ #
    # Step 6 - Singular value decay and cumulative energy plots          #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    modes_plot = min(40, r_max)

    ax = axes[0]
    ax.semilogy(np.arange(1, modes_plot + 1), sigma[:modes_plot], 'o-', ms=4, lw=1.5, color='steelblue')
    ax.set_xlabel("Mode index  $k$", fontsize=12)
    ax.set_ylabel(r"Singular value  $\sigma_k$", fontsize=12)
    ax.set_title("Singular value decay  (semi-log)", fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axes[1]
    ax.plot(np.arange(1, modes_plot + 1), cum_energy[:modes_plot] * 100, 's-', ms=4, lw=1.5, color='tomato')
    ax.axhline(99.9, color='gray', ls='--', lw=1.2, label=r'$\eta = 99.9\,\%$')
    ax.axhline(99.99, color='black', ls=':', lw=1.2, label=r'$\eta = 99.99\,\%$')
    ax.axvline(r_999, color='gray', ls='--', lw=1.2)
    ax.axvline(r_9999, color='black', ls=':', lw=1.2)
    ax.set_xlabel("Number of modes  $r$", fontsize=12)
    ax.set_ylabel("Cumulative energy  [%]", fontsize=12)
    ax.set_title("Cumulative energy captured", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.5)

    fig.suptitle(f"SVD spectrum of QoI snapshot matrix  (Ny={Ny}, M_train={m_train})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_spectrum.png", dpi=150)
    print(f"Saved: {output_dir}/svd_spectrum.png")

    # ------------------------------------------------------------------ #
    # Step 7 - Leading POD modes                                          #
    # ------------------------------------------------------------------ #
    N_MODES_PLOT = min(6, R)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for k, ax in enumerate(axes.ravel()):
        if k >= N_MODES_PLOT:
            ax.set_visible(False)
            continue
        mode = U[:, k].copy()
        if mode[np.argmax(np.abs(mode))] < 0:
            mode = -mode
        frac = rel_energy[k] * 100
        ax.plot(x_qoi, mode, lw=1.8, color=f'C{k}')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel("x  [cm]", fontsize=10)
        ax.set_ylabel(r"$\psi_k(x)$", fontsize=10)
        ax.set_title(f"Mode {k+1}  ({frac:.2f} %,  $\\sigma_{k+1}={sigma[k]:.3f}$)", fontsize=10)
        ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle(f"Leading {N_MODES_PLOT} POD modes  Psi  (QoI domain [{QOI_XMIN:.3f}, {QOI_XMAX:.3f}] cm)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_modes.png", dpi=150)
    print(f"Saved: {output_dir}/svd_modes.png")

    # ------------------------------------------------------------------ #
    # Step 8 - Reconstruction error vs rank                               #
    # ------------------------------------------------------------------ #
    nY_tr = np.linalg.norm(Y_tr, 'fro')
    ranks = np.arange(1, min(r_max + 1, 41))
    frob_errors = np.array([np.sqrt(energy[r:].sum()) / nY_tr for r in ranks])

    RANK_LIST = sorted(set([1, 2, 3, min(5, r_max), R]))
    SNAP_INDICES = [0, 1, 2]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors_r = plt.get_cmap('cool')(np.linspace(0, 1, len(RANK_LIST)))
    for col_idx, snap_idx in enumerate(SNAP_INDICES):
        ax = axes[0, col_idx] if col_idx < 2 else axes[1, 0]
        ax.plot(x_qoi, Y_tr[:, snap_idx], 'k-', lw=2, label='Full QoI snapshot')
        for r_val, col in zip(RANK_LIST, colors_r):
            Psi_r = U[:, :r_val]
            y_recon = Psi_r @ (Psi_r.T @ Y_tr[:, snap_idx])
            rel_err = np.linalg.norm(y_recon - Y_tr[:, snap_idx]) / np.linalg.norm(Y_tr[:, snap_idx])
            ax.plot(x_qoi, y_recon, '--', lw=1.3, color=col, label=f'r={r_val}  (err={rel_err:.1e})')
        ax.set_xlabel("x  [cm]", fontsize=11)
        ax.set_ylabel(r"$y(\mu)$", fontsize=11)
        ax.set_title(f"Training snapshot {snap_idx} reconstruction "
                     f"({_mu_label(X_tr[snap_idx], sampler)})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)

    ax_r = axes[1, 1]
    ax_r.semilogy(ranks, frob_errors, 'o-', ms=4, lw=1.5, color='darkgreen')
    for r_val in [r_999, r_9999]:
        ax_r.axvline(r_val, ls='--', lw=1, color='gray', label=f'r={r_val}')
    ax_r.set_xlabel("Rank  $r$", fontsize=11)
    ax_r.set_ylabel(r"$\|Y - Y_r\|_F \;/\; \|Y\|_F$", fontsize=11)
    ax_r.set_title("Relative Frobenius reconstruction error vs rank", fontsize=11)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, which='both', ls='--', alpha=0.5)

    fig.suptitle("POD reconstruction quality  (QoI snapshots, geometry parameter)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_reconstruction.png", dpi=150)
    print(f"Saved: {output_dir}/svd_reconstruction.png")

    # ------------------------------------------------------------------ #
    # eps_proj on the held-out test set vs rank                          #
    # ------------------------------------------------------------------ #
    RANK_TEST = sorted(set([1, 2, 3, min(5, r_max), R]))
    eps_by_rank = {}
    for r_val in RANK_TEST:
        Psi_r = U[:, :r_val]
        proj = Psi_r @ (Psi_r.T @ Y_te)
        resid = np.linalg.norm(Y_te - proj, axis=0)
        norms = np.linalg.norm(Y_te, axis=0)
        eps_by_rank[r_val] = resid / norms

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([eps_by_rank[r] for r in RANK_TEST], tick_labels=[str(r) for r in RANK_TEST],
                     patch_artist=True, medianprops=dict(color='black', lw=2))
    colors_box = plt.get_cmap('cool')(np.linspace(0, 1, len(RANK_TEST)))
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.set_xlabel("Rank  $r$", fontsize=12)
    ax.set_ylabel(r"$\varepsilon_{\mathrm{proj}}(\mu)$", fontsize=12)
    ax.set_title(f"Projection error on test set  (M_test={m_test})\n"
                 r"$\varepsilon_{\mathrm{proj}} = \|y - \Psi_r\Psi_r^\top y\|_2 \;/\; \|y\|_2$", fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_proj_error.png", dpi=150)
    print(f"Saved: {output_dir}/svd_proj_error.png")

    # ------------------------------------------------------------------ #
    # Summary                                                            #
    # ------------------------------------------------------------------ #
    print("\n=== SVD summary (geometry parameter) ===")
    print(f"  QoI interval          : [{QOI_XMIN:.4f}, {QOI_XMAX:.4f}] cm")
    print(f"  QoI dimension Ny      : {Ny}")
    print(f"  Train snapshots M_tr  : {m_train}")
    print(f"  Test  snapshots M_te  : {m_test}")
    print(f"  Snapshot matrix Y_tr  : {Y_tr.shape}")
    print(f"  Coefficient matrix    : {A_alpha.shape}  (R x M_train)")
    print(f"  sigma_1               : {sigma[0]:.4f}")
    if r_max > 1:
        print(f"  sigma_1 / sigma_2     : {sigma[0]/sigma[1]:.3f}")
    print(f"  Energy in mode 1      : {rel_energy[0]*100:.4f} %")
    print(f"  Modes for eta = 99.9  % : {r_999}")
    print(f"  Modes for eta = 99.99 % : {r_9999}")
    print(f"  Chosen rank R         : {R}")
    print(f"  ||Y_tr-Y_R||_F/||Y_tr||_F : {frob_errors[R-1]:.2e}")
    print(f"\n  Test eps_proj (r={R}) -- median : {np.median(eps_by_rank[R]):.2e},  "
          f"max : {np.max(eps_by_rank[R]):.2e}")

    return dict(Psi=Psi, A_alpha=A_alpha, X_tr=X_tr, X_te=X_te, Y_tr=Y_tr, Y_te=Y_te,
                x_qoi=x_qoi, qoi_mask=qoi_mask, R=R, sampler=sampler)


if __name__ == "__main__":
    main()
