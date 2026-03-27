"""
QoI-side Snapshot SVD / Proper Orthogonal Decomposition (POD)

Pipeline
-------------------------------------------
1. Generate full-state snapshots  Phi (N_cells x M_samples).
2. Apply the QoI masking operator H to obtain  Y = H Phi  (Ny x M_samples)
3. Plot representative masked snapshots
4. Compute the thin SVD of Y
5. Rank selection via energy criterion  Σ_{k=1}^r σ_k^2 / Σ σ_k^2 ≥ η
6. Plot singular value decay and cumulative energy
7. Visualise leading POD modes on the QoI sub-domain
8. Reconstruction error vs rank
-------
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from lhs_generation import DiffusionParamSampler
from second_order_solver import solve_diffusion, build_mesh


def main(
    L=10.0,
    n_bins=5,
    n_cells=200,
    m_train=400,
    m_test=100,
    seed=1,
    d_bounds=(0.2, 2.0),
    sigma_a_bounds=(0.05, 1.0),
    q_bounds=(0.0, 2.0),
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs",
    energy_threshold=0.999,
):
    QOI_XMIN = qoi_xmin if qoi_xmin is not None else L / 3
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else 2.0 * L / 3

    os.makedirs(output_dir, exist_ok=True)
    layer_bounds = np.linspace(0.0, L, n_bins + 1)

    # ------------------------------------------------------------------ #
    # Step 1 – Build snapshot matrices for training and test sets        #
    # ------------------------------------------------------------------ #
    # Training set: LHS for space-filling coverage.
    # Test set: uniform random within bounds.
    sampler = DiffusionParamSampler(n_bins, d_bounds, sigma_a_bounds, q_bounds)
    X_tr = sampler.sample(m_train, random_state=seed)
    X_te = sampler.sample_random(m_test, random_state=seed + 1)

    # Retrieve mesh geometry
    x_centers, x_faces, _ = build_mesh(L, n_cells)

    # ------------------------------------------------------------------ #
    # Step 2 – QoI masking operator H                                    #
    # ------------------------------------------------------------------ #
    # Cell i occupies [x_faces[i], x_faces[i+1]].
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi    = x_centers[qoi_mask]
    Ny       = qoi_mask.sum()

    print(f"QoI interval  : [{QOI_XMIN:.3f}, {QOI_XMAX:.3f}] cm  (Ny={Ny} of {n_cells} cells)")

    def solve_qoi_batch(X_params, label):
        """Solve diffusion for each row of X_params, apply H, return (Ny, M) matrix."""
        M = len(X_params)
        Y_out = np.empty((Ny, M))
        print(f"  Solving {M} {label} snapshots")
        for i, g in enumerate(X_params):
            D_i, Siga_i, q_i = sampler.unpack(g)
            _, phi_i = solve_diffusion(
                L, n_cells, layer_bounds, D_i, Siga_i, q_i,
                bc_left=('dirichlet', 0.0),
                bc_right=('dirichlet', 0.0),
            )
            Y_out[:, i] = phi_i[qoi_mask]
        return Y_out

    Y_tr = solve_qoi_batch(X_tr, "training")   # (Ny, M_TRAIN)
    Y_te = solve_qoi_batch(X_te, "test")       # (Ny, M_TEST)

    print(f"Y_tr : {Y_tr.shape},  Y_te : {Y_te.shape}")

    # ------------------------------------------------------------------ #
    # Save samples and QoI values to text files                          #
    # ------------------------------------------------------------------ #
    param_header = "  ".join(sampler.param_names)
    x_header     = "  ".join([f"x={xi:.4f}" for xi in x_qoi])

    for tag, X_out, Y_out in [("train", X_tr, Y_tr), ("test", X_te, Y_te)]:
        np.savetxt(f"{output_dir}/samples_{tag}.txt",    X_out,   header=param_header, fmt="%.6e", comments="# ")
        np.savetxt(f"{output_dir}/qoi_values_{tag}.txt", Y_out.T, header=x_header,     fmt="%.6e", comments="# ")
        print(f"Saved: samples_{tag}.txt  ({X_out.shape[0]} x {X_out.shape[1]}),  "
              f"qoi_values_{tag}.txt  ({Y_out.shape[1]} x {Y_out.shape[0]})")

    # ------------------------------------------------------------------ #
    # Step 3 – Plot representative masked snapshots                      #
    # ------------------------------------------------------------------ #
    N_EXAMPLES = 8
    rng_plot   = np.random.default_rng(0)
    ex_idx     = rng_plot.choice(m_train, size=N_EXAMPLES, replace=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap('tab10').colors
    for k, idx in enumerate(ex_idx):
        ax.plot(x_qoi, Y_tr[:, idx], color=colors[k % 10], lw=1.4,
                label=f'Sample {idx}')

    # Mark QoI boundary
    ax.axvline(QOI_XMIN, color='k', ls='--', lw=1.2, label=f'QoI boundary')
    ax.axvline(QOI_XMAX, color='k', ls='--', lw=1.2)

    # Layer boundaries that fall inside the QoI
    for xb in layer_bounds[1:-1]:
        if QOI_XMIN <= xb <= QOI_XMAX:
            ax.axvline(xb, color='gray', ls=':', lw=0.8)

    ax.set_xlabel("x  [cm]", fontsize=12)
    ax.set_ylabel(r"$y(\mu) = H\phi(\mu)$", fontsize=12)
    ax.set_title(
        f"Representative QoI snapshots  (mask: [{QOI_XMIN:.2f}, {QOI_XMAX:.2f}] cm,  "
        f"Ny={Ny})",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qoi_snapshots.png", dpi=150)
    print(f"Saved: {output_dir}/qoi_snapshots.png")

    # ------------------------------------------------------------------ #
    # Step 4 – Thin SVD of the QoI snapshot matrix Y                     #
    # ------------------------------------------------------------------ #
    # Y = U Σ V^T
    # Ψ := U[:, :r]  is the rank-r POD basis
    print("\nComputing thin SVD of Y_tr  (training snapshots only)")
    U, sigma, Vh = np.linalg.svd(Y_tr, full_matrices=False)
    r_max = len(sigma)
    print(f"Singular values (first 10): {sigma[:10].round(4)}")

    # ------------------------------------------------------------------ #
    # Step 5 – Energy criterion and rank selection                       #
    # ------------------------------------------------------------------ #
    energy       = sigma ** 2
    total_energy = energy.sum()
    rel_energy   = energy / total_energy
    cum_energy   = np.cumsum(rel_energy)

    # Document specifies η = {0.999, 0.9999}
    r_999  = int(np.searchsorted(cum_energy, 0.999))  + 1
    r_9999 = int(np.searchsorted(cum_energy, 0.9999)) + 1
    print(f"\nModes for 99.9  % energy : {r_999}")
    print(f"Modes for 99.99 % energy : {r_9999}")

    # Chosen rank for subsequent steps
    R = int(np.searchsorted(cum_energy, energy_threshold)) + 1
    Psi = U[:, :R]
    print(f"Using r = {R}  (η = {energy_threshold})")

    # ------------------------------------------------------------------ #
    # Section 8 – Coefficient matrix                                     #
    # ------------------------------------------------------------------ #
    # Each column α^(i) = Ψ^T y^(i) is the reduced coordinate vector for training snapshot i.
    # This is a simple if Ψ is known.
    A_alpha = Psi.T @ Y_tr
    print(f"Coefficient matrix A_alpha: {A_alpha.shape}  (R x M_train)")

    # Save Ψ and A_alpha — inputs needed by the GPR surrogate (Section 9)
    np.savetxt(f"{output_dir}/pod_basis.txt",         Psi,       fmt="%.6e",
               header=f"POD basis Psi  shape ({Ny} x {R})  — columns are spatial modes", comments="# ")
    np.savetxt(f"{output_dir}/coefficients_train.txt", A_alpha.T, fmt="%.6e",
               header=f"Reduced coefficients A_alpha.T  shape ({m_train} x {R})  — rows are alpha^(i) = Psi^T y^(i)", comments="# ")
    print(f"Saved: {output_dir}/pod_basis.txt            ({Ny} x {R})")
    print(f"Saved: {output_dir}/coefficients_train.txt   ({m_train} x {R})")

    # ------------------------------------------------------------------ #
    # Step 6 – Singular value decay and cumulative energy plots          #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    modes_plot = min(40, r_max)

    ax = axes[0]
    ax.semilogy(np.arange(1, modes_plot + 1), sigma[:modes_plot], 'o-',
                ms=4, lw=1.5, color='steelblue')
    ax.set_xlabel("Mode index  $k$", fontsize=12)
    ax.set_ylabel(r"Singular value  $\sigma_k$", fontsize=12)
    ax.set_title("Singular value decay  (semi-log)", fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axes[1]
    ax.plot(np.arange(1, modes_plot + 1), cum_energy[:modes_plot] * 100,
            's-', ms=4, lw=1.5, color='tomato')
    ax.axhline(99.9,  color='gray',  ls='--', lw=1.2, label=r'$\eta = 99.9\,\%$')
    ax.axhline(99.99, color='black', ls=':',  lw=1.2, label=r'$\eta = 99.99\,\%$')
    ax.axvline(r_999,  color='gray',  ls='--', lw=1.2)
    ax.axvline(r_9999, color='black', ls=':',  lw=1.2)
    ax.set_xlabel("Number of modes  $r$", fontsize=12)
    ax.set_ylabel("Cumulative energy  [%]", fontsize=12)
    ax.set_title("Cumulative energy captured", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.5)

    fig.suptitle(
        f"SVD spectrum of QoI snapshot matrix  "
        f"(Ny={Ny}, M_train={m_train})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_spectrum.png", dpi=150)
    print(f"Saved: {output_dir}/svd_spectrum.png")

    # ------------------------------------------------------------------ #
    # Step 7 – Leading POD modes on the QoI sub-domain                   #
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
        ax.set_title(
            f"Mode {k+1}  ({frac:.2f} %,  $\\sigma_{k+1}={sigma[k]:.3f}$)",
            fontsize=10,
        )
        ax.grid(True, ls='--', alpha=0.4)
        for xb in layer_bounds[1:-1]:
            if QOI_XMIN <= xb <= QOI_XMAX:
                ax.axvline(xb, color='gray', ls=':', lw=0.7)

    fig.suptitle(
        f"Leading {N_MODES_PLOT} POD modes  Ψ  (QoI domain  "
        f"[{QOI_XMIN:.2f}, {QOI_XMAX:.2f}] cm)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_modes.png", dpi=150)
    print(f"Saved: {output_dir}/svd_modes.png")

    # ------------------------------------------------------------------ #
    # Step 8 – Reconstruction error vs rank                              #
    # ------------------------------------------------------------------ #
    # For rank r, the rank-r approximation of Y is  Yr = U_r Σ_r Vh_r
    # which equals  Ψ_r (Ψ_r^T Y).
    # Relative Frobenius-norm error:  ||Y - Yr||_F / ||Y||_F = sqrt(Σ_{k>r} σ_k^2) / ||σ||_2

    nY_tr = np.linalg.norm(Y_tr, 'fro')
    ranks       = np.arange(1, min(r_max + 1, 41))
    frob_errors = np.array([
        np.sqrt(energy[r:].sum()) / nY_tr for r in ranks
    ])

    RANK_LIST   = sorted(set([1, 2, 3, 5, 10, R]))
    SNAP_INDICES = [0, 1, 2]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- Top row: one subplot per example snapshot ---
    colors_r = plt.get_cmap('cool')(np.linspace(0, 1, len(RANK_LIST)))
    for col_idx, snap_idx in enumerate(SNAP_INDICES):
        ax = axes[0, col_idx] if col_idx < 2 else axes[1, 0]
        ax.plot(x_qoi, Y_tr[:, snap_idx], 'k-', lw=2, label='Full QoI snapshot')
        for r_val, col in zip(RANK_LIST, colors_r):
            Psi_r   = U[:, :r_val]
            y_recon = Psi_r @ (Psi_r.T @ Y_tr[:, snap_idx])
            rel_err = np.linalg.norm(y_recon - Y_tr[:, snap_idx]) / np.linalg.norm(Y_tr[:, snap_idx])
            ax.plot(x_qoi, y_recon, '--', lw=1.3, color=col,
                    label=f'r={r_val}  (err={rel_err:.1e})')
        ax.set_xlabel("x  [cm]", fontsize=11)
        ax.set_ylabel(r"$y(\mu)$", fontsize=11)
        ax.set_title(f"Training snapshot {snap_idx} reconstruction", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)

    # --- Bottom right: Frobenius error vs rank ---
    ax_r = axes[1, 1]
    ax_r.semilogy(ranks, frob_errors, 'o-', ms=4, lw=1.5, color='darkgreen')
    for r_val in [r_999, r_9999]:
        ax_r.axvline(r_val, ls='--', lw=1, color='gray',
                     label=f'r={r_val} (η={99.9 if r_val==r_999 else 99.99:.2f}%)')
    ax_r.set_xlabel("Rank  $r$", fontsize=11)
    ax_r.set_ylabel(r"$\|Y - Y_r\|_F \;/\; \|Y\|_F$", fontsize=11)
    ax_r.set_title("Relative Frobenius reconstruction error vs rank", fontsize=11)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, which='both', ls='--', alpha=0.5)

    fig.suptitle("POD reconstruction quality  (QoI snapshots)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_reconstruction.png", dpi=150)
    print(f"Saved: {output_dir}/svd_reconstruction.png")

    # ------------------------------------------------------------------ #
    # Section 8 – ε_proj on the held-out test set vs rank                #
    # ------------------------------------------------------------------ #
    # For each test snapshot y and rank r:
    #   ε_proj(µ) = ||y - Ψ_r Ψ_r^T y||_2 / ||y||_2
    RANK_TEST = sorted(set([1, 2, 3, 5, 10, R]))
    eps_by_rank = {}
    for r_val in RANK_TEST:
        Psi_r  = U[:, :r_val]                          # (Ny, r)
        proj   = Psi_r @ (Psi_r.T @ Y_te)              # (Ny, M_TEST)
        resid  = np.linalg.norm(Y_te - proj, axis=0)   # (M_TEST,)
        norms  = np.linalg.norm(Y_te, axis=0)          # (M_TEST,)
        eps_by_rank[r_val] = resid / norms

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        [eps_by_rank[r] for r in RANK_TEST],
        tick_labels=[str(r) for r in RANK_TEST],
        patch_artist=True,
        medianprops=dict(color='black', lw=2),
    )
    colors_box = plt.get_cmap('cool')(np.linspace(0, 1, len(RANK_TEST)))
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.set_xlabel("Rank  $r$", fontsize=12)
    ax.set_ylabel(r"$\varepsilon_{\mathrm{proj}}(\mu)$", fontsize=12)
    ax.set_title(
        f"Projection error on test set  (M_test={m_test})\n"
        r"$\varepsilon_{\mathrm{proj}} = \|y - \Psi_r\Psi_r^\top y\|_2 \;/\; \|y\|_2$",
        fontsize=12,
    )
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/svd_proj_error.png", dpi=150)
    print(f"Saved: {output_dir}/svd_proj_error.png")

    plt.show()

    # ------------------------------------------------------------------ #
    # Summary                                                            #
    # ------------------------------------------------------------------ #
    print("\n=== SVD summary ===")
    print(f"  QoI interval          : [{QOI_XMIN:.3f}, {QOI_XMAX:.3f}] cm")
    print(f"  QoI dimension Ny      : {Ny}")
    print(f"  Train snapshots M_tr  : {m_train}")
    print(f"  Test  snapshots M_te  : {m_test}")
    print(f"  Snapshot matrix Y_tr  : {Y_tr.shape}")
    print(f"  Coefficient matrix    : {A_alpha.shape}  (R x M_train)")
    print(f"  sigma_1               : {sigma[0]:.4f}")
    print(f"  sigma_1 / sigma_2     : {sigma[0]/sigma[1]:.3f}")
    print(f"  Energy in mode 1      : {rel_energy[0]*100:.2f} %")
    print(f"  Modes for η = 99.9  % : {r_999}")
    print(f"  Modes for η = 99.99 % : {r_9999}")
    print(f"  Chosen rank R         : {R}")
    print(f"  ||Y_tr-Y_R||_F/||Y_tr||_F : {frob_errors[R-1]:.2e}")
    print(f"\n  Test ε_proj (r={R}) — median : {np.median(eps_by_rank[R]):.2e},  "
          f"max : {np.max(eps_by_rank[R]):.2e}")


if __name__ == "__main__":
    main()
