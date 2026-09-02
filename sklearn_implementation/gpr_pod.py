"""
Scikit-learn mirror of pytorch_implementation/gpr_pod.py

Gaussian Process Regression on POD expansion coefficients using
sklearn.gaussian_process.GaussianProcessRegressor.

Kernel equivalence to the GPyTorch implementation
--------------------------------------------------
GPyTorch:  ScaleKernel(RBFKernel(ard_num_dims=p)) + GaussianLikelihood
sklearn:   ConstantKernel() * RBF(length_scale=ones(p)) + WhiteKernel()
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh  


def main(
    n_restarts=5,
    n_bins=5,
    d_bounds=(0.2, 2.0),
    sigma_a_bounds=(0.05, 1.0),
    q_bounds=(0.0, 2.0),
    L=10.0,
    n_cells=200,
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs",
):
    QOI_XMIN = qoi_xmin if qoi_xmin is not None else L / 3
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else 2.0 * L / 3

    # ------------------------------------------------------------------ #
    # 1. Load SVD/POD outputs                                            #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("GPR (sklearn) on POD expansion coefficients")
    print("=" * 60)
    print("\nStep 1: Loading SVD/POD outputs")

    Psi         = np.loadtxt(f"{output_dir}/pod_basis.txt",          comments='#')  # (Ny, R)
    alpha_train = np.loadtxt(f"{output_dir}/coefficients_train.txt", comments='#')  # (M_tr, R)
    X_tr        = np.loadtxt(f"{output_dir}/samples_train.txt",      comments='#')  # (M_tr, p)
    X_te        = np.loadtxt(f"{output_dir}/samples_test.txt",       comments='#')  # (M_te, p)
    Y_te_T      = np.loadtxt(f"{output_dir}/qoi_values_test.txt",    comments='#')  # (M_te, Ny)
    Y_te        = Y_te_T.T                                                           # (Ny, M_te)

    Ny, R    = Psi.shape
    M_tr, p  = X_tr.shape
    M_te     = X_te.shape[0]

    print(f"  POD basis    Ψ  : {Ny} × {R}  (Ny spatial, R modes)")
    print(f"  Coefficients α  : {M_tr} × {R}  training samples")
    print(f"  Test set        : {M_te} samples,  p = {p} parameters")

    x_centers, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi    = x_centers[qoi_mask]

    # ------------------------------------------------------------------ #
    # 2. Normalise inputs, standardise outputs                           #
    # ------------------------------------------------------------------ #
    print("\nStep 2: Normalising inputs and standardising outputs")

    # Standardise inputs (zero mean, unit variance), fit on training data only
    x_scaler  = StandardScaler()
    X_tr_norm = x_scaler.fit_transform(X_tr)   # (M_tr, p)
    X_te_norm = x_scaler.transform(X_te)       # (M_te, p)

    # Standardise each coefficient dimension independently
    alpha_mu  = alpha_train.mean(axis=0)                            # (R,)
    alpha_sig = alpha_train.std(axis=0)                             # (R,)
    alpha_sig[alpha_sig == 0] = 1.0                                 # guard degenerate modes
    alpha_z   = (alpha_train - alpha_mu) / alpha_sig                # (M_tr, R)  ~ N(0,1)

    # ------------------------------------------------------------------ #
    # 3. Kernel definition                                               #
    # ------------------------------------------------------------------ #
    # Mirrors GPyTorch:  ScaleKernel(RBFKernel(ard_num_dims=p)) + GaussianLikelihood
    #
    # ConstantKernel  ≡  output scale (σ_f^2)
    # RBF             ≡  ARD RBF with one length-scale per input dimension
    # WhiteKernel     ≡  Gaussian observation noise (σ_n^2)
    def _make_kernel(p):
        return (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=np.ones(p), length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )

    # ------------------------------------------------------------------ #
    # 4. Train one GP per expansion coefficient                          #
    # ------------------------------------------------------------------ #
    print(f"\nStep 3: Training {R} GP(s)  (n_restarts_optimizer={n_restarts})\n")

    gp_models  = []
    lml_values = []          # log-marginal likelihood after fitting (sklearn maximises this)

    for k in range(R):
        y_k = alpha_z[:, k]

        gpr = GaussianProcessRegressor(
            kernel=_make_kernel(p),
            n_restarts_optimizer=n_restarts,
            normalize_y=False,    # we already standardised
            copy_X_train=True,
        )
        gpr.fit(X_tr_norm, y_k)
        gp_models.append(gpr)
        lml_values.append(gpr.log_marginal_likelihood_value_)

        kernel     = gpr.kernel_
        # Extract learned hyperparameters for display
        noise_var  = kernel.k2.noise_level              # WhiteKernel noise_level = σ_n^2
        out_scale  = kernel.k1.k1.constant_value        # ConstantKernel = σ_f^2

        print(f"  Mode {k+1:2d}/{R}:  log-MLL = {lml_values[-1]:8.3f}"
              f"   noise σ^2 = {noise_var:.2e}"
              f"   output scale = {out_scale:.3f}")

    # ------------------------------------------------------------------ #
    # 5. Predict and un-standardise                                      #
    # ------------------------------------------------------------------ #
    print("\nStep 4: Predicting on test set")

    alpha_pred_z     = np.zeros((M_te, R))   # predictive mean
    alpha_pred_z_std = np.zeros((M_te, R))   # predictive std

    for k, gpr in enumerate(gp_models):
        mean_k, std_k            = gpr.predict(X_te_norm, return_std=True)
        alpha_pred_z[:, k]     = mean_k
        alpha_pred_z_std[:, k] = std_k

    # Un-standardise: α^ = μ_α + σ_α · z~
    alpha_pred     = alpha_pred_z     * alpha_sig + alpha_mu    # (M_te, R)
    alpha_pred_std = alpha_pred_z_std * alpha_sig               # (M_te, R)

    # True test coefficients for parity plots
    alpha_te_true = (Psi.T @ Y_te).T                            # (M_te, R)

    # ------------------------------------------------------------------ #
    # 6. Reconstruct flux and propagate uncertainty                      #
    # ------------------------------------------------------------------ #
    Y_pred = Psi @ alpha_pred.T    # (Ny, M_te)  reconstructed QoI

    # Pointwise flux variance via independent-GP propagation
    Y_pred_std = np.sqrt((Psi ** 2) @ (alpha_pred_std ** 2).T)  # (Ny, M_te)

    # Relative errors
    eps_y    = (np.linalg.norm(Y_te - Y_pred,               axis=0) /
                np.linalg.norm(Y_te,                         axis=0))
    eps_proj = (np.linalg.norm(Y_te - Psi @ (Psi.T @ Y_te), axis=0) /
                np.linalg.norm(Y_te,                         axis=0))
    eps_alpha = (np.linalg.norm(alpha_te_true - alpha_pred,  axis=1) /
                 np.linalg.norm(alpha_te_true,               axis=1))

    print(f"\n{'='*55}")
    print(f"Test reconstruction errors  (M_test={M_te}, R={R})")
    print(f"  ε_y   (GPR field):  median = {np.median(eps_y):.2e},  max = {np.max(eps_y):.2e}")
    print(f"  ε_proj (POD proj):  median = {np.median(eps_proj):.2e},  max = {np.max(eps_proj):.2e}"
          f"  <- lower bound")
    print(f"  ε_α   (coeff err):  median = {np.median(eps_alpha):.2e},  max = {np.max(eps_alpha):.2e}")
    print(f"{'='*55}")

    # ------------------------------------------------------------------ #
    # 7. Diagnostic plots                                                #
    # ------------------------------------------------------------------ #

    # ---------- (a) Log-marginal likelihood per mode ------------------- #
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, R + 1), lml_values, color='steelblue', alpha=0.8)
    ax.set_xlabel("POD mode", fontsize=12)
    ax.set_ylabel("Log-marginal likelihood", fontsize=12)
    ax.set_title(f"GP optimised log-MLL per mode  (R = {R})", fontsize=13)
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_sklearn_lml.png", dpi=150)
    print(f"\nSaved: {output_dir}/gpr_sklearn_lml.png")

    # ---------- (b) Parity plots for leading modes ---------------------- #
    n_show = min(6, R)
    n_cols = 3
    n_rows = (n_show + 2) // 3
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4.2 * n_rows),
                              squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= n_show:
            ax.set_visible(False)
            continue
        true    = alpha_te_true[:, k]
        pred    = alpha_pred[:, k]
        sigma_k = alpha_pred_std[:, k]
        lo = min(true.min(), (pred - 2 * sigma_k).min())
        hi = max(true.max(), (pred + 2 * sigma_k).max())
        ax.errorbar(true, pred, yerr=2 * sigma_k, fmt='o', ms=4, lw=0.8,
                    alpha=0.65, color='steelblue', ecolor='lightsteelblue',
                    label=r'pred $\pm 2\sigma$')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='ideal (y = x)')
        ax.set_xlabel(fr'True $\alpha_{k+1}$', fontsize=11)
        ax.set_ylabel(fr'Predicted $\alpha_{k+1}$', fontsize=11)
        ax.set_title(f'Mode {k+1}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)
    fig.suptitle("GP (sklearn): Predicted vs True expansion coefficients  (test set)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_sklearn_coefficient_parity.png", dpi=150)
    print(f"Saved: {output_dir}/gpr_sklearn_coefficient_parity.png")

    # ---------- (c) Error distribution ---------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(eps_proj, bins=20, alpha=0.65, color='tomato',
            label=r'$\varepsilon_{\mathrm{proj}}$ (POD lower bound)')
    ax.hist(eps_y,    bins=20, alpha=0.65, color='steelblue',
            label=r'$\varepsilon_y$ (GPR field)')
    ax.set_xlabel(r"Relative error $\varepsilon$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Test reconstruction error distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)

    ax = axes[1]
    ax.semilogy(np.sort(eps_proj), 's--', ms=4, lw=1.2, color='tomato',
                label=r'$\varepsilon_{\mathrm{proj}}$ (POD lower bound)')
    ax.semilogy(np.sort(eps_y),   'o-',  ms=4, lw=1.2, color='steelblue',
                label=r'$\varepsilon_y$ (GPR field)')
    ax.set_xlabel("Test sample (sorted by error)", fontsize=12)
    ax.set_ylabel(r"Relative error $\varepsilon$", fontsize=12)
    ax.set_title("Sorted reconstruction error", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    fig.suptitle(r"$\varepsilon_y$ vs $\varepsilon_{\mathrm{proj}}$"
                 f"  (M_train={M_tr}, R={R})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_sklearn_error_comparison.png", dpi=150)
    print(f"Saved: {output_dir}/gpr_sklearn_error_comparison.png")

    # ---------- (d) Sample flux reconstructions with ±2σ band ---------- #
    n_plot   = min(4, M_te)
    idx_plot = np.arange(n_plot)

    fig, axes = plt.subplots(1, n_plot, figsize=(4.5 * n_plot, 4.2), sharey=False)
    if n_plot == 1:
        axes = [axes]

    for ax, idx in zip(axes, idx_plot):
        ax.fill_between(
            x_qoi,
            Y_pred[:, idx] - 2 * Y_pred_std[:, idx],
            Y_pred[:, idx] + 2 * Y_pred_std[:, idx],
            alpha=0.25, color='steelblue', label=r'$\pm 2\sigma$ GP band',
        )
        ax.plot(x_qoi, Y_te[:, idx],   'k-',  lw=2.0, label=r'True $y(\mu)$')
        ax.plot(x_qoi, Y_pred[:, idx], 'b--', lw=1.5, label=r'GPR $\hat{y}(\mu)$')
        ax.set_xlabel("x  [cm]", fontsize=11)
        ax.set_ylabel(r"$\phi(x;\,\mu)$", fontsize=11)
        ax.set_title(
            f"Test sample {idx}\n"
            fr"$\varepsilon_y={eps_y[idx]:.1e}$"
            fr"   $\varepsilon_{{\mathrm{{proj}}}}={eps_proj[idx]:.1e}$",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle(
        r"GPR (sklearn) flux reconstruction on QoI window  $[L/3,\;2L/3]$",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_sklearn_flux_reconstruction.png", dpi=150)
    print(f"Saved: {output_dir}/gpr_sklearn_flux_reconstruction.png")

    plt.show()

    # ------------------------------------------------------------------ #
    # Summary table                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*55}")
    print("Summary")
    print(f"{'='*55}")
    print(f"  POD rank R              : {R}")
    print(f"  Training samples M_tr   : {M_tr}")
    print(f"  Test samples M_te       : {M_te}")
    print(f"  GP input dimension p    : {p}")
    print(f"  Optimizer restarts      : {n_restarts}")
    print(f"\n  Median ε_y     : {np.median(eps_y):.2e}")
    print(f"  Mean   ε_y     : {np.mean(eps_y):.2e}")
    print(f"  Max    ε_y     : {np.max(eps_y):.2e}")
    print(f"  Median ε_proj  : {np.median(eps_proj):.2e}")
    print(f"  Mean   ε_proj  : {np.mean(eps_proj):.2e}")
    print(f"  Max    ε_proj  : {np.max(eps_proj):.2e}")
    print(f"  Median ε_alpha : {np.median(eps_alpha):.2e}")
    print(f"  Mean   ε_alpha : {np.mean(eps_alpha):.2e}")
    print(f"  Max    ε_alpha : {np.max(eps_alpha):.2e}")


    return dict(
        eps_y=eps_y, eps_proj=eps_proj, eps_alpha=eps_alpha,
        alpha_te_true=alpha_te_true, alpha_pred=alpha_pred, alpha_pred_std=alpha_pred_std,
        Y_te=Y_te, Y_pred=Y_pred, Y_pred_std=Y_pred_std,
        x_qoi=x_qoi, R=R, M_tr=M_tr, M_te=M_te, p=p,
    )


if __name__ == "__main__":
    main()
