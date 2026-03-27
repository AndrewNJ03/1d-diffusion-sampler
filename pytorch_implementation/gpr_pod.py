"""
Week 4 — Gaussian Process Regression on POD expansion coefficients

Pipeline
--------
Inputs:
    output_graphs/pod_basis.txt           — POD basis Ψ  (Ny × R)
    output_graphs/coefficients_train.txt  — expansion coefficients α  (M_train × R)
    output_graphs/samples_train.txt       — training parameters  X_train  (M_train × p)
    output_graphs/samples_test.txt        — test parameters  X_test  (M_test × p)
    output_graphs/qoi_values_test.txt     — true test QoI vectors  Y_test  (M_test × Ny)
"""

import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from second_order_solver import build_mesh   # for reconstructing x_qoi


def main(
    n_epochs=3000,
    lr=0.05,
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
    print("GPR on POD expansion coefficients")
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

    # QoI spatial coordinates for plotting
    x_centers, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi    = x_centers[qoi_mask]

    # ------------------------------------------------------------------ #
    # 2. Normalise inputs, standardise outputs                           #
    # ------------------------------------------------------------------ #
    print("\nStep 2: Normalising inputs and standardising outputs")

    # Build (p, 2) bounds array matching lhs_generation.DiffusionParamSampler order:
    #   [D_1...D_N | Σa_1...Σa_N | q_1...q_N]
    # Accepts either a scalar (lo, hi) tuple or a per-layer (n_bins, 2) array.
    n = n_bins

    def _expand(b):
        b = np.asarray(b, dtype=float)
        if b.shape == (2,):
            b = np.tile(b, (n, 1))
        return b

    bounds = np.vstack([_expand(d_bounds), _expand(sigma_a_bounds), _expand(q_bounds)])  # (p, 2)

    X_lo  = bounds[:, 0]
    X_rng = bounds[:, 1] - bounds[:, 0]
    X_rng[X_rng == 0] = 1.0              # fixed dims: numerator is also 0 → result stays 0

    X_tr_norm = (X_tr - X_lo) / X_rng    # (M_tr, p)  ∈ [0, 1]
    X_te_norm = (X_te - X_lo) / X_rng    # (M_te, p)  ∈ [0, 1]

    # Standardise each coefficient dimension independently
    alpha_mu  = alpha_train.mean(axis=0)                            # (R,)
    alpha_sig = alpha_train.std(axis=0)                             # (R,)
    alpha_sig[alpha_sig == 0] = 1.0                                 # guard degenerate modes
    alpha_z   = (alpha_train - alpha_mu) / alpha_sig                # (M_tr, R)  ~ N(0,1)

    # Torch tensors (float64 throughout — GPs benefit from double precision)
    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float64)
    X_te_t = torch.tensor(X_te_norm, dtype=torch.float64)

    # ------------------------------------------------------------------ #
    # 3. GPyTorch ExactGP definition                                     #
    # ------------------------------------------------------------------ #
    class ExactGPModel(gpytorch.models.ExactGP):
        """
        Exact GP with:
          - constant mean module
          - output-scaled ARD RBF kernel (one length-scale per input dimension)

        ARD (Automatic Relevance Determination) allows the model to learn which
        of the p=15 parameters actually influence each expansion coefficient.
        """

        def __init__(self, X_train, y_train, likelihood):
            super().__init__(X_train, y_train, likelihood)
            self.mean_module  = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=p))

        def forward(self, x):
            mean  = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    # ------------------------------------------------------------------ #
    # 4. Train one GP per expansion coefficient                          #
    # ------------------------------------------------------------------ #
    print(f"\nStep 3: Training {R} GP(s)  x  {n_epochs} epochs each\n")

    gp_models      = []
    gp_likelihoods = []
    train_losses   = []

    for k in range(R):
        y_k = torch.tensor(alpha_z[:, k], dtype=torch.float64)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        model      = ExactGPModel(X_tr_t, y_k, likelihood).double()

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        losses_k = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = -mll(model(X_tr_t), y_k)
            loss.backward()
            optimizer.step()
            losses_k.append(loss.item())

        gp_models.append(model)
        gp_likelihoods.append(likelihood)
        train_losses.append(losses_k)

        print(f"  Mode {k+1:2d}/{R}:  final −MLL = {losses_k[-1]:8.4f}"
              f"   noise σ^2 = {likelihood.noise.item():.2e}"
              f"   output scale = {model.covar_module.outputscale.item():.3f}")

    # ------------------------------------------------------------------ #
    # 5. Predict and un-standardise                                      #
    # ------------------------------------------------------------------ #
    print("\nStep 4: Predicting on test set")

    alpha_pred_z     = np.zeros((M_te, R))   # predictive mean
    alpha_pred_z_std = np.zeros((M_te, R))   # predictive std

    for k, (model, likelihood) in enumerate(zip(gp_models, gp_likelihoods)):
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(X_te_t))
        alpha_pred_z[:, k]     = pred.mean.numpy()
        alpha_pred_z_std[:, k] = pred.variance.sqrt().numpy()

    # Un-standardise: α^= μ_α + σ_α · z~
    alpha_pred     = alpha_pred_z     * alpha_sig + alpha_mu    # (M_te, R)
    alpha_pred_std = alpha_pred_z_std * alpha_sig               # (M_te, R)

    # True test coefficients for parity plots
    alpha_te_true = (Psi.T @ Y_te).T                            # (M_te, R)

    # ------------------------------------------------------------------ #
    # 6. Reconstruct flux and propagate uncertainty                      #
    # ------------------------------------------------------------------ #
    Y_pred = Psi @ alpha_pred.T    # (Ny, M_te)  reconstructed QoI

    # Pointwise flux variance via independent-GP propagation:
    #   Var[φ_i] = Σ_k  Ψ_{ik}^2  Var[α_k]
    Y_pred_std = np.sqrt((Psi ** 2) @ (alpha_pred_std ** 2).T)  # (Ny, M_te)

    # Relative errors (document Section 8–9 notation)
    # ε_y   : field reconstruction error from GPR  (eq. in Sec. 9)
    # ε_proj: POD projection error — lower bound   (eq. in Sec. 8)
    # ε_α   : coefficient prediction error         (eq. in Sec. 9)
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

    # ---------- (a) Training convergence -------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.get_cmap('viridis')
    for k in range(R):
        ax.semilogy(
            train_losses[k], lw=1,
            color=cmap(k / max(R - 1, 1)),
            alpha=0.85,
            label=f'mode {k+1}' if R <= 8 else None,
        )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("$-$ log-marginal likelihood", fontsize=12)
    ax.set_title(f"GP training convergence  (R = {R} modes)", fontsize=13)
    if R <= 8:
        ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_training_loss.png", dpi=150)
    print(f"\nSaved: {output_dir}/gpr_training_loss.png")

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
    fig.suptitle("GP: Predicted vs True expansion coefficients  (test set)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_coefficient_parity.png", dpi=150)
    print(f"Saved: {output_dir}/gpr_coefficient_parity.png")

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
    plt.savefig(f"{output_dir}/gpr_error_comparison.png", dpi=150)
    print(f"Saved: {output_dir}/gpr_error_comparison.png")

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
        r"GPR flux reconstruction on QoI window  $[L/3,\;2L/3]$",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gpr_flux_reconstruction.png", dpi=150)
    print(f"Saved: {output_dir}/gpr_flux_reconstruction.png")

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
    print(f"  GP epochs               : {n_epochs}")
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
