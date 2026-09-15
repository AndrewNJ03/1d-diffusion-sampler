"""
ge_gpr_combined.py — GE-GPR vs. standard GPR, fully-coupled 5-parameter
study: mu = (void_thickness, steepness, D_teflon, Sigma_a_teflon, q_teflon)
varying simultaneously in one LHS design, one forward solve, one adjoint
solve per POD mode, one GE-GPR fit -- instead of the 5 one-at-a-time
studies in gpr_vs_gegpr_comparison/.

Runs the full pipeline end to end: Stage 1 (svd_analysis_combined.main())
builds the POD basis and writes its artifacts into output_dir, then this
module loads them, computes all 5 adjoint partials per training sample via
adjoint_combined.sensitivity_wrt_combined_param(), and fits GE-GPR/std-GPR
on top. Set run_svd_stage=False to reuse artifacts already in output_dir.

Reuses GEGaussianProcess, the sklearn kernel factory, and the generic
plotting helpers from sklearn_implementation/ge_gpr.py completely
unchanged -- that machinery only sees (X, y, gradients) and does not care
how many physical dimensions mu has or what they mean.
"""

import os
import sys
import time

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'void_teflon_geometry'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sklearn_implementation'))

from second_order_solver import build_mesh                             # noqa: E402
from adjoint_solver import build_adjoint_rhs, solve_adjoint            # noqa: E402
from params_combined import CombinedParamSampler                        # noqa: E402
from adjoint_combined import forward_system_combined, sensitivity_wrt_combined_param  # noqa: E402
import svd_analysis_combined                                            # noqa: E402
from ge_gpr import (                                                    # noqa: E402
    GEGaussianProcess, _make_sklearn_kernel, _default_lc_sizes,
    _load_pod_artifacts, _gradients_to_standardized,
    _plot_error_comparison, _plot_parity, _plot_flux_samples, _plot_learning_curve,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Adjoint gradients: all 5 partials from one adjoint solve per mode            #
# ──────────────────────────────────────────────────────────────────────────── #

def _compute_training_gradients(
    X_tr_subset: np.ndarray,
    Psi: np.ndarray,
    L: float,
    n_cells: int,
    core_radius: float,
    shell_width: float,
    mat_core, mat_void, mat_shield,
    bc_left: tuple,
    bc_right: tuple,
    qoi_mask: np.ndarray,
    param_names,
) -> np.ndarray:
    """
    G_full[i, k, j] = d(alpha_k(mu^i)) / d(param_names[j]), for all 5
    combined parameters, one forward + R adjoint solves per training
    sample (each adjoint solve yields all 5 partials at once).
    """
    M_ge, _ = X_tr_subset.shape
    Ny, R = Psi.shape
    G_full = np.zeros((M_ge, R, len(param_names)))

    for i in range(M_ge):
        mu = X_tr_subset[i]
        vt, steepness = float(mu[0]), float(mu[1])
        phi, A, rhs, dx, D_cell, xc, mat_teflon = forward_system_combined(
            mu, L, n_cells, core_radius, shell_width, mat_core, mat_void, mat_shield,
            bc_left, bc_right,
        )
        for k in range(R):
            g_k = build_adjoint_rhs(Psi[:, k], qoi_mask)
            lambda_k = solve_adjoint(A, g_k)
            for j, name in enumerate(param_names):
                total, _parts = sensitivity_wrt_combined_param(
                    name, lambda_k, phi, dx, D_cell, xc, L, core_radius,
                    vt, shell_width, mat_void, mat_teflon, steepness,
                    bc_left[0], bc_right[0],
                )
                G_full[i, k, j] = total

    return G_full


# ──────────────────────────────────────────────────────────────────────────── #
# Main entry point                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def main(
    L: float = 1.0,
    core_radius: float = 0.16921,
    shield_thickness: float = 0.07801,
    shell_width: float = 0.25278,
    mat_core=(1.0, 0.5, 1.0),
    mat_void=(1.8, 0.01, 0.0),
    mat_shield=(0.3, 0.9, 0.0),
    n_cells: int = 3000,
    void_bounds=(0.001, 0.25),
    steepness_bounds=(200.0, 200.0),
    d_teflon_bounds=(0.5, 2.0),
    sigma_a_teflon_bounds=(0.05, 1.5),
    q_teflon_bounds=(0.0, 2.0),
    qoi_xmin: float = None,
    qoi_xmax: float = None,
    output_dir: str = "output_graphs_combined",
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
    # Stage 1: SVD/POD
    run_svd_stage: bool = True,
    energy_threshold: float = 0.9999,
    m_train: int = 200,
    m_test: int = 60,
    seed: int = 1,
    # GE-GPR training budget
    m_ge: int = 80,
    n_restarts: int = 2,
    noise_f_init: float = 1e-3,
    noise_d_init: float = 1e-3,
    jitter: float = 1e-6,
    run_learning_curve: bool = True,
    learning_curve_sizes=None,
):
    os.makedirs(output_dir, exist_ok=True)

    QOI_XMIN = qoi_xmin if qoi_xmin is not None else core_radius
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else core_radius + shell_width

    print("=" * 60)
    print("GE-GPR vs Std-GPR — combined 5-parameter study")
    print("=" * 60)

    if run_svd_stage:
        print(f"\nStep 0: Running SVD/POD stage (energy_threshold={energy_threshold})")
        svd_analysis_combined.main(
            L=L, core_radius=core_radius, shield_thickness=shield_thickness, shell_width=shell_width,
            mat_core=mat_core, mat_void=mat_void, mat_shield=mat_shield, n_cells=n_cells,
            void_bounds=void_bounds, steepness_bounds=steepness_bounds,
            d_teflon_bounds=d_teflon_bounds, sigma_a_teflon_bounds=sigma_a_teflon_bounds,
            q_teflon_bounds=q_teflon_bounds,
            m_train=m_train, m_test=m_test, seed=seed,
            qoi_xmin=qoi_xmin, qoi_xmax=qoi_xmax, output_dir=output_dir,
            energy_threshold=energy_threshold, bc_left=bc_left, bc_right=bc_right,
        )

    print("\nStep 1: Loading SVD/POD artifacts")
    Psi, alpha_train, X_tr, X_te, Y_te = _load_pod_artifacts(output_dir)
    Ny, R = Psi.shape
    M_tr, p = X_tr.shape
    M_te = X_te.shape[0]

    sampler = CombinedParamSampler(
        void_bounds, steepness_bounds, d_teflon_bounds, sigma_a_teflon_bounds, q_teflon_bounds,
    )
    active_idx = np.where(~sampler._fixed_mask)[0]
    if len(active_idx) < 1:
        raise ValueError("At least one of the 5 combined parameters must be active.")
    p_act = len(active_idx)
    m_ge = min(m_ge, M_tr)

    print(f"  POD basis  Psi : {Ny} x {R}")
    print(f"  Train samples  : {M_tr}  (using {m_ge} for GE-GPR)")
    print(f"  Test samples   : {M_te},  p = {p},  active = {sampler.active_names}")

    print("\nStep 2: Normalising inputs and standardising outputs")
    bounds = sampler.bounds
    X_lo = bounds[:, 0]
    X_rng = bounds[:, 1] - bounds[:, 0]
    X_rng[X_rng == 0] = 1.0

    X_tr_norm = (X_tr - X_lo) / X_rng
    X_te_norm = (X_te - X_lo) / X_rng

    alpha_mu = alpha_train.mean(axis=0)
    alpha_sig = alpha_train.std(axis=0)
    alpha_sig[alpha_sig == 0] = 1.0
    alpha_z = (alpha_train - alpha_mu) / alpha_sig

    X_ge_norm = X_tr_norm[:m_ge]
    alpha_ge_z = alpha_z[:m_ge]

    _, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi = np.linspace(QOI_XMIN, QOI_XMAX, qoi_mask.sum())

    print(f"\nStep 3: Computing adjoint gradients for {m_ge} training samples "
          f"({R} mode{'s' if R > 1 else ''} each, {p_act} active of 5 parameters) ...")
    t0 = time.perf_counter()
    G_full = _compute_training_gradients(
        X_tr[:m_ge], Psi, L, n_cells, core_radius, shell_width,
        mat_core, mat_void, mat_shield, bc_left, bc_right, qoi_mask, sampler.param_names,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    G_z_norm = _gradients_to_standardized(G_full, X_rng, alpha_sig)
    G_act_z_norm = G_z_norm[:, :, active_idx]

    print(f"\nStep 4: Fitting GE-GPR and standard GPR "
          f"(m_ge={m_ge}, R={R}, p_act={p_act}, restarts={n_restarts})")

    ge_models, std_models = [], []
    for k in range(R):
        print(f"\n  Mode {k + 1}/{R}")

        t_ge = time.perf_counter()
        ge_gp = GEGaussianProcess(
            active_dims=np.arange(p_act), n_restarts=n_restarts,
            noise_f_init=noise_f_init, noise_d_init=noise_d_init, jitter=jitter,
        )
        ge_gp.fit(X_ge_norm[:, active_idx], alpha_ge_z[:, k], G_act_z_norm[:, k, :])
        ge_models.append(ge_gp)
        dt_ge = time.perf_counter() - t_ge

        t_std = time.perf_counter()
        std_gpr = GaussianProcessRegressor(
            kernel=_make_sklearn_kernel(p_act), n_restarts_optimizer=n_restarts, normalize_y=False,
        )
        std_gpr.fit(X_ge_norm[:, active_idx], alpha_ge_z[:, k])
        std_models.append(std_gpr)
        dt_std = time.perf_counter() - t_std

        print(f"    GE-GPR  log-MLL = {ge_gp.lml_:8.3f}   ({dt_ge:.1f}s)")
        print(f"    Std-GPR log-MLL = {std_gpr.log_marginal_likelihood_value_:8.3f}   ({dt_std:.1f}s)")

    print("\nStep 5: Predicting on test set")
    X_te_act = X_te_norm[:, active_idx]
    alpha_ge_z_pred = np.zeros((M_te, R))
    alpha_std_z_pred = np.zeros((M_te, R))
    for k in range(R):
        alpha_ge_z_pred[:, k] = ge_models[k].predict(X_te_act)
        alpha_std_z_pred[:, k] = std_models[k].predict(X_te_act)

    alpha_ge_pred = alpha_ge_z_pred * alpha_sig + alpha_mu
    alpha_std_pred = alpha_std_z_pred * alpha_sig + alpha_mu
    alpha_te_true = (Psi.T @ Y_te).T

    Y_ge_pred = Psi @ alpha_ge_pred.T
    Y_std_pred = Psi @ alpha_std_pred.T

    norms_te = np.linalg.norm(Y_te, axis=0)
    eps_ge = np.linalg.norm(Y_te - Y_ge_pred, axis=0) / norms_te
    eps_std = np.linalg.norm(Y_te - Y_std_pred, axis=0) / norms_te
    eps_proj = np.linalg.norm(Y_te - Psi @ (Psi.T @ Y_te), axis=0) / norms_te

    print(f"\n{'=' * 60}")
    print(f"Test reconstruction errors  (m_ge={m_ge}, M_te={M_te}, R={R}, p_act={p_act})")
    print(f"  eps_y  GE-GPR      :  median={np.median(eps_ge):.2e}  max={np.max(eps_ge):.2e}")
    print(f"  eps_y  Std GPR     :  median={np.median(eps_std):.2e}  max={np.max(eps_std):.2e}")
    print(f"  eps_proj (POD lb)  :  median={np.median(eps_proj):.2e}  max={np.max(eps_proj):.2e}")
    print(f"{'=' * 60}")

    lc_results = {}
    if run_learning_curve and m_ge >= 10:
        lc_sizes = learning_curve_sizes or _default_lc_sizes(m_ge)
        lc_sizes = [s for s in lc_sizes if s <= m_ge]
        if lc_sizes:
            print(f"\nStep 6: Learning curve  (sizes: {lc_sizes})")
            lc_ge_med, lc_ge_p75, lc_std_med, lc_std_p75 = [], [], [], []
            for m_lc in lc_sizes:
                X_lc = X_ge_norm[:m_lc, active_idx]

                ge_preds_z = np.zeros((M_te, R))
                for k in range(R):
                    gp = GEGaussianProcess(
                        active_dims=np.arange(p_act), n_restarts=n_restarts,
                        noise_f_init=noise_f_init, noise_d_init=noise_d_init, jitter=jitter,
                    )
                    gp.fit(X_lc, alpha_ge_z[:m_lc, k], G_act_z_norm[:m_lc, k, :])
                    ge_preds_z[:, k] = gp.predict(X_te_act)
                Y_ge_lc = Psi @ ((ge_preds_z * alpha_sig + alpha_mu).T)
                eps_lc_ge = np.linalg.norm(Y_te - Y_ge_lc, axis=0) / norms_te

                std_preds_z = np.zeros((M_te, R))
                for k in range(R):
                    gpr = GaussianProcessRegressor(
                        kernel=_make_sklearn_kernel(p_act), n_restarts_optimizer=n_restarts, normalize_y=False,
                    )
                    gpr.fit(X_lc, alpha_ge_z[:m_lc, k])
                    std_preds_z[:, k] = gpr.predict(X_te_act)
                Y_std_lc = Psi @ ((std_preds_z * alpha_sig + alpha_mu).T)
                eps_lc_std = np.linalg.norm(Y_te - Y_std_lc, axis=0) / norms_te

                lc_ge_med.append(np.median(eps_lc_ge))
                lc_ge_p75.append(np.percentile(eps_lc_ge, 75))
                lc_std_med.append(np.median(eps_lc_std))
                lc_std_p75.append(np.percentile(eps_lc_std, 75))
                print(f"  M={m_lc:4d}:  GE-GPR eps_y={lc_ge_med[-1]:.2e}   Std-GPR eps_y={lc_std_med[-1]:.2e}")

            lc_results = dict(sizes=lc_sizes, ge_med=lc_ge_med, ge_p75=lc_ge_p75,
                               std_med=lc_std_med, std_p75=lc_std_p75)

    print("\nStep 7: Generating plots")
    _plot_error_comparison(eps_ge, eps_std, eps_proj, m_ge, R, output_dir)
    _plot_parity(alpha_te_true, alpha_ge_pred, alpha_std_pred, R, m_ge, output_dir)
    _plot_flux_samples(Y_te, Y_ge_pred, Y_std_pred, eps_ge, eps_std, x_qoi, output_dir)
    if lc_results:
        _plot_learning_curve(lc_results, output_dir)

    print(f"\nGE-GPR (combined) comparison complete.  All plots saved to {output_dir}/")

    return dict(
        eps_ge=eps_ge, eps_std=eps_std, eps_proj=eps_proj,
        alpha_te_true=alpha_te_true, alpha_ge_pred=alpha_ge_pred, alpha_std_pred=alpha_std_pred,
        lc_results=lc_results, p_act=p_act, active_names=sampler.active_names,
    )


if __name__ == "__main__":
    main()
