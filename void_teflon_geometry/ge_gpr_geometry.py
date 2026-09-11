"""
ge_gpr_geometry.py — Gradient-Enhanced GPR vs. standard GPR, void_teflon geometry.

Compares GE-GPR (trained on POD coefficients *and* their adjoint gradients)
against standard GPR (function values only) as surrogates alpha_k(a, b) for
the two active switch/geometry parameters of geometry.sigmoid_switch:

    a = steepness                       (sigmoid steepness)
    b = core_radius + void_thickness    (Void/Teflon interface center)

params.GeometryParamSampler samples mu = (void_thickness, steepness), not
(b, a) directly, but b is an affine shift of void_thickness
(b = void_thickness + core_radius, a fixed constant), so this module makes
no physics changes: it reuses void_thickness/steepness sampling, forward
solves, and adjoint gradients exactly as adjoint_geometry.py already
provides them, and only relabels void_thickness -> b for presentation
(d(alpha_k)/d(void_thickness) == d(alpha_k)/db and d(alpha_k)/d(steepness)
== d(alpha_k)/da, unchanged by the shift).

Runs the full pipeline end to end: Stage 1 (svd_analysis_geometry.main())
builds the POD basis at the requested `energy_threshold` and writes its
artifacts (pod_basis.txt, samples_train.txt, coefficients_train.txt,
samples_test.txt, qoi_values_test.txt) into output_dir, then Stage 2 (this
module) loads them and fits GE-GPR/std-GPR on top. Set run_svd_stage=False
to skip Stage 1 and reuse artifacts already sitting in output_dir (e.g. when
iterating on GE-GPR fitting only). Either way, BOTH void_thickness and
steepness must be active (real, non-degenerate bounds for each) -- e.g.
steepness_bounds=(100.0, 400.0) -- otherwise there is only one parameter to
compare surrogates over.

Reuses the generic GEGaussianProcess class, sklearn kernel factory, and
plotting helpers from sklearn_implementation/ge_gpr.py unchanged: that
machinery only sees (X, y, gradients) and does not care what the input
dimensions physically represent.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sklearn_implementation'))

from second_order_solver import build_mesh                            # noqa: E402
from adjoint_solver import build_adjoint_rhs, solve_adjoint            # noqa: E402
from params import GeometryParamSampler                                # noqa: E402
from adjoint_geometry import _forward_system, sensitivity_wrt_param    # noqa: E402
from geometry import material_boundaries                               # noqa: E402
import svd_analysis_geometry                                           # noqa: E402
from ge_gpr import (                                                   # noqa: E402
    GEGaussianProcess, _make_sklearn_kernel, _default_lc_sizes,
    _load_pod_artifacts, _gradients_to_standardized,
    _plot_error_comparison, _plot_parity, _plot_flux_samples, _plot_learning_curve,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Adjoint gradients d(alpha_k)/d(void_thickness), d(alpha_k)/d(steepness)      #
# ──────────────────────────────────────────────────────────────────────────── #

def _compute_training_gradients(
    X_tr_subset: np.ndarray,
    Psi: np.ndarray,
    L: float,
    n_cells: int,
    core_radius: float,
    shell_width: float,
    mat_core, mat_void, mat_teflon, mat_shield,
    bc_left: tuple,
    bc_right: tuple,
    qoi_mask: np.ndarray,
    param_names,
) -> np.ndarray:
    """
    G_full[i, k, j] = d(alpha_k(mu^i)) / d(param_names[j]), computed via the
    adjoint (adjoint_geometry.sensitivity_wrt_param), one forward + R adjoint
    solves per training sample.
    """
    M_ge, _ = X_tr_subset.shape
    Ny, R = Psi.shape
    G_full = np.zeros((M_ge, R, len(param_names)))

    for i in range(M_ge):
        mu = X_tr_subset[i]
        vt, steepness = float(mu[0]), float(mu[1])
        phi, A, rhs, dx, D_cell, xc = _forward_system(
            mu, L, n_cells, core_radius, shell_width,
            mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
        )
        for k in range(R):
            g_k = build_adjoint_rhs(Psi[:, k], qoi_mask)
            lambda_k = solve_adjoint(A, g_k)
            for j, name in enumerate(param_names):
                total, _parts = sensitivity_wrt_param(
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
    # Physical / mesh (must match the SVD stage that produced the artifacts)
    L: float = 1.0,
    core_radius: float = 0.16921,
    shield_thickness: float = 0.07801,
    shell_width: float = 0.25278,
    mat_core=(1.0, 0.5, 1.0),
    mat_void=(1.8, 0.01, 0.0),
    mat_teflon=(0.6, 0.4, 0.0),
    mat_shield=(0.3, 0.9, 0.0),
    n_cells: int = 3500,
    void_bounds=(0.001, 0.25),
    steepness_bounds=(100.0, 400.0),
    qoi_xmin: float = None,
    qoi_xmax: float = None,
    output_dir: str = "output_graphs_geometry_2param",
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
    # Stage 1: SVD/POD (svd_analysis_geometry.main()) -- set run_svd_stage=
    # False to skip and reuse artifacts already in output_dir instead.
    run_svd_stage: bool = True,
    energy_threshold: float = 0.99999,
    m_train: int = 300,
    m_test: int = 60,
    seed: int = 1,
    # GE-GPR training budget
    m_ge: int = 100,
    # Hyperparameter optimisation
    n_restarts: int = 3,
    noise_f_init: float = 1e-3,
    noise_d_init: float = 1e-3,
    jitter: float = 1e-6,
    # Learning-curve comparison
    run_learning_curve: bool = True,
    learning_curve_sizes=None,
):
    """
    GE-GPR vs. standard-GPR surrogate comparison over (a, b) for the
    void_teflon geometry, mirroring sklearn_implementation/ge_gpr.py's
    Stage-4 structure but sourced from adjoint_geometry.py's gradients.
    """
    os.makedirs(output_dir, exist_ok=True)

    QOI_XMIN = qoi_xmin if qoi_xmin is not None else core_radius
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else core_radius + shell_width

    print("=" * 60)
    print("GE-GPR vs Std-GPR — void_teflon geometry  (a=steepness, b=interface center)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 0. Stage 1: SVD/POD basis at the requested energy_threshold        #
    # ------------------------------------------------------------------ #
    if run_svd_stage:
        print(f"\nStep 0: Running SVD/POD stage (energy_threshold={energy_threshold})")
        svd_analysis_geometry.main(
            L=L, core_radius=core_radius, shield_thickness=shield_thickness,
            shell_width=shell_width,
            mat_core=mat_core, mat_void=mat_void, mat_teflon=mat_teflon, mat_shield=mat_shield,
            n_cells=n_cells, void_bounds=void_bounds, steepness_bounds=steepness_bounds,
            m_train=m_train, m_test=m_test, seed=seed,
            qoi_xmin=qoi_xmin, qoi_xmax=qoi_xmax, output_dir=output_dir,
            energy_threshold=energy_threshold, bc_left=bc_left, bc_right=bc_right,
        )

    # ------------------------------------------------------------------ #
    # 1. Load SVD/POD artifacts                                          #
    # ------------------------------------------------------------------ #
    print("\nStep 1: Loading SVD/POD artifacts")
    Psi, alpha_train, X_tr, X_te, Y_te = _load_pod_artifacts(output_dir)

    Ny, R   = Psi.shape
    M_tr, p = X_tr.shape
    M_te    = X_te.shape[0]

    sampler = GeometryParamSampler(void_bounds, steepness_bounds)
    active_idx = np.where(~sampler._fixed_mask)[0]
    if len(active_idx) < 1:
        raise ValueError(
            "At least one of void_thickness (b) or steepness (a) must be "
            "active (non-degenerate bounds) to compare surrogates -- both "
            f"are fixed. Active parameters: {sampler.active_names}."
        )
    p_act = len(active_idx)
    m_ge  = min(m_ge, M_tr)

    print(f"  POD basis  Psi : {Ny} x {R}")
    print(f"  Train samples  : {M_tr}  (using {m_ge} for GE-GPR)")
    print(f"  Test samples   : {M_te},  p = {p},  active = {sampler.active_names}")

    # ------------------------------------------------------------------ #
    # 2. Normalise inputs, standardise outputs                           #
    # ------------------------------------------------------------------ #
    print("\nStep 2: Normalising inputs and standardising outputs")

    bounds = sampler.bounds            # (p, 2), in (void_thickness, steepness) space
    X_lo   = bounds[:, 0]
    X_rng  = bounds[:, 1] - bounds[:, 0]
    X_rng[X_rng == 0] = 1.0

    X_tr_norm = (X_tr - X_lo) / X_rng  # (M_tr, p) in [0,1]; unaffected by the b-shift
    X_te_norm = (X_te - X_lo) / X_rng

    alpha_mu  = alpha_train.mean(axis=0)
    alpha_sig = alpha_train.std(axis=0)
    alpha_sig[alpha_sig == 0] = 1.0
    alpha_z   = (alpha_train - alpha_mu) / alpha_sig

    X_ge_norm  = X_tr_norm[:m_ge]
    alpha_ge_z = alpha_z[:m_ge]

    _, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)
    x_qoi    = np.linspace(QOI_XMIN, QOI_XMAX, qoi_mask.sum())

    # ------------------------------------------------------------------ #
    # 3. Adjoint gradients d(alpha_k)/d(void_thickness) == d(alpha_k)/db #
    #    and d(alpha_k)/d(steepness) == d(alpha_k)/da                    #
    # ------------------------------------------------------------------ #
    print(f"\nStep 3: Computing adjoint gradients d(alpha_k)/da, d(alpha_k)/db "
          f"for {m_ge} training samples ({R} mode{'s' if R > 1 else ''} each) ...")
    t0 = time.perf_counter()

    G_full = _compute_training_gradients(
        X_tr[:m_ge], Psi, L, n_cells, core_radius, shell_width,
        mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
        qoi_mask, sampler.param_names,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Chain rule to standardised/normalised space (b's constant shift by
    # core_radius has zero derivative, so this is identical whether the
    # second input dimension is called void_thickness or b).
    G_z_norm = _gradients_to_standardized(G_full, X_rng, alpha_sig)
    G_act_z_norm = G_z_norm[:, :, active_idx]

    # ------------------------------------------------------------------ #
    # 4. Fit GE-GP and standard GP (per POD mode) on the same subset     #
    # ------------------------------------------------------------------ #
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
        ge_gp.fit(
            X_ge_norm[:, active_idx], alpha_ge_z[:, k], G_act_z_norm[:, k, :],
        )
        ge_models.append(ge_gp)
        dt_ge = time.perf_counter() - t_ge

        t_std = time.perf_counter()
        std_gpr = GaussianProcessRegressor(
            kernel=_make_sklearn_kernel(p_act),
            n_restarts_optimizer=n_restarts, normalize_y=False,
        )
        std_gpr.fit(X_ge_norm[:, active_idx], alpha_ge_z[:, k])
        std_models.append(std_gpr)
        dt_std = time.perf_counter() - t_std

        print(f"    GE-GPR  log-MLL = {ge_gp.lml_:8.3f}   ({dt_ge:.1f}s)")
        print(f"    Std-GPR log-MLL = {std_gpr.log_marginal_likelihood_value_:8.3f}"
              f"   ({dt_std:.1f}s)")

    # ------------------------------------------------------------------ #
    # 5. Predict on test set                                             #
    # ------------------------------------------------------------------ #
    print("\nStep 5: Predicting on test set")

    X_te_act = X_te_norm[:, active_idx]
    alpha_ge_z_pred  = np.zeros((M_te, R))
    alpha_std_z_pred = np.zeros((M_te, R))

    for k in range(R):
        alpha_ge_z_pred[:, k]  = ge_models[k].predict(X_te_act)
        alpha_std_z_pred[:, k] = std_models[k].predict(X_te_act)

    alpha_ge_pred  = alpha_ge_z_pred  * alpha_sig + alpha_mu
    alpha_std_pred = alpha_std_z_pred * alpha_sig + alpha_mu
    alpha_te_true  = (Psi.T @ Y_te).T

    Y_ge_pred  = Psi @ alpha_ge_pred.T
    Y_std_pred = Psi @ alpha_std_pred.T

    norms_te = np.linalg.norm(Y_te, axis=0)
    eps_ge   = np.linalg.norm(Y_te - Y_ge_pred,  axis=0) / norms_te
    eps_std  = np.linalg.norm(Y_te - Y_std_pred, axis=0) / norms_te
    eps_proj = np.linalg.norm(Y_te - Psi @ (Psi.T @ Y_te), axis=0) / norms_te

    print(f"\n{'=' * 60}")
    print(f"Test reconstruction errors  (m_ge={m_ge}, M_te={M_te}, R={R})")
    print(f"  eps_y  GE-GPR      :  median={np.median(eps_ge):.2e}  max={np.max(eps_ge):.2e}")
    print(f"  eps_y  Std GPR     :  median={np.median(eps_std):.2e}  max={np.max(eps_std):.2e}")
    print(f"  eps_proj (POD lb)  :  median={np.median(eps_proj):.2e}  max={np.max(eps_proj):.2e}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------ #
    # 6. Learning-curve comparison                                        #
    # ------------------------------------------------------------------ #
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
                        kernel=_make_sklearn_kernel(p_act),
                        n_restarts_optimizer=n_restarts, normalize_y=False,
                    )
                    gpr.fit(X_lc, alpha_ge_z[:m_lc, k])
                    std_preds_z[:, k] = gpr.predict(X_te_act)

                Y_std_lc = Psi @ ((std_preds_z * alpha_sig + alpha_mu).T)
                eps_lc_std = np.linalg.norm(Y_te - Y_std_lc, axis=0) / norms_te

                lc_ge_med.append(np.median(eps_lc_ge))
                lc_ge_p75.append(np.percentile(eps_lc_ge, 75))
                lc_std_med.append(np.median(eps_lc_std))
                lc_std_p75.append(np.percentile(eps_lc_std, 75))

                print(f"  M={m_lc:4d}:  GE-GPR eps_y={lc_ge_med[-1]:.2e}"
                      f"   Std-GPR eps_y={lc_std_med[-1]:.2e}")

            lc_results = dict(
                sizes=lc_sizes,
                ge_med=lc_ge_med, ge_p75=lc_ge_p75,
                std_med=lc_std_med, std_p75=lc_std_p75,
            )

    # ------------------------------------------------------------------ #
    # 7. Plots (generic ones reused unchanged from sklearn_implementation) #
    # ------------------------------------------------------------------ #
    print("\nStep 7: Generating plots")
    vt_col = sampler.param_names.index("void_thickness")
    a_col  = sampler.param_names.index("steepness")

    # Each plotted test sample has its own material boundaries (the moving
    # Void/Teflon interface plus any fixed boundaries); pass through all of
    # them that actually fall inside the QoI window.
    n_flux_plot = min(4, M_te)
    switch_x = []
    for i in range(n_flux_plot):
        boundaries = material_boundaries(L, core_radius, shell_width, void_thickness=X_te[i, vt_col])
        in_window = boundaries[(boundaries >= QOI_XMIN) & (boundaries <= QOI_XMAX)]
        switch_x.append(in_window)

    _plot_error_comparison(eps_ge, eps_std, eps_proj, m_ge, R, output_dir)
    _plot_parity(alpha_te_true, alpha_ge_pred, alpha_std_pred, R, m_ge, output_dir)
    _plot_flux_samples(Y_te, Y_ge_pred, Y_std_pred, eps_ge, eps_std, x_qoi, output_dir, switch_x=switch_x)
    if lc_results:
        _plot_learning_curve(lc_results, output_dir)

    b_te = core_radius + X_te[:, vt_col]
    a_te = X_te[:, a_col]
    _plot_error_map_ab(b_te, a_te, eps_ge, eps_std, m_ge, output_dir)

    print(f"\nGE-GPR (geometry) comparison complete.  All plots saved to {output_dir}/")

    return dict(
        eps_ge=eps_ge, eps_std=eps_std, eps_proj=eps_proj,
        alpha_te_true=alpha_te_true,
        alpha_ge_pred=alpha_ge_pred, alpha_std_pred=alpha_std_pred,
        lc_results=lc_results,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Plotting helper specific to this module: error over the (a, b) plane         #
# ──────────────────────────────────────────────────────────────────────────── #

def _plot_error_map_ab(b_te, a_te, eps_ge, eps_std, m_ge, output_dir):
    vmin = max(min(eps_ge.min(), eps_std.min()), 1e-8)
    vmax = max(eps_ge.max(), eps_std.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, eps, label, cmap in zip(
        axes, [eps_ge, eps_std], ['GE-GPR', 'Std GPR'], ['viridis', 'inferno'],
    ):
        sc = ax.scatter(b_te, a_te, c=eps, cmap=cmap, s=45,
                         norm=LogNorm(vmin=vmin, vmax=vmax),
                         edgecolors='k', linewidths=0.3)
        ax.set_xlabel(r'$b$ = interface center  [cm]', fontsize=11)
        ax.set_ylabel(r'$a$ = steepness', fontsize=11)
        ax.set_title(f'{label} test-set error', fontsize=12)
        ax.grid(True, ls='--', alpha=0.3)
        fig.colorbar(sc, ax=ax, label=r'$\varepsilon_y$')

    fig.suptitle(f'Test reconstruction error over (a, b)  (m_ge={m_ge})', fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/gegpr_error_map_ab.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
