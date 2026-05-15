"""
ge_gpr.py — Gradient-Enhanced Gaussian Process Regression (GE-GPR)

Section 16 of the workflow document.

GE-GPR augments the training data with adjoint-derived gradients ∂α_k/∂µ,
building a joint GP model over function values *and* derivatives.

Kernel (ARD-RBF, log-parameterised):
    k(x, x') = σ_f² exp(-½ Σ_j r_j²/ℓ_j²),   r_j = x_j - x'_j

Derivative kernel identities used to build the augmented covariance matrix:

    Cov(f(x),   ∂f/∂x'_m)     = k(x,x') · r_m / ℓ_m²
    Cov(∂f/∂x_m, ∂f/∂x'_n)   = k(x,x') · [δ_{mn}/ℓ_m² − r_m r_n / (ℓ_m² ℓ_n²)]

Augmented observation vector for one POD mode α_k (M training points, p_act
active gradient dimensions):

    y_aug = [α_k(µ¹), …, α_k(µᴹ),
             ∂α_k/∂x_0(µ¹), …, ∂α_k/∂x_{p-1}(µ¹),
             ∂α_k/∂x_0(µ²), …,
             …,
             ∂α_k/∂x_{p-1}(µᴹ)]

All inputs are normalised to [0,1]; outputs are standardised.
Gradients are chain-rule–corrected to the standardised/normalised spaces.

Comparison with standard sklearn GPR (same training subset) produces the
learning-curve plot requested by the document.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh, assign_material_properties, assemble_system
from lhs_generation import DiffusionParamSampler
from adjoint_solver import (
    build_adjoint_rhs, solve_adjoint, compute_full_gradient, get_layer_assignment,
)


# ──────────────────────────────────────────────────────────────────────────── #
# GE-GP core class                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class GEGaussianProcess:
    """
    Gradient-Enhanced Gaussian Process with ARD-RBF kernel.

    The augmented training set consists of M function values plus M*p_act
    derivative observations, giving M*(1+p_act) training targets in total.

    Hyperparameters (all in log space):
        θ = [log σ_f, log ℓ_0, …, log ℓ_{p_act-1}, log σ_nf, log σ_nd]

    where σ_nf = noise on function values, σ_nd = noise on derivative values.

    Parameters
    ----------
    active_dims : array-like of int
        Column indices of the (normalised) input X to use as gradient dimensions.
        Should correspond to the non-fixed parameters.
    n_restarts : int
        Number of random restarts for L-BFGS-B hyperparameter optimisation.
    noise_f_init : float
        Initial (and lower-bound) noise level on function observations.
    noise_d_init : float
        Initial (and lower-bound) noise level on derivative observations.
    """

    def __init__(
        self,
        active_dims: np.ndarray,
        n_restarts: int = 3,
        noise_f_init: float = 1e-3,
        noise_d_init: float = 1e-3,
        jitter: float = 1e-6,
    ):
        self.active_dims = np.asarray(active_dims, dtype=int)
        self.p_act = int(len(self.active_dims))
        self.n_restarts = n_restarts
        self.noise_f_init = noise_f_init
        self.noise_d_init = noise_d_init
        self.jitter = jitter

        # Populated by fit()
        self.theta_ = None
        self.X_train_ = None
        self.y_aug_ = None
        self.L_ = None
        self.alpha_ = None
        self.lml_ = None

    # ------------------------------------------------------------------ #
    # Internal: augmented kernel matrix                                  #
    # ------------------------------------------------------------------ #

    def _parse_theta(self, theta):
        p = self.p_act
        sf2  = np.exp(2.0 * theta[0])
        ells2 = np.exp(2.0 * theta[1:1 + p])
        nf2  = np.exp(2.0 * theta[1 + p])
        nd2  = np.exp(2.0 * theta[2 + p])
        return sf2, ells2, nf2, nd2

    def _build_aug_K(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Build the (M*(1+p_act)) × (M*(1+p_act)) augmented kernel matrix.

        Layout (row / column blocks):
            [ K_ff + σ_nf² I   |  K_fd           ]
            [ K_fd^T           |  K_dd + σ_nd² I ]

        K_ff[i,j]            = k(xⁱ, xʲ)
        K_fd[i, j*p+m]       = Cov(f(xⁱ), ∂f/∂x'_m(xʲ))
                              = k(xⁱ,xʲ) · (xⁱ_m − xʲ_m) / ℓ_m²
        K_dd[i*p+m, j*p+n]  = Cov(∂f/∂x_m(xⁱ), ∂f/∂x'_n(xʲ))
                              = k(xⁱ,xʲ) · [δ_{mn}/ℓ_m² − r_m r_n/(ℓ_m² ℓ_n²)]
        """
        sf2, ells2, nf2, nd2 = self._parse_theta(theta)
        p = self.p_act
        M = X.shape[0]
        X_a = X[:, self.active_dims]            # (M, p_act)

        # Pairwise differences: R[i,j,m] = x^i_m − x^j_m
        R = X_a[:, None, :] - X_a[None, :, :]  # (M, M, p_act)
        D2 = np.sum(R ** 2 / ells2, axis=2)    # (M, M)
        Kff = sf2 * np.exp(-0.5 * D2)          # (M, M)

        # K_fd: (M, M*p_act)
        Kfd = (Kff[:, :, None] * R) / ells2    # (M, M, p_act)
        Kfd = Kfd.reshape(M, M * p)

        # K_dd: (M*p_act, M*p_act)
        # δ_{mn}/ℓ_m² — outer product of R/ℓ²
        delta = np.eye(p)[None, None, :, :] / ells2[None, None, :, None]  # (1,1,p,p)
        RmRn  = (R[:, :, :, None] * R[:, :, None, :]) / (ells2[:, None] * ells2[None, :])
        Kdd_blk = Kff[:, :, None, None] * (delta - RmRn)  # (M, M, p, p)
        Kdd = Kdd_blk.transpose(0, 2, 1, 3).reshape(M * p, M * p)

        n_aug = M * (1 + p)
        K = np.empty((n_aug, n_aug))
        K[:M, :M] = Kff + nf2 * np.eye(M)
        K[:M, M:] = Kfd
        K[M:, :M] = Kfd.T
        K[M:, M:] = Kdd + nd2 * np.eye(M * p)
        K += self.jitter * np.eye(n_aug)
        return K

    # ------------------------------------------------------------------ #
    # Internal: negative log-marginal likelihood                         #
    # ------------------------------------------------------------------ #

    def _neg_lml(self, theta: np.ndarray, X: np.ndarray, y_aug: np.ndarray) -> float:
        try:
            K = self._build_aug_K(X, theta)
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_aug))
        lml = (
            -0.5 * float(y_aug @ alpha)
            - np.sum(np.log(np.diag(L)))
            - 0.5 * len(y_aug) * np.log(2.0 * np.pi)
        )
        return -lml

    # ------------------------------------------------------------------ #
    # Public API: fit / predict                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        G_active: np.ndarray,
    ) -> "GEGaussianProcess":
        """
        Fit the GE-GP.

        Parameters
        ----------
        X : (M, p)       normalised inputs ∈ [0,1]^p
        y : (M,)         standardised function values
        G_active : (M, p_act)  standardised gradients ∂z/∂x_m for active dims,
                               i.e. chain-rule–corrected to the z/x_norm space.
        """
        p = self.p_act
        y_aug = np.concatenate([y, G_active.ravel(order='C')])
        # G_active.ravel('C') → [G[0,0],…,G[0,p-1], G[1,0],…,G[M-1,p-1]]
        # y_aug[M + i*p + m] = G_active[i, m]  matches the K_aug block layout.

        self.X_train_ = X.copy()
        self.y_aug_   = y_aug

        theta_dim = 3 + p          # log_sf  +  log_ells (p)  +  log_nf  +  log_nd
        bounds = (
            [(-5.0, 5.0)] * (1 + p)   # sf, ells
            + [(-10.0, 0.0), (-10.0, 0.0)]  # nf, nd  (keep noise ≤ 1)
        )

        theta0 = np.zeros(theta_dim)
        theta0[1 + p] = np.log(self.noise_f_init)
        theta0[2 + p] = np.log(self.noise_d_init)

        best_theta, best_nlml = theta0.copy(), np.inf
        rng = np.random.default_rng(0)

        for restart in range(self.n_restarts + 1):
            if restart == 0:
                theta_i = theta0.copy()
            else:
                theta_i = np.concatenate([
                    rng.uniform(-2.0, 2.0, size=1 + p),
                    rng.uniform(-8.0, -2.0, size=2),
                ])

            result = minimize(
                self._neg_lml, theta_i,
                args=(X, y_aug),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6},
            )
            if result.fun < best_nlml:
                best_nlml = result.fun
                best_theta = result.x.copy()

        self.theta_ = best_theta
        K = self._build_aug_K(X, best_theta)
        self.L_     = np.linalg.cholesky(K)
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y_aug))
        self.lml_   = -best_nlml
        return self

    def predict(
        self,
        X_test: np.ndarray,
        return_std: bool = False,
    ):
        """
        Predict function values f(x*) at test points.

        Returns mean (and optionally std) in the *standardised* output space.
        """
        sf2, ells2, _, _ = self._parse_theta(self.theta_)
        p = self.p_act
        M = self.X_train_.shape[0]
        M_te = X_test.shape[0]

        X_tr_a = self.X_train_[:, self.active_dims]
        X_te_a = X_test[:, self.active_dims]

        R_star = X_te_a[:, None, :] - X_tr_a[None, :, :]     # (M_te, M, p)
        D2_star = np.sum(R_star ** 2 / ells2, axis=2)         # (M_te, M)
        kff_s = sf2 * np.exp(-0.5 * D2_star)                  # (M_te, M)

        # Cov(f(x*), ∂f/∂x'_m(xʲ)) = kff_s[te,j] · (x*_m − xʲ_m) / ℓ_m²
        kfd_s = (kff_s[:, :, None] * R_star) / ells2          # (M_te, M, p)
        kfd_s = kfd_s.reshape(M_te, M * p)                    # (M_te, M*p)

        k_star = np.concatenate([kff_s, kfd_s], axis=1)       # (M_te, M*(1+p))
        mean = k_star @ self.alpha_                            # (M_te,)

        if not return_std:
            return mean

        v = np.linalg.solve(self.L_, k_star.T)                # (n_aug, M_te)
        var = np.maximum(sf2 - np.sum(v ** 2, axis=0), 0.0)
        return mean, np.sqrt(var)


# ──────────────────────────────────────────────────────────────────────────── #
# Gradient computation helper                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _compute_training_gradients(
    X_tr_subset: np.ndarray,
    Psi: np.ndarray,
    sampler: DiffusionParamSampler,
    L: float,
    n_cells: int,
    layer_bounds: np.ndarray,
    bc_left: tuple,
    bc_right: tuple,
    qoi_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute adjoint gradients ∂α_k/∂µ_j for a subset of training samples.

    Parameters
    ----------
    X_tr_subset : (M_ge, p)
    Psi         : (Ny, R)

    Returns
    -------
    G_full : (M_ge, R, p)   G_full[i, k, j] = ∂α_k(µ^i)/∂µ_j
    """
    M_ge, p = X_tr_subset.shape
    Ny, R   = Psi.shape
    n_layers = len(layer_bounds) - 1
    G_full   = np.zeros((M_ge, R, p))

    for i in range(M_ge):
        mu = X_tr_subset[i]
        D_l, Siga_l, q_l = sampler.unpack(mu)
        x_centers, _, dx = build_mesh(L, n_cells)
        D_cell, Siga_cell, q_cell = assign_material_properties(
            x_centers, layer_bounds, D_l, Siga_l, q_l,
        )
        A, rhs = assemble_system(dx, D_cell, Siga_cell, q_cell, bc_left, bc_right)
        phi    = spsolve(A, rhs)
        layer_asgn = get_layer_assignment(x_centers, layer_bounds)

        for k in range(R):
            g_k      = build_adjoint_rhs(Psi[:, k], qoi_mask)
            lambda_k = solve_adjoint(A, g_k)
            G_full[i, k, :] = compute_full_gradient(
                lambda_k, phi, rhs, dx, D_cell, layer_asgn, n_layers,
                bc_left[0], bc_right[0],
            )

    return G_full


# ──────────────────────────────────────────────────────────────────────────── #
# Standard-GPR helper (sklearn, on a specified subset)                         #
# ──────────────────────────────────────────────────────────────────────────── #

def _make_sklearn_kernel(p_active: int):
    return (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=np.ones(p_active), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )


def _fit_eval_std_gpr(
    X_tr_norm: np.ndarray,
    alpha_z: np.ndarray,
    X_te_norm: np.ndarray,
    alpha_te_true: np.ndarray,
    Y_te: np.ndarray,
    Psi: np.ndarray,
    active_idx: np.ndarray,
    n_restarts: int,
) -> dict:
    """Fit standard sklearn GPR on X_tr_norm[:, active_idx] and evaluate."""
    M = X_tr_norm.shape[0]
    R = Psi.shape[1]
    X_tr_act = X_tr_norm[:, active_idx]
    X_te_act = X_te_norm[:, active_idx]
    p_act = X_tr_act.shape[1]

    gp_models = []
    for k in range(R):
        gpr = GaussianProcessRegressor(
            kernel=_make_sklearn_kernel(p_act),
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
        )
        gpr.fit(X_tr_act, alpha_z[:, k])
        gp_models.append(gpr)

    M_te = X_te_norm.shape[0]
    alpha_pred_z = np.column_stack([gp.predict(X_te_act) for gp in gp_models])

    return alpha_pred_z, gp_models


# ──────────────────────────────────────────────────────────────────────────── #
# Per-layer bounds helper (mirrors run_pipeline.py)                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _per_layer_bounds(scalar_bounds, n_layers, fixed_mats, key):
    b = np.asarray(scalar_bounds, dtype=float)
    if b.ndim == 1:
        bounds = [[b[0], b[1]] for _ in range(n_layers)]
    else:
        bounds = b.tolist()
    if fixed_mats:
        for mat_idx, vals in fixed_mats.items():
            if key in vals:
                v = vals[key]
                bounds[mat_idx - 1] = [v, v]
    return bounds


# ──────────────────────────────────────────────────────────────────────────── #
# Main entry point                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def main(
    # Physical / mesh (must match the SVD stage that produced the artifacts)
    L: float = 10.0,
    n_bins: int = 3,
    n_cells: int = 200,
    d_bounds=(0.5, 2.0),
    sigma_a_bounds=(0.05, 1.5),
    q_bounds=(1.0, 1.0),
    qoi_xmin: float = 3.0,
    qoi_xmax: float = 7.0,
    output_dir: str = "output_graphs_sklearn",
    fixed_mats: dict = None,
    # GE-GPR training budget
    m_ge: int = 100,           # training samples to use for GE-GPR (≤ M_train)
    # Hyperparameter optimisation
    n_restarts: int = 3,
    noise_f_init: float = 1e-3,
    noise_d_init: float = 1e-3,
    jitter: float = 1e-6,
    # Learning-curve comparison
    run_learning_curve: bool = True,
    learning_curve_sizes=None,  # e.g. [10, 20, 40, 60, 80, 100]
):
    """
    Stage 4: Gradient-Enhanced GPR surrogate for the POD coefficients.

    Loads SVD/POD artifacts written by Stage 1 (svd_analysis), then:
      1. Computes adjoint gradients for m_ge training samples.
      2. Trains one GE-GP per POD mode.
      3. Evaluates on the held-out test set.
      4. Compares with standard sklearn GPR on the same training subset.
      5. Optionally produces a learning-curve plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    bc_left  = ('dirichlet', 0.0)
    bc_right = ('dirichlet', 0.0)
    layer_bounds = np.linspace(0.0, L, n_bins + 1)

    print("=" * 60)
    print("GE-GPR — Gradient-Enhanced Gaussian Process Regression")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load SVD/POD artifacts                                          #
    # ------------------------------------------------------------------ #
    print("\nStep 1: Loading SVD/POD artifacts")
    Psi         = np.loadtxt(f"{output_dir}/pod_basis.txt",          comments='#')
    alpha_train = np.loadtxt(f"{output_dir}/coefficients_train.txt", comments='#')
    X_tr        = np.loadtxt(f"{output_dir}/samples_train.txt",      comments='#')
    X_te        = np.loadtxt(f"{output_dir}/samples_test.txt",       comments='#')
    Y_te_T      = np.loadtxt(f"{output_dir}/qoi_values_test.txt",    comments='#')
    Y_te        = Y_te_T.T          # (Ny, M_te)

    if Psi.ndim == 1:
        Psi = Psi[:, None]
    if alpha_train.ndim == 1:
        alpha_train = alpha_train[:, None]

    Ny, R   = Psi.shape
    M_tr, p = X_tr.shape
    M_te    = X_te.shape[0]
    m_ge    = min(m_ge, M_tr)

    print(f"  POD basis  Ψ  : {Ny} × {R}")
    print(f"  Train samples : {M_tr}  (using {m_ge} for GE-GPR)")
    print(f"  Test samples  : {M_te},  p = {p}")

    # ------------------------------------------------------------------ #
    # 2. Rebuild sampler, normalise inputs                               #
    # ------------------------------------------------------------------ #
    print("\nStep 2: Building sampler and normalising inputs")

    d_b  = _per_layer_bounds(d_bounds,       n_bins, fixed_mats, "D")
    sa_b = _per_layer_bounds(sigma_a_bounds, n_bins, fixed_mats, "sigma_a")
    q_b  = _per_layer_bounds(q_bounds,       n_bins, fixed_mats, "q")
    sampler     = DiffusionParamSampler(n_bins, d_b, sa_b, q_b)
    active_idx  = np.where(~sampler._fixed_mask)[0]   # non-fixed param indices
    p_act       = len(active_idx)

    bounds  = sampler.bounds            # (p, 2)
    X_lo    = bounds[:, 0]
    X_rng   = bounds[:, 1] - bounds[:, 0]
    X_rng[X_rng == 0] = 1.0            # guard fixed dims

    X_tr_norm = (X_tr - X_lo) / X_rng  # (M_tr, p) ∈ [0,1]
    X_te_norm = (X_te - X_lo) / X_rng  # (M_te, p)

    # Standardise POD coefficients using *full* training set statistics
    alpha_mu  = alpha_train.mean(axis=0)   # (R,)
    alpha_sig = alpha_train.std(axis=0)    # (R,)
    alpha_sig[alpha_sig == 0] = 1.0
    alpha_z   = (alpha_train - alpha_mu) / alpha_sig   # (M_tr, R)

    # Subset for GE-GPR training
    X_ge_norm = X_tr_norm[:m_ge]          # (m_ge, p)
    alpha_ge_z = alpha_z[:m_ge]           # (m_ge, R)

    # QoI mask (must match svd_analysis)
    _, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < qoi_xmax) & (x_faces[1:] > qoi_xmin)
    x_qoi    = np.linspace(qoi_xmin, qoi_xmax, qoi_mask.sum())

    print(f"  Active (non-fixed) parameters: {p_act}/{p}")
    print(f"  Active indices: {active_idx.tolist()}")
    print(f"  Active names  : {[sampler.param_names[j] for j in active_idx]}")

    # ------------------------------------------------------------------ #
    # 3. Compute adjoint gradients for the GE-GPR training subset        #
    # ------------------------------------------------------------------ #
    print(f"\nStep 3: Computing adjoint gradients for {m_ge} training samples "
          f"({R} mode{'s' if R>1 else ''} each) …")
    t0 = time.perf_counter()

    G_full = _compute_training_gradients(
        X_tr[:m_ge], Psi, sampler, L, n_cells, layer_bounds,
        bc_left, bc_right, qoi_mask,
    )
    # G_full : (m_ge, R, p)   — ∂α_k(µ^i)/∂µ_j  in original param space

    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ------------------------------------------------------------------ #
    # 4. Chain-rule: convert gradients to standardised/normalised spaces #
    #                                                                     #
    #   z_k = (α_k − μ_k) / σ_k      x_j = (µ_j − lo_j) / rng_j       #
    #   ∂z_k/∂x_j = (∂α_k/∂µ_j) · rng_j / σ_k                          #
    # ------------------------------------------------------------------ #
    # G_z_norm[i, k, j] = G_full[i, k, j] * X_rng[j] / alpha_sig[k]
    G_z_norm = G_full * X_rng[None, None, :] / alpha_sig[None, :, None]
    # Shape: (m_ge, R, p);  restrict to active dims for GE-GPR
    G_act_z_norm = G_z_norm[:, :, active_idx]   # (m_ge, R, p_act)

    # ------------------------------------------------------------------ #
    # 5. Fit GE-GP and standard GP (per POD mode) on the same subset     #
    # ------------------------------------------------------------------ #
    print(f"\nStep 4: Fitting GE-GPR and standard GPR "
          f"(m_ge={m_ge}, R={R}, p_act={p_act}, restarts={n_restarts})")

    ge_models  = []
    std_models = []

    for k in range(R):
        print(f"\n  Mode {k+1}/{R}")

        # --- GE-GPR ---
        t_ge = time.perf_counter()
        ge_gp = GEGaussianProcess(
            active_dims=np.arange(p_act),   # indices into the p_act-dim active subspace
            n_restarts=n_restarts,
            noise_f_init=noise_f_init,
            noise_d_init=noise_d_init,
            jitter=jitter,
        )
        # Pass only the active columns of X_ge_norm (active dims only)
        ge_gp.fit(
            X_ge_norm[:, active_idx],      # (m_ge, p_act)
            alpha_ge_z[:, k],              # (m_ge,)
            G_act_z_norm[:, k, :],         # (m_ge, p_act)
        )
        ge_models.append(ge_gp)
        dt_ge = time.perf_counter() - t_ge

        # --- Standard sklearn GPR (same subset, active dims only) ---
        t_std = time.perf_counter()
        std_gpr = GaussianProcessRegressor(
            kernel=_make_sklearn_kernel(p_act),
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
        )
        std_gpr.fit(X_ge_norm[:, active_idx], alpha_ge_z[:, k])
        std_models.append(std_gpr)
        dt_std = time.perf_counter() - t_std

        print(f"    GE-GPR  log-MLL = {ge_gp.lml_:8.3f}   ({dt_ge:.1f}s)")
        print(f"    Std-GPR log-MLL = {std_gpr.log_marginal_likelihood_value_:8.3f}"
              f"   ({dt_std:.1f}s)")

    # ------------------------------------------------------------------ #
    # 6. Predict on test set (GE-GPR and standard GPR)                   #
    # ------------------------------------------------------------------ #
    print("\nStep 5: Predicting on test set")

    X_te_act = X_te_norm[:, active_idx]   # (M_te, p_act)

    alpha_ge_z_pred   = np.zeros((M_te, R))
    alpha_ge_z_std    = np.zeros((M_te, R))
    alpha_std_z_pred  = np.zeros((M_te, R))

    for k in range(R):
        alpha_ge_z_pred[:, k], alpha_ge_z_std[:, k] = ge_models[k].predict(
            X_te_act, return_std=True,
        )
        alpha_std_z_pred[:, k] = std_models[k].predict(X_te_act)

    # Un-standardise → original α space
    alpha_ge_pred  = alpha_ge_z_pred  * alpha_sig + alpha_mu   # (M_te, R)
    alpha_std_pred = alpha_std_z_pred * alpha_sig + alpha_mu   # (M_te, R)

    # True test coefficients
    alpha_te_true = (Psi.T @ Y_te).T                           # (M_te, R)

    # Reconstruct QoI fields
    Y_ge_pred  = Psi @ alpha_ge_pred.T    # (Ny, M_te)
    Y_std_pred = Psi @ alpha_std_pred.T   # (Ny, M_te)

    # Relative reconstruction errors
    norms_te = np.linalg.norm(Y_te, axis=0)            # (M_te,)
    eps_ge   = np.linalg.norm(Y_te - Y_ge_pred,  axis=0) / norms_te
    eps_std  = np.linalg.norm(Y_te - Y_std_pred, axis=0) / norms_te
    eps_proj = (np.linalg.norm(Y_te - Psi @ (Psi.T @ Y_te), axis=0) / norms_te)

    print(f"\n{'='*60}")
    print(f"Test reconstruction errors  (m_ge={m_ge}, M_te={M_te}, R={R})")
    print(f"  ε_y  GE-GPR      :  median={np.median(eps_ge):.2e}  max={np.max(eps_ge):.2e}")
    print(f"  ε_y  Std GPR     :  median={np.median(eps_std):.2e}  max={np.max(eps_std):.2e}")
    print(f"  ε_proj (POD lb)  :  median={np.median(eps_proj):.2e}  max={np.max(eps_proj):.2e}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 7. Learning-curve comparison                                        #
    # ------------------------------------------------------------------ #
    lc_results = {}
    if run_learning_curve and m_ge >= 10:
        lc_sizes = learning_curve_sizes or _default_lc_sizes(m_ge)
        # Keep only sizes ≤ m_ge (gradients already computed)
        lc_sizes = [s for s in lc_sizes if s <= m_ge]

        if lc_sizes:
            print(f"\nStep 6: Learning curve  (sizes: {lc_sizes})")
            lc_ge_med, lc_ge_p75  = [], []
            lc_std_med, lc_std_p75 = [], []

            for m_lc in lc_sizes:
                # Subset
                X_lc  = X_ge_norm[:m_lc, active_idx]
                y_lc  = {}
                G_lc  = {}
                for k in range(R):
                    y_lc[k] = alpha_ge_z[:m_lc, k]
                    G_lc[k] = G_act_z_norm[:m_lc, k, :]

                # Fit GE-GPR
                ge_preds_z = np.zeros((M_te, R))
                for k in range(R):
                    gp = GEGaussianProcess(
                        active_dims=np.arange(p_act),
                        n_restarts=n_restarts,
                        noise_f_init=noise_f_init,
                        noise_d_init=noise_d_init,
                        jitter=jitter,
                    )
                    gp.fit(X_lc, y_lc[k], G_lc[k])
                    ge_preds_z[:, k] = gp.predict(X_te_act)

                Y_ge_lc = Psi @ ((ge_preds_z * alpha_sig + alpha_mu).T)
                eps_lc_ge = np.linalg.norm(Y_te - Y_ge_lc, axis=0) / norms_te

                # Fit std GPR
                std_preds_z = np.zeros((M_te, R))
                for k in range(R):
                    gpr = GaussianProcessRegressor(
                        kernel=_make_sklearn_kernel(p_act),
                        n_restarts_optimizer=n_restarts,
                        normalize_y=False,
                    )
                    gpr.fit(X_lc, y_lc[k])
                    std_preds_z[:, k] = gpr.predict(X_te_act)

                Y_std_lc = Psi @ ((std_preds_z * alpha_sig + alpha_mu).T)
                eps_lc_std = np.linalg.norm(Y_te - Y_std_lc, axis=0) / norms_te

                lc_ge_med.append(np.median(eps_lc_ge))
                lc_ge_p75.append(np.percentile(eps_lc_ge, 75))
                lc_std_med.append(np.median(eps_lc_std))
                lc_std_p75.append(np.percentile(eps_lc_std, 75))

                print(f"  M={m_lc:4d}:  GE-GPR ε_y={lc_ge_med[-1]:.2e}"
                      f"   Std-GPR ε_y={lc_std_med[-1]:.2e}")

            lc_results = dict(
                sizes=lc_sizes,
                ge_med=lc_ge_med, ge_p75=lc_ge_p75,
                std_med=lc_std_med, std_p75=lc_std_p75,
            )

    # ------------------------------------------------------------------ #
    # 8. Plots                                                            #
    # ------------------------------------------------------------------ #
    print("\nStep 7: Generating plots")
    _plot_error_comparison(eps_ge, eps_std, eps_proj, m_ge, R, output_dir)
    _plot_parity(alpha_te_true, alpha_ge_pred, alpha_std_pred, R, m_ge, output_dir)
    _plot_flux_samples(Y_te, Y_ge_pred, Y_std_pred, eps_ge, eps_std,
                       x_qoi, output_dir)
    if lc_results:
        _plot_learning_curve(lc_results, output_dir)

    print(f"\nGE-GPR stage complete.  All plots saved to {output_dir}/")

    return dict(
        eps_ge=eps_ge, eps_std=eps_std, eps_proj=eps_proj,
        alpha_te_true=alpha_te_true,
        alpha_ge_pred=alpha_ge_pred, alpha_std_pred=alpha_std_pred,
        lc_results=lc_results,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Plotting helpers                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def _default_lc_sizes(m_max: int):
    """Default learning-curve sample sizes spaced below m_max."""
    candidates = [10, 15, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200]
    return [s for s in candidates if s <= m_max]


def _plot_error_comparison(eps_ge, eps_std, eps_proj, m_ge, R, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(eps_proj, bins=20, alpha=0.55, color='gray',
            label=r'$\varepsilon_{\mathrm{proj}}$ (POD lower bound)')
    ax.hist(eps_std, bins=20, alpha=0.65, color='tomato',
            label=r'$\varepsilon_y$ Std GPR')
    ax.hist(eps_ge,  bins=20, alpha=0.65, color='steelblue',
            label=r'$\varepsilon_y$ GE-GPR')
    ax.set_xlabel(r'Relative error $\varepsilon_y$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error distribution (test set)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    ax = axes[1]
    ax.semilogy(np.sort(eps_proj), 's--', ms=4, lw=1.2, color='gray',
                label=r'$\varepsilon_{\mathrm{proj}}$')
    ax.semilogy(np.sort(eps_std), 'o-',  ms=4, lw=1.2, color='tomato',
                label=r'Std GPR')
    ax.semilogy(np.sort(eps_ge),  '^-',  ms=4, lw=1.2, color='steelblue',
                label='GE-GPR')
    ax.set_xlabel('Test sample (sorted by error)', fontsize=12)
    ax.set_ylabel(r'$\varepsilon_y$', fontsize=12)
    ax.set_title('Sorted error (semi-log)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    fig.suptitle(
        f'GE-GPR vs Std-GPR — test reconstruction errors\n'
        f'(m_ge={m_ge}, R={R})', fontsize=12,
    )
    plt.tight_layout()
    path = f"{output_dir}/gegpr_error_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def _plot_parity(alpha_te_true, alpha_ge_pred, alpha_std_pred, R, m_ge, output_dir):
    n_show = min(R, 4)
    fig, axes = plt.subplots(2, n_show, figsize=(4.5 * n_show, 8), squeeze=False)

    for k in range(n_show):
        true = alpha_te_true[:, k]
        for row, (pred, label, color) in enumerate([
            (alpha_ge_pred[:, k],  'GE-GPR',  'steelblue'),
            (alpha_std_pred[:, k], 'Std GPR', 'tomato'),
        ]):
            ax = axes[row, k]
            lo = min(true.min(), pred.min())
            hi = max(true.max(), pred.max())
            ax.scatter(true, pred, s=15, alpha=0.6, color=color)
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2)
            ax.set_xlabel(fr'True $\alpha_{k+1}$', fontsize=10)
            ax.set_ylabel(fr'Predicted $\alpha_{k+1}$ ({label})', fontsize=10)
            ax.set_title(f'Mode {k+1} — {label}', fontsize=10)
            ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle(
        f'Parity plots: GE-GPR vs Std-GPR  (m_ge={m_ge})', fontsize=12,
    )
    plt.tight_layout()
    path = f"{output_dir}/gegpr_parity.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def _plot_flux_samples(Y_te, Y_ge_pred, Y_std_pred, eps_ge, eps_std, x_qoi, output_dir):
    n_plot = min(4, Y_te.shape[1])
    fig, axes = plt.subplots(1, n_plot, figsize=(4.5 * n_plot, 4.2), sharey=False)
    if n_plot == 1:
        axes = [axes]

    for ax, idx in zip(axes, range(n_plot)):
        ax.plot(x_qoi, Y_te[:, idx],      'k-',  lw=2.0, label=r'True $y(\mu)$')
        ax.plot(x_qoi, Y_ge_pred[:, idx], 'b--', lw=1.5, label=r'GE-GPR $\hat{y}$')
        ax.plot(x_qoi, Y_std_pred[:, idx],'r:',  lw=1.5, label=r'Std-GPR $\hat{y}$')
        ax.set_xlabel('x  [cm]', fontsize=10)
        ax.set_ylabel(r'$\phi(x;\mu)$', fontsize=10)
        ax.set_title(
            f'Sample {idx}\n'
            fr'GE: $\varepsilon$={eps_ge[idx]:.1e}  '
            fr'Std: $\varepsilon$={eps_std[idx]:.1e}',
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('Flux reconstruction: GE-GPR vs Std-GPR', fontsize=12)
    plt.tight_layout()
    path = f"{output_dir}/gegpr_flux_reconstruction.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def _plot_learning_curve(lc: dict, output_dir: str):
    sizes   = lc['sizes']
    ge_med  = np.array(lc['ge_med'])
    ge_p75  = np.array(lc['ge_p75'])
    std_med = np.array(lc['std_med'])
    std_p75 = np.array(lc['std_p75'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(sizes, std_med, 'o-',  color='tomato',    lw=2, ms=6,
                label='Std GPR  (median)')
    ax.fill_between(sizes, std_med, std_p75,
                    color='tomato', alpha=0.15, label='Std GPR  (75th pct)')
    ax.semilogy(sizes, ge_med,  '^--', color='steelblue', lw=2, ms=6,
                label='GE-GPR  (median)')
    ax.fill_between(sizes, ge_med, ge_p75,
                    color='steelblue', alpha=0.15, label='GE-GPR  (75th pct)')

    ax.set_xlabel('Training samples  $M$', fontsize=12)
    ax.set_ylabel(r'Median $\varepsilon_y$ on test set', fontsize=12)
    ax.set_title('Learning curve: GE-GPR vs Standard GPR', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    path = f"{output_dir}/gegpr_learning_curve.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main(fixed_mats={1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}})
