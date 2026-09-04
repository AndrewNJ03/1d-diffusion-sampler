"""
adjoint_geometry.py — Adjoint sensitivities w.r.t. the geometry/switch
parameters mu = (void_thickness, steepness).

Mirrors sklearn_implementation/adjoint_stage.py (Sections 13-15 of the
workflow document), generalized to whichever of the two parameters are
"active" (non-degenerate bounds, per params.GeometryParamSampler) — exactly
the same fixed-vs-active convention DiffusionParamSampler uses for
materials, so both void_thickness and steepness are parameterizeable in the
same sense.

Reuses, unmodified, from common/adjoint_solver.py:
  - build_adjoint_rhs(psi_k, qoi_mask)   g_k = H^T psi_k
  - solve_adjoint(A, g_k)                A^T lambda_k = g_k
  - alpha_via_adjoint(lambda_k, rhs)     alpha_k = lambda_k^T s   (identity check)
  - _interior_mask(N, bc_left, bc_right) rows not overwritten by Dirichlet BCs

New here (chain rule through the smooth switch):
  Differentiating the per-cell fields D(x), Sigma_a(x), q(x) built by
  geometry.void_teflon_field() w.r.t. each parameter gives, for cells in
  the Void/Teflon shell (elsewhere the field is a fixed hard value, so the
  derivative is 0), with s_i = sigmoid_switch(r_i; a, b), a = steepness,
  b = r_interface = core_radius + void_thickness (the interface center,
  passed directly as sigmoid_switch's own `b` argument):

  d(field_i)/d(void_thickness) = (val_teflon - val_void) * s_i (1 - s_i) * (-a)

      Only b depends on void_thickness (db/d(vt) = 1, and d(argument)/db =
      -a); this is a one-signed bump concentrated at the interface.

  d(field_i)/d(steepness)      = (val_teflon - val_void) * s_i (1 - s_i) * (r_i - r_interface)

      b does not depend on steepness (a) at all in this parameterization, so
      the switch's argument a*(r-b) differentiates w.r.t. a directly:
      d(argument)/da = (r - b) = (r - r_interface): an *odd* function about
      the interface (increasing steepness pulls values on each side further
      from the halfway point, in opposite directions) rather than the
      one-signed void_thickness bump.

  Given either per-cell direction vector, the scalar sensitivity of a
  reduced coefficient is

      d(alpha_k)/d(param) = -lambda_k^T (dA/d(param)) phi + lambda_k^T (ds/d(param))

  decomposed exactly as in adjoint_solver.py's per-layer sensitivity_q /
  sensitivity_sigma_a / sensitivity_D, except the per-layer indicator
  vector is replaced by the continuous per-cell direction vector above
  (sensitivity_D generalizes to sensitivity_D_directional: dotting the
  face-coupling derivatives against a direction vector reduces exactly to
  the per-layer formula when that direction is a 0/1 layer indicator).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh, assemble_system      # noqa: E402
from adjoint_solver import (                                     # noqa: E402
    build_adjoint_rhs, solve_adjoint, alpha_via_adjoint, _interior_mask,
)
from geometry import (                                           # noqa: E402
    sigmoid_switch, assign_void_teflon_properties, solve_void_teflon,
)
from params import GeometryParamSampler                           # noqa: E402


# ------------------------------------------------------------------ #
# Chain rule: per-cell field derivatives w.r.t. each parameter        #
# ------------------------------------------------------------------ #

def field_derivative_wrt_void_thickness(
    x_centers, L, core_radius, void_thickness, shell_width,
    val_void, val_teflon, steepness,
):
    """
    d(field)/d(void_thickness), per cell. Only the smooth Void/Teflon
    interface term depends on void_thickness; cells outside the shell have
    zero derivative (see module docstring).
    """
    center = 0.5 * L
    r = np.abs(x_centers - center)
    r_interface = core_radius + void_thickness

    s = sigmoid_switch(r, steepness, r_interface)
    dfield = (val_teflon - val_void) * s * (1.0 - s) * (-steepness)

    in_shell = (r >= core_radius) & (r <= core_radius + shell_width)
    return np.where(in_shell, dfield, 0.0)


def field_derivative_wrt_steepness(
    x_centers, L, core_radius, void_thickness, shell_width,
    val_void, val_teflon, steepness,
):
    """
    d(field)/d(steepness), per cell. Because b (the interface center) does
    not depend on steepness (a) in this parameterization, the switch
    argument a*(r - b) differentiates directly: d(argument)/da = (r - b) =
    (r - r_interface), an odd-symmetric bump about the interface (see
    module docstring).
    """
    center = 0.5 * L
    r = np.abs(x_centers - center)
    r_interface = core_radius + void_thickness

    s = sigmoid_switch(r, steepness, r_interface)
    dfield = (val_teflon - val_void) * s * (1.0 - s) * (r - r_interface)

    in_shell = (r >= core_radius) & (r <= core_radius + shell_width)
    return np.where(in_shell, dfield, 0.0)


_FIELD_DERIVATIVE_FUNCS = {
    "void_thickness": field_derivative_wrt_void_thickness,
    "steepness": field_derivative_wrt_steepness,
}


def geometry_field_derivatives(
    param_name, x_centers, L, core_radius, void_thickness, shell_width,
    mat_void, mat_teflon, steepness,
):
    """Per-cell (dD, dSigma_a, dq) w.r.t. the named parameter, one call each."""
    func = _FIELD_DERIVATIVE_FUNCS[param_name]
    D_void, Sa_void, q_void = mat_void
    D_tef, Sa_tef, q_tef = mat_teflon

    dD = func(x_centers, L, core_radius, void_thickness, shell_width, D_void, D_tef, steepness)
    dSa = func(x_centers, L, core_radius, void_thickness, shell_width, Sa_void, Sa_tef, steepness)
    dq = func(x_centers, L, core_radius, void_thickness, shell_width, q_void, q_tef, steepness)
    return dD, dSa, dq


# ------------------------------------------------------------------ #
# Directional generalization of adjoint_solver.sensitivity_D          #
# ------------------------------------------------------------------ #

def sensitivity_D_directional(lambda_k, phi, dx, D_cell, direction,
                               bc_left='dirichlet', bc_right='dirichlet'):
    """
    d(alpha_k)/d(param) contribution through D(x), for a continuous per-cell
    direction vector `direction` = dD/d(param) (instead of a per-layer 0/1
    indicator as in adjoint_solver.sensitivity_D — this reduces to that
    formula exactly when `direction` is such an indicator).

    Derivation identical to adjoint_solver.sensitivity_D (Section 15):
    d(alpha_k)/d(param) = -lambda_k^T (dA/d(param)) phi, with dA/d(param)
    assembled from the analytic face-coupling derivatives d(beta_f)/d(D_i).
    """
    N = len(dx)
    interior = _interior_mask(N, bc_left, bc_right)
    d = D_cell / dx
    lam = np.where(interior, lambda_k, 0.0)

    i = np.arange(N - 1)
    ip1 = i + 1
    sum_d = d[i] + d[ip1]
    denom = sum_d ** 2
    safe = denom > 0.0

    dbeta_dDi = np.zeros(N - 1)
    dbeta_dDip1 = np.zeros(N - 1)
    dbeta_dDi[safe] = 2.0 * d[ip1][safe] ** 2 / denom[safe] / dx[i][safe]
    dbeta_dDip1[safe] = 2.0 * d[i][safe] ** 2 / denom[safe] / dx[ip1][safe]

    dphi = phi[i] - phi[ip1]
    lam_diff = lam[i] - lam[ip1]

    contrib = (dbeta_dDi * direction[i] + dbeta_dDip1 * direction[ip1]) * dphi * lam_diff
    return -float(contrib.sum())


def sensitivity_from_directions(
    lambda_k, phi, dx, D_cell, dD, dSa, dq,
    bc_left='dirichlet', bc_right='dirichlet',
):
    """
    Full d(alpha_k)/d(param) given per-cell direction vectors (dD, dSa, dq)
    = d(field)/d(param), decomposed into the D, Sigma_a, and q channels
    (directional generalizations of eqs. 14-15 and Section 15's
    D-sensitivity in the workflow document). Works identically regardless
    of which parameter the direction vectors came from.

    Returns
    -------
    total  : float, the combined sensitivity
    parts  : dict with the D / Sigma_a / q channel contributions
    """
    N = len(dx)
    interior = _interior_mask(N, bc_left, bc_right)

    dD_term = sensitivity_D_directional(lambda_k, phi, dx, D_cell, dD, bc_left, bc_right)
    dSa_term = -float((lambda_k[interior] * phi[interior]) @ (dSa[interior] * dx[interior]))
    dq_term = float(lambda_k[interior] @ (dq[interior] * dx[interior]))

    total = dD_term + dSa_term + dq_term
    return total, dict(D=dD_term, Sigma_a=dSa_term, q=dq_term)


def sensitivity_wrt_param(
    param_name, lambda_k, phi, dx, D_cell, x_centers, L, core_radius,
    void_thickness, shell_width, mat_void, mat_teflon, steepness,
    bc_left='dirichlet', bc_right='dirichlet',
):
    """d(alpha_k)/d(param_name), param_name in {'void_thickness', 'steepness'}."""
    dD, dSa, dq = geometry_field_derivatives(
        param_name, x_centers, L, core_radius, void_thickness, shell_width,
        mat_void, mat_teflon, steepness,
    )
    return sensitivity_from_directions(lambda_k, phi, dx, D_cell, dD, dSa, dq, bc_left, bc_right)


# ------------------------------------------------------------------ #
# Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _forward_system(mu, L, n_cells, core_radius, shell_width,
                     mat_core, mat_void, mat_teflon, mat_shield,
                     bc_left, bc_right):
    """Full forward solve; returns (phi, A, rhs, dx, D_cell, x_centers)."""
    vt, steepness = float(mu[0]), float(mu[1])
    x_centers, x_faces, dx = build_mesh(L, n_cells)
    D_cell, Sa_cell, q_cell, _ = assign_void_teflon_properties(
        x_centers, L, core_radius, vt, shell_width,
        mat_core, mat_void, mat_teflon, mat_shield, steepness,
    )
    A, rhs = assemble_system(dx, D_cell, Sa_cell, q_cell, bc_left, bc_right)
    phi = spsolve(A, rhs)
    return phi, A, rhs, dx, D_cell, x_centers


def _alpha_fd(mu, j, eps, psi_k, qoi_mask, L, n_cells, core_radius, shell_width,
              mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right):
    """Central-difference approximation of d(alpha_k)/d(mu_j)."""
    def _alpha(mu_):
        phi_, _, _, _, _, _ = _forward_system(
            mu_, L, n_cells, core_radius, shell_width,
            mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
        )
        return float(psi_k @ phi_[qoi_mask])

    mu_p, mu_m = np.array(mu, dtype=float), np.array(mu, dtype=float)
    mu_p[j] += eps
    mu_m[j] -= eps
    return (_alpha(mu_p) - _alpha(mu_m)) / (2.0 * eps)


def _fd_eps(param_name, value, lo, hi, void_thickness, shell_width, fd_eps):
    """Central-difference step, clamped to stay inside physically valid bounds."""
    eps = fd_eps * max(abs(value), 1e-3)
    if param_name == "void_thickness":
        margin = min(value - 0.0, shell_width - void_thickness, void_thickness)
    else:
        margin = value  # steepness > 0; keep the perturbed value positive
    if hi > lo:
        margin = min(margin, value - lo, hi - value)
    if margin > 0:
        eps = min(eps, 0.4 * margin)
    return eps


# ------------------------------------------------------------------ #
# Main entry point                                                    #
# ------------------------------------------------------------------ #

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
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs_geometry",
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
    n_verify=10,
    n_fd_check=5,
    n_sweep=60,
    fd_eps=1e-5,
    mode_indices=(0, 1),
    expansion_point=None,
):
    QOI_XMIN = qoi_xmin if qoi_xmin is not None else core_radius
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else core_radius + shell_width

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Adjoint Stage (geometry) — void_thickness / steepness sensitivities")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 0. Load POD artifacts written by svd_analysis_geometry              #
    # ------------------------------------------------------------------ #
    Psi = np.loadtxt(f"{output_dir}/pod_basis.txt", comments='#')
    X_tr = np.loadtxt(f"{output_dir}/samples_train.txt", comments='#')

    if Psi.ndim == 1:
        Psi = Psi[:, None]
    if X_tr.ndim == 1:
        X_tr = X_tr[:, None]
    Ny, R = Psi.shape
    M_tr = X_tr.shape[0]

    sampler = GeometryParamSampler(void_bounds, steepness_bounds)
    active_idx = np.where(~sampler._fixed_mask)[0]
    param_names = sampler.param_names

    x_centers, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)

    mode_indices = [k for k in mode_indices if k < R]
    if not mode_indices:
        mode_indices = [0]
    k0 = mode_indices[0]

    print(f"  POD basis: {Ny}x{R},  training samples: {M_tr}")
    print(f"  Active parameter(s): {[param_names[j] for j in active_idx] or '(none — all fixed)'}")
    print(f"  Modes analysed: {[m + 1 for m in mode_indices]}")

    # ------------------------------------------------------------------ #
    # 1. Adjoint identity:  alpha_k (direct) vs lambda_k^T s (adjoint)    #
    # ------------------------------------------------------------------ #
    n_verify = min(n_verify, M_tr)
    print(f"\nStep 1: Identity check  ({n_verify} samples, {R} modes)")

    alpha_direct = np.zeros((n_verify, R))
    alpha_adjoint = np.zeros((n_verify, R))

    for s in range(n_verify):
        mu = X_tr[s]
        phi, A, rhs, dx, D_cell, _ = _forward_system(
            mu, L, n_cells, core_radius, shell_width,
            mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
        )
        for k in range(R):
            alpha_direct[s, k] = float(Psi[:, k] @ phi[qoi_mask])
            g_k = build_adjoint_rhs(Psi[:, k], qoi_mask)
            lambda_k = solve_adjoint(A, g_k)
            alpha_adjoint[s, k] = alpha_via_adjoint(lambda_k, rhs)

    err_id = np.abs(alpha_direct - alpha_adjoint) / (np.abs(alpha_direct) + 1e-15)
    print(f"  Max  relative identity error: {err_id.max():.3e}")
    print(f"  Mean relative identity error: {err_id.mean():.3e}")

    n_show = min(len(mode_indices), 3)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4.5), squeeze=False)
    for ki, k in enumerate(mode_indices[:n_show]):
        ax = axes[0, ki]
        ad = alpha_direct[:, k]
        aa = alpha_adjoint[:, k]
        ax.scatter(ad, aa, s=40, alpha=0.75, color='steelblue', edgecolors='navy', linewidths=0.4)
        lo, hi = min(ad.min(), aa.min()), max(ad.max(), aa.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
        ax.set_xlabel(r'$\alpha_k$ direct  ($\psi_k^\top y$)', fontsize=11)
        ax.set_ylabel(r'$\alpha_k$ adjoint  ($\lambda_k^\top s$)', fontsize=11)
        ax.set_title(f'Mode {k + 1}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)
    fig.suptitle('Adjoint identity verification (geometry parameters)', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adjoint_identity_geometry.png", dpi=150)
    print(f"  Saved: {output_dir}/adjoint_identity_geometry.png")

    if len(active_idx) == 0:
        print("\nNo active parameters (all fixed) — skipping FD check and sweep.")
        return dict(alpha_direct=alpha_direct, alpha_adjoint=alpha_adjoint)

    # ------------------------------------------------------------------ #
    # 2. FD gradient verification (all active parameters)                #
    # ------------------------------------------------------------------ #
    n_fd = min(n_fd_check, M_tr)
    print(f"\nStep 2: FD gradient check  ({n_fd} samples, "
          f"params {[param_names[j] for j in active_idx]}, modes {[m+1 for m in mode_indices]})")

    analytic_grads = np.zeros((n_fd, len(active_idx), len(mode_indices)))
    fd_grads = np.zeros((n_fd, len(active_idx), len(mode_indices)))

    for s in range(n_fd):
        mu = X_tr[s]
        vt, steepness = float(mu[0]), float(mu[1])
        phi, A, rhs, dx, D_cell, xc = _forward_system(
            mu, L, n_cells, core_radius, shell_width,
            mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
        )

        for ki, k in enumerate(mode_indices):
            psi_k = Psi[:, k]
            g_k = build_adjoint_rhs(psi_k, qoi_mask)
            lambda_k = solve_adjoint(A, g_k)

            for ai, j in enumerate(active_idx):
                name = param_names[j]
                total, _parts = sensitivity_wrt_param(
                    name, lambda_k, phi, dx, D_cell, xc, L, core_radius,
                    vt, shell_width, mat_void, mat_teflon, steepness, bc_left[0], bc_right[0],
                )
                analytic_grads[s, ai, ki] = total

                lo_j, hi_j = sampler.bounds[j]
                eps = _fd_eps(name, float(mu[j]), lo_j, hi_j, vt, shell_width, fd_eps)
                fd_grads[s, ai, ki] = _alpha_fd(
                    mu, j, eps, psi_k, qoi_mask, L, n_cells, core_radius, shell_width,
                    mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
                )

        rel_err = (np.abs(analytic_grads[s] - fd_grads[s]) / (np.abs(fd_grads[s]) + 1e-14))
        print(f"  Sample {s} (vt={vt:.4f}, a={steepness:.1f}): "
              f"max FD rel err = {rel_err.max():.3e}  mean = {rel_err.mean():.3e}")

    fig, axes = plt.subplots(1, len(active_idx), figsize=(6 * len(active_idx), 5), squeeze=False)
    for ai, j in enumerate(active_idx):
        ax = axes[0, ai]
        flat_a = analytic_grads[:, ai, :].ravel()
        flat_f = fd_grads[:, ai, :].ravel()
        ax.scatter(flat_f, flat_a, s=25, alpha=0.65, color='steelblue')
        lo, hi = min(flat_f.min(), flat_a.min()), max(flat_f.max(), flat_a.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
        ax.set_xlabel(fr'FD gradient  $\partial\alpha_k/\partial\,\mathrm{{{param_names[j]}}}$', fontsize=11)
        ax.set_ylabel(r'Analytic (adjoint) gradient', fontsize=11)
        ax.set_title(f'{param_names[j]}  ({n_fd} samples, modes {[m+1 for m in mode_indices]})', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adjoint_gradient_check_geometry.png", dpi=150)
    print(f"  Saved: {output_dir}/adjoint_gradient_check_geometry.png")

    # ------------------------------------------------------------------ #
    # 3. 1D sensitivity sweep (each active parameter, others held at      #
    #    the midpoint of their own bounds — trivial for a fixed param)    #
    # ------------------------------------------------------------------ #
    mu_nom = 0.5 * (sampler.bounds[:, 0] + sampler.bounds[:, 1])
    sweep_params = active_idx[:min(2, len(active_idx))]
    print(f"\nStep 3: 1D sweep  ({n_sweep} pts, params: {[param_names[j] for j in sweep_params]})")

    for j in sweep_params:
        name = param_names[j]
        lo_j, hi_j = sampler.bounds[j]
        p_vals = np.linspace(lo_j, hi_j, n_sweep)

        alpha_sw = np.zeros((n_sweep, len(mode_indices)))
        dalpha_sw = np.zeros((n_sweep, len(mode_indices)))
        dalpha_parts = {k: {'D': np.zeros(n_sweep), 'Sigma_a': np.zeros(n_sweep), 'q': np.zeros(n_sweep)}
                        for k in mode_indices}

        for si, p_val in enumerate(p_vals):
            mu = mu_nom.copy()
            mu[j] = p_val
            vt, steepness = float(mu[0]), float(mu[1])
            phi, A, rhs, dx, D_cell, xc = _forward_system(
                mu, L, n_cells, core_radius, shell_width,
                mat_core, mat_void, mat_teflon, mat_shield, bc_left, bc_right,
            )
            for ki, k in enumerate(mode_indices):
                psi_k = Psi[:, k]
                alpha_sw[si, ki] = float(psi_k @ phi[qoi_mask])
                g_k = build_adjoint_rhs(psi_k, qoi_mask)
                lambda_k = solve_adjoint(A, g_k)
                total, parts = sensitivity_wrt_param(
                    name, lambda_k, phi, dx, D_cell, xc, L, core_radius,
                    vt, shell_width, mat_void, mat_teflon, steepness, bc_left[0], bc_right[0],
                )
                dalpha_sw[si, ki] = total
                for key in ('D', 'Sigma_a', 'q'):
                    dalpha_parts[k][key][si] = parts[key]

        mu_0 = float(expansion_point) if expansion_point is not None else 0.5 * (lo_j + hi_j)
        mu_0 = float(np.clip(mu_0, lo_j, hi_j))
        exp_idx = int(np.argmin(np.abs(p_vals - mu_0)))
        mu_0 = p_vals[exp_idx]

        nm = len(mode_indices)
        fig, axes = plt.subplots(2, nm, figsize=(5.5 * nm, 8), squeeze=False)

        for ki, k in enumerate(mode_indices):
            alpha_0 = alpha_sw[exp_idx, ki]
            grad_0 = dalpha_sw[exp_idx, ki]
            alpha_taylor = alpha_0 + grad_0 * (p_vals - mu_0)

            ax = axes[0, ki]
            ax.plot(p_vals, alpha_sw[:, ki], 'k-', lw=2, label=fr'True $\alpha_k({name})$')
            ax.plot(p_vals, alpha_taylor, 'b--', lw=1.8,
                    label=fr'Taylor  $\alpha_k(p_0)+g_0\,(p-p_0)$')
            ax.axvline(mu_0, color='gray', ls=':', lw=1.0)
            ax.scatter([mu_0], [alpha_0], s=70, color='red', zorder=5, label=fr'$p_0={mu_0:.3g}$')
            ax.set_xlabel(name, fontsize=11)
            ax.set_ylabel(fr'$\alpha_{k + 1}$', fontsize=11)
            ax.set_title(f'Mode {k + 1}: coefficient + Taylor expansion', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, ls='--', alpha=0.4)

            ax = axes[1, ki]
            ax.plot(p_vals, dalpha_sw[:, ki], 'k-', lw=2, label='Adjoint gradient (total)')
            ax.plot(p_vals, dalpha_parts[k]['D'], lw=1.2, color='tab:orange', label='D channel')
            ax.plot(p_vals, dalpha_parts[k]['Sigma_a'], lw=1.2, color='tab:green', label=r'$\Sigma_a$ channel')
            ax.plot(p_vals, dalpha_parts[k]['q'], lw=1.2, color='tab:purple', label='q channel')
            ax.axvline(mu_0, color='gray', ls=':', lw=1.0)
            ax.scatter([mu_0], [grad_0], s=70, color='red', zorder=5, label=fr'$p_0={mu_0:.3g}$')
            ax.set_xlabel(name, fontsize=11)
            ax.set_ylabel(fr'$\partial\alpha_{k + 1}/\partial\,{name}$', fontsize=11)
            ax.set_title(f'Mode {k + 1}: adjoint sensitivity  (channel breakdown)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, ls='--', alpha=0.4)

        fig.suptitle(f'1D sensitivity sweep — {name}', fontsize=13)
        plt.tight_layout()
        fname = f"{output_dir}/adjoint_sweep_{name}.png"
        plt.savefig(fname, dpi=150)
        print(f"  Saved: {fname}")

    print("\nAdjoint stage (geometry) complete.")
    return dict(
        alpha_direct=alpha_direct, alpha_adjoint=alpha_adjoint,
        analytic_grads=analytic_grads, fd_grads=fd_grads,
    )


if __name__ == "__main__":
    main()
