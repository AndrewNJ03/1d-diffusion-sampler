"""
adjoint_stage.py — Stage 3 of the sklearn ROM pipeline.

Implements the adjoint deliverables from Sections 14–15 of the workflow document:

  1. Identity verification: α_k (direct) vs α_k = λ_k^T s (adjoint)
  2. Gradient verification: analytic ∂α_k/∂µ vs central finite differences
  3. 1D sensitivity sweep: α_k(µ) and ∂α_k/∂µ_j vs µ_j for leading active parameters

Outputs (saved to output_dir):
  adjoint_identity.png        — parity plot, direct vs adjoint α_k
  adjoint_gradient_check.png  — scatter, analytic vs FD gradient over several samples
  adjoint_sweep_<param>.png   — 1D sweep plots (one file per swept parameter)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh, assign_material_properties, assemble_system
from lhs_generation import DiffusionParamSampler
from adjoint_solver import (
    build_adjoint_rhs, solve_adjoint, alpha_via_adjoint,
    get_layer_assignment, compute_full_gradient,
)


# ------------------------------------------------------------------ #
# Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _forward_system(mu, sampler, L, n_cells, layer_bounds, bc_left, bc_right):
    """Full forward solve; returns (phi, A, rhs, dx, D_cell, layer_assignment)."""
    D_l, Siga_l, q_l = sampler.unpack(mu)
    x_centers, _, dx = build_mesh(L, n_cells)
    D_cell, Siga_cell, q_cell = assign_material_properties(
        x_centers, layer_bounds, D_l, Siga_l, q_l
    )
    A, rhs = assemble_system(dx, D_cell, Siga_cell, q_cell, bc_left, bc_right)
    phi    = spsolve(A, rhs)
    layer_asgn = get_layer_assignment(x_centers, layer_bounds)
    return phi, A, rhs, dx, D_cell, layer_asgn


def _alpha_fd(mu, j, eps, psi_k, qoi_mask, sampler, L, n_cells, layer_bounds, bc_left, bc_right):
    """Central-difference approximation of ∂α_k/∂µ_j."""
    def _alpha(mu_):
        phi_, _, _, _, _, _ = _forward_system(
            mu_, sampler, L, n_cells, layer_bounds, bc_left, bc_right
        )
        return float(psi_k @ phi_[qoi_mask])

    mu_p, mu_m = mu.copy(), mu.copy()
    mu_p[j] += eps
    mu_m[j] -= eps
    return (_alpha(mu_p) - _alpha(mu_m)) / (2.0 * eps)


def _per_layer_bounds(scalar_bounds, n_layers, fixed_mats, key):
    b = np.asarray(scalar_bounds, dtype=float)
    if b.ndim == 1:
        # scalar (lo, hi) — expand to per-layer
        bounds = [[b[0], b[1]] for _ in range(n_layers)]
    else:
        # already (n_layers, 2) — use as-is
        bounds = b.tolist()
    if fixed_mats:
        for mat_idx, vals in fixed_mats.items():
            if key in vals:
                v = vals[key]
                bounds[mat_idx - 1] = [v, v]
    return bounds


# ------------------------------------------------------------------ #
# Main entry point                                                   #
# ------------------------------------------------------------------ #

def main(
    L=10.0,
    n_bins=3,
    n_cells=200,
    d_bounds=(0.5, 2.0),
    sigma_a_bounds=(0.05, 1.5),
    q_bounds=(1.0, 1.0),
    qoi_xmin=3.0,
    qoi_xmax=7.0,
    output_dir="output_graphs_sklearn",
    fixed_mats=None,
    n_verify=10,        # samples used for the α identity check
    n_fd_check=3,       # samples used for FD gradient verification
    n_sweep=60,         # points in each 1D parameter sweep
    fd_eps=1e-5,        # central-difference step size
    mode_indices=(0,),  # 0-based POD mode indices to analyse
    expansion_point=1.2,  # parameter value at which the Taylor expansion is built
):
    os.makedirs(output_dir, exist_ok=True)

    bc_left  = ('dirichlet', 0.0)
    bc_right = ('dirichlet', 0.0)
    layer_bounds = np.linspace(0.0, L, n_bins + 1)

    # ------------------------------------------------------------------ #
    # 0. Load POD artifacts written by svd_analysis                      #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Adjoint Stage — Sensitivities and Verification")
    print("=" * 60)

    Psi  = np.loadtxt(f"{output_dir}/pod_basis.txt",     comments='#')
    X_tr = np.loadtxt(f"{output_dir}/samples_train.txt", comments='#')

    if Psi.ndim == 1:
        Psi = Psi[:, None]
    Ny, R  = Psi.shape
    M_tr   = X_tr.shape[0]

    # Rebuild sampler for unpack() and parameter metadata
    d_b  = _per_layer_bounds(d_bounds,       n_bins, fixed_mats, "D")
    sa_b = _per_layer_bounds(sigma_a_bounds, n_bins, fixed_mats, "sigma_a")
    q_b  = _per_layer_bounds(q_bounds,       n_bins, fixed_mats, "q")
    sampler     = DiffusionParamSampler(n_bins, d_b, sa_b, q_b)
    param_names = sampler.param_names
    active_idx  = np.where(~sampler._fixed_mask)[0]  # non-fixed parameter indices

    # QoI mask (must match svd_analysis.py)
    _, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < qoi_xmax) & (x_faces[1:] > qoi_xmin)

    # Clamp mode_indices to available rank
    mode_indices = [k for k in mode_indices if k < R]
    if not mode_indices:
        mode_indices = [0]
    k0 = mode_indices[0]

    print(f"  POD basis: {Ny}×{R},  training samples: {M_tr}")
    print(f"  Active parameters ({len(active_idx)}/{sampler.p}): "
          f"{[param_names[j] for j in active_idx]}")
    print(f"  Modes analysed: {[m + 1 for m in mode_indices]}")

    # ------------------------------------------------------------------ #
    # 1. Adjoint identity:  α_k (direct) vs λ_k^T s (adjoint)           #
    # ------------------------------------------------------------------ #
    n_verify = min(n_verify, M_tr)
    print(f"\nStep 1: Identity check  ({n_verify} samples, {R} modes)")

    alpha_direct  = np.zeros((n_verify, R))
    alpha_adjoint = np.zeros((n_verify, R))

    for s in range(n_verify):
        phi, A, rhs, dx, D_cell, layer_asgn = _forward_system(
            X_tr[s], sampler, L, n_cells, layer_bounds, bc_left, bc_right
        )
        for k in range(R):
            alpha_direct[s, k] = float(Psi[:, k] @ phi[qoi_mask])
            g_k       = build_adjoint_rhs(Psi[:, k], qoi_mask)
            lambda_k  = solve_adjoint(A, g_k)
            alpha_adjoint[s, k] = alpha_via_adjoint(lambda_k, rhs)

    err_id = np.abs(alpha_direct - alpha_adjoint) / (np.abs(alpha_direct) + 1e-15)
    print(f"  Max  relative identity error: {err_id.max():.3e}")
    print(f"  Mean relative identity error: {err_id.mean():.3e}")

    n_show  = min(len(mode_indices), 3)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4.5), squeeze=False)
    for ki, k in enumerate(mode_indices[:n_show]):
        ax = axes[0, ki]
        ad = alpha_direct[:, k]
        aa = alpha_adjoint[:, k]
        ax.scatter(ad, aa, s=40, alpha=0.75, color='steelblue',
                   edgecolors='navy', linewidths=0.4)
        lo, hi = min(ad.min(), aa.min()), max(ad.max(), aa.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
        ax.set_xlabel(r'$\alpha_k$ direct  ($\psi_k^\top y$)', fontsize=11)
        ax.set_ylabel(r'$\alpha_k$ adjoint  ($\lambda_k^\top s$)', fontsize=11)
        ax.set_title(f'Mode {k + 1}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)
    fig.suptitle('Adjoint identity verification', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adjoint_identity.png", dpi=150)
    print(f"  Saved: {output_dir}/adjoint_identity.png")

    # ------------------------------------------------------------------ #
    # 2. FD gradient verification                                        #
    # ------------------------------------------------------------------ #
    n_fd = min(n_fd_check, M_tr)
    print(f"\nStep 2: FD gradient check  ({n_fd} samples, mode {k0 + 1})")

    analytic_grads = np.zeros((n_fd, len(active_idx)))
    fd_grads       = np.zeros((n_fd, len(active_idx)))
    psi_k0 = Psi[:, k0]

    for s in range(n_fd):
        mu = X_tr[s]
        phi, A, rhs, dx, D_cell, layer_asgn = _forward_system(
            mu, sampler, L, n_cells, layer_bounds, bc_left, bc_right
        )
        g_k0     = build_adjoint_rhs(psi_k0, qoi_mask)
        lambda_k0 = solve_adjoint(A, g_k0)
        full_grad = compute_full_gradient(
            lambda_k0, phi, rhs, dx, D_cell, layer_asgn, n_bins,
            bc_left[0], bc_right[0],
        )
        analytic_grads[s] = full_grad[active_idx]

        for ai, j in enumerate(active_idx):
            eps_j = fd_eps * max(abs(mu[j]), 1e-3)
            fd_grads[s, ai] = _alpha_fd(
                mu, j, eps_j, psi_k0, qoi_mask,
                sampler, L, n_cells, layer_bounds, bc_left, bc_right,
            )

        rel_err = np.abs(analytic_grads[s] - fd_grads[s]) / (np.abs(fd_grads[s]) + 1e-14)
        print(f"  Sample {s}: max FD rel err = {rel_err.max():.3e}  "
              f"mean = {rel_err.mean():.3e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    flat_a = analytic_grads.ravel()
    flat_f = fd_grads.ravel()
    ax.scatter(flat_f, flat_a, s=25, alpha=0.65, color='steelblue')
    lo, hi = min(flat_f.min(), flat_a.min()), max(flat_f.max(), flat_a.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='y = x')
    ax.set_xlabel(r'FD gradient  $\partial\alpha_k/\partial\mu_j$', fontsize=12)
    ax.set_ylabel(r'Analytic gradient', fontsize=12)
    ax.set_title(f'Gradient verification  (mode {k0 + 1}, {n_fd} samples)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adjoint_gradient_check.png", dpi=150)
    print(f"  Saved: {output_dir}/adjoint_gradient_check.png")

    # ------------------------------------------------------------------ #
    # 3. 1D sensitivity sweep  (up to 2 non-fixed parameters)            #
    # ------------------------------------------------------------------ #
    sweep_params = active_idx[:min(2, len(active_idx))]
    print(f"\nStep 3: 1D sweep  ({n_sweep} pts, params: "
          f"{[param_names[j] for j in sweep_params]})")

    # Nominal point: midpoint of each parameter's range
    mu_nom = 0.5 * (sampler.bounds[:, 0] + sampler.bounds[:, 1])

    for j in sweep_params:
        lo_j, hi_j = sampler.bounds[j]
        mu_j_vals   = np.linspace(lo_j, hi_j, n_sweep)

        alpha_sw  = np.zeros((n_sweep, len(mode_indices)))
        dalpha_sw = np.zeros((n_sweep, len(mode_indices)))

        for si, mu_j in enumerate(mu_j_vals):
            mu = mu_nom.copy()
            mu[j] = mu_j
            phi, A, rhs, dx, D_cell, layer_asgn = _forward_system(
                mu, sampler, L, n_cells, layer_bounds, bc_left, bc_right
            )
            for ki, k in enumerate(mode_indices):
                psi_k = Psi[:, k]
                alpha_sw[si, ki] = float(psi_k @ phi[qoi_mask])
                g_k      = build_adjoint_rhs(psi_k, qoi_mask)
                lambda_k = solve_adjoint(A, g_k)
                full_grad = compute_full_gradient(
                    lambda_k, phi, rhs, dx, D_cell, layer_asgn, n_bins,
                    bc_left[0], bc_right[0],
                )
                dalpha_sw[si, ki] = full_grad[j]

        # Taylor expansion about expansion_point (clamped to parameter range)
        mu_0 = float(np.clip(expansion_point, lo_j, hi_j))
        exp_idx = int(np.argmin(np.abs(mu_j_vals - mu_0)))
        mu_0 = mu_j_vals[exp_idx]   # snap to the nearest sweep point

        nm = len(mode_indices)
        fig, axes = plt.subplots(2, nm, figsize=(5 * nm, 8), squeeze=False)

        for ki, k in enumerate(mode_indices):
            alpha_0 = alpha_sw[exp_idx, ki]
            grad_0  = dalpha_sw[exp_idx, ki]
            alpha_taylor = alpha_0 + grad_0 * (mu_j_vals - mu_0)

            # Top: true curve + first-order Taylor approximation
            ax = axes[0, ki]
            ax.plot(mu_j_vals, alpha_sw[:, ki], 'k-', lw=2,
                    label=r'True $\alpha_k(\mu)$')
            ax.plot(mu_j_vals, alpha_taylor, 'b--', lw=1.8,
                    label=fr'Taylor  $\alpha_k(\mu_0)+g_0\,(\mu-\mu_0)$')
            ax.axvline(mu_0, color='gray', ls=':', lw=1.0)
            ax.scatter([mu_0], [alpha_0], s=70, color='red', zorder=5,
                       label=fr'$\mu_0={mu_0:.3g}$')
            ax.set_xlabel(param_names[j], fontsize=11)
            ax.set_ylabel(fr'$\alpha_{k + 1}(\mu)$', fontsize=11)
            ax.set_title(f'Mode {k + 1}: coefficient + Taylor expansion', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, ls='--', alpha=0.4)

            # Bottom: adjoint gradient + linear + quadratic Taylor expansions
            # d²α/dµ² (slope of gradient) via central differences
            if exp_idx == 0:
                dgrad_0 = ((dalpha_sw[1, ki] - dalpha_sw[0, ki])
                           / (mu_j_vals[1] - mu_j_vals[0]))
            elif exp_idx == n_sweep - 1:
                dgrad_0 = ((dalpha_sw[-1, ki] - dalpha_sw[-2, ki])
                           / (mu_j_vals[-1] - mu_j_vals[-2]))
            else:
                dgrad_0 = ((dalpha_sw[exp_idx + 1, ki] - dalpha_sw[exp_idx - 1, ki])
                           / (mu_j_vals[exp_idx + 1] - mu_j_vals[exp_idx - 1]))

            # d³α/dµ³ (curvature of gradient) via second-order central difference
            if 1 <= exp_idx <= n_sweep - 2:
                h = mu_j_vals[exp_idx + 1] - mu_j_vals[exp_idx]
                d2grad_0 = ((dalpha_sw[exp_idx + 1, ki]
                             - 2.0 * dalpha_sw[exp_idx, ki]
                             + dalpha_sw[exp_idx - 1, ki]) / h**2)
            else:
                d2grad_0 = 0.0

            dm = mu_j_vals - mu_0
            dalpha_taylor_lin  = grad_0 + dgrad_0 * dm
            dalpha_taylor_quad = dalpha_taylor_lin + 0.5 * d2grad_0 * dm**2

            ax = axes[1, ki]
            ax.plot(mu_j_vals, dalpha_sw[:, ki], 'k-', lw=2,
                    label='Adjoint gradient')
            ax.plot(mu_j_vals, dalpha_taylor_lin, 'b--', lw=1.8,
                    label=fr'Linear Taylor  [$g_0\'\'={dgrad_0:.3g}$]')
            ax.plot(mu_j_vals, dalpha_taylor_quad, color='darkorange', ls='-.', lw=1.8,
                    label=fr'Quadratic Taylor  [$g_0\'\'\'={d2grad_0:.3g}$]')
            ax.axvline(mu_0, color='gray', ls=':', lw=1.0)
            ax.scatter([mu_0], [grad_0], s=70, color='red', zorder=5,
                       label=fr'$\mu_0={mu_0:.3g}$')
            ax.set_xlabel(param_names[j], fontsize=11)
            ax.set_ylabel(
                r'$\partial\alpha_{' + str(k + 1) + r'}/\partial\,' + param_names[j] + '$',
                fontsize=11,
            )
            ax.set_title(f'Mode {k + 1}: adjoint sensitivity', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, ls='--', alpha=0.4)

        fig.suptitle(f'1D sensitivity sweep — {param_names[j]}', fontsize=13)
        plt.tight_layout()
        fname = f"{output_dir}/adjoint_sweep_{param_names[j]}.png"
        plt.savefig(fname, dpi=150)
        print(f"  Saved: {fname}")

    print("\nAdjoint stage complete.")
    return dict(
        alpha_direct=alpha_direct,
        alpha_adjoint=alpha_adjoint,
        analytic_grads=analytic_grads,
        fd_grads=fd_grads,
    )


if __name__ == "__main__":
    main(fixed_mats={1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}})
