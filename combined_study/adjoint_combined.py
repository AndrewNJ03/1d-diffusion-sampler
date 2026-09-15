"""
adjoint_combined.py — Adjoint sensitivities for the fully-coupled
5-parameter study, mu = (void_thickness, steepness, D_teflon,
Sigma_a_teflon, q_teflon).

Extends adjoint_geometry.py's two switch-parameter sensitivities
(void_thickness, steepness) with three new ones for the Teflon region's
material triple, previously fixed constants in mat_teflon.
adjoint_geometry.sensitivity_from_directions() is reused completely
unchanged -- it is already generic to arbitrary per-cell (dD, dSigma_a, dq)
direction vectors, so combining geometry and material sensitivities is a
matter of supplying one new direction-vector function per material
channel, not new adjoint math (see that module's docstring, and the
feasibility note this is built from: sensitivity_D_directional already
"reduces exactly to the per-layer formula when that direction is a 0/1
layer indicator").

New here: d(field)/d(val_teflon), for whichever material channel val_teflon
belongs to. void_teflon_field() blends

    field(x) = val_void + (val_teflon - val_void) * s(x)

across the shell (s = sigmoid_switch, s->0 towards Void, s->1 towards
Teflon; field is a fixed hard value outside the shell). Differentiating
w.r.t. val_teflon directly gives

    d(field)/d(val_teflon) = s(x)   within the shell, 0 elsewhere

-- the same shell domain field_derivative_wrt_void_thickness/steepness use
in adjoint_geometry.py. The other two material channels get an all-zero
direction vector for that parameter (e.g. D_teflon only perturbs the D(x)
field, not Sigma_a(x) or q(x)), and one adjoint solve per POD mode still
yields all 5 partials at once, exactly as for the 2-parameter case.
"""

import os
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'void_teflon_geometry'))

from second_order_solver import build_mesh, assemble_system           # noqa: E402
from adjoint_solver import build_adjoint_rhs, solve_adjoint           # noqa: E402
from geometry import sigmoid_switch, assign_void_teflon_properties    # noqa: E402
from adjoint_geometry import (                                        # noqa: E402
    geometry_field_derivatives, sensitivity_from_directions,
)
from params_combined import CombinedParamSampler                      # noqa: E402


# ------------------------------------------------------------------ #
# New: per-cell field derivative w.r.t. a Teflon material value       #
# ------------------------------------------------------------------ #

def field_derivative_wrt_teflon_value(
    x_centers, L, core_radius, void_thickness, shell_width, steepness,
):
    """
    d(field)/d(val_teflon), per cell, for whichever material channel
    val_teflon belongs to (D, Sigma_a, or q -- the derivative has the same
    form for all three, since void_teflon_field() blends linearly in
    val_teflon within the shell and is independent of it elsewhere).
    """
    center = 0.5 * L
    r = np.abs(x_centers - center)
    r_interface = core_radius + void_thickness

    s = sigmoid_switch(r, steepness, r_interface)
    in_shell = (r >= core_radius) & (r <= core_radius + shell_width)
    return np.where(in_shell, s, 0.0)


_MATERIAL_CHANNELS = {"D_teflon": 0, "Sigma_a_teflon": 1, "q_teflon": 2}


def combined_field_derivatives(
    param_name, x_centers, L, core_radius, void_thickness, shell_width,
    mat_void, mat_teflon, steepness,
):
    """
    Per-cell (dD, dSigma_a, dq) w.r.t. param_name, for any of the 5
    combined parameters. void_thickness/steepness delegate to
    adjoint_geometry.geometry_field_derivatives() unchanged; the three
    Teflon material parameters use field_derivative_wrt_teflon_value() on
    their own channel and zero on the other two.
    """
    if param_name in ("void_thickness", "steepness"):
        return geometry_field_derivatives(
            param_name, x_centers, L, core_radius, void_thickness, shell_width,
            mat_void, mat_teflon, steepness,
        )

    channel = _MATERIAL_CHANNELS[param_name]
    d_val = field_derivative_wrt_teflon_value(
        x_centers, L, core_radius, void_thickness, shell_width, steepness,
    )
    zeros = np.zeros_like(d_val)
    dD, dSa, dq = zeros, zeros, zeros
    if channel == 0:
        dD = d_val
    elif channel == 1:
        dSa = d_val
    else:
        dq = d_val
    return dD, dSa, dq


def sensitivity_wrt_combined_param(
    param_name, lambda_k, phi, dx, D_cell, x_centers, L, core_radius,
    void_thickness, shell_width, mat_void, mat_teflon, steepness,
    bc_left='dirichlet', bc_right='dirichlet',
):
    """d(alpha_k)/d(param_name) for any of the 5 combined parameters."""
    dD, dSa, dq = combined_field_derivatives(
        param_name, x_centers, L, core_radius, void_thickness, shell_width,
        mat_void, mat_teflon, steepness,
    )
    return sensitivity_from_directions(lambda_k, phi, dx, D_cell, dD, dSa, dq, bc_left, bc_right)


# ------------------------------------------------------------------ #
# Forward solve for the combined 5D parameter vector                  #
# ------------------------------------------------------------------ #

def forward_system_combined(
    mu, L, n_cells, core_radius, shell_width, mat_core, mat_void, mat_shield,
    bc_left, bc_right,
):
    """
    Full forward solve for mu = (void_thickness, steepness, D_teflon,
    Sigma_a_teflon, q_teflon) -- mat_teflon is read from mu[2:5] instead of
    a fixed constant, everything else identical to
    adjoint_geometry._forward_system().

    Returns
    -------
    phi, A, rhs, dx, D_cell, x_centers, mat_teflon
    """
    vt, steepness, d_t, sa_t, q_t = (float(v) for v in mu)
    mat_teflon = (d_t, sa_t, q_t)

    x_centers, x_faces, dx = build_mesh(L, n_cells)
    D_cell, Sa_cell, q_cell, _ = assign_void_teflon_properties(
        x_centers, L, core_radius, vt, shell_width,
        mat_core, mat_void, mat_teflon, mat_shield, steepness,
    )
    A, rhs = assemble_system(dx, D_cell, Sa_cell, q_cell, bc_left, bc_right)
    phi = spsolve(A, rhs)
    return phi, A, rhs, dx, D_cell, x_centers, mat_teflon


def _alpha_fd(mu, j, eps, psi_k, qoi_mask, L, n_cells, core_radius, shell_width,
              mat_core, mat_void, mat_shield, bc_left, bc_right):
    """Central-difference approximation of d(alpha_k)/d(mu_j), any of the 5 dims."""
    def _alpha(mu_):
        phi_, _, _, _, _, _, _ = forward_system_combined(
            mu_, L, n_cells, core_radius, shell_width,
            mat_core, mat_void, mat_shield, bc_left, bc_right,
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
    elif param_name == "steepness":
        margin = value  # steepness > 0; keep the perturbed value positive
    else:
        margin = max(abs(value), 1e-3)  # material values: no sign constraint assumed
    if hi > lo:
        margin = min(margin, value - lo, hi - value)
    if margin > 0:
        eps = min(eps, 0.4 * margin)
    return eps


# ------------------------------------------------------------------ #
# Verification: adjoint identity + FD gradient check, all 5 params    #
# ------------------------------------------------------------------ #

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
    qoi_xmin=None,
    qoi_xmax=None,
    output_dir="output_graphs_combined",
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
    n_verify=10,
    n_fd_check=5,
    fd_eps=1e-5,
    mode_indices=(0, 1),
):
    """
    Verifies the combined adjoint machinery the same way
    adjoint_geometry.main() does for the 2-parameter case: an identity
    check (adjoint alpha_k vs. direct projection) and a finite-difference
    gradient check across all 5 (or however many are active) parameters,
    reusing the POD basis/training samples already written by
    svd_analysis_combined.main() to output_dir.
    """
    QOI_XMIN = qoi_xmin if qoi_xmin is not None else core_radius
    QOI_XMAX = qoi_xmax if qoi_xmax is not None else core_radius + shell_width

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Adjoint Stage (combined 5-parameter) — verification")
    print("=" * 60)

    Psi = np.loadtxt(f"{output_dir}/pod_basis.txt", comments='#')
    X_tr = np.loadtxt(f"{output_dir}/samples_train.txt", comments='#')
    if Psi.ndim == 1:
        Psi = Psi[:, None]
    if X_tr.ndim == 1:
        X_tr = X_tr[:, None]
    Ny, R = Psi.shape
    M_tr = X_tr.shape[0]

    sampler = CombinedParamSampler(
        void_bounds, steepness_bounds, d_teflon_bounds, sigma_a_teflon_bounds, q_teflon_bounds,
    )
    active_idx = np.where(~sampler._fixed_mask)[0]
    param_names = sampler.param_names

    x_centers, x_faces, _ = build_mesh(L, n_cells)
    qoi_mask = (x_faces[:-1] < QOI_XMAX) & (x_faces[1:] > QOI_XMIN)

    mode_indices = [k for k in mode_indices if k < R] or [0]

    print(f"  POD basis: {Ny}x{R},  training samples: {M_tr}")
    print(f"  Active parameters: {sampler.active_names}")
    print(f"  Modes analysed: {[m + 1 for m in mode_indices]}")

    # ------------------------------------------------------------------ #
    # 1. Adjoint identity                                                 #
    # ------------------------------------------------------------------ #
    n_verify = min(n_verify, M_tr)
    print(f"\nStep 1: Identity check  ({n_verify} samples, {R} modes)")

    alpha_direct = np.zeros((n_verify, R))
    alpha_adjoint = np.zeros((n_verify, R))

    for s in range(n_verify):
        mu = X_tr[s]
        phi, A, rhs, dx, D_cell, xc, mat_teflon = forward_system_combined(
            mu, L, n_cells, core_radius, shell_width, mat_core, mat_void, mat_shield,
            bc_left, bc_right,
        )
        for k in range(R):
            alpha_direct[s, k] = float(Psi[:, k] @ phi[qoi_mask])
            g_k = build_adjoint_rhs(Psi[:, k], qoi_mask)
            lambda_k = solve_adjoint(A, g_k)
            alpha_adjoint[s, k] = float(lambda_k @ rhs)

    err_id = np.abs(alpha_direct - alpha_adjoint) / (np.abs(alpha_direct) + 1e-15)
    print(f"  Max  relative identity error: {err_id.max():.3e}")
    print(f"  Mean relative identity error: {err_id.mean():.3e}")

    if len(active_idx) == 0:
        print("\nNo active parameters (all fixed) — skipping FD check.")
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
        vt, steepness, mat_teflon = sampler.unpack(mu)
        phi, A, rhs, dx, D_cell, xc, _ = forward_system_combined(
            mu, L, n_cells, core_radius, shell_width, mat_core, mat_void, mat_shield,
            bc_left, bc_right,
        )

        for ki, k in enumerate(mode_indices):
            psi_k = Psi[:, k]
            g_k = build_adjoint_rhs(psi_k, qoi_mask)
            lambda_k = solve_adjoint(A, g_k)

            for ai, j in enumerate(active_idx):
                name = param_names[j]
                total, _parts = sensitivity_wrt_combined_param(
                    name, lambda_k, phi, dx, D_cell, xc, L, core_radius,
                    vt, shell_width, mat_void, mat_teflon, steepness, bc_left[0], bc_right[0],
                )
                analytic_grads[s, ai, ki] = total

                lo_j, hi_j = sampler.bounds[j]
                eps = _fd_eps(name, float(mu[j]), lo_j, hi_j, vt, shell_width, fd_eps)
                fd_grads[s, ai, ki] = _alpha_fd(
                    mu, j, eps, psi_k, qoi_mask, L, n_cells, core_radius, shell_width,
                    mat_core, mat_void, mat_shield, bc_left, bc_right,
                )

        rel_err = np.abs(analytic_grads[s] - fd_grads[s]) / (np.abs(fd_grads[s]) + 1e-14)
        print(f"  Sample {s}: max FD rel err = {rel_err.max():.3e}  mean = {rel_err.mean():.3e}")

    print("\nAdjoint stage (combined) complete.")
    return dict(
        alpha_direct=alpha_direct, alpha_adjoint=alpha_adjoint,
        analytic_grads=analytic_grads, fd_grads=fd_grads,
    )


if __name__ == "__main__":
    main()
