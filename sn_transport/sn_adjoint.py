"""
Discrete adjoint sensitivities for the DD S_N system in sn_solver.py.

For any linear functional Q = g^T u of the S_N unknowns, solve

    A^T lambda = g          (one solve per functional, reused for every source)

so that Q = lambda^T s and, for any parameter p,

    dQ/dp = lambda^T ( ds/dp - (dA/dp) u ).

Reshape lambda to (N, N_ang) like the edge unknowns; Lambda_i = sum_n lambda[i, n].
Because Sigma_t, Sigma_s and q enter row (i, n) of A u = s only through cell i,
the per-cell derivatives are

    dQ/dSigma_t,i = - h_i  sum_n lambda[i,n] psi_bar[i,n]     (DD collision term; incident flux included)
    dQ/dSigma_s,i = (h_i / 2) Lambda_i phi_i                   (scattering source)
    dQ/dq_i       = (h_i / 2) Lambda_i                         (fixed source)

and a per-material derivative is the sum over that material's cells
(per_material()). Replacing the 0/1 material indicator by any per-cell
direction vector gives the derivative along that direction, as in
common/adjoint_solver.py.

Interpretation: dividing block n of A^T lambda = g by w_n shows that
lambda[:, n] / w_n is the discrete adjoint angular flux, which is swept in the
-mu_n direction (A^T is upper bidiagonal per direction where A is lower).

The first-collision source of an uncollided beam, q_i = Sigma_s,i phi_u,i,
adds the chain-rule terms in dlog_phi_u_dsigma_t().
"""

import numpy as np


def solve_adjoint(lu, g: np.ndarray) -> np.ndarray:
    """A^T lambda = g, reusing the forward LU factorization (scipy SuperLU)."""
    return lu.solve(g, trans='T')


def dQ_dsigma_t_cell(lam2d, psi_bar, h):
    return -h * np.sum(lam2d * psi_bar, axis=1)


def dQ_dsigma_s_cell(lam2d, phi, h):
    return 0.5 * h * phi * lam2d.sum(axis=1)


def dQ_dq_cell(lam2d, h):
    return 0.5 * h * lam2d.sum(axis=1)


def per_material(cell_values, mat_id, n_mat):
    """Sum a per-cell derivative over each material's cells -> (n_mat,)."""
    return np.bincount(mat_id, weights=cell_values, minlength=n_mat)


def _dlogE(a):
    """d log E / da for E(a) = (1 - e^{-a}) / a, i.e. 1/expm1(a) - 1/a (-> -1/2 as a -> 0)."""
    a = np.asarray(a, dtype=float)
    small = a < 1e-4
    out = np.empty_like(a)
    out[small] = -0.5 + a[small] / 12.0
    out[~small] = 1.0 / np.expm1(a[~small]) - 1.0 / a[~small]
    return out


def dlog_phi_u_dsigma_t(h, sigma_t, mat_id, n_mat, mu0):
    """
    d log(phi_u,i) / d Sigma_t,m for the exact uncollided cell average
    phi_u,i = (J0/mu0) exp(-tau_{i-1/2}/mu0) E(Sigma_t,i h_i / mu0):

        = - (upstream path length of material m before cell i) / mu0
          + [cell i is material m] (h_i / mu0) dlogE(Sigma_t,i h_i / mu0)

    Returns (N, n_mat).
    """
    onehot = np.eye(n_mat)[mat_id]                                  # (N, n_mat)
    hm = h[:, None] * onehot
    upstream = np.vstack([np.zeros((1, n_mat)), np.cumsum(hm, axis=0)[:-1]])
    return -upstream / mu0 + hm / mu0 * _dlogE(sigma_t * h / mu0)[:, None]
