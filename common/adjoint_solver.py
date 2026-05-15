"""
Adjoint solver and parameter sensitivities for the 1D diffusion sandbox.

Implements Sections 13–15 of the workflow document:
  - Adjoint RHS construction:  g_k = H^T ψ_k  (embeds POD mode into full space)
  - Adjoint solve:              A^T λ_k = g_k
  - Verification identity:     α_k = λ_k^T s   (eq. 8/12)
  - Sensitivities (eq. 13):
      ∂α_k/∂q_m    (eq. 14)  — RHS perturbation only
      ∂α_k/∂Σa_m   (eq. 15)  — diagonal absorption perturbation
      ∂α_k/∂D_m             — face-coupling perturbation (Section 15)
"""

import numpy as np
from scipy.sparse.linalg import spsolve


def build_adjoint_rhs(psi_k: np.ndarray, qoi_mask: np.ndarray) -> np.ndarray:
    """
    Build adjoint RHS  g_k = H^T ψ_k ∈ R^N.

    H is the QoI masking operator (Ny × N, implemented as a boolean index).
    H^T ψ_k places the Ny-vector ψ_k at the QoI cell positions and zeros elsewhere.
    """
    N = len(qoi_mask)
    g_k = np.zeros(N)
    g_k[qoi_mask] = psi_k
    return g_k


def solve_adjoint(A, g_k: np.ndarray) -> np.ndarray:
    """Solve A^T λ_k = g_k.  Returns λ_k ∈ R^N."""
    return spsolve(A.T.tocsr(), g_k)


def alpha_via_adjoint(lambda_k: np.ndarray, rhs: np.ndarray) -> float:
    """Adjoint identity: α_k = λ_k^T s  (eq. 8/12 in the document)."""
    return float(lambda_k @ rhs)


def get_layer_assignment(x_centers: np.ndarray, layer_bounds: np.ndarray) -> np.ndarray:
    """
    Integer array (N,) with 0-based layer index for each cell.

    Matches the logic in assign_material_properties() in second_order_solver.py.
    """
    assignment = np.searchsorted(layer_bounds[1:], x_centers, side='right')
    return np.minimum(assignment, len(layer_bounds) - 2)


def _interior_mask(N: int, bc_left: str, bc_right: str) -> np.ndarray:
    """
    Boolean mask identifying rows that are NOT overwritten by Dirichlet BCs.

    Dirichlet rows have their equation replaced by φ_i = value, so neither
    absorption, source, nor diffusion coefficients appear there.
    """
    interior = np.ones(N, dtype=bool)
    if bc_left == 'dirichlet':
        interior[0] = False
    if bc_right == 'dirichlet':
        interior[N - 1] = False
    return interior


def sensitivity_q(
    lambda_k: np.ndarray,
    dx: np.ndarray,
    layer_assignment: np.ndarray,
    n_layers: int,
    bc_left: str = 'dirichlet',
    bc_right: str = 'dirichlet',
) -> np.ndarray:
    """
    ∂α_k/∂q_m = Σ_{i ∈ layer m, interior} (λ_k)_i Δx_i   (eq. 14).

    For Dirichlet BCs the RHS at boundary rows is overwritten (set to the
    boundary value), so q does not appear there and those cells are excluded.
    """
    N = len(dx)
    interior = _interior_mask(N, bc_left, bc_right)
    grads = np.zeros(n_layers)
    for m in range(n_layers):
        mask = (layer_assignment == m) & interior
        grads[m] = float(lambda_k[mask] @ dx[mask])
    return grads


def sensitivity_sigma_a(
    lambda_k: np.ndarray,
    phi: np.ndarray,
    dx: np.ndarray,
    layer_assignment: np.ndarray,
    n_layers: int,
    bc_left: str = 'dirichlet',
    bc_right: str = 'dirichlet',
) -> np.ndarray:
    """
    ∂α_k/∂Σa_m = -Σ_{i ∈ layer m, interior} (λ_k)_i φ_i Δx_i   (eq. 15).

    Dirichlet rows are excluded because A[0,0] = A[N-1,N-1] = 1 regardless
    of Σa, so ∂A[bc_row, bc_row]/∂Σa_m = 0.
    """
    N = len(dx)
    interior = _interior_mask(N, bc_left, bc_right)
    grads = np.zeros(n_layers)
    for m in range(n_layers):
        mask = (layer_assignment == m) & interior
        grads[m] = -float((lambda_k[mask] * phi[mask]) @ dx[mask])
    return grads


def sensitivity_D(
    lambda_k: np.ndarray,
    phi: np.ndarray,
    dx: np.ndarray,
    D_cell: np.ndarray,
    layer_assignment: np.ndarray,
    n_layers: int,
    bc_left: str = 'dirichlet',
    bc_right: str = 'dirichlet',
) -> np.ndarray:
    """
    ∂α_k/∂D_m via analytic face-coupling derivatives (Section 15, eq. 13).

    For face f between cells i and i+1 (d = D/dx, half-cell conductance):

        ∂β_f/∂D_i     = 2 d_{i+1}^2 / (d_i + d_{i+1})^2 / dx_i
        ∂β_f/∂D_{i+1} = 2 d_i^2     / (d_i + d_{i+1})^2 / dx_{i+1}

    Contributions to rows i and i+1 are included only when those rows are
    interior (not overwritten by Dirichlet BC).

    Derivation: ∂α_k/∂D_m = -λ_k^T (∂A/∂D_m) φ  (∂s/∂D_m = 0 for Dirichlet).
    """
    N = len(dx)
    interior = _interior_mask(N, bc_left, bc_right)
    d = D_cell / dx   # half-cell conductances (N,)

    # Accumulate λ^T (∂A/∂D_m) φ per layer; negate at the end.
    grads = np.zeros(n_layers)

    for f in range(N - 1):
        i, ip1 = f, f + 1
        sum_d = d[i] + d[ip1]
        if sum_d == 0.0:
            continue
        denom = sum_d ** 2

        dbeta_dDi   = 2.0 * d[ip1] ** 2 / denom / dx[i]
        dbeta_dDip1 = 2.0 * d[i]   ** 2 / denom / dx[ip1]

        # λ^T contributions at rows i and i+1 (zero if Dirichlet row)
        lam_i   = lambda_k[i]   if interior[i]   else 0.0
        lam_ip1 = lambda_k[ip1] if interior[ip1] else 0.0
        dphi = phi[i] - phi[ip1]

        # Each face contributes to the layer of cell i and the layer of cell i+1
        lam_diff = lam_i - lam_ip1
        grads[layer_assignment[i]]   += dbeta_dDi   * dphi * lam_diff
        grads[layer_assignment[ip1]] += dbeta_dDip1 * dphi * lam_diff

    return -grads   # ∂α_k/∂D_m = -λ^T (∂A/∂D_m) φ


def compute_full_gradient(
    lambda_k: np.ndarray,
    phi: np.ndarray,
    rhs: np.ndarray,
    dx: np.ndarray,
    D_cell: np.ndarray,
    layer_assignment: np.ndarray,
    n_layers: int,
    bc_left: str = 'dirichlet',
    bc_right: str = 'dirichlet',
) -> np.ndarray:
    """
    Full gradient ∂α_k/∂µ where µ = [D_1…D_n | Σa_1…Σa_n | q_1…q_n].

    Returns
    -------
    grad : (3*n_layers,)  ordered as [∂/∂D, ∂/∂Σa, ∂/∂q] matching DiffusionParamSampler.
    """
    dD  = sensitivity_D(lambda_k, phi, dx, D_cell, layer_assignment, n_layers, bc_left, bc_right)
    dSa = sensitivity_sigma_a(lambda_k, phi, dx, layer_assignment, n_layers, bc_left, bc_right)
    dq  = sensitivity_q(lambda_k, dx, layer_assignment, n_layers, bc_left, bc_right)
    return np.concatenate([dD, dSa, dq])
