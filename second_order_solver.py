"""
1D one-group neutron diffusion solver
Cell-centered finite volume, second-order accurate.

PDE:  -d/dx [ D(x) dφ/dx ] + Σ_a(x) φ(x) = q(x),   x ∈ (0, L)

Boundary conditions supported:
  - Dirichlet:  φ(0) = φ_L,  φ(L) = φ_R
  - Robin:      φ(0) + a_L D(0) φ'(0) = b_L
                φ(L) + a_R D(L) φ'(L) = b_R
"""

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve


def build_mesh(L: float, N: int):
    """
    Uniform cell-centered mesh on [0, L] with N cells.

    Returns
    -------
    x_centers : (N,)   cell-center coordinates
    x_faces   : (N+1,) face coordinates  (x_faces[0]=0, x_faces[N]=L)
    dx        : (N,)   cell widths (uniform = L/N)
    """
    x_faces = np.linspace(0.0, L, N + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    dx = np.diff(x_faces)
    return x_centers, x_faces, dx


def assign_material_properties(x_centers, layer_bounds, D_layers, Sigma_a_layers, q_layers):
    """
    Assign piecewise-constant material properties to each cell.

    Parameters
    ----------
    x_centers    : (N,) cell-center coordinates
    layer_bounds : (N_mat+1,) layer boundary positions, e.g. [0, x1, x2, L]
    D_layers     : (N_mat,) diffusion coefficient per layer
    Sigma_a_layers : (N_mat,) absorption cross section per layer
    q_layers     : (N_mat,) volumetric source per layer

    Returns
    -------
    D, Sigma_a, q : (N,) arrays of cell-wise material properties
    """
    N = len(x_centers)
    D       = np.empty(N)
    Sigma_a = np.empty(N)
    q       = np.empty(N)

    for i, xc in enumerate(x_centers):
        # find which layer this cell center belongs to
        m = np.searchsorted(layer_bounds[1:], xc, side='right')
        m = min(m, len(D_layers) - 1)   # clamp to last layer for cells at x=L
        D[i]       = D_layers[m]
        Sigma_a[i] = Sigma_a_layers[m]
        q[i]       = q_layers[m]

    return D, Sigma_a, q


def _interface_beta(D, dx):
    """
    Compute face-centered β_{i+1/2} for i = 0 … N-2
    using the harmonic mean of half-cells.

        d_i       = D_i / dx_i
        β_{i+1/2} = 2 d_i d_{i+1} / (d_i + d_{i+1})

    Returns
    -------
    beta : (N-1,) array  (beta[i] corresponds to face x_{i+1/2})
    """
    d = D / dx                                       # (N,)  half-cell conductances
    beta = 2.0 * d[:-1] * d[1:] / (d[:-1] + d[1:])   # harmonic mean, (N-1,)
    return beta


def assemble_system(dx, D, Sigma_a, q,
                    bc_left=('dirichlet', 0.0),
                    bc_right=('dirichlet', 0.0)):
    """
    Assemble the linear system  A φ = rhs  for the 1D diffusion equation.

    Interior equation (i = 1 … N-2, 0-indexed):
      -β_{i+1/2}(φ_{i+1} - φ_i) + β_{i-1/2}(φ_i - φ_{i-1})
        + Σ_a,i φ_i dx_i = q_i dx_i

    Boundary conditions
    -------------------
    Dirichlet: ('dirichlet', value)
        Equation: φ_0 = value   (or φ_{N-1} = value)
        Implemented by modifying the first (last) row of A and rhs.

    Robin: ('robin', a, b)
        φ + a D φ' = b  at the boundary face.
        Discretised using a one-sided first-order difference for φ' at the face.

        Left boundary (face at x=0, between virtual face and cell 0):
          φ_0 + a_L D_0 (φ_0 - φ_ghost) / (dx_0/2) = b_L
          with flux continuity: J_0 = β_0_left (φ_0 - φ_ghost)
          → substitute into cell 0 balance.

        The Robin BC at the left face gives:
          β_{-1/2} = D_0 / (dx_0/2)        (boundary half-cell conductance)
          φ_ghost  = (b_L - φ_0) * dx_0 / (2 a_L D_0) + φ_0
                   solved to eliminate the ghost value.

        Resulting left-face contribution to cell 0:
          J_{-1/2} = (φ_0 - b_L) / a_L     (when a_L ≠ 0)
          β_left_eff φ_0  side → adds  (1/a_L) to diagonal, (b_L/a_L) to rhs.

    Parameters
    ----------
    dx       : (N,) cell widths
    D        : (N,) diffusion coefficients
    Sigma_a  : (N,) absorption cross sections
    q        : (N,) volumetric sources
    bc_left  : tuple  ('dirichlet', val) or ('robin', a, b)
    bc_right : tuple  ('dirichlet', val) or ('robin', a, b)

    Returns
    -------
    A   : (N, N) sparse CSR matrix
    rhs : (N,) right-hand side vector
    """
    N    = len(dx)
    beta = _interface_beta(D, dx)           # (N-1,) interior face conductances

    # ------------------------------------------------------------------ #
    # Build tridiagonal part for interior cells i = 1 … N-2              #
    # ------------------------------------------------------------------ #
    # Diagonal contribution from interior faces:
    A   = lil_matrix((N, N))
    rhs = q * dx                             # (N,) source term

    # --- interior rows i = 1 … N-2 ---
    for i in range(1, N - 1):
        A[i, i - 1] = -beta[i - 1]
        A[i, i]     =  beta[i - 1] + beta[i] + Sigma_a[i] * dx[i]
        A[i, i + 1] = -beta[i]

    # ------------------------------------------------------------------ #
    # Boundary rows                                                        #
    # ------------------------------------------------------------------ #
    # --- Left boundary (row 0) ---
    bc_type_L = bc_left[0].lower()
    if bc_type_L == 'dirichlet':
        phi_L = bc_left[1]
        A[0, 0] = 1.0
        rhs[0]  = phi_L

    elif bc_type_L == 'robin':
        a_L = bc_left[1]
        b_L = bc_left[2]
        if abs(a_L) < 1e-14:
            raise ValueError("Robin a_L ≈ 0: use Dirichlet instead.")
        A[0, 0] =  beta[0] + 1.0 / a_L + Sigma_a[0] * dx[0]
        A[0, 1] = -beta[0]
        rhs[0]  = q[0] * dx[0] + b_L / a_L

    else:
        raise ValueError(f"Unknown BC type '{bc_left[0]}'. Use 'dirichlet' or 'robin'.")

    # --- Right boundary (row N-1) ---
    bc_type_R = bc_right[0].lower()
    if bc_type_R == 'dirichlet':
        phi_R = bc_right[1]
        A[N - 1, N - 1] = 1.0
        rhs[N - 1]       = phi_R

    elif bc_type_R == 'robin':
        a_R = bc_right[1]
        b_R = bc_right[2]
        if abs(a_R) < 1e-14:
            raise ValueError("Robin a_R ≈ 0: use Dirichlet instead.")
        A[N - 1, N - 2] = -beta[N - 2]
        A[N - 1, N - 1] =  beta[N - 2] + 1.0 / a_R + Sigma_a[N - 1] * dx[N - 1]
        rhs[N - 1]       = q[N - 1] * dx[N - 1] + b_R / a_R

    else:
        raise ValueError(f"Unknown BC type '{bc_right[0]}'. Use 'dirichlet' or 'robin'.")

    return A.tocsr(), rhs


def solve_diffusion(L, N, layer_bounds, D_layers, Sigma_a_layers, q_layers,
                    bc_left=('dirichlet', 0.0),
                    bc_right=('dirichlet', 0.0)):
    """
    Full forward solve for the 1D diffusion equation.

    Parameters
    ----------
    L              : float   slab length
    N              : int     number of cells
    layer_bounds   : (N_mat+1,) material layer boundary positions
    D_layers       : (N_mat,) diffusion coefficients per layer
    Sigma_a_layers : (N_mat,) absorption cross sections per layer
    q_layers       : (N_mat,) volumetric sources per layer
    bc_left        : tuple   left BC specification
    bc_right       : tuple   right BC specification

    Returns
    -------
    x_centers : (N,) cell-center coordinates
    phi       : (N,) scalar flux solution
    """
    x_centers, x_faces, dx = build_mesh(L, N)
    D, Sigma_a, q = assign_material_properties(
        x_centers, layer_bounds, D_layers, Sigma_a_layers, q_layers
    )
    A, rhs = assemble_system(dx, D, Sigma_a, q, bc_left, bc_right)
    phi = spsolve(A, rhs)
    return x_centers, phi


# ------------------------------------------------------------------ #
# Quick verification                                                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # --- Homogeneous slab, Dirichlet BCs, uniform source ---
    # Manufactured solution for D=1, Σ_a=1, q=1, φ(0)=φ(L)=0:
    #   φ(x) = (1 - cosh(x - L/2) / cosh(L/2)) / Σ_a
    L = 10.0
    N = 200

    D_val     = 1.0
    Sigma_val = 1.0
    q_val     = 1.0

    layer_bounds   = [0.0, L]
    D_layers       = [D_val]
    Sigma_a_layers = [Sigma_val]
    q_layers       = [q_val]

    x, phi = solve_diffusion(
        L, N, layer_bounds, D_layers, Sigma_a_layers, q_layers,
        bc_left=('dirichlet', 0.0),
        bc_right=('dirichlet', 0.0),
    )

    # Manufactured solution
    kappa   = np.sqrt(Sigma_val / D_val)
    phi_exact = (q_val / Sigma_val) * (1.0 - np.cosh(kappa * (x - L / 2)) / np.cosh(kappa * L / 2))

    l2_err = np.linalg.norm(phi - phi_exact) / np.linalg.norm(phi_exact)
    print(f"Homogeneous slab — relative L2 error: {l2_err:.3e}  (N={N})")

    plt.figure(figsize=(8, 4))
    plt.plot(x, phi_exact, 'k-',  lw=2,   label='Exact')
    plt.plot(x, phi,       'r--', lw=1.5, label=f'FD (N={N})')
    plt.xlabel('x'); plt.ylabel('φ(x)')
    plt.title('1D diffusion — homogeneous slab verification')
    plt.legend(); plt.tight_layout()
    plt.savefig('output_graphs/verification_homogeneous.png', dpi=150)
    plt.show()

    # --- 3-layer slab ---
    L = 10.0
    N = 200

    layer_bounds   = [0.0, L/3, 2*L/3, L]
    D_layers       = [1.0,  0.5,  1.5]
    Sigma_a_layers = [0.1,  1.0,  0.2]
    q_layers       = [1.0,  0.0,  0.5]

    x3, phi3 = solve_diffusion(
        L, N, layer_bounds, D_layers, Sigma_a_layers, q_layers,
        bc_left=('dirichlet', 0.0),
        bc_right=('dirichlet', 0.0),
    )

    plt.figure(figsize=(8, 4))
    plt.plot(x3, phi3, 'b-', lw=2)
    for xb in layer_bounds[1:-1]:
        plt.axvline(xb, color='gray', ls='--', lw=1, label='layer boundary')
    plt.xlabel('x'); plt.ylabel('φ(x)')
    plt.title('1D diffusion — 3-layer slab')
    plt.tight_layout()
    plt.savefig('output_graphs/verification_3layer.png', dpi=150)
    plt.show()
