"""
1D one-group discrete-ordinates (S_N) transport solver for a layered slab.

PDE (isotropic scattering and source):

    mu dpsi/dx + Sigma_t(x) psi(x, mu) = ( Sigma_s(x) phi(x) + q(x) ) / 2,
    phi(x) = int_{-1}^{1} psi(x, mu) dmu  ~=  sum_n w_n psi_n(x)

Angle : Gauss-Legendre S_N (weights sum to 2).
Space : diamond difference (DD) on a cell-centered mesh whose faces sit on
        every region interface.
BCs   : prescribed incoming angular flux psi_inc[n] on the inflow boundary of
        each direction (left for mu_n > 0, right for mu_n < 0); zero = vacuum.

Unknowns
--------
For each cell i and direction n the unknown is the OUTFLOW edge flux
e[i, n] (right face of cell i if mu_n > 0, left face if mu_n < 0). The inflow
edge is the upwind cell's outflow edge, or psi_inc[n] on the boundary. The DD
cell balance for row (i, n), with a_n = |mu_n| and b_i = Sigma_t,i h_i / 2, is

    (a_n + b_i) e[i,n] + (b_i - a_n) e_in[i,n]  -  (h_i Sigma_s,i / 2) phi_i  =  h_i q_i / 2,
    phi_i = sum_m w_m (e[i,m] + e_in[i,m]) / 2 .

Stacking every row gives one sparse linear system  A u = s  (u = e.ravel(),
cell-major: index i*N_ang + n), the transport analogue of assemble_system()
in common/second_order_solver.py. Sigma_t, Sigma_s and q all enter A and s
linearly and cell-locally, which is what makes the adjoint sensitivities in
sn_adjoint.py simple. The same discretization is also solved the classical
way, by transport sweeps inside source iteration, as an independent check.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


# ------------------------------------------------------------------ #
# Angular quadrature and mesh                                        #
# ------------------------------------------------------------------ #

def gauss_legendre(n_angles: int):
    """S_N Gauss-Legendre directions (ascending) and weights (sum = 2)."""
    if n_angles % 2:
        raise ValueError("n_angles must be even (no mu = 0 direction).")
    return np.polynomial.legendre.leggauss(n_angles)


def build_slab_mesh(regions, cells_per_cm: float, materials):
    """
    Mesh with faces on every region interface.

    Parameters
    ----------
    regions      : sequence of (material_name, thickness)
    cells_per_cm : target cell density (each region gets round(t * density) >= 1 cells)
    materials    : tuple of material names; mat_id indexes into it

    Returns
    -------
    x_faces    : (N+1,) face coordinates
    h          : (N,)   cell widths
    mat_id     : (N,)   material index per cell
    interfaces : (n_regions+1,) region boundary coordinates
    """
    faces = [0.0]
    mat_id = []
    for name, t in regions:
        n = max(1, int(round(t * cells_per_cm)))
        x0 = faces[-1]
        faces.extend(x0 + t * np.arange(1, n + 1) / n)
        mat_id.extend([materials.index(name)] * n)
    x_faces = np.asarray(faces)
    interfaces = np.concatenate([[0.0], np.cumsum([t for _, t in regions])])
    return x_faces, np.diff(x_faces), np.asarray(mat_id), interfaces


# ------------------------------------------------------------------ #
# Discretization helpers                                             #
# ------------------------------------------------------------------ #

def _upwind_index(N: int, mu: np.ndarray) -> np.ndarray:
    """(N, N_ang) index of the upwind cell for each (i, n); -1 on the inflow boundary."""
    i = np.arange(N)[:, None]
    up = np.where(mu[None, :] > 0, i - 1, i + 1)
    up[(up < 0) | (up >= N)] = -1
    return up


def _psi_inc_or_zero(psi_inc, n_ang):
    return np.zeros(n_ang) if psi_inc is None else np.asarray(psi_inc, dtype=float)


def inflow_edges(e: np.ndarray, mu: np.ndarray, psi_inc=None) -> np.ndarray:
    """(N, N_ang) inflow edge flux for every cell and direction."""
    N, n_ang = e.shape
    psi_inc = _psi_inc_or_zero(psi_inc, n_ang)
    up = _upwind_index(N, mu)
    e_up = e[np.clip(up, 0, N - 1), np.arange(n_ang)[None, :]]
    return np.where(up >= 0, e_up, psi_inc[None, :])


def cell_average(e: np.ndarray, mu: np.ndarray, psi_inc=None) -> np.ndarray:
    """DD cell-averaged angular flux, (N, N_ang)."""
    return 0.5 * (e + inflow_edges(e, mu, psi_inc))


def scalar_flux(psi_bar: np.ndarray, w: np.ndarray) -> np.ndarray:
    return psi_bar @ w


# ------------------------------------------------------------------ #
# Assembled system  A u = s                                          #
# ------------------------------------------------------------------ #

def assemble_system(h, sigma_t, sigma_s, q, mu, w, psi_inc=None):
    """
    Assemble the DD S_N system A u = s (see module docstring).

    Parameters
    ----------
    h, sigma_t, sigma_s, q : (N,) cell widths and cell-wise data
    mu, w                  : (N_ang,) quadrature
    psi_inc                : (N_ang,) incoming boundary flux per direction (None = vacuum)

    Returns
    -------
    A   : (N*N_ang, N*N_ang) sparse CSC matrix
    rhs : (N*N_ang,)
    """
    N, n_ang = len(h), len(mu)
    psi_inc = _psi_inc_or_zero(psi_inc, n_ang)

    a = np.abs(mu)[None, :]                          # (1, N_ang)
    b = (0.5 * sigma_t * h)[:, None]                 # (N, 1)
    sc = (0.25 * h * sigma_s)[:, None]               # (N, 1): (h Sigma_s / 2) * (1/2 of the DD average)
    idx = np.arange(N * n_ang).reshape(N, n_ang)
    up = _upwind_index(N, mu)
    interior = up >= 0
    up_idx = np.where(interior, up * n_ang + np.arange(n_ang)[None, :], -1)

    rhs = np.broadcast_to((0.5 * h * q)[:, None], (N, n_ang)).copy()
    rows, cols, vals = [], [], []

    # Streaming + collision: outflow edge
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append((a + b).ravel())

    # Streaming + collision: inflow edge (boundary value moves to the RHS)
    coef_in = b - a                                  # (N, N_ang)
    rows.append(idx[interior]); cols.append(up_idx[interior]); vals.append(coef_in[interior])
    rhs[~interior] -= coef_in[~interior] * np.broadcast_to(psi_inc, (N, n_ang))[~interior]

    # Scattering: row (i, n) couples to both edges of cell i for every direction m
    r3 = np.broadcast_to(idx[:, :, None], (N, n_ang, n_ang))           # [i, n, m] -> row (i, n)
    c_out = np.broadcast_to(idx[:, None, :], (N, n_ang, n_ang))        # -> col (i, m)
    c_in = np.broadcast_to(up_idx[:, None, :], (N, n_ang, n_ang))      # -> col (up_m(i), m)
    v3 = np.broadcast_to(-sc[:, :, None] * w[None, None, :], (N, n_ang, n_ang))
    rows.append(r3.ravel()); cols.append(c_out.ravel()); vals.append(v3.ravel())
    m_int = c_in >= 0
    rows.append(r3[m_int]); cols.append(c_in[m_int]); vals.append(v3[m_int])
    # Incident flux inside the scattering source of boundary cells
    bnd_scatter = ((~interior) * (w * psi_inc)[None, :]).sum(axis=1)  # (N,)
    rhs += (sc[:, 0] * bnd_scatter)[:, None]

    A = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N * n_ang, N * n_ang),
    ).tocsc()
    return A, rhs.ravel()


def exit_current_functionals(N: int, mu: np.ndarray, w: np.ndarray):
    """
    Linear functionals on u = e.ravel() giving the exiting partial currents

        J_left  = sum_{mu_n < 0} w_n |mu_n| psi_n(0),   J_right = sum_{mu_n > 0} w_n mu_n psi_n(L).

    Returns g_left, g_right, each (N*N_ang,).
    """
    n_ang = len(mu)
    g_left = np.zeros(N * n_ang)
    g_right = np.zeros(N * n_ang)
    neg, pos = mu < 0, mu > 0
    g_left[np.arange(n_ang)[neg]] = w[neg] * np.abs(mu[neg])                     # cell 0
    g_right[(N - 1) * n_ang + np.arange(n_ang)[pos]] = w[pos] * mu[pos]          # cell N-1
    return g_left, g_right


# ------------------------------------------------------------------ #
# Classical solve: transport sweeps + source iteration               #
# ------------------------------------------------------------------ #

def sweep(h, sigma_t, S, mu, psi_inc=None) -> np.ndarray:
    """
    One DD transport sweep for an isotropic emission density S (N,), i.e. the
    full right-hand side (Sigma_s phi + q)/2. Returns outflow edges e (N, N_ang).
    """
    N, n_ang = len(h), len(mu)
    psi_inc = _psi_inc_or_zero(psi_inc, n_ang)
    e = np.empty((N, n_ang))
    for sel, cells in ((mu > 0, range(N)), (mu < 0, range(N - 1, -1, -1))):
        a = np.abs(mu[sel])
        psi = psi_inc[sel].copy()
        for i in cells:
            b = 0.5 * sigma_t[i] * h[i]
            psi = ((a - b) * psi + h[i] * S[i]) / (a + b)
            e[i, sel] = psi
    return e


def source_iteration(h, sigma_t, sigma_s, q, mu, w, psi_inc=None, tol=1e-12, max_iter=20000):
    """
    Source iteration on the scalar flux. Converges at rate ~ max(c); slow for
    c -> 1 in optically thick regions (no acceleration is applied).

    Returns e (N, N_ang), phi (N,), n_iter.
    """
    phi = np.zeros(len(h))
    for it in range(1, max_iter + 1):
        e = sweep(h, sigma_t, 0.5 * (sigma_s * phi + q), mu, psi_inc)
        phi_new = scalar_flux(cell_average(e, mu, psi_inc), w)
        change = np.max(np.abs(phi_new - phi)) / max(np.max(np.abs(phi_new)), 1e-300)
        phi = phi_new
        if change < tol:
            return e, phi, it
    raise RuntimeError(f"Source iteration did not converge in {max_iter} iterations (last change {change:.2e}).")


def solve_sn(h, sigma_t, sigma_s, q, mu, w, psi_inc=None, method="direct", **si_kwargs):
    """
    Forward S_N solve.

    method : "direct" (sparse LU of A u = s, exact for the discrete system) or
             "source_iteration" (sweeps, converged to si_kwargs['tol']).

    Returns e (N, N_ang) outflow edges, psi_bar (N, N_ang), phi (N,).
    """
    N, n_ang = len(h), len(mu)
    if method == "direct":
        A, rhs = assemble_system(h, sigma_t, sigma_s, q, mu, w, psi_inc)
        e = splu(A).solve(rhs).reshape(N, n_ang)
    elif method == "source_iteration":
        e, _, _ = source_iteration(h, sigma_t, sigma_s, q, mu, w, psi_inc, **si_kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'.")
    psi_bar = cell_average(e, mu, psi_inc)
    return e, psi_bar, scalar_flux(psi_bar, w)


# ------------------------------------------------------------------ #
# Exact uncollided beam                                              #
# ------------------------------------------------------------------ #

def _E(a):
    """E(a) = (1 - exp(-a)) / a, the cell-average attenuation factor (E(0) = 1)."""
    a = np.asarray(a, dtype=float)
    out = np.ones_like(a)
    nz = a > 0
    out[nz] = -np.expm1(-a[nz]) / a[nz]
    return out


def uncollided_beam(h, sigma_t, mu0: float, J0: float = 1.0):
    """
    Uncollided flux of a mono-directional beam with incident current J0 entering
    at x = 0 in direction mu0 in (0, 1]:

        psi_u(x, mu) = (J0 / mu0) exp(-tau(x)/mu0) delta(mu - mu0),   tau(x) = int_0^x Sigma_t.

    The cell average of phi_u is exact (no spatial discretization error):

        phi_u,i = (J0/mu0) exp(-tau_{i-1/2}/mu0) E(Sigma_t,i h_i / mu0).

    Returns phi_u (N,) and the uncollided current transmitted through x = L.
    """
    if not (0.0 < mu0 <= 1.0):
        raise ValueError("mu0 must be in (0, 1].")
    tau_cum = np.cumsum(sigma_t * h)
    tau_left = np.concatenate([[0.0], tau_cum[:-1]])
    phi_u = (J0 / mu0) * np.exp(-tau_left / mu0) * _E(sigma_t * h / mu0)
    J_trans = J0 * np.exp(-tau_cum[-1] / mu0)
    return phi_u, J_trans


# ------------------------------------------------------------------ #
# Verification                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from slab_params import MATERIALS, DEFAULT_REGIONS, DEFAULT_THETA_TRUE, SlabParamSampler

    out = "output_graphs"
    os.makedirs(out, exist_ok=True)
    mu, w = gauss_legendre(16)

    # --- 1. Pure absorber, uniform source: exact cell averages at fixed quadrature ---
    # psi_n(x) = q/(2 St) (1 - exp(-St d/|mu_n|)), d = distance from the inflow boundary.
    L, St, qv = 4.0, 1.0, 1.0
    print("1. Pure absorber, uniform source, vacuum BCs (spatial convergence of DD)")
    errs, Ns = [], [20, 40, 80, 160, 320, 640]
    for N in Ns:
        xf = np.linspace(0.0, L, N + 1); h = np.diff(xf)
        _, _, phi = solve_sn(h, np.full(N, St), np.zeros(N), np.full(N, qv), mu, w)
        dl = np.where(mu[None, :] > 0, xf[:-1, None], L - xf[1:, None])   # inflow-side distance
        dr = dl + h[:, None]
        am = np.abs(mu)[None, :]
        psi_ex = qv / (2 * St) * (1 - am / (St * h[:, None]) * (np.exp(-St * dl / am) - np.exp(-St * dr / am)))
        errs.append(np.linalg.norm(phi - psi_ex @ w) / np.linalg.norm(psi_ex @ w))
    for k, N in enumerate(Ns):
        ratio = f"  ratio {errs[k-1]/errs[k]:.2f}" if k else ""
        print(f"   N={N:4d}  rel L2 err = {errs[k]:.3e}{ratio}")

    # --- 2. Sweeps + source iteration vs. direct solve (scattering, both BC types) ---
    print("\n2. Source iteration vs. direct LU (5-region slab, truth materials, beam on mu_n=max)")
    sampler = SlabParamSampler()
    st_m, c_m = sampler.unpack(np.array(DEFAULT_THETA_TRUE))
    xf, h, mat, interfaces = build_slab_mesh(DEFAULT_REGIONS, 50, MATERIALS)
    N = len(h)
    sigma_t, sigma_s = st_m[mat], (c_m * st_m)[mat]
    n0 = np.argmax(mu)
    psi_inc = np.zeros_like(mu); psi_inc[n0] = 1.0 / (w[n0] * mu[n0])   # unit incident current
    e_d, _, phi_d = solve_sn(h, sigma_t, sigma_s, np.zeros(N), mu, w, psi_inc)
    e_s, phi_s, n_it = source_iteration(h, sigma_t, sigma_s, np.zeros(N), mu, w, psi_inc, tol=1e-13)
    print(f"   SI iterations = {n_it},  max |phi_SI - phi_LU| / max phi = "
          f"{np.max(np.abs(phi_s - phi_d)) / phi_d.max():.2e}")

    # --- 3. Particle balance: J_in = J_left + J_right + absorption (exact for DD) ---
    print("\n3. Global particle balance (uncollided-split solve, mu0 = 0.6)")
    gL, gR = exit_current_functionals(N, mu, w)
    for c_test in (c_m, np.ones(3)):
        ss = (c_test * st_m)[mat]
        phi_u, J_unc = uncollided_beam(h, sigma_t, 0.6)
        e_c, _, phi_c = solve_sn(h, sigma_t, ss, ss * phi_u, mu, w)
        JL, JR = gL @ e_c.ravel(), gR @ e_c.ravel() + J_unc
        absorbed = np.sum((sigma_t - ss) * h * (phi_c + phi_u))
        print(f"   c = {np.round(c_test, 2)}:  J_L={JL:.6f}  J_R={JR:.6f}  absorbed={absorbed:.6f}  "
              f"1 - (sum) = {1 - JL - JR - absorbed:.2e}")

    # --- 4. Uncollided split vs. beam carried by the S_N quadrature itself ---
    print(f"\n4. Uncollided split vs. direct S_N beam at mu0 = mu_n = {mu[n0]:.4f} (mesh refinement)")
    for cpc in (12.5, 25, 50, 100, 200):
        xf, h, mat, _ = build_slab_mesh(DEFAULT_REGIONS, cpc, MATERIALS)
        N = len(h); st, ss = st_m[mat], (c_m * st_m)[mat]
        gL, gR = exit_current_functionals(N, mu, w)
        e_b, _, _ = solve_sn(h, st, ss, np.zeros(N), mu, w, psi_inc)
        phi_u, J_unc = uncollided_beam(h, st, mu[n0])
        e_c, _, _ = solve_sn(h, st, ss, ss * phi_u, mu, w)
        dJL = gL @ e_c.ravel() - gL @ e_b.ravel()
        dJR = gR @ e_c.ravel() + J_unc - gR @ e_b.ravel()
        print(f"   N={N:4d}  |dJ_L|={abs(dJL):.2e}  |dJ_R|={abs(dJR):.2e}")

    # --- 5. Flux shapes for the truth materials at several beam angles ---
    xf, h, mat, interfaces = build_slab_mesh(DEFAULT_REGIONS, 50, MATERIALS)
    N = len(h); st, ss = st_m[mat], (c_m * st_m)[mat]
    xc = 0.5 * (xf[:-1] + xf[1:])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, deg in enumerate((0, 45, 75)):
        mu0 = np.cos(np.deg2rad(deg))
        phi_u, _ = uncollided_beam(h, st, mu0)
        _, _, phi_c = solve_sn(h, st, ss, ss * phi_u, mu, w)
        ax.semilogy(xc, phi_c + phi_u, color=f"C{k}", lw=1.8, label=f"total, {deg} deg")
        ax.semilogy(xc, phi_u, color=f"C{k}", ls="--", lw=1.0, label=f"uncollided, {deg} deg")
    for xb in interfaces[1:-1]:
        ax.axvline(xb, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("x  [cm]"); ax.set_ylabel(r"$\phi(x)$ per unit incident current")
    ax.set_title("Scalar flux, truth materials (beam enters at x = 0)")
    ax.legend(fontsize=8, ncol=3); plt.tight_layout()
    plt.savefig(f"{out}/flux_truth_angles.png", dpi=150)
    print(f"\nSaved: {out}/flux_truth_angles.png")
