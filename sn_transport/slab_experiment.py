"""
Measurement model for the layered-slab inverse problem.

Experiment: a mono-directional beam of known shape (incidence cosine mu0, set
by the experimenter) and uncertain strength J0 enters one face of

    | Shield | Dielectric | Core | Dielectric | Shield |

There is no internal source. For each mu0 the measured data are the exiting
partial currents on the left and right faces. Everything here is per unit
incident current; J0 multiplies all currents and is handled by the caller.

Forward model (sn_solver.py): exact uncollided beam + DD S_N for the collided
flux driven by the first-collision source q = Sigma_s phi_u, with vacuum
incoming flux. Collided S_N matrix A depends on theta but NOT on mu0, so one
LU factorization serves every angle, and the two exit-current functionals
need only two adjoint solves per theta for the full Jacobian at all angles:

    cost(theta) = 1 LU + K forward solves + 2 adjoint solves   (K angles).

Run this file directly to verify the adjoint (identity check + central FD).
"""

import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))
from sn_solver import (  # noqa: E402
    gauss_legendre, build_slab_mesh, assemble_system, exit_current_functionals,
    cell_average, scalar_flux, uncollided_beam,
)
from sn_adjoint import (  # noqa: E402
    solve_adjoint, dQ_dsigma_t_cell, dQ_dsigma_s_cell, dQ_dq_cell,
    per_material, dlog_phi_u_dsigma_t,
)
from slab_params import MATERIALS, DEFAULT_REGIONS, SlabParamSampler  # noqa: E402


class SlabExperiment:
    """
    Parameters
    ----------
    regions      : sequence of (material, thickness), physical left to right
    materials    : material order used by theta
    cells_per_cm : mesh density
    n_angles     : S_N order (Gauss-Legendre)
    source_side  : "left" or "right" face the beam enters through
    """

    def __init__(self, regions=DEFAULT_REGIONS, materials=MATERIALS, cells_per_cm=50,
                 n_angles=16, source_side="left"):
        if source_side not in ("left", "right"):
            raise ValueError("source_side must be 'left' or 'right'.")
        self.source_side = source_side
        self.materials = tuple(materials)
        self.n_mat = len(self.materials)
        self.sampler = SlabParamSampler(materials=self.materials)

        # Internally the beam always enters at x = 0; a right-side source mirrors the slab.
        internal = list(regions) if source_side == "left" else list(regions)[::-1]
        self.x_faces, self.h, self.mat_id, self.interfaces = build_slab_mesh(
            internal, cells_per_cm, self.materials)
        self.N = len(self.h)
        self.mu, self.w = gauss_legendre(n_angles)
        self.n_ang = n_angles
        self.g_near, self.g_far = exit_current_functionals(self.N, self.mu, self.w)
        self.T_mat = per_material(self.h, self.mat_id, self.n_mat)   # thickness per material

    # ------------------------------------------------------------------ #

    def _cells(self, theta):
        sigma_t_m, c_m = self.sampler.unpack(theta)
        return sigma_t_m, c_m, sigma_t_m[self.mat_id], (c_m * sigma_t_m)[self.mat_id]

    def _factor(self, sigma_t, sigma_s):
        A, _ = assemble_system(self.h, sigma_t, sigma_s, np.zeros(self.N), self.mu, self.w)
        return splu(A)

    def _forward_angle(self, lu, sigma_t, sigma_s, mu0):
        """Collided solve for one beam angle. Returns dict of state."""
        phi_u, J_unc = uncollided_beam(self.h, sigma_t, mu0)
        q = sigma_s * phi_u
        rhs = np.repeat(0.5 * self.h * q, self.n_ang)
        u = lu.solve(rhs)
        e = u.reshape(self.N, self.n_ang)
        psi_bar = cell_average(e, self.mu)
        return dict(u=u, rhs=rhs, psi_bar=psi_bar, phi_c=scalar_flux(psi_bar, self.w),
                    phi_u=phi_u, J_unc=J_unc)

    def _to_physical(self, near, far):
        """(near, far) w.r.t. the source face -> (left, right)."""
        return (near, far) if self.source_side == "left" else (far, near)

    # ------------------------------------------------------------------ #

    def currents(self, theta, mu0s):
        """Exiting currents per unit incident current. Returns (K, 2): [J_left, J_right]."""
        _, _, st, ss = self._cells(theta)
        lu = self._factor(st, ss)
        out = np.empty((len(mu0s), 2))
        for k, mu0 in enumerate(mu0s):
            s = self._forward_angle(lu, st, ss, mu0)
            out[k] = self._to_physical(self.g_near @ s["u"], self.g_far @ s["u"] + s["J_unc"])
        return out

    def currents_and_jacobian(self, theta, mu0s, return_identity=False):
        """
        Exiting currents and their adjoint gradients w.r.t. all p = 2 n_mat parameters.

        Returns
        -------
        J  : (K, 2)     [J_left, J_right] per unit incident current
        dJ : (K, 2, p)  dJ/dtheta, theta = (Sigma_t per material, c per material)
        If return_identity, also the max relative adjoint-identity error |g^T u - lambda^T s| / |g^T u|.
        """
        st_m, c_m, st, ss = self._cells(theta)
        lu = self._factor(st, ss)
        lams = [solve_adjoint(lu, g).reshape(self.N, self.n_ang) for g in (self.g_near, self.g_far)]

        K = len(mu0s)
        J = np.empty((K, 2))
        dJ = np.empty((K, 2, 2 * self.n_mat))
        id_err = 0.0
        for k, mu0 in enumerate(mu0s):
            s = self._forward_angle(lu, st, ss, mu0)
            dlogphi_u = dlog_phi_u_dsigma_t(self.h, st, self.mat_id, self.n_mat, mu0)
            vals, grads = [], []
            for which, (g, lam) in enumerate(zip((self.g_near, self.g_far), lams)):
                Qc = g @ s["u"]
                id_err = max(id_err, abs(Qc - lam.ravel() @ s["rhs"]) / max(abs(Qc), 1e-300))

                dq = dQ_dq_cell(lam, self.h)
                # Sigma_t at fixed Sigma_s: collision term + attenuation of the first-collision source
                d_st = (per_material(dQ_dsigma_t_cell(lam, s["psi_bar"], self.h), self.mat_id, self.n_mat)
                        + (dq * ss * s["phi_u"]) @ dlogphi_u)
                # Sigma_s at fixed Sigma_t: scattering of collided flux + first-collision source
                d_ss = per_material(dQ_dsigma_s_cell(lam, s["phi_c"] + s["phi_u"], self.h),
                                    self.mat_id, self.n_mat)
                if which == 1:                       # far face also sees the uncollided beam
                    Qc += s["J_unc"]
                    d_st += -s["J_unc"] * self.T_mat / mu0
                # (Sigma_t, Sigma_s) -> (Sigma_t, c) with Sigma_s = c Sigma_t
                vals.append(Qc)
                grads.append(np.concatenate([d_st + c_m * d_ss, st_m * d_ss]))
            J[k] = self._to_physical(*vals)
            dJ[k] = self._to_physical(*grads)
        return (J, dJ, id_err) if return_identity else (J, dJ)


# ------------------------------------------------------------------ #
# Verification                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = "output_graphs"
    os.makedirs(out, exist_ok=True)
    mu0s = np.cos(np.deg2rad([0.0, 45.0, 80.0]))
    fd_rel = 1e-6

    for side in ("left", "right"):
        exp = SlabExperiment(source_side=side)
        names = exp.sampler.param_names
        thetas = exp.sampler.sample_random(3, random_state=7)
        print(f"Source on the {side} face  (N={exp.N} cells, S{exp.n_ang}, p={len(names)})")

        an_all, fd_all = [], []
        for t, theta in enumerate(thetas):
            J, dJ, id_err = exp.currents_and_jacobian(theta, mu0s, return_identity=True)
            fd = np.empty_like(dJ)
            for j in range(len(theta)):
                eps = fd_rel * theta[j]
                tp, tm = theta.copy(), theta.copy()
                tp[j] += eps; tm[j] -= eps
                fd[:, :, j] = (exp.currents(tp, mu0s) - exp.currents(tm, mu0s)) / (2 * eps)
            rel = np.abs(dJ - fd) / (np.abs(fd) + 1e-12 * np.abs(J)[:, :, None])
            worst = rel.max(axis=(0, 1))
            print(f"  theta #{t}: identity err = {id_err:.1e},  max FD rel err per param: "
                  + "  ".join(f"{n}={v:.1e}" for n, v in zip(names, worst)))
            an_all.append(dJ.ravel()); fd_all.append(fd.ravel())

        if side == "left":
            an, fdv = np.concatenate(an_all), np.concatenate(fd_all)
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(fdv, an, s=18, alpha=0.7)
            lim = [min(fdv.min(), an.min()), max(fdv.max(), an.max())]
            ax.plot(lim, lim, 'k--', lw=1, label='y = x')
            ax.set_xlabel(r'central FD  $\partial J/\partial\theta_j$')
            ax.set_ylabel('adjoint')
            ax.set_title(f'Adjoint vs FD: 3 samples x {len(mu0s)} angles x 2 faces x 6 params')
            ax.legend(); ax.grid(ls='--', alpha=0.4); plt.tight_layout()
            plt.savefig(f"{out}/adjoint_gradient_check.png", dpi=150)

    # Timing for one posterior evaluation
    exp = SlabExperiment()
    theta = np.array(exp.sampler.bounds.mean(axis=1))
    mu0s = np.cos(np.deg2rad(np.linspace(0, 80, 8)))
    t0 = time.perf_counter()
    for _ in range(20):
        exp.currents_and_jacobian(theta, mu0s)
    t1 = time.perf_counter()
    for _ in range(20):
        exp.currents(theta, mu0s)
    t2 = time.perf_counter()
    print(f"\nTiming (8 angles): currents+Jacobian {1e3*(t1-t0)/20:.1f} ms,  currents only {1e3*(t2-t1)/20:.1f} ms")
    print(f"Saved: {out}/adjoint_gradient_check.png")
