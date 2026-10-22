"""
Geometry, materials, and parameter sampling for the layered-slab inverse problem.

Slab layout (left to right), three distinct materials:

    | Shield | Dielectric | Core | Dielectric | Shield |

Parameter vector (p = 6), material order (shield, dielectric, core):

    theta = (Sigma_t_shield, Sigma_t_dielectric, Sigma_t_core,
             c_shield,       c_dielectric,       c_core)

with c = Sigma_s / Sigma_t the scattering ratio, so Sigma_s = c * Sigma_t and
Sigma_a = (1 - c) * Sigma_t. Any (Sigma_t > 0, 0 <= c <= 1) is physical, which is
why the slab is parameterized by (Sigma_t, c) rather than (Sigma_t, Sigma_s).

Fixed parameters follow the repository convention: degenerate bounds (v, v).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from lhs_generation import latin_hypercube_sample  # noqa: E402


MATERIALS = ("shield", "dielectric", "core")

# (material, thickness [cm]) from left to right
DEFAULT_REGIONS = (
    ("shield",     0.5),
    ("dielectric", 1.0),
    ("core",       1.0),
    ("dielectric", 1.0),
    ("shield",     0.5),
)

# Prior bounds per material, ordered as MATERIALS
DEFAULT_SIGMA_T_BOUNDS = ((1.0, 3.0), (0.2, 1.0), (0.5, 2.0))   # [1/cm]
DEFAULT_C_BOUNDS       = ((0.05, 0.95), (0.05, 0.95), (0.05, 0.95))

# Synthetic "truth" used by the verification and inverse-UQ drivers
DEFAULT_THETA_TRUE = (2.0, 0.6, 1.2, 0.3, 0.9, 0.5)


class SlabParamSampler:
    """
    Bounds, sampling, and unpacking for theta = (Sigma_t per material, c per material).

    Parameters
    ----------
    sigma_t_bounds : array-like, shape (n_mat, 2) or (2,)
    c_bounds       : array-like, shape (n_mat, 2) or (2,)
    materials      : tuple of material names (defines the order)
    """

    def __init__(self, sigma_t_bounds=DEFAULT_SIGMA_T_BOUNDS, c_bounds=DEFAULT_C_BOUNDS,
                 materials=MATERIALS):
        self.materials = tuple(materials)
        self.n_mat = len(self.materials)

        def _broadcast(b, name):
            b = np.asarray(b, dtype=float)
            if b.shape == (2,):
                b = np.tile(b, (self.n_mat, 1))
            elif b.shape != (self.n_mat, 2):
                raise ValueError(f"{name} must have shape (2,) or ({self.n_mat}, 2), got {b.shape}.")
            return b

        st_b = _broadcast(sigma_t_bounds, "sigma_t_bounds")
        c_b = _broadcast(c_bounds, "c_bounds")
        if np.any(st_b[:, 0] <= 0.0):
            raise ValueError("Sigma_t lower bounds must be positive.")
        if np.any(c_b[:, 0] < 0.0) or np.any(c_b[:, 1] > 1.0):
            raise ValueError("Scattering-ratio bounds must lie in [0, 1].")

        self.bounds = np.vstack([st_b, c_b])          # (p, 2)
        self.p = self.bounds.shape[0]
        self._fixed_mask = (self.bounds[:, 1] - self.bounds[:, 0]) == 0.0
        self._fixed_value = self.bounds[:, 0]

    @property
    def param_names(self):
        return ([f"Sigma_t_{m}" for m in self.materials]
                + [f"c_{m}" for m in self.materials])

    @property
    def active_idx(self):
        return np.where(~self._fixed_mask)[0]

    def sample(self, n_samples: int, random_state=None) -> np.ndarray:
        """Latin hypercube design inside the bounds (fixed columns exact)."""
        bounds_lhs = self.bounds.copy()
        bounds_lhs[self._fixed_mask, 1] += 1.0
        X = latin_hypercube_sample(n_samples, bounds_lhs, random_state=random_state)
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def sample_random(self, n_samples: int, random_state=None) -> np.ndarray:
        """I.i.d. uniform draws inside the bounds (fixed columns exact)."""
        rng = np.random.default_rng(random_state)
        X = rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_samples, self.p))
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def in_bounds(self, theta) -> bool:
        theta = np.asarray(theta)
        return bool(np.all(theta >= self.bounds[:, 0]) and np.all(theta <= self.bounds[:, 1]))

    def unpack(self, theta):
        """theta (p,) -> (sigma_t_mat, c_mat), each (n_mat,)."""
        theta = np.asarray(theta, dtype=float)
        n = self.n_mat
        return theta[:n], theta[n:2 * n]
