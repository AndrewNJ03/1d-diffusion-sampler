"""
params_combined.py — Parameter sampling for the fully-coupled 5-parameter
study, mu = (void_thickness, steepness, D_teflon, Sigma_a_teflon, q_teflon).

Combines params.GeometryParamSampler's two switch parameters with three
material properties of the Teflon shell region -- previously fixed
constants in void_teflon_geometry's mat_teflon tuple -- promoted to
sampled dimensions. This reconstructs, as one combined 5D design, exactly
the "5 parameters" enumerated one-at-a-time in
gpr_vs_gegpr_comparison/README.md:

    void_thickness, steepness            (geometry, void_teflon_geometry/)
    D, Sigma_a, q  (of the Teflon shell)  (material, here relabeled *_teflon)

As with GeometryParamSampler/DiffusionParamSampler, any entry is held fixed
simply by giving it degenerate (lo == hi) bounds -- there is no separate
fixed-vs-active flag.

Public API
----------
CombinedParamSampler
    Bounds container + LHS / random sampling for the 5D mu vector.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from lhs_generation import latin_hypercube_sample  # noqa: E402


class CombinedParamSampler:
    """
    LHS sampler for the combined geometry + material parameter vector

        mu = (void_thickness, steepness, D_teflon, Sigma_a_teflon, q_teflon),
        p = 5

    Any entry can be held fixed by giving it degenerate bounds (lo == hi).

    Parameters
    ----------
    void_bounds, steepness_bounds : (lo, hi)
        Same meaning as params.GeometryParamSampler.
    d_teflon_bounds, sigma_a_teflon_bounds, q_teflon_bounds : (lo, hi)
        Bounds on the Teflon region's (D, Sigma_a, q) -- these replace the
        fixed mat_teflon constants used by the single-parameter geometry
        studies.
    """

    def __init__(
        self,
        void_bounds,
        steepness_bounds,
        d_teflon_bounds,
        sigma_a_teflon_bounds,
        q_teflon_bounds,
    ):
        self.bounds = np.asarray(
            [void_bounds, steepness_bounds, d_teflon_bounds,
             sigma_a_teflon_bounds, q_teflon_bounds],
            dtype=float,
        )
        self.p = 5

        self._fixed_mask = (self.bounds[:, 1] - self.bounds[:, 0]) == 0.0
        self._fixed_value = self.bounds[:, 0]

    @property
    def param_names(self):
        return ["void_thickness", "steepness", "D_teflon", "Sigma_a_teflon", "q_teflon"]

    @property
    def active_names(self):
        """Names of the non-degenerate (actually sampled) parameters."""
        return [n for n, fixed in zip(self.param_names, self._fixed_mask) if not fixed]

    def sample(self, n_samples: int, random_state=None) -> np.ndarray:
        """LHS design, shape (n_samples, 5). Fixed dims come back exact."""
        bounds_lhs = self.bounds.copy()
        bounds_lhs[self._fixed_mask, 1] += 1.0   # dummy non-zero width for LHS
        X = latin_hypercube_sample(n_samples, bounds_lhs, random_state=random_state)
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def sample_random(self, n_samples: int, random_state=None) -> np.ndarray:
        """Uniform random design, shape (n_samples, 5). Fixed dims come back exact."""
        rng = np.random.default_rng(random_state)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(n_samples, self.p))
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def unpack(self, mu: np.ndarray):
        """
        Extract (void_thickness, steepness, mat_teflon) from a length-5
        parameter vector, mat_teflon = (D, Sigma_a, q) ready to pass
        straight into geometry.assign_void_teflon_properties/solve_void_teflon
        in place of a fixed constant tuple.
        """
        vt, steepness, d_t, sa_t, q_t = (float(v) for v in mu)
        return vt, steepness, (d_t, sa_t, q_t)
