"""
Parameter sampling module for the geometry-parameterized Void/Teflon slab.

Mirrors the role of common/lhs_generation.py's DiffusionParamSampler, but
the sampled quantities are geometry / switch parameters — void_thickness
(the Void/Teflon interface location) and the sigmoid steepness — rather
than the per-layer material vector (D, Sigma_a, q). Reuses the generic
latin_hypercube_sample() core routine unchanged.

Following the workflow document's design choice (Section 3): there is no
special-cased "fixed vs. active" logic. A parameter is held fixed simply by
giving it degenerate (zero-width) bounds, exactly as DiffusionParamSampler
does for materials. So steepness is "parameterizeable" in the same sense
void_thickness always was: pass a real (lo, hi) range to make it an active,
LHS-sampled / adjoint-differentiated parameter, or lo == hi to hold it at a
constant value.

Public API
----------
GeometryParamSampler
    Bounds container + LHS / random sampling for
    mu = (void_thickness, steepness).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from lhs_generation import latin_hypercube_sample  # noqa: E402


class GeometryParamSampler:
    """
    LHS sampler for the geometry / switch parameter vector

        mu = (void_thickness, steepness),   p = 2

    Either entry can be held fixed by giving it degenerate bounds
    (lo == hi), matching DiffusionParamSampler's convention for materials.

    Parameters
    ----------
    void_bounds : (lo, hi)
        Bounds on the Void shell thickness. Must satisfy
        0 <= lo <= hi <= shell_width (the fixed Void + Teflon shell width).
    steepness_bounds : (lo, hi)
        Bounds on the sigmoid steepness `a`. lo == hi holds it fixed
        (the default, matching the original single-value behavior).
    """

    def __init__(self, void_bounds, steepness_bounds=(200.0, 200.0)):
        self.bounds = np.asarray([void_bounds, steepness_bounds], dtype=float)
        self.p = 2

        self._fixed_mask = (self.bounds[:, 1] - self.bounds[:, 0]) == 0.0
        self._fixed_value = self.bounds[:, 0]

    @property
    def param_names(self):
        return ["void_thickness", "steepness"]

    @property
    def active_names(self):
        """Names of the non-degenerate (actually sampled) parameters."""
        return [n for n, fixed in zip(self.param_names, self._fixed_mask) if not fixed]

    def sample(self, n_samples: int, random_state=None) -> np.ndarray:
        """LHS design, shape (n_samples, 2). Fixed dims come back exact."""
        bounds_lhs = self.bounds.copy()
        bounds_lhs[self._fixed_mask, 1] += 1.0   # dummy non-zero width for LHS

        X = latin_hypercube_sample(n_samples, bounds_lhs, random_state=random_state)
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def sample_random(self, n_samples: int, random_state=None) -> np.ndarray:
        """Uniform random design, shape (n_samples, 2). Fixed dims come back exact."""
        rng = np.random.default_rng(random_state)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(n_samples, self.p))
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def unpack(self, mu: np.ndarray):
        """Extract (void_thickness, steepness) from a length-2 parameter vector."""
        return float(mu[0]), float(mu[1])
