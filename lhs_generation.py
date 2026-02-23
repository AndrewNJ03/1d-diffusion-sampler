"""
Latin Hypercube Sampling (LHS) generator

Parameter vector:
    g = (D_1, ..., D_Nmat,  Σa_1, ..., Σa_Nmat,  q_1, ..., q_Nmat)^T
    p = 3 * Nmat

Public API
----------
latin_hypercube_sample(n_samples, bounds, random_state)
    Core LHS: returns an (n_samples, p) array scaled to the supplied bounds.

DiffusionParamSampler
    Convenience class that stores per-parameter bounds, calls the core LHS,
    and unpacks the result into the arrays expected by solve_diffusion().
"""

import numpy as np


# ------------------------------------------------------------------ #
# Core LHS routine                                                     #
# ------------------------------------------------------------------ #

def latin_hypercube_sample(
    n_samples: int,
    bounds: np.ndarray,
    random_state=None,
) -> np.ndarray:
    """
    Generate a Latin Hypercube sample.

    Each dimension is divided into `n_samples` equal-width strata.
    One point is drawn uniformly at random from each stratum, and
    the strata assignments are independently shuffled across dimensions,
    giving a space-filling design with exact one-sample-per-stratum
    marginal coverage.

    Parameters
    ----------
    n_samples : int
        Number of sample points M.
    bounds : array-like, shape (p, 2)
        Lower and upper bounds for each of the p parameters.
        bounds[j] = [lower_j, upper_j].
    random_state : int, numpy.random.Generator, or None
        Seed or generator for reproducibility.

    Returns
    -------
    X : ndarray, shape (n_samples, p)
        LHS design matrix.  Row i is sample g^(i).
    """
    rng = np.random.default_rng(random_state)
    bounds = np.asarray(bounds, dtype=float)

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (p, 2).")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Each upper bound must be strictly greater than its lower bound.")

    p = bounds.shape[0]
    M = n_samples

    # Unit-hypercube LHS in [0, 1)^p
    # For each dimension: sample a jitter in [0, 1/M) per stratum,
    # then shuffle stratum order independently.
    strata = np.arange(M, dtype=float)                    # (M,)
    X_unit = np.empty((M, p))
    for j in range(p):
        perm = rng.permutation(M)                         # shuffle strata
        jitter = rng.uniform(0.0, 1.0, size=M)           # one jitter per stratum
        X_unit[:, j] = (strata[perm] + jitter) / M       # scaled to [0, 1)

    # Scale to [lower_j, upper_j]
    lo = bounds[:, 0]                                     # (p,)
    hi = bounds[:, 1]                                     # (p,)
    X = lo + X_unit * (hi - lo)

    return X


# ------------------------------------------------------------------ #
# Convenience class for the diffusion problem                          #
# ------------------------------------------------------------------ #

class DiffusionParamSampler:
    """
    LHS sampler for the 1D diffusion parameter vector.

    The parameter vector:
        g = (D_1,...,D_Nmat, Σa_1,...,Σa_Nmat, q_1,...,q_Nmat)

    Parameters
    ----------
    n_layers : int
        Number of material layers N_mat.
    D_bounds : array-like, shape (n_layers, 2) or (2,)
        Bounds [lo, hi] for the diffusion coefficient of each layer.
        If shape (2,) the same bounds are used for all layers.
    sigma_a_bounds : array-like, shape (n_layers, 2) or (2,)
        Bounds for Σa per layer.
    q_bounds : array-like, shape (n_layers, 2) or (2,)
        Bounds for the volumetric source q per layer.
        To fix a layer's source, set lo = hi = value (degenerate distribution).
    """

    def __init__(self, n_layers, D_bounds, sigma_a_bounds, q_bounds):
        self.n_layers = n_layers
        p = 3 * n_layers

        # Expand scalar bounds to per-layer arrays
        def _broadcast(b, name):
            b = np.asarray(b, dtype=float)
            if b.shape == (2,):
                b = np.tile(b, (n_layers, 1))
            elif b.shape != (n_layers, 2):
                raise ValueError(
                    f"{name} must have shape (2,) or ({n_layers}, 2), got {b.shape}."
                )
            return b

        D_b      = _broadcast(D_bounds,      "D_bounds")
        Siga_b   = _broadcast(sigma_a_bounds, "sigma_a_bounds")
        q_b      = _broadcast(q_bounds,       "q_bounds")

        # Stack into full bounds array shape (p, 2): [D..., Σa..., q...]
        self.bounds = np.vstack([D_b, Siga_b, q_b])  # (p, 2)
        self.p = p

        # Track degenerate (fixed) dimensions
        self._fixed_mask  = (self.bounds[:, 1] - self.bounds[:, 0]) == 0.0
        self._fixed_value = self.bounds[:, 0]

    @property
    def param_names(self):
        """Readable name for each of the p parameters."""
        names = []
        for m in range(1, self.n_layers + 1):
            names.append(f"D_{m}")
        for m in range(1, self.n_layers + 1):
            names.append(f"Sigma_a_{m}")
        for m in range(1, self.n_layers + 1):
            names.append(f"q_{m}")
        return names

    def sample(self, n_samples: int, random_state=None) -> np.ndarray:
        """
        Draw n_samples LHS points from the parameter space.

        Parameters
        ----------
        n_samples : int
            Number of samples M.
        random_state : int or numpy.random.Generator, optional

        Returns
        -------
        X : ndarray, shape (n_samples, p)
            Design matrix; row i is g^(i).
            Columns: [D_1,...,D_Nmat, Σa_1,...,Σa_Nmat, q_1,...,q_Nmat].
        """
        # For fixed dims, temporarily widen bounds slightly so LHS doesn't
        # receive zero-width intervals, then overwrite afterwards.
        bounds_lhs = self.bounds.copy()
        bounds_lhs[self._fixed_mask, 1] += 1.0  # dummy non-zero width

        X = latin_hypercube_sample(n_samples, bounds_lhs, random_state=random_state)

        # Restore exact fixed values
        X[:, self._fixed_mask] = self._fixed_value[self._fixed_mask]
        return X

    def unpack(self, mu: np.ndarray):
        """
        Unpack a single parameter vector g (shape (p,)) into the three
        layer-property arrays expected by solve_diffusion().

        Returns
        -------
        D_layers       : (n_layers,) diffusion coefficients
        Sigma_a_layers : (n_layers,) absorption cross sections
        q_layers       : (n_layers,) volumetric sources
        """
        n = self.n_layers
        D_layers       = mu[0:n]
        Sigma_a_layers = mu[n:2*n]
        q_layers       = mu[2*n:3*n]
        return D_layers, Sigma_a_layers, q_layers

    def unpack_batch(self, X: np.ndarray):
        """
        Unpack design matrix X (shape M×p) into three (M, n_layers) arrays.

        Returns
        -------
        D_mat       : (M, n_layers)
        Sigma_a_mat : (M, n_layers)
        q_mat       : (M, n_layers)
        """
        n = self.n_layers
        return X[:, 0:n], X[:, n:2*n], X[:, 2*n:3*n]


# ------------------------------------------------------------------ #
# Quick demonstration                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- Configuration ----
    N_MAT  = 3     # number of material layers
    M      = 200   # training samples
    SEED   = 42

    # Physical parameter bounds
    D_bounds       = [0.2, 2.0]   # diffusion coefficient  [cm]
    sigma_a_bounds = [0.05, 2.0]  # absorption cross section [cm^-1]
    q_bounds       = [0.0, 2.0]   # volumetric source [n cm^-3 s^-1]

    sampler = DiffusionParamSampler(N_MAT, D_bounds, sigma_a_bounds, q_bounds)

    print(f"Parameter names ({sampler.p} total): {sampler.param_names}")
    print(f"Bounds:\n{sampler.bounds}")

    X_train = sampler.sample(M, random_state=SEED)
    print(f"\nTraining design matrix shape: {X_train.shape}")
    print(f"Sample 0 (g^(0)): {X_train[0]}")

    # ---- Verify marginal coverage (each stratum hit exactly once) ----
    M_check = 50
    X_check = latin_hypercube_sample(M_check, sampler.bounds, random_state=0)
    for j in range(sampler.p):
        lo, hi = sampler.bounds[j]
        if hi == lo:
            continue
        scaled = (X_check[:, j] - lo) / (hi - lo)   # map to [0,1)
        stratum_idx = np.floor(scaled * M_check).astype(int)
        assert len(np.unique(stratum_idx)) == M_check, \
            f"LHS marginal coverage failed for parameter {j}"
    print("\nMarginal coverage check passed: each stratum occupied exactly once per dimension.")

    # ---- Plot pairwise scatter for the first 4 parameters ----
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.ravel()
    pairs = [(0, 3), (0, 6), (3, 6), (1, 4)]   # (D_1,Σa_1), (D_1,q_1), (Σa_1,q_1), (D_2,Σa_2)
    names = sampler.param_names

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(X_train[:, i], X_train[:, j], s=8, alpha=0.6, color='steelblue')
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_title(f"{names[i]} vs {names[j]}")

    fig.suptitle(f"LHS design  (M={M}, N_mat={N_MAT})", fontsize=13)
    plt.tight_layout()
    plt.savefig("output_graphs/lhs_pairwise_scatter.png", dpi=150)
    print("Saved: output_graphs/lhs_pairwise_scatter.png")

    # ---- Plot marginal histograms for all 9 parameters ----
    fig2, axes2 = plt.subplots(3, 3, figsize=(10, 8))
    for k, ax in enumerate(axes2.ravel()):
        ax.hist(X_train[:, k], bins=20, color='steelblue', edgecolor='white', linewidth=0.4)
        ax.set_xlabel(names[k])
        ax.set_ylabel("count")
    fig2.suptitle(f"LHS marginal distributions  (M={M})", fontsize=13)
    plt.tight_layout()
    plt.savefig("output_graphs/lhs_marginals.png", dpi=150)
    print("Saved: output_graphs/lhs_marginals.png")

    plt.show()
