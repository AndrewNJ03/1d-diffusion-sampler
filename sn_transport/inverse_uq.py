"""
inverse_uq.py: Bayesian inference of (Sigma_t, c) for Shield / Dielectric / Core
from exiting currents measured at several beam angles.

Unknowns   z = (theta_active, log J0), theta = (Sigma_t x3, c x3), J0 = beam strength
Data       d_k,f = log of the measured exiting current on face f in {left, right}
           for beam angle mu0_k (K angles -> 2K data)
Model      d_k,f = log J0 + log J_f(theta; mu0_k) + eps,   eps ~ N(0, sigma_meas^2 + sigma_model^2)
           (relative measurement error plus an allowance for S_N discretization
           error; J0 enters additively in log space)
Prior      theta uniform on the sampler box, log J0 ~ N(log J0_nominal, sigma_J0^2)

The synthetic data come from a finer mesh and quadrature than the inference
model, so the inference does not reuse the exact model that made the data.

Steps
-----
1. Angle sensitivity scan at the truth: d log J_f / d log theta_j vs. beam angle.
2. Identifiability and design comparison at the truth (adjoint Jacobian only,
   no data): singular values of d log J / d log(theta, J0), and the Laplace
   posterior std for several angle sets.
3. Synthetic data from the truth (J0 drawn from its prior, lognormal noise).
4. MAP by bounded Gauss-Newton (scipy least_squares, adjoint Jacobian), multi-start from LHS.
5. Laplace approximation at the MAP.
6. Adaptive Metropolis MCMC (Haario et al. 2001), several chains started
   around the MAP, split-R-hat per parameter. Samples saved to posterior_samples.txt.

Laplace covariances replace the uniform prior box by a Gaussian of the same
variance, (hi - lo)^2 / 12. Without that, directions the data do not inform
would get infinite (or, with a pseudo-inverse, zero) variance.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
from slab_experiment import SlabExperiment  # noqa: E402
from slab_params import DEFAULT_THETA_TRUE  # noqa: E402


class LogPosterior:
    """Residuals, Jacobian and log posterior in z = (theta_active, log J0)."""

    def __init__(self, exp, mu0s, log_data, sigma, J0_nominal, sigma_J0, theta_fixed):
        self.exp, self.mu0s = exp, np.asarray(mu0s)
        self.d = np.asarray(log_data).ravel()
        self.sigma_meas, self.sigma_J0 = sigma, sigma_J0
        self.logJ0_nom = np.log(J0_nominal)
        self.theta_fixed = np.asarray(theta_fixed, dtype=float)
        self.active = exp.sampler.active_idx
        self.lo = exp.sampler.bounds[self.active, 0]
        self.hi = exp.sampler.bounds[self.active, 1]
        self.prior_sd = (self.hi - self.lo) / np.sqrt(12.0)
        self.names = [exp.sampler.param_names[j] for j in self.active] + ["log_J0"]

    def theta(self, z):
        th = self.theta_fixed.copy()
        th[self.active] = z[:-1]
        return th

    def residuals(self, z):
        J = self.exp.currents(self.theta(z), self.mu0s)
        r = (self.d - z[-1] - np.log(J).ravel()) / self.sigma_meas
        return np.append(r, (z[-1] - self.logJ0_nom) / self.sigma_J0)

    def jacobian(self, z):
        J, dJ = self.exp.currents_and_jacobian(self.theta(z), self.mu0s)
        dlog = (dJ / J[:, :, None]).reshape(-1, dJ.shape[2])[:, self.active]
        Jr = np.zeros((self.d.size + 1, len(z)))
        Jr[:-1, :-1] = -dlog / self.sigma_meas
        Jr[:-1, -1] = -1.0 / self.sigma_meas
        Jr[-1, -1] = 1.0 / self.sigma_J0
        return Jr

    def __call__(self, z):
        if np.any(z[:-1] < self.lo) or np.any(z[:-1] > self.hi):
            return -np.inf
        r = self.residuals(z)
        return -0.5 * r @ r


def laplace_cov(Jr, prior_sd_theta):
    """
    Gauss-Newton posterior covariance (Jr^T Jr + P)^{-1}, P = diag(1/prior_sd^2)
    on the theta block (log J0's Gaussian prior is already a row of Jr).
    """
    P = np.zeros(Jr.shape[1])
    P[:-1] = 1.0 / np.asarray(prior_sd_theta) ** 2
    return np.linalg.inv(Jr.T @ Jr + np.diag(P))


def split_rhat(chains):
    """Split-R-hat (Gelman et al.) per parameter; chains: (n_chains, n_samples, d)."""
    n = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :n], chains[:, n:2 * n]], axis=0)
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = n * halves.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def adaptive_metropolis(logpost, z0, cov0, n_steps, rng, adapt_start=1000, adapt_every=200):
    d = len(z0)
    sd = 2.38 ** 2 / d
    chol = np.linalg.cholesky(sd * cov0 + 1e-12 * np.eye(d))
    chain = np.empty((n_steps, d))
    z, lp = z0.copy(), logpost(z0)
    n_acc = 0
    for t in range(n_steps):
        prop = z + chol @ rng.standard_normal(d)
        lp_prop = logpost(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            z, lp = prop, lp_prop
            n_acc += 1
        chain[t] = z
        if t >= adapt_start and t % adapt_every == 0:
            C = np.cov(chain[t // 2:t + 1].T)
            try:
                chol = np.linalg.cholesky(sd * C + 1e-12 * np.eye(d))
            except np.linalg.LinAlgError:
                pass
    return chain, n_acc / n_steps


def main(
    theta_true=DEFAULT_THETA_TRUE,
    beam_angles_deg=(0.0, 30.0, 45.0, 60.0, 70.0, 80.0),
    J0_nominal=1.0,
    J0_rel_std=0.05,          # std of log J0 (~ relative uncertainty of the beam strength)
    sigma_meas=0.01,          # std of log(measured current) (~ 1 % relative noise)
    sigma_model=0.003,        # allowance for S_N discretization error of the inference model
    source_side="left",
    model_cells_per_cm=25,    # inference model (S16 at 25 cells/cm: <~0.35 % current error)
    model_n_angles=16,
    data_cells_per_cm=100,    # finer model that generates the synthetic data
    data_n_angles=48,
    n_map_starts=6,
    n_chains=4,
    n_mcmc=10000,             # steps per chain
    burn_in=3000,
    seed=0,
    output_dir="output_graphs_inverse",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    exp = SlabExperiment(cells_per_cm=model_cells_per_cm, n_angles=model_n_angles,
                         source_side=source_side)
    sampler = exp.sampler
    names = sampler.param_names
    theta_true = np.asarray(theta_true, dtype=float)
    mu0s = np.cos(np.deg2rad(beam_angles_deg))
    sigma = np.hypot(sigma_meas, sigma_model)
    if not sampler.in_bounds(theta_true):
        raise ValueError("theta_true lies outside the prior box.")

    print("=" * 60)
    print(f"Inverse UQ: {len(sampler.active_idx)} active of {sampler.p} parameters + log J0")
    print(f"  model N={exp.N}, S{exp.n_ang}; beam on {source_side} face; angles {list(beam_angles_deg)} deg")
    print(f"  noise: sigma_meas={sigma_meas}, sigma_model={sigma_model}, sigma_J0={J0_rel_std}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Angle sensitivity scan                                          #
    # ------------------------------------------------------------------ #
    deg_grid = np.linspace(0.0, 85.0, 35)
    J_g, dJ_g = exp.currents_and_jacobian(theta_true, np.cos(np.deg2rad(deg_grid)))
    S = dJ_g * theta_true[None, None, :] / J_g[:, :, None]          # d log J / d log theta
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].semilogy(deg_grid, J_g[:, 0], label="left face")
    axes[0].semilogy(deg_grid, J_g[:, 1], label="right face")
    axes[0].set_title("Exiting current per unit incident current")
    for f, face in enumerate(("left", "right")):
        ax = axes[f + 1]
        for j, n in enumerate(names):
            ax.plot(deg_grid, S[:, f, j], color=f"C{j % 3}", ls="-" if j < 3 else "--", label=n)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(rf"$\partial\log J_{{\mathrm{{{face}}}}}/\partial\log\theta_j$")
    for ax in axes:
        ax.set_xlabel("beam angle from normal [deg]"); ax.grid(ls="--", alpha=0.4)
        for a in beam_angles_deg:
            ax.axvline(a, color="gray", ls=":", lw=0.7)
    axes[0].legend(); axes[2].legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(f"{output_dir}/angle_sensitivity.png", dpi=150)
    print(f"\nStep 1: saved {output_dir}/angle_sensitivity.png")

    # ------------------------------------------------------------------ #
    # 2. Identifiability and design comparison (no data needed)          #
    # ------------------------------------------------------------------ #
    J_c, dJ_c = exp.currents_and_jacobian(theta_true, mu0s)
    G = (dJ_c * theta_true[None, None, :] / J_c[:, :, None]).reshape(-1, sampler.p)[:, sampler.active_idx]
    G = np.hstack([G, np.ones((G.shape[0], 1))])                    # d log J / d log J0 = 1
    sv, Vt = np.linalg.svd(G, full_matrices=False)[1:]
    labels = [names[j] for j in sampler.active_idx] + ["log_J0"]
    print("\nStep 2: identifiability at the truth (chosen angle set)")
    print("  singular values of d log J / d log(theta, J0): " + "  ".join(f"{v:.2e}" for v in sv))
    print("  weakest direction: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(labels, Vt[-1])
                                             if abs(v) > 0.05))

    designs = {
        "normal only": (0.0,),
        "0, 60": (0.0, 60.0),
        "chosen set": tuple(beam_angles_deg),
        "12 angles 0-82": tuple(np.linspace(0.0, 82.0, 12)),
    }
    print("\n  Laplace posterior std / |truth| at theta_true, by angle design (log J0: absolute)")
    print("  " + f"{'design':>16s}" + "".join(f"{n:>20s}" for n in labels))
    design_std = {}
    for label, angs in designs.items():
        lp_d = LogPosterior(exp, np.cos(np.deg2rad(angs)), np.zeros((len(angs), 2)),
                            sigma, J0_nominal, J0_rel_std, theta_true)
        z_t = np.append(theta_true[lp_d.active], np.log(J0_nominal))
        std = np.sqrt(np.diag(laplace_cov(lp_d.jacobian(z_t), lp_d.prior_sd)))
        rel = std / np.append(np.abs(z_t[:-1]), 1.0)
        design_std[label] = rel
        print("  " + f"{label:>16s}" + "".join(f"{v:>20.3g}" for v in rel))
    prior_rel = np.append(lp_d.prior_sd / np.abs(z_t[:-1]), J0_rel_std)
    print("  " + f"{'prior only':>16s}" + "".join(f"{v:>20.3g}" for v in prior_rel))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / (len(designs) + 1)
    for i, (label, rel) in enumerate(list(design_std.items()) + [("prior only", prior_rel)]):
        ax.bar(np.arange(len(rel)) + i * width, rel, width, label=label,
               color="lightgray" if label == "prior only" else None)
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2, labels, rotation=20)
    ax.set_yscale("log"); ax.set_ylabel("Laplace std / |truth|  (log J0: absolute)")
    ax.set_title(f"Linearized posterior width vs. angle design  ($\\sigma$={sigma:.4f}, "
                 f"$\\sigma_{{J0}}$={J0_rel_std})")
    ax.legend(fontsize=8); ax.grid(axis="y", ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig(f"{output_dir}/design_comparison.png", dpi=150)

    # ------------------------------------------------------------------ #
    # 3. Synthetic data from a finer model                               #
    # ------------------------------------------------------------------ #
    exp_data = SlabExperiment(cells_per_cm=data_cells_per_cm, n_angles=data_n_angles,
                              source_side=source_side)
    J_fine = exp_data.currents(theta_true, mu0s)
    logJ0_true = np.log(J0_nominal) + J0_rel_std * rng.standard_normal()
    log_data = logJ0_true + np.log(J_fine) + sigma_meas * rng.standard_normal((len(mu0s), 2))
    lp = LogPosterior(exp, mu0s, log_data, sigma, J0_nominal, J0_rel_std, theta_true)
    z_true = np.append(theta_true[lp.active], logJ0_true)
    model_err = np.abs(exp.currents(theta_true, mu0s) / J_fine - 1).max()
    print(f"\nStep 3: synthetic data (N={exp_data.N}, S{exp_data.n_ang}), J0_true = {np.exp(logJ0_true):.4f}; "
          f"inference-model error at truth = {model_err:.2e}")

    # ------------------------------------------------------------------ #
    # 4. MAP: multi-start bounded Gauss-Newton                           #
    # ------------------------------------------------------------------ #
    print(f"\nStep 4: MAP from {n_map_starts} LHS starts")
    starts = sampler.sample(n_map_starts, random_state=seed + 1)[:, lp.active]
    best = None
    t0 = time.perf_counter()
    for s, th0 in enumerate(starts):
        res = least_squares(lp.residuals, np.append(th0, np.log(J0_nominal)), jac=lp.jacobian,
                            bounds=(np.append(lp.lo, -np.inf), np.append(lp.hi, np.inf)),
                            method="trf", x_scale="jac")
        print(f"  start {s}: cost = {res.cost:9.4f}  ({res.nfev} evals)  "
              + " ".join(f"{v:.3f}" for v in res.x))
        if best is None or res.cost < best.cost:
            best = res
    z_map = best.x
    print(f"  done in {time.perf_counter() - t0:.1f}s; best cost {best.cost:.4f} "
          f"(~ {len(lp.d) / 2:.0f} expected for noise-level misfit)")

    # ------------------------------------------------------------------ #
    # 5. Laplace                                                         #
    # ------------------------------------------------------------------ #
    cov_lap = laplace_cov(lp.jacobian(z_map), lp.prior_sd)
    std_lap = np.sqrt(np.diag(cov_lap))

    # ------------------------------------------------------------------ #
    # 6. Adaptive Metropolis, several chains                             #
    # ------------------------------------------------------------------ #
    print(f"\nStep 6: adaptive Metropolis, {n_chains} chains x {n_mcmc} steps")
    t0 = time.perf_counter()
    chains, accs = [], []
    eps_box = 1e-6 * (lp.hi - lp.lo)
    for c in range(n_chains):
        z0 = rng.multivariate_normal(z_map, cov_lap)
        z0[:-1] = np.clip(z0[:-1], lp.lo + eps_box, lp.hi - eps_box)
        ch, acc = adaptive_metropolis(lp, z0, cov_lap, n_mcmc, rng)
        chains.append(ch[burn_in:]); accs.append(acc)
        print(f"  chain {c}: acceptance {acc:.2f}")
    chains = np.array(chains)
    rhat = split_rhat(chains)
    post = chains.reshape(-1, chains.shape[2])
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    np.savetxt(f"{output_dir}/posterior_samples.txt", post, fmt="%.6e",
               header="  ".join(lp.names), comments="# ")

    lo95, hi95 = np.percentile(post, [2.5, 97.5], axis=0)
    print(f"\n{'param':>20s}{'truth':>9s}{'MAP':>9s}{'Lap sd':>9s}{'mean':>9s}{'sd':>9s}"
          f"{'95% CI':>21s}{'R-hat':>7s}")
    for i, n in enumerate(lp.names):
        flag = "" if lo95[i] <= z_true[i] <= hi95[i] else "  (truth outside CI)"
        print(f"{n:>20s}{z_true[i]:>9.4f}{z_map[i]:>9.4f}{std_lap[i]:>9.4f}{post[:, i].mean():>9.4f}"
              f"{post[:, i].std():>9.4f}   [{lo95[i]:7.4f}, {hi95[i]:7.4f}]{rhat[i]:>7.3f}{flag}")
    if np.any(rhat > 1.05):
        print("  WARNING: R-hat > 1.05 for some parameters; run longer chains before trusting the CIs.")

    d = len(lp.names)
    ncol = 4
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.3 * nrow))
    for i, ax in enumerate(axes.ravel()):
        if i >= d:
            ax.set_visible(False); continue
        ax.hist(post[:, i], bins=50, density=True, color="steelblue", alpha=0.7, label="MCMC")
        xs = np.linspace(*ax.get_xlim(), 200)
        ax.plot(xs, np.exp(-0.5 * ((xs - z_map[i]) / std_lap[i]) ** 2) / (std_lap[i] * np.sqrt(2 * np.pi)),
                "k-", lw=1.2, label="Laplace")
        ax.axvline(z_true[i], color="red", lw=1.5, label="truth")
        ax.set_title(f"{lp.names[i]}  (R-hat {rhat[i]:.2f})", fontsize=10)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"Posterior marginals ({len(post)} samples, {len(mu0s)} angles, $\\sigma$={sigma:.4f})")
    plt.tight_layout(); plt.savefig(f"{output_dir}/posterior_marginals.png", dpi=150)

    corr = np.corrcoef(post.T)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for a in range(d):
        for b in range(d):
            ax.text(b, a, f"{corr[a, b]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(d), lp.names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(d), lp.names, fontsize=8)
    fig.colorbar(im); ax.set_title("Posterior correlation (MCMC)")
    plt.tight_layout(); plt.savefig(f"{output_dir}/posterior_correlation.png", dpi=150)

    print(f"\nSaved plots and posterior_samples.txt in {output_dir}/")
    return dict(z_true=z_true, z_map=z_map, cov_laplace=cov_lap, samples=post, names=lp.names,
                rhat=rhat, acceptance=accs, design_std=design_std, singular_values=sv)


if __name__ == "__main__":
    main()
