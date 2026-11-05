"""
band_plots.py: like-for-like predictive uncertainty for GE-GPR and standard GPR,
and a figure that keeps the bands readable when they are thin.

sklearn's return_std includes the fitted WhiteKernel noise, while
GEGaussianProcess.predict returns the latent std by default, so plotting the two
directly compares different quantities. predict_gp and predict_ge return both
stds for both models:

    sd_latent  uncertainty in the underlying function
    sd_total   sd_latent plus the fitted observation noise

A well-fitted GE-GPR band can be a fraction of a percent of the response range,
which is under a pixel on a mean-curve plot. plot_band_panels therefore adds a
panel of deviations from the GE-GPR mean and a panel of sigma on a log axis.
No std is inflated to make a band visible; held-out coverage in
compare_ge_vs_gp.py is the test of whether a band can be trusted.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

GE_COLOR, GP_COLOR = "#2a78d6", "#eb6834"
GE_STYLE, GP_STYLE = "-", "--"
THIN_BAND_FRACTION = 0.01   # warn when the widest +/-2 sigma is below this share of the y-range
BOUND_TOL = 1e-3            # log-space distance from a bound that counts as "at the bound"


def _at_bound(names, theta, bounds) -> list:
    """Entries like "ell_0:upper" for each hyperparameter at a bound."""
    out = []
    for name, t, (lo, hi) in zip(names, theta, bounds):
        if t - lo <= BOUND_TOL:
            out.append(f"{name}:lower")
        elif hi - t <= BOUND_TOL:
            out.append(f"{name}:upper")
    return out


def matched_gp_kernel(p: int):
    """
    C * ARD-RBF + White with the hyperparameter ranges of GEGaussianProcess, whose
    bounds are on log sigma_f, log ell and log sigma_n (see theta_bounds()).
    """
    return (ConstantKernel(1.0, constant_value_bounds=(np.exp(-10.0), np.exp(10.0)))
            * RBF(length_scale=np.ones(p), length_scale_bounds=(np.exp(-5.0), np.exp(5.0)))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(np.exp(-20.0), 1.0)))


def predict_gp(gpr, X: np.ndarray, y_mu: float = 0.0, y_sd: float = 1.0) -> dict:
    """Mean and both stds of a fitted sklearn GPR whose kernel ends in + WhiteKernel."""
    mean, sd_total = gpr.predict(X, return_std=True)
    noise_var = float(gpr.kernel_.k2.noise_level)
    sd_latent = np.sqrt(np.maximum(sd_total ** 2 - noise_var, 0.0))
    return dict(mean=mean * y_sd + y_mu, sd_latent=sd_latent * y_sd, sd_total=sd_total * y_sd)


def predict_ge(ge_gp, X: np.ndarray, y_mu: float = 0.0, y_sd: float = 1.0) -> dict:
    """Mean and both stds of a fitted GEGaussianProcess."""
    mean, sd_latent = ge_gp.predict(X, return_std=True)
    _, sd_total = ge_gp.predict(X, return_std=True, include_noise=True)
    return dict(mean=mean * y_sd + y_mu, sd_latent=sd_latent * y_sd, sd_total=sd_total * y_sd)


def ge_hyperparameters(ge_gp) -> dict:
    """Fitted values in the standardized space, and the names of any at a bound."""
    p = ge_gp.p_act
    gradient_noise = ([f"noise_gradient_{i}" for i in range(p)]
                      if getattr(ge_gp, "per_dim_gradient_noise", False) else ["noise_gradient"])
    names = ["sigma_f"] + [f"ell_{i}" for i in range(p)] + ["noise_value"] + gradient_noise
    theta = np.asarray(ge_gp.theta_)
    bounds = np.array(ge_gp.theta_bounds())
    return dict(values=dict(zip(names, np.exp(theta).tolist())),
                at_bound=_at_bound(names, theta, bounds), lml=float(ge_gp.lml_))


def gp_hyperparameters(gpr) -> dict:
    """As ge_hyperparameters, for a fitted matched_gp_kernel GPR, reported as stds."""
    kernel = gpr.kernel_
    ell = np.atleast_1d(kernel.k1.k2.length_scale)
    names = ["sigma_f"] + [f"ell_{i}" for i in range(len(ell))] + ["noise_value"]
    values = [float(np.sqrt(kernel.k1.k1.constant_value))] + ell.tolist() \
        + [float(np.sqrt(kernel.k2.noise_level))]
    return dict(values=dict(zip(names, values)),
                at_bound=_at_bound(names, kernel.theta, kernel.bounds),
                lml=float(gpr.log_marginal_likelihood_value_))


def report_bands(label: str, pred: dict, y_range: float, hyper: dict = None) -> float:
    """Print band size relative to the y-range and fitted hyperparameters; returns the share."""
    share = 4.0 * float(np.max(pred["sd_total"])) / max(y_range, 1e-300)
    print(f"{label}: widest +/-2 sigma band = {share:.3%} of the y-range "
          f"(latent sigma max {np.max(pred['sd_latent']):.3e}, "
          f"with noise {np.max(pred['sd_total']):.3e})")
    if share < THIN_BAND_FRACTION:
        print(f"  NOTE: band under {THIN_BAND_FRACTION:.0%} of the y-range, so it is not visible "
              "on the mean panel. Use the deviation and sigma panels.")
    if hyper is not None:
        vals = ", ".join(f"{k}={v:.3g}" for k, v in hyper["values"].items())
        print(f"  fitted: {vals}; log-MLL {hyper['lml']:.3f}")
        # A deterministic solve has no observation noise, so noise at its floor is expected.
        floor = [b for b in hyper["at_bound"] if b.startswith("noise") and b.endswith(":lower")]
        other = [b for b in hyper["at_bound"] if b not in floor]
        if floor:
            print(f"  noise at its lower bound ({', '.join(floor)}): the data look noise-free "
                  "at this floor, so the band is set by the kernel alone.")
        if other:
            print(f"  WARNING: at a bound: {', '.join(other)}. The data do not determine "
                  "these; treat the band as a statement about the prior, not the data.")
    return share


def plot_band_panels(grid: np.ndarray, ge: dict, gp: dict, X_train: np.ndarray, y_train: np.ndarray,
                     xlabel: str, ylabel: str, title: str, path, X_test=None, y_test=None,
                     annotate=None) -> None:
    """
    Three panels on a shared x axis: means with +/-2 sigma bands, deviation from the
    GE-GPR mean with the same bands, and sigma on a log axis. Bands include the
    fitted noise for both models; the sigma panel also shows the latent std.
    """
    fig, (ax_mean, ax_dev, ax_sd) = plt.subplots(
        3, 1, figsize=(7.5, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.4, 1.4]))
    models = [("GE-GPR", ge, GE_COLOR, GE_STYLE), ("Standard GP", gp, GP_COLOR, GP_STYLE)]
    for name, pred, color, style in models:
        lo, hi = pred["mean"] - 2 * pred["sd_total"], pred["mean"] + 2 * pred["sd_total"]
        ax_mean.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0, label=f"{name} ±2σ")
        ax_mean.plot(grid, pred["mean"], style, color=color, lw=2, label=f"{name} mean")
        ax_dev.fill_between(grid, lo - ge["mean"], hi - ge["mean"], color=color, alpha=0.18, lw=0)
        ax_dev.plot(grid, pred["mean"] - ge["mean"], style, color=color, lw=2)
        for key, ls, tag in (("sd_total", style, "σ with noise"), ("sd_latent", ":", "latent σ")):
            sd = np.where(pred[key] > 0, pred[key], np.nan)   # a log axis cannot show exact zeros
            ax_sd.plot(grid, sd, ls, color=color, lw=2 if key == "sd_total" else 1.5,
                       label=f"{name} {tag}")

    # The 1-D slices of a multi-parameter fit have no training data on the slice.
    if X_train is not None and len(X_train):
        ge_at_train = np.interp(X_train, grid, ge["mean"])
        ax_mean.scatter(X_train, y_train, s=36, color="#0b0b0b", edgecolor="white", lw=1, zorder=5,
                        label="training cases")
        ax_dev.scatter(X_train, y_train - ge_at_train, s=36, color="#0b0b0b", edgecolor="white",
                       lw=1, zorder=5)
    if X_test is not None and len(X_test):
        ge_at_test = np.interp(X_test, grid, ge["mean"])
        ax_mean.scatter(X_test, y_test, s=30, facecolor="none", edgecolor="#52514e", zorder=4,
                        label="held-out cases")
        ax_dev.scatter(X_test, y_test - ge_at_test, s=30, facecolor="none", edgecolor="#52514e",
                       zorder=4)
    if annotate is not None and X_train is not None:
        for cid, xi, yi in zip(annotate, X_train, y_train):
            ax_mean.annotate(f"C{cid}", (xi, yi), xytext=(5, 5), textcoords="offset points",
                             fontsize=8, color="#52514e")

    ax_mean.set_ylabel(ylabel)
    ax_mean.set_title(title)
    ax_mean.legend(fontsize=8, frameon=False)
    ax_dev.axhline(0.0, color="#52514e", lw=0.8)
    ax_dev.set_ylabel("deviation from\nGE-GPR mean")
    ax_sd.set_yscale("log")
    ax_sd.set_ylabel("predictive σ")
    ax_sd.set_xlabel(xlabel)
    ax_sd.legend(fontsize=8, frameon=False, ncol=2, loc="best")
    for ax in (ax_mean, ax_dev, ax_sd):
        ax.grid(True, color="#e4e3df", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
