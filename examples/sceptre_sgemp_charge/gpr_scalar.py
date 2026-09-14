"""
gpr_scalar.py — Plain GPR surrogate for the SCEPTRE net SGEMP charge q_net(void_thickness).

Loads output/qoi_scalar.txt (written by extract_qoi.py) and fits a single
sklearn GaussianProcessRegressor with this project's standard kernel family
(ConstantKernel * RBF + WhiteKernel; mirrors sklearn_implementation/gpr_pod.py),
inputs standardised with StandardScaler, outputs standardised manually.

With only 2 of 11 planned cases dumped so far, this is a pipeline-validation
run, not a meaningful surrogate fit -- kept case-count-agnostic so it works
unchanged once more case_*_forward_1d.csv files exist.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from extract_qoi import OUT_DIR

N_PLANNED_CASES = 11


def _make_kernel(p: int):
    return (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=np.ones(p), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )


def load_qoi_table(path: Path = None):
    path = Path(path) if path is not None else OUT_DIR / "qoi_scalar.txt"
    table = np.loadtxt(path, comments="#")
    case_ids = table[:, 0].astype(int)
    X = table[:, 1:2]   # void_thickness, (M, 1)
    y = table[:, 2]     # q_net,          (M,)
    return case_ids, X, y


def main(n_restarts: int = 10, out_dir: Path = OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GPR (sklearn) — scalar SGEMP charge QoI")
    print("=" * 60)

    case_ids, X, y = load_qoi_table()
    M = len(y)
    print(f"Loaded M = {M} case(s): {case_ids.tolist()}")
    if M < N_PLANNED_CASES:
        print(f"NOTE: only {M}/{N_PLANNED_CASES} planned cases have a dumped CSV -- "
              f"treat this fit as pipeline validation, not a converged surrogate.")

    x_scaler = StandardScaler()
    X_norm = x_scaler.fit_transform(X)

    y_mu, y_sd = float(np.mean(y)), float(np.std(y))
    if y_sd == 0.0:
        y_sd = 1.0
    y_z = (y - y_mu) / y_sd

    gpr = GaussianProcessRegressor(
        kernel=_make_kernel(p=1),
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
        copy_X_train=True,
    )
    gpr.fit(X_norm, y_z)

    k = gpr.kernel_
    print(f"\nlog-marginal-likelihood = {gpr.log_marginal_likelihood_value_:.3f}")
    print(f"output scale sigma_f^2  = {k.k1.k1.constant_value:.3e}")
    print(f"length scale ell        = {np.atleast_1d(k.k1.k2.length_scale)}")
    print(f"noise level  sigma_n^2  = {k.k2.noise_level:.3e}")

    # Predict on a fine grid spanning the sampled range for a diagnostic plot.
    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    grid_norm = x_scaler.transform(grid)
    mu_z, sd_z = gpr.predict(grid_norm, return_std=True)
    mu = mu_z * y_sd + y_mu
    sd = sd_z * y_sd

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid[:, 0], mu, "-", color="steelblue", lw=2, label="GP mean")
    ax.fill_between(grid[:, 0], mu - 2 * sd, mu + 2 * sd,
                     color="steelblue", alpha=0.2, label=r"$\pm 2\sigma$")
    ax.scatter(X[:, 0], y, color="tomato", zorder=5, label="training cases")
    for cid, xi, yi in zip(case_ids, X[:, 0], y):
        ax.annotate(f"C{cid}", (xi, yi), xytext=(4, 4),
                     textcoords="offset points", fontsize=8, color="dimgray")
    ax.set_xlabel("void_thickness")
    ax.set_ylabel(r"$q_{\mathrm{net}}$")
    ax.set_title(f"Std GPR — SGEMP net charge  (M={M})")
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "gpr_scalar_prediction.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    return dict(gp=gpr, x_scaler=x_scaler, y_mu=y_mu, y_sd=y_sd,
                X=X, y=y, case_ids=case_ids)


if __name__ == "__main__":
    main()
