"""
ge_gpr_scalar.py — Gradient-enhanced GPR surrogate for q_net(void_thickness).

Reuses GEGaussianProcess from sklearn_implementation/ge_gpr.py unmodified.
This external SCEPTRE solver has no adjoint/sensitivity output (only forward
flux dumps), so the gradient dq_net/d(void_thickness) fed to GE-GPR is a
finite-difference approximation (np.gradient across the sorted training
cases) -- a rough placeholder, explicitly noted, valid until more cases are
generated and/or a real sensitivity source becomes available.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from extract_qoi import OUT_DIR
from gpr_scalar import load_qoi_table, _make_kernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sklearn_implementation"))
from ge_gpr import GEGaussianProcess  # noqa: E402

N_PLANNED_CASES = 11


def finite_difference_gradient(X: np.ndarray, y: np.ndarray):
    """
    dq_net/d(void_thickness) via np.gradient on the (sorted) training cases:
    central differences at interior points, one-sided at the ends. With
    only 2 points this degenerates to a single secant slope for both.
    """
    order = np.argsort(X[:, 0])
    x_sorted = X[order, 0]
    y_sorted = y[order]
    dy_dx_sorted = np.gradient(y_sorted, x_sorted)

    dy_dx = np.empty_like(dy_dx_sorted)
    dy_dx[order] = dy_dx_sorted
    return dy_dx


def main(n_restarts: int = 10, out_dir: Path = OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GE-GPR (finite-difference gradients) — scalar SGEMP charge QoI")
    print("=" * 60)

    case_ids, X, y = load_qoi_table()
    M = len(y)
    print(f"Loaded M = {M} case(s): {case_ids.tolist()}")
    if M < N_PLANNED_CASES:
        print(f"NOTE: only {M}/{N_PLANNED_CASES} planned cases have a dumped CSV -- "
              f"the finite-difference gradient below is a rough placeholder "
              f"until more cases exist (or a real sensitivity/adjoint source is used).")

    x_scaler = StandardScaler()
    X_norm = x_scaler.fit_transform(X)
    scale_X = x_scaler.scale_  # (1,)

    y_mu, y_sd = float(np.mean(y)), float(np.std(y))
    if y_sd == 0.0:
        y_sd = 1.0
    y_z = (y - y_mu) / y_sd

    dy_dx_phys = finite_difference_gradient(X, y)
    print(f"Finite-difference dq_net/d(void_thickness) = {dy_dx_phys}")

    # Chain rule to the (standardised y) / (StandardScaler-normalised x) space:
    #   dz/dx_norm = (dy/dx_phys) * scale_X / y_sd
    G_active = (dy_dx_phys * scale_X / y_sd).reshape(-1, 1)  # (M, 1)

    ge_gp = GEGaussianProcess(
        active_dims=np.array([0]),
        n_restarts=n_restarts,
        noise_f_init=1e-3,
        noise_d_init=1e-3,
    )
    ge_gp.fit(X_norm, y_z, G_active)
    print(f"\nGE-GPR log-MLL = {ge_gp.lml_:.3f}")

    # Standard GPR on the same data, for comparison (mirrors ge_gpr.py's main()).
    std_gpr = GaussianProcessRegressor(
        kernel=_make_kernel(p=1),
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
    )
    std_gpr.fit(X_norm, y_z)
    print(f"Std GPR  log-MLL = {std_gpr.log_marginal_likelihood_value_:.3f}")

    grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    grid_norm = x_scaler.transform(grid)

    mu_ge_z, sd_ge_z = ge_gp.predict(grid_norm, return_std=True)
    mu_ge, sd_ge = mu_ge_z * y_sd + y_mu, sd_ge_z * y_sd

    mu_std_z, sd_std_z = std_gpr.predict(grid_norm, return_std=True)
    mu_std, sd_std = mu_std_z * y_sd + y_mu, sd_std_z * y_sd

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid[:, 0], mu_ge, "-", color="steelblue", lw=2, label="GE-GPR mean")
    ax.fill_between(grid[:, 0], mu_ge - 2 * sd_ge, mu_ge + 2 * sd_ge,
                     color="steelblue", alpha=0.18, label=r"GE-GPR $\pm 2\sigma$")
    ax.plot(grid[:, 0], mu_std, "--", color="tomato", lw=2, label="Std GPR mean")
    ax.fill_between(grid[:, 0], mu_std - 2 * sd_std, mu_std + 2 * sd_std,
                     color="tomato", alpha=0.12, label=r"Std GPR $\pm 2\sigma$")
    ax.scatter(X[:, 0], y, color="black", zorder=5, label="training cases")
    for cid, xi, yi in zip(case_ids, X[:, 0], y):
        ax.annotate(f"C{cid}", (xi, yi), xytext=(4, 4),
                     textcoords="offset points", fontsize=8, color="dimgray")
    ax.set_xlabel("void_thickness")
    ax.set_ylabel(r"$q_{\mathrm{net}}$")
    ax.set_title(f"GE-GPR vs Std GPR — SGEMP net charge  (M={M})")
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / "ge_gpr_scalar_prediction.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    return dict(ge_gp=ge_gp, std_gpr=std_gpr, x_scaler=x_scaler,
                y_mu=y_mu, y_sd=y_sd, X=X, y=y, case_ids=case_ids,
                G_active=G_active)


if __name__ == "__main__":
    main()
