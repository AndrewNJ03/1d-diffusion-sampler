"""
ge_gpr_scalar.py — Gradient-enhanced GPR surrogate for the SGEMP charge q(void_thickness).

Reuses GEGaussianProcess from sklearn_implementation/ge_gpr.py unmodified.

gradient_source="adjoint" (default): values and gradients both come from
void_sensitivity.py's output/qoi_void_adjoint.txt, i.e. q = <Q+, psi> and the
adjoint-form dq/dt_v in case_key units. Values and gradients must describe the
same functional, so extract_qoi.py's q_net is NOT used on this path. Cases
without an adjoint flux dump (NaN gradient) are dropped.

gradient_source="fd": the old placeholder. q_net from extract_qoi.py's
output/qoi_scalar.txt with np.gradient across the sorted cases. Only for
checking the pipeline when no adjoint dumps exist.
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


def load_adjoint_table(path: Path = None, allow_failed: bool = False):
    """
    Read void_sensitivity.py's qoi_void_adjoint.txt.
    Returns case_ids, X (M, 1) void thickness, q (M,), dq/dt_v (M,). Drops cases
    without an adjoint gradient and, unless allow_failed, cases whose boundary
    identity failed (status 2). Cases the identity could not check (status 1,
    internal forward source) are kept.
    """
    path = Path(path) if path is not None else OUT_DIR / "qoi_void_adjoint.txt"
    table = np.atleast_2d(np.loadtxt(path, comments="#"))
    if table.shape[1] < 9:
        raise ValueError(f"{path} was written by an older void_sensitivity.py; rerun it")
    status = table[:, 7].astype(int)
    has_grad = np.isfinite(table[:, 8])
    failed = has_grad & (status == 2)
    keep = has_grad & (allow_failed | (status != 2))
    for mask, why in ((~has_grad, "no adjoint gradient"),
                      (failed & ~keep, "boundary identity failed")):
        if mask.any():
            print(f"Dropping {int(mask.sum())} case(s), {why}: {table[mask, 0].astype(int).tolist()}")
    unchecked = keep & (status == 1)
    if unchecked.any():
        print(f"{int(unchecked.sum())} case(s) not checked by the boundary identity (internal "
              f"source or electron-only adjoint): {table[unchecked, 0].astype(int).tolist()}")
    table = table[keep]
    return table[:, 0].astype(int), table[:, 1:2], table[:, 2], table[:, 8]


def main(n_restarts: int = 10, out_dir: Path = OUT_DIR, gradient_source: str = "adjoint",
         table_path: Path = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"GE-GPR ({gradient_source} gradients) — scalar SGEMP charge QoI")
    print("=" * 60)

    if table_path is None:
        table_path = out_dir / ("qoi_void_adjoint.txt" if gradient_source == "adjoint"
                                else "qoi_scalar.txt")
    if gradient_source == "adjoint":
        case_ids, X, y, dy_dx_phys = load_adjoint_table(table_path)
        qlabel = r"$q = \langle Q^\dagger, \psi\rangle$"
    elif gradient_source == "fd":
        case_ids, X, y = load_qoi_table(table_path)
        dy_dx_phys = finite_difference_gradient(X, y)
        qlabel = r"$q_{\mathrm{net}}$"
    else:
        raise ValueError("gradient_source must be 'adjoint' or 'fd'")
    M = len(y)
    if M < 2:
        raise ValueError(f"need at least 2 cases with values and gradients, got {M}")
    if len(np.unique(X[:, 0])) < M:
        raise ValueError("duplicate void thicknesses in the training set")
    print(f"Loaded M = {M} case(s): {case_ids.tolist()}")
    print(f"dq/d(void_thickness) = {dy_dx_phys}")

    x_scaler = StandardScaler()
    X_norm = x_scaler.fit_transform(X)
    scale_X = x_scaler.scale_  # (1,)

    y_mu, y_sd = float(np.mean(y)), float(np.std(y))
    if y_sd == 0.0:
        y_sd = 1.0
    y_z = (y - y_mu) / y_sd

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
    ax.set_ylabel(qlabel)
    ax.set_title(f"GE-GPR vs Std GPR — SGEMP charge, {gradient_source} gradients  (M={M})")
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
    # python3 ge_gpr_scalar.py [adjoint|fd]
    main(gradient_source=sys.argv[1] if len(sys.argv) > 1 else "adjoint")
