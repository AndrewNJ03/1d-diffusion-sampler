"""
gpr_scalar_geometry_grid.py — Standard-GPR learning curve on the 1000-case
void/dielectric/shield thickness grid, q_net(void_thickness,
dielectric_thickness, shield_thickness).

Reads output/qoi_void_dielectric_shield_grid.txt (written by extract_qoi.py
from data/constantR_voidDielectricShield_1000_monoenergy/summary(2).csv's
response_1d column). Sibling of gpr_scalar_void_sweep.py, generalised from
p=1 to p=3 active geometry parameters. The underlying data is a full
10x10x10 factorial grid (10 levels each), not a sparse design, so a genuine
multi-parameter GPR fit is meaningful here.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

from extract_qoi import OUT_DIR, SOURCES

DATA_PATH = OUT_DIR / "qoi_void_dielectric_shield_grid.txt"
PLOT_OUT_DIR = Path(__file__).resolve().parent / "output_void_dielectric_shield_grid"

PARAM_COLS = SOURCES["void_dielectric_shield_grid"].param_cols  # ("void_thickness", "dielectric_thickness", "shield_thickness")
LEARNING_CURVE_SIZES = [10, 20, 30, 50, 75, 100, 200, 400, 600, 800]


def _make_kernel(p: int):
    return (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=np.ones(p), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )


def load_geometry_grid(path: Path = DATA_PATH):
    """Load (case_ids, X=[void,dielectric,shield]_thickness, y=q_net) from extract_qoi.py's output table."""
    table = np.loadtxt(path, comments="#")
    case_ids = table[:, 0].astype(int)
    X = table[:, 1:1 + len(PARAM_COLS)]
    y = table[:, 1 + len(PARAM_COLS)]
    return case_ids, X, y


def relative_l2_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


def main(n_restarts: int = 5, test_frac: float = 0.2, seed: int = 1, out_dir: Path = PLOT_OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Standard GPR — SGEMP net charge vs. (void, dielectric, shield) thickness")
    print("=" * 60)

    case_ids, X, y = load_geometry_grid()
    M, p = X.shape
    print(f"Loaded M = {M} cases, p = {p} active parameters from {DATA_PATH.name}")
    for j, name in enumerate(PARAM_COLS):
        print(f"  {name:20s}: [{X[:, j].min():.4g}, {X[:, j].max():.4g}], "
              f"{len(np.unique(X[:, j]))} unique levels")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    n_test = int(round(test_frac * M))
    test_idx, train_pool_idx = perm[:n_test], perm[n_test:]
    M_pool = len(train_pool_idx)
    print(f"Held-out test set: {n_test} cases;  training pool: {M_pool} cases")

    X_test, y_test = X[test_idx], y[test_idx]

    x_scaler = StandardScaler().fit(X[train_pool_idx])
    y_mu, y_sd = float(y[train_pool_idx].mean()), float(y[train_pool_idx].std())
    if y_sd == 0.0:
        y_sd = 1.0

    def _fit_predict(m):
        idx = train_pool_idx[:m]
        X_tr, y_tr = X[idx], y[idx]
        X_tr_n = x_scaler.transform(X_tr)
        y_tr_z = (y_tr - y_mu) / y_sd

        gpr = GaussianProcessRegressor(
            kernel=_make_kernel(p=p),
            n_restarts_optimizer=n_restarts,
            normalize_y=False,
        )
        gpr.fit(X_tr_n, y_tr_z)

        X_te_n = x_scaler.transform(X_test)
        y_pred_z = gpr.predict(X_te_n)
        y_pred = y_pred_z * y_sd + y_mu
        return gpr, y_pred

    print(f"\nLearning curve  (sizes: {LEARNING_CURVE_SIZES}, "
          f"test_frac={test_frac}, restarts={n_restarts})")
    curve_eps = []
    last_gpr, last_pred = None, None
    for m in LEARNING_CURVE_SIZES:
        if m > M_pool:
            print(f"  M={m:4d}:  skipped (exceeds training pool of {M_pool})")
            continue
        gpr, y_pred = _fit_predict(m)
        eps = relative_l2_error(y_pred, y_test)
        curve_eps.append((m, eps))
        ell = np.atleast_1d(gpr.kernel_.k1.k2.length_scale)
        print(f"  M={m:4d}:  eps_q = {eps:.4e}   "
              f"(ell_void={ell[0]:.3g}, ell_diel={ell[1]:.3g}, ell_shield={ell[2]:.3g})")
        last_gpr, last_pred = gpr, y_pred

    m_final, eps_final = curve_eps[-1]
    print(f"\nFinal (M={m_final}) held-out relative error eps_q = {eps_final:.4e}")

    # --- Plots ---------------------------------------------------------- #
    ms = [c[0] for c in curve_eps]
    epss = [c[1] for c in curve_eps]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ms, epss, "o-", color="seagreen")
    ax.set_xlabel("Training-set size $M$")
    ax.set_ylabel(r"held-out relative error $\varepsilon_q$")
    ax.set_yscale("log")
    ax.set_title(r"Standard GPR learning curve — $q_{net}$(void, dielectric, shield)")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "gpr_learning_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, last_pred, s=14, color="darkorange", alpha=0.7)
    lims = [min(y_test.min(), last_pred.min()), max(y_test.max(), last_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(r"true $q_{net}$")
    ax.set_ylabel(r"predicted $q_{net}$")
    ax.set_title(f"Parity — standard GPR  (M={m_final}, p=3)")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "gpr_parity.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved plots to {out_dir}")

    np.savetxt(
        out_dir / "learning_curve.csv",
        np.array(curve_eps),
        delimiter=",", header="M,eps_q", comments="",
    )
    print(f"Saved: {out_dir / 'learning_curve.csv'}")

    return dict(curve=curve_eps, gpr=last_gpr, x_scaler=x_scaler,
                y_mu=y_mu, y_sd=y_sd, X=X, y=y,
                train_pool_idx=train_pool_idx, test_idx=test_idx)


if __name__ == "__main__":
    main()
