"""
coarse_gpr_delam.py
-------------------
Coarse Gaussian Process Regression surrogate for the center-conductor
charge deposition from the aljac coaxial-cable delamination sweep.

Inputs (3-D parameter space):
    theta_0  -- void centroid angle      [deg]
    dtheta   -- void half-angular-width  [deg]
    delta    -- void radial thickness    [cm]

Output (scalar QoI):
    Q_center = charge deposited in block "center_conductor_Al"  [C]

Design notes (because N is small):
    * 10 SCEPTRE cases after dropping Case 1 (effectively no-delam baseline).
    * Sklearn GPR with ARD-RBF + WhiteKernel + ConstantKernel
      (mirrors the kernel used in this project's gpr_pod.py).
    * Inputs normalised to [0,1]; outputs standardised. The parameter
      scales differ by ~4 orders of magnitude (delta vs theta), so
      ARD-on-raw would have terrible conditioning.
    * Leave-one-out cross-validation is the honest error metric here --
      with N=10 a held-out test set isn't viable.
    * 1-D slices at the nominal point are produced so you can eyeball
      what the surrogate believes about each parameter direction.

Usage:
    Place this script in the same directory as the
    Charge_Deposition_Total_Block_Case*.txt files and sceptre_case_key.txt,
    then:
        python coarse_gpr_delam.py
    Or pass an explicit data directory:
        python coarse_gpr_delam.py --data-dir /path/to/files
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, WhiteKernel,
)


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

TARGET_BLOCK = "center_conductor_Al"   # QoI block name to extract
DROP_CASES   = {1}                     # Case 1 -- baseline (effectively no void)

PARAM_NAMES  = ["theta_0 [deg]", "dtheta [deg]", "delta [cm]"]
PARAM_KEYS   = ["theta_0", "dtheta", "delta"]

OUT_DIR = "gpr_coarse_out"


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #

def load_case_key(path: Path) -> dict[int, dict]:
    """
    Parse sceptre_case_key.txt -> {case_id: {theta_0, dtheta, delta}}.

    Expected format:
        Case #, theta_0, dtheta, delta
        0, 90, 30, 0.002
        ...
    """
    cases = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith("case"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            cid = int(parts[0])
            cases[cid] = {
                "theta_0": float(parts[1]),
                "dtheta":  float(parts[2]),
                "delta":   float(parts[3]),
            }
    return cases


def load_deposition_value(path: Path, block_name: str) -> float:
    """
    Parse a Charge_Deposition_Total_Block_CaseN.txt file and return the
    deposition Value for the named block.

    File format:
        # CHARGE DEPOSITION
        # Block, Block Name, Volume, Value
        1, center_conductor_Al, 0.006546835392, -6.43324567e-09
        ...
    """
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            if parts[1] == block_name:
                return float(parts[3])
    raise ValueError(f"Block '{block_name}' not found in {path}")


def assemble_dataset(data_dir: Path, target_block: str, drop_cases: set[int]):
    """
    Build the (X, y, case_ids) arrays for the GP.

    Returns
    -------
    X : (N, 3) ndarray of [theta_0, dtheta, delta] in physical units
    y : (N,)   ndarray of the target block's charge deposition  [C]
    case_ids : list[int] of length N, ordered like the rows of X
    """
    case_key = load_case_key(data_dir / "sceptre_case_key.txt")

    pattern = re.compile(r"Charge_Deposition_Total_Block_Case(\d+)\.txt$")
    rows, ys, ids = [], [], []

    for path in sorted(data_dir.glob("Charge_Deposition_Total_Block_Case*.txt")):
        m = pattern.search(path.name)
        if not m:
            continue
        cid = int(m.group(1))
        if cid in drop_cases:
            continue
        if cid not in case_key:
            print(f"  [skip] Case {cid}: not in case key")
            continue

        params = case_key[cid]
        try:
            q = load_deposition_value(path, target_block)
        except ValueError as e:
            print(f"  [skip] Case {cid}: {e}")
            continue

        rows.append([params["theta_0"], params["dtheta"], params["delta"]])
        ys.append(q)
        ids.append(cid)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(ys,   dtype=float)
    return X, y, ids


# --------------------------------------------------------------------------- #
# GP utilities                                                                #
# --------------------------------------------------------------------------- #

def normalize_X(X: np.ndarray, X_ref: np.ndarray | None = None):
    """Min-max scale columns of X to [0,1] using X_ref's range (default = X itself)."""
    if X_ref is None:
        X_ref = X
    lo = X_ref.min(axis=0)
    hi = X_ref.max(axis=0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X - lo) / rng, lo, rng


def standardise_y(y: np.ndarray, y_ref: np.ndarray | None = None):
    if y_ref is None:
        y_ref = y
    mu = float(np.mean(y_ref))
    sd = float(np.std(y_ref, ddof=0))
    if sd == 0.0:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def make_kernel(p: int) -> "Kernel":
    """ARD-RBF * constant + white noise. Bounds widened for small-N stability."""
    return (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=np.ones(p), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e0))
    )


def fit_gp(X_norm: np.ndarray, y_z: np.ndarray, n_restarts: int = 20):
    gp = GaussianProcessRegressor(
        kernel=make_kernel(X_norm.shape[1]),
        n_restarts_optimizer=n_restarts,
        normalize_y=False,         # we standardise manually
        alpha=1e-10,               # WhiteKernel handles noise
        random_state=0,
    )
    gp.fit(X_norm, y_z)
    return gp


def predict(gp, X_norm, y_mu, y_sd, return_std=True):
    """Predict in standardised space, then unstandardise mean and std."""
    if return_std:
        mu_z, sd_z = gp.predict(X_norm, return_std=True)
        return mu_z * y_sd + y_mu, sd_z * y_sd
    mu_z = gp.predict(X_norm, return_std=False)
    return mu_z * y_sd + y_mu


# --------------------------------------------------------------------------- #
# Leave-one-out cross-validation                                              #
# --------------------------------------------------------------------------- #

def loo_cv(X: np.ndarray, y: np.ndarray, n_restarts: int = 20):
    """
    Honest leave-one-out CV. Per held-out fold we re-normalise/standardise
    using only the training subset, then evaluate on the held-out point.
    """
    N = len(y)
    y_pred = np.empty(N)
    y_std  = np.empty(N)

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False

        X_tr, y_tr = X[mask], y[mask]
        X_te       = X[i:i + 1]

        Xn_tr, lo, rng = normalize_X(X_tr)
        yz_tr, mu, sd  = standardise_y(y_tr)

        gp = fit_gp(Xn_tr, yz_tr, n_restarts=n_restarts)

        Xn_te = (X_te - lo) / rng
        mu_p, sd_p = predict(gp, Xn_te, mu, sd, return_std=True)
        y_pred[i] = mu_p[0]
        y_std[i]  = sd_p[0]

    return y_pred, y_std


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_parity(y_true, y_pred, y_std, case_ids, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=1, label="y = x")
    ax.errorbar(
        y_true, y_pred, yerr=y_std,
        fmt="o", color="steelblue", ecolor="lightsteelblue",
        capsize=3, ms=6, label="LOO predictions ±1σ",
    )
    for x, yp, cid in zip(y_true, y_pred, case_ids):
        ax.annotate(f"C{cid}", (x, yp), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="dimgray")
    ax.set_xlabel("True charge deposition  [C]")
    ax.set_ylabel("LOO predicted  [C]")
    ax.set_title(f"GPR LOO parity — {TARGET_BLOCK}")
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_slices(gp, X, y, lo, rng, y_mu, y_sd, out_path):
    """
    1-D slices through the surrogate at the nominal (median) point.
    Sweeps each parameter across its sampled range with the others fixed.
    """
    nominal = np.median(X, axis=0)
    n_grid = 80

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        sweep = np.linspace(X[:, j].min(), X[:, j].max(), n_grid)
        Xq = np.tile(nominal, (n_grid, 1))
        Xq[:, j] = sweep
        Xq_n = (Xq - lo) / rng

        mu, sd = predict(gp, Xq_n, y_mu, y_sd, return_std=True)
        ax.plot(sweep, mu, "-", color="steelblue", lw=2, label="GP mean")
        ax.fill_between(sweep, mu - 2 * sd, mu + 2 * sd,
                        color="steelblue", alpha=0.18, label="±2σ")

        # Overlay training points whose other coordinates are close to nominal
        # so the visual comparison is at least somewhat fair (loose tolerance).
        other_dims = [k for k in range(X.shape[1]) if k != j]
        tol = np.array([0.5 * (X[:, k].max() - X[:, k].min())
                        if (X[:, k].max() - X[:, k].min()) > 0 else 1.0
                        for k in other_dims])
        close = np.all(
            np.abs(X[:, other_dims] - nominal[other_dims]) <= tol,
            axis=1,
        )
        ax.scatter(X[close, j], y[close], color="tomato",
                   s=45, zorder=5, label="training pts")
        ax.axvline(nominal[j], color="gray", ls=":", lw=1,
                   label=f"nominal = {nominal[j]:g}")

        ax.set_xlabel(name)
        if j == 0:
            ax.set_ylabel(f"{TARGET_BLOCK} charge  [C]")
        ax.set_title(f"Slice along {PARAM_KEYS[j]}")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"GP 1-D slices at nominal point  "
        f"(θ₀={nominal[0]:g}°, Δθ={nominal[1]:g}°, δ={nominal[2]:g})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing the .txt input files.")
    parser.add_argument("--out-dir",  type=str, default=OUT_DIR,
                        help="Output directory for plots and CSV.")
    parser.add_argument("--n-restarts", type=int, default=20,
                        help="GP optimiser restarts (use 20+ for small N).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Coarse GPR — center-conductor charge deposition")
    print("=" * 64)
    print(f"Data directory  : {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Target block    : {TARGET_BLOCK}")
    print(f"Dropping cases  : {sorted(DROP_CASES)}")

    # 1. Load
    X, y, case_ids = assemble_dataset(data_dir, TARGET_BLOCK, DROP_CASES)
    print(f"\nLoaded N = {len(y)} cases")
    print(f"  case_ids = {case_ids}")
    print(f"  X shape  = {X.shape}")
    print(f"  y range  = [{y.min():.3e}, {y.max():.3e}]")
    print(f"  y std    = {np.std(y):.3e}")

    # 2. Fit on all data
    Xn, lo, rng        = normalize_X(X)
    yz, y_mu, y_sd     = standardise_y(y)
    gp                 = fit_gp(Xn, yz, n_restarts=args.n_restarts)

    # Extract learned hyperparameters
    k = gp.kernel_
    # ConstantKernel * RBF + WhiteKernel
    const   = k.k1.k1.constant_value
    lengths = k.k1.k2.length_scale          # ARD: one per dim, in normalised space
    noise   = k.k2.noise_level
    print("\nLearned hyperparameters (normalised input space):")
    print(f"  log-marginal-likelihood = {gp.log_marginal_likelihood_value_:.3f}")
    print(f"  output scale  σ_f²      = {const:.3e}")
    print(f"  noise level   σ_n²      = {noise:.3e}")
    for name, ell in zip(PARAM_KEYS, np.atleast_1d(lengths)):
        # Translate back to physical units for interpretability.
        phys_ell = ell * rng[PARAM_KEYS.index(name)]
        print(f"  length scale  ℓ_{name:7s} = {ell:.3f}   "
              f"(≈ {phys_ell:.4g} in physical units)")

    # 3. LOO-CV
    print("\nRunning leave-one-out cross-validation...")
    y_loo, sd_loo = loo_cv(X, y, n_restarts=args.n_restarts)

    err  = y_loo - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    # R² (1 - SS_res/SS_tot)
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # Standardised residuals: should be ~N(0,1) if the GP is well-calibrated.
    z = err / np.where(sd_loo > 0, sd_loo, 1.0)

    print(f"\nLOO results:")
    print(f"  RMSE = {rmse:.3e}")
    print(f"  MAE  = {mae:.3e}")
    print(f"  R²   = {r2:.4f}")
    print(f"  mean |z-score| = {np.mean(np.abs(z)):.2f}   (≈1 means well-calibrated)")

    # 4. Save per-case table
    csv_path = out_dir / "loo_predictions.csv"
    with open(csv_path, "w") as f:
        f.write("case,theta_0,dtheta,delta,y_true,y_loo_pred,y_loo_std,abs_err,z_score\n")
        for i, cid in enumerate(case_ids):
            f.write(
                f"{cid},{X[i,0]:g},{X[i,1]:g},{X[i,2]:g},"
                f"{y[i]:.6e},{y_loo[i]:.6e},{sd_loo[i]:.6e},"
                f"{abs(err[i]):.6e},{z[i]:+.3f}\n"
            )
    print(f"\nWrote {csv_path}")

    # 5. Plots
    plot_parity(y, y_loo, sd_loo, case_ids, out_dir / "parity_loo.png")
    plot_slices(gp, X, y, lo, rng, y_mu, y_sd, out_dir / "slices_nominal.png")
    print(f"Wrote {out_dir / 'parity_loo.png'}")
    print(f"Wrote {out_dir / 'slices_nominal.png'}")

    print("\nDone.")
    print("-" * 64)
    print("Reminder: with N=10 across only theta_0 ∈ {0, 90}, the θ₀")
    print("length scale is poorly identified. Treat predictions at")
    print("intermediate θ₀ (e.g. 45°) as extrapolation -- the ±2σ band")
    print("on the slice plot is the GP's own admission of that.")


if __name__ == "__main__":
    main()
