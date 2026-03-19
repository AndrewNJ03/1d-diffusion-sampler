"""
POD sensitivity study — two independent analyses:

Study 1 – Snapshot convergence
    Vary M_TRAIN ∈ {50, 100, 200, 400} with a fixed QoI window.
    For each M_TRAIN, build an independent LHS, compute the SVD, and record:
      - the first 10 singular values,
      - modes needed for η = 99.9 % and η = 99.99 %,
      - median test projection error ε_proj at the chosen rank.

Study 2 – QoI window comparison
    Fix M_TRAIN = 400 and sweep several [xmin, xmax] intervals.
    For each window, record the same SVD metrics to compare how
    spatial localisation affects the dimensionality of the snapshot manifold.

Both studies share the same test set (M_TEST = 100, SEED = 2) so results
are directly comparable.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from lhs_generation import DiffusionParamSampler
from second_order_solver import solve_diffusion, build_mesh

# ------------------------------------------------------------------ #
# Shared configuration                                                #
# ------------------------------------------------------------------ #
L            = 10.0
N_BINS       = 5
N_CELLS      = 200
layer_bounds = np.linspace(0.0, L, N_BINS + 1)

D_bounds       = [0.2, 2.0]
sigma_a_bounds = [0.05, 1.0]
q_bounds       = [0.0, 2.0]

M_TEST    = 100
SEED_TEST = 2       # fixed test seed — same across both studies
SEED_BASE = 10      # base seed for training sets (incremented per run)

OUTPUT_DIR = "output_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sampler = DiffusionParamSampler(N_BINS, D_bounds, sigma_a_bounds, q_bounds)
x_centers, x_faces, _ = build_mesh(L, N_CELLS)

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def build_qoi_mask(xmin, xmax):
    """Boolean mask selecting cells whose interval overlaps [xmin, xmax]."""
    return (x_faces[:-1] < xmax) & (x_faces[1:] > xmin)


def solve_qoi_batch(X_params, mask):
    """Solve diffusion for each parameter row; return (Ny, M) QoI matrix."""
    Ny    = mask.sum()
    M     = len(X_params)
    Y_out = np.empty((Ny, M))
    for i, g in enumerate(X_params):
        D_i, Siga_i, q_i = sampler.unpack(g)
        _, phi_i = solve_diffusion(
            L, N_CELLS, layer_bounds, D_i, Siga_i, q_i,
            bc_left=('dirichlet', 0.0),
            bc_right=('dirichlet', 0.0),
        )
        Y_out[:, i] = phi_i[mask]
    return Y_out


def svd_metrics(Y_tr, Y_te):
    """
    Compute SVD of Y_tr and return a dict of key metrics evaluated on Y_te.

    Returns
    -------
    dict with keys:
        sigma       : (r_max,) singular values
        r_999       : modes for 99.9 % energy
        r_9999      : modes for 99.99 % energy
        eps_median  : median test ε_proj at rank r_999
        eps_max     : max    test ε_proj at rank r_999
        Psi         : (Ny, r_999) POD basis
    """
    U, sigma, _ = np.linalg.svd(Y_tr, full_matrices=False)

    energy     = sigma ** 2
    cum_energy = np.cumsum(energy) / energy.sum()

    r_999  = int(np.searchsorted(cum_energy, 0.999))  + 1
    r_9999 = int(np.searchsorted(cum_energy, 0.9999)) + 1

    Psi_r = U[:, :r_999]
    proj  = Psi_r @ (Psi_r.T @ Y_te)
    eps   = np.linalg.norm(Y_te - proj, axis=0) / np.linalg.norm(Y_te, axis=0)

    return dict(
        sigma      = sigma,
        r_999      = r_999,
        r_9999     = r_9999,
        eps_median = float(np.median(eps)),
        eps_max    = float(np.max(eps)),
        Psi        = Psi_r,
    )


# ================================================================== #
# Study 1 – Snapshot convergence                                      #
# ================================================================== #
print("=" * 60)
print("Study 1 – Snapshot convergence")
print("=" * 60)

QOI_XMIN_S1 = L / 3
QOI_XMAX_S1 = 2.0 * L / 3
mask_s1     = build_qoi_mask(QOI_XMIN_S1, QOI_XMAX_S1)
Ny_s1       = mask_s1.sum()

M_TRAIN_LIST = [50, 100, 200, 400]

# Fixed test set for Study 1
X_te_s1 = sampler.sample(M_TEST, random_state=SEED_TEST)
Y_te_s1 = solve_qoi_batch(X_te_s1, mask_s1)
print(f"QoI window : [{QOI_XMIN_S1:.2f}, {QOI_XMAX_S1:.2f}] cm,  Ny = {Ny_s1}")
print(f"Test set   : {M_TEST} samples\n")

results_s1 = {}
for k, M in enumerate(M_TRAIN_LIST):
    X_tr = sampler.sample(M, random_state=SEED_BASE + k)
    Y_tr = solve_qoi_batch(X_tr, mask_s1)
    m    = svd_metrics(Y_tr, Y_te_s1)
    results_s1[M] = m
    print(f"  M={M:4d}  r_999={m['r_999']}  r_9999={m['r_9999']}"
          f"  σ_1={m['sigma'][0]:.2f}"
          f"  ε_proj median={m['eps_median']:.2e}  max={m['eps_max']:.2e}")

# --- Plot 1a: singular value decay for each M ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
colors_m = plt.get_cmap('plasma')(np.linspace(0.1, 0.9, len(M_TRAIN_LIST)))
for col, M in zip(colors_m, M_TRAIN_LIST):
    sig = results_s1[M]['sigma']
    n   = min(20, len(sig))
    ax.semilogy(np.arange(1, n + 1), sig[:n], 'o-', ms=4, lw=1.5,
                color=col, label=f'M={M}')
ax.set_xlabel("Mode index  $k$", fontsize=12)
ax.set_ylabel(r"$\sigma_k$", fontsize=12)
ax.set_title("Singular value decay vs training size", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', ls='--', alpha=0.5)

# --- Plot 1b: r_999 and median ε_proj vs M ---
ax = axes[1]
r_vals     = [results_s1[M]['r_999']      for M in M_TRAIN_LIST]
eps_medians = [results_s1[M]['eps_median'] for M in M_TRAIN_LIST]
eps_maxs    = [results_s1[M]['eps_max']    for M in M_TRAIN_LIST]

ax2 = ax.twinx()
ax.plot(M_TRAIN_LIST, r_vals, 's-', ms=7, lw=1.8, color='steelblue',
        label='Rank $r$ (η=99.9%)')
ax2.semilogy(M_TRAIN_LIST, eps_medians, 'o--', ms=6, lw=1.5, color='tomato',
             label='Median $\\varepsilon_{proj}$')
ax2.semilogy(M_TRAIN_LIST, eps_maxs,    '^:', ms=6, lw=1.2, color='darkorange',
             label='Max $\\varepsilon_{proj}$')

ax.set_xlabel("Training set size  $M$", fontsize=12)
ax.set_ylabel("Rank $r$ (99.9% energy)", fontsize=12, color='steelblue')
ax2.set_ylabel("Test $\\varepsilon_{proj}$", fontsize=12, color='tomato')
ax.set_title("Rank and test error vs training size", fontsize=12)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(True, ls='--', alpha=0.4)

fig.suptitle(
    f"Study 1: Snapshot convergence  "
    f"(QoI=[{QOI_XMIN_S1:.2f},{QOI_XMAX_S1:.2f}] cm)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/study1_convergence.png", dpi=150)
print(f"\nSaved: {OUTPUT_DIR}/study1_convergence.png")


# ================================================================== #
# Study 2 – QoI window comparison                                     #
# ================================================================== #
print("\n" + "=" * 60)
print("Study 2 – QoI window comparison")
print("=" * 60)

# Windows to compare: vary position and width across the slab
QOI_WINDOWS = [
    (0.0,      L / 3,    "left third     [0.00, 3.33]"),
    (L / 3,    2*L / 3,  "middle third   [3.33, 6.67]"),
    (2*L / 3,  L,        "right third    [6.67, 10.0]"),
    (L / 4,    3*L / 4,  "middle half    [2.50, 7.50]"),
    (0.0,      L,        "full domain    [0.00, 10.0]"),
]

M_TRAIN_S2 = 400
X_tr_s2    = sampler.sample(M_TRAIN_S2, random_state=SEED_BASE + 99)

results_s2 = {}
for xmin, xmax, label in QOI_WINDOWS:
    mask = build_qoi_mask(xmin, xmax)
    Ny   = mask.sum()
    Y_tr = solve_qoi_batch(X_tr_s2, mask)
    X_te = sampler.sample(M_TEST,   random_state=SEED_TEST)
    Y_te = solve_qoi_batch(X_te,    mask)
    m    = svd_metrics(Y_tr, Y_te)
    results_s2[label] = {**m, 'Ny': Ny, 'xmin': xmin, 'xmax': xmax}
    print(f"  {label}  Ny={Ny:3d}  r_999={m['r_999']}  r_9999={m['r_9999']}"
          f"  σ_1={m['sigma'][0]:.2f}"
          f"  ε median={m['eps_median']:.2e}")

# --- Plot 2a: singular value decay per window ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
colors_w = plt.get_cmap('tab10').colors
for col, (label, m) in zip(colors_w, results_s2.items()):
    sig = m['sigma']
    n   = min(15, len(sig))
    ax.semilogy(np.arange(1, n + 1), sig[:n], 'o-', ms=4, lw=1.5,
                color=col, label=label.split()[0] + ' ' + label.split()[1])
ax.set_xlabel("Mode index  $k$", fontsize=12)
ax.set_ylabel(r"$\sigma_k$", fontsize=12)
ax.set_title("Singular value decay by QoI window", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, which='both', ls='--', alpha=0.5)

# --- Plot 2b: r_999, r_9999, median ε vs window ---
ax = axes[1]
labels_short = [v.split()[0] + '\n' + v.split()[1] for v in results_s2]
r999_vals  = [results_s2[l]['r_999']      for l in results_s2]
r9999_vals = [results_s2[l]['r_9999']     for l in results_s2]
eps_vals   = [results_s2[l]['eps_median'] for l in results_s2]
x_pos      = np.arange(len(labels_short))
width      = 0.3

bars1 = ax.bar(x_pos - width/2, r999_vals,  width, label='r (99.9%)',  color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, r9999_vals, width, label='r (99.99%)', color='navy',      alpha=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_short, fontsize=9)
ax.set_ylabel("Rank  $r$", fontsize=12)
ax.set_title("Rank needed by QoI window", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, axis='y', ls='--', alpha=0.5)

ax2 = ax.twinx()
ax2.semilogy(x_pos, eps_vals, 'D-', ms=7, lw=1.5, color='tomato',
             label='Median $\\varepsilon_{proj}$')
ax2.set_ylabel("Median $\\varepsilon_{proj}$", fontsize=12, color='tomato')
ax2.legend(fontsize=9, loc='upper right')

fig.suptitle(
    f"Study 2: QoI window comparison  (M_train={M_TRAIN_S2})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/study2_qoi_windows.png", dpi=150)
print(f"\nSaved: {OUTPUT_DIR}/study2_qoi_windows.png")

plt.show()

# ------------------------------------------------------------------ #
# Summary table                                                        #
# ------------------------------------------------------------------ #
print("\n=== Study 1 summary: convergence ===")
print(f"  {'M':>6}  {'r_999':>6}  {'r_9999':>7}  {'eps_median':>12}  {'eps_max':>10}")
for M in M_TRAIN_LIST:
    m = results_s1[M]
    print(f"  {M:6d}  {m['r_999']:6d}  {m['r_9999']:7d}  "
          f"{m['eps_median']:12.2e}  {m['eps_max']:10.2e}")

print("\n=== Study 2 summary: QoI windows ===")
print(f"  {'Window':<30}  {'Ny':>4}  {'r_999':>6}  {'r_9999':>7}  {'eps_median':>12}")
for label, m in results_s2.items():
    print(f"  {label:<30}  {m['Ny']:4d}  {m['r_999']:6d}  {m['r_9999']:7d}  "
          f"{m['eps_median']:12.2e}")
