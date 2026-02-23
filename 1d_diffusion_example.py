"""
1D diffusion example: 5 layer slab problem.

Uses:
  - lhs_generation.py   : DiffusionParamSampler to draw LHS parameter sets
  - second_order_solver.py : solve_diffusion() to compute the scalar flux

Problem setup
-------------
  Domain : x ∈ [0, L],  L = 10 cm
  Layers : 5 equal-width bins, each of width L/5 = 2 cm
  BCs    : zero-flux Dirichlet on both ends (φ(0) = φ(L) = 0)
  Mesh   : N = 100 cells (20 cells per layer)

A batch of M LHS samples is solved and percentile flux profiles are plotted.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lhs_generation import DiffusionParamSampler
from second_order_solver import solve_diffusion

# ------------------------------------------------------------------ #
# Problem geometry                                                     #
# ------------------------------------------------------------------ #
L      = 10.0           # slab length [cm]
N_BINS = 5              # number of material layers / bins
N_CELLS = 100           # total finite-volume cells

layer_bounds = np.linspace(0.0, L, N_BINS + 1)   # [0, 2, 4, 6, 8, 10]

print("=== 5-layer diffusion problem ===")
print(f"Layer bounds : {layer_bounds}")

# ------------------------------------------------------------------ #
# LHS parameter study                                                  #
# ------------------------------------------------------------------ #
M_SAMPLES = 1000
SEED      = 42

D_bounds       = [0.2, 2.0]    # [cm]
sigma_a_bounds = [0.05, 1.0]   # [cm^-1]
q_bounds       = [0.0, 2.0]    # [n cm^-3 s^-1]

sampler = DiffusionParamSampler(N_BINS, D_bounds, sigma_a_bounds, q_bounds)
X = sampler.sample(M_SAMPLES, random_state=SEED)

print(f"\nLHS design: {M_SAMPLES} samples, {sampler.p} parameters")

# Solve for every LHS sample and collect flux profiles
phi_all = np.empty((M_SAMPLES, N_CELLS))
x = None
for i, g in enumerate(X):
    D_i, Siga_i, q_i = sampler.unpack(g)
    x_i, phi_i = solve_diffusion(
        L, N_CELLS, layer_bounds, D_i, Siga_i, q_i,
        bc_left=('dirichlet', 0.0),
        bc_right=('dirichlet', 0.0),
    )
    if x is None:
        x = x_i
    phi_all[i] = phi_i

# Percentile Counter
# 0 - 9
lower_percentiles = list(range(0, 10, 1))
# 10 - 90
midle_percentiles = list(range(10, 100, 10))
# 91 - 100
highr_percentiles = list(range(90, 101, 1))

PERCENTILES = lower_percentiles + midle_percentiles + highr_percentiles
phi_pcts = np.percentile(phi_all, PERCENTILES, axis=0)  # shape (9, N_CELLS)

# ------------------------------------------------------------------ #
# Plot                                                                 #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(9, 5))

cmap = plt.get_cmap('plasma')
norm = matplotlib.colors.Normalize(vmin=PERCENTILES[0], vmax=PERCENTILES[-1])
colors = [cmap(norm(pct)) for pct in PERCENTILES]

for pct, pct_vals, color in zip(PERCENTILES, phi_pcts, colors):
    ax.plot(x, pct_vals, color=color, lw=1.5)

# Layer boundaries
for xb in layer_bounds[1:-1]:
    ax.axvline(xb, color='gray', ls=':', lw=0.8)

ax.set_xlabel('x  [cm]', fontsize=12)
ax.set_ylabel('φ(x)  [a.u.]', fontsize=12)
ax.set_title(f'1D diffusion – 5-layer slab  (N_cells={N_CELLS}, M_LHS={M_SAMPLES})', fontsize=13)

sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Percentile', fontsize=11)
cbar_ticks = [0, 5, 10, 25, 50, 75, 90, 95, 100]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'P{p}' for p in cbar_ticks])
plt.tight_layout()
plt.savefig('output_graphs/5layer_diffusion.png', dpi=150)
print("\nSaved: output_graphs/5layer_diffusion.png")
plt.show()
