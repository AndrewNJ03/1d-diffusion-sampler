"""
5-layer diffusion study: varying diffusion coefficient in layer 5 only.

Geometry
--------
  Domain       : x ∈ [0, 10] cm, 5 equal layers of width 2 cm each
  Layer bounds : [0, 2, 4, 6, 8, 10]

Fixed parameters (layers 1-5)
------------------------------
  D       = 1.0  cm          (layers 1-4, fixed)
  Sigma_a = 0.5  cm^-1       (all layers)
  q       = 1.0  n/cm^3/s    (all layers)

Varying parameter
-----------------
  D_5 ∈ [0.2, 2.0]  cm    (layer 5 only, sampled via LHS)

BCs    : zero-flux Dirichlet on both ends (φ(0) = φ(10) = 0)
Mesh   : N = 200 cells
Plot   : solutions restricted to x ∈ [1, 5]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from second_order_solver import solve_diffusion
from lhs_generation import DiffusionParamSampler
from masking_function import mask_solution

# ------------------------------------------------------------------ #
# Geometry                                                            #
# ------------------------------------------------------------------ #
L            = 10.0
N_LAYERS     = 5
N_CELLS      = 200
layer_bounds = np.linspace(0.0, L, N_LAYERS + 1)   # [0,2,4,6,8,10]

PLOT_MIN = 1.0
PLOT_MAX = 5.0

# ------------------------------------------------------------------ #
# Parameter setup                                                      #
# ------------------------------------------------------------------ #
D_FIXED     = 1.0
D5_RANGE    = [0.2, 2.0]   # only layer 5 varies
SIGMA_FIXED = 0.5
Q_FIXED     = 1.0

M_SAMPLES = 40
SEED      = 0

# Degenerate bounds [v, v] → fixed; only D for layer 5 is open
D_bounds       = [[D_FIXED, D_FIXED]] * (N_LAYERS - 1) + [D5_RANGE]
sigma_a_bounds = [[SIGMA_FIXED, SIGMA_FIXED]] * N_LAYERS
q_bounds       = [[Q_FIXED, Q_FIXED]] * N_LAYERS

sampler = DiffusionParamSampler(N_LAYERS, D_bounds, sigma_a_bounds, q_bounds)
X = sampler.sample(M_SAMPLES, random_state=SEED)

# D_5 values actually sampled (column index N_LAYERS - 1 = 4)
D5_values = X[:, N_LAYERS - 1]
print(f"D_5 sample range: [{D5_values.min():.3f}, {D5_values.max():.3f}]")

# ------------------------------------------------------------------ #
# Solve for each sample                                               #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))

cmap   = plt.get_cmap('plasma')
norm   = plt.Normalize(vmin=D5_RANGE[0], vmax=D5_RANGE[1])

phi_at_x1 = []
phi_at_x5 = []

for i, g in enumerate(X):
    D_i, Siga_i, q_i = sampler.unpack(g)
    x_full, phi_full = solve_diffusion(
        L, N_CELLS, layer_bounds, D_i, Siga_i, q_i,
        bc_left=('dirichlet', 0.0),
        bc_right=('dirichlet', 0.0),
    )
    x_m, phi_m = mask_solution(x_full, phi_full, PLOT_MIN, PLOT_MAX)
    phi_at_x1.append(phi_m[0])
    phi_at_x5.append(phi_m[-1])
    ax.plot(x_m, phi_m, color=cmap(norm(D5_values[i])), lw=1.0, alpha=0.8)

print(f"Max flux difference at x≈{PLOT_MIN}: {max(phi_at_x1) - min(phi_at_x1):.4f}")
print(f"Max flux difference at x≈{PLOT_MAX}: {max(phi_at_x5) - min(phi_at_x5):.4f}")

# ------------------------------------------------------------------ #
# Colorbar and labels                                                #
# ------------------------------------------------------------------ #
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("$D_5$  [cm]", fontsize=11)

ax.set_xlabel("x  [cm]", fontsize=12)
ax.set_ylabel("φ(x)  [a.u.]", fontsize=12)
ax.set_title(
    f"5-layer slab – varying $D_5 \\in {D5_RANGE}$\n"
    f"solution shown on $x \\in [{PLOT_MIN}, {PLOT_MAX}]$",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("output_graphs/varying_D_layer5.png", dpi=150)
print("Saved: output_graphs/varying_D_layer5.png")
plt.show()
