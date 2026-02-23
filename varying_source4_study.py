"""
5-layer diffusion study: varying source in layer 5 only.

Geometry
--------
  Domain       : x ∈ [0, 10] cm, 5 equal layers of width 2 cm each
  Layer bounds : [0, 2, 4, 6, 8, 10]

Fixed parameters (layers 1-5)
------------------------------
  D       = 1.0  cm          (all layers)
  Sigma_a = 0.5  cm^-1       (all layers)
  q       = 1.0  n/cm^3/s    (layers 1-3, 5 fixed)

Varying parameter
-----------------
  q_4 ∈ [0, 3]  n/cm^3/s    (layer 5 only, sampled via LHS)

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
SIGMA_FIXED = 0.5
Q_FIXED     = 1.0
Q4_RANGE    = [0.0, 3.0]   # only layer 4 varies

M_SAMPLES = 40
SEED      = 0

# Degenerate bounds [v, v] → fixed; only q for layer 4 is open
D_bounds       = [[D_FIXED,     D_FIXED]]     * N_LAYERS
sigma_a_bounds = [[SIGMA_FIXED, SIGMA_FIXED]] * N_LAYERS
q_bounds       = [[Q_FIXED, Q_FIXED]] * (N_LAYERS - 2) + [Q4_RANGE] + [[Q_FIXED, Q_FIXED]]

sampler = DiffusionParamSampler(N_LAYERS, D_bounds, sigma_a_bounds, q_bounds)
X = sampler.sample(M_SAMPLES, random_state=SEED)

# q_4 values actually sampled
q4_values = X[:, -2]
print(f"q_4 sample range: [{q4_values.min():.3f}, {q4_values.max():.3f}]")

# ------------------------------------------------------------------ #
# Solve for each sample                                               #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))

cmap   = plt.get_cmap('plasma')
norm   = plt.Normalize(vmin=Q4_RANGE[0], vmax=Q4_RANGE[1])

for i, g in enumerate(X):
    D_i, Siga_i, q_i = sampler.unpack(g)
    x_full, phi_full = solve_diffusion(
        L, N_CELLS, layer_bounds, D_i, Siga_i, q_i,
        bc_left=('dirichlet', 0.0),
        bc_right=('dirichlet', 0.0),
    )
    x_m, phi_m = mask_solution(x_full, phi_full, PLOT_MIN, PLOT_MAX)
    ax.plot(x_m, phi_m, color=cmap(norm(q4_values[i])), lw=1.0, alpha=0.8)

# ------------------------------------------------------------------ #
# Colorbar and labels                                                  #
# ------------------------------------------------------------------ #
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("$q_4$  [n cm$^{-3}$ s$^{-1}$]", fontsize=11)

ax.set_xlabel("x  [cm]", fontsize=12)
ax.set_ylabel("φ(x)  [a.u.]", fontsize=12)
ax.set_title(
    f"5-layer slab – varying $q_4 \\in {Q4_RANGE}$\n"
    f"solution shown on $x \\in [{PLOT_MIN}, {PLOT_MAX}]$",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("output_graphs/varying_source_layer4.png", dpi=150)
print("Saved: output_graphs/varying_source_layer4.png")
plt.show()
