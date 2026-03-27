"""
1D diffusion example: 5 layer slab problem.

Uses:
  - lhs_generation.py   : DiffusionParamSampler to draw LHS parameter sets
  - second_order_solver.py : solve_diffusion() to compute the scalar flux

A batch of M LHS samples is solved and 5 example flux profiles are plotted.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lhs_generation import DiffusionParamSampler
from second_order_solver import solve_diffusion

# ------------------------------------------------------------------ #
# Problem geometry                                                   #
# ------------------------------------------------------------------ #
L      = 10.0           # slab length [cm]
N_BINS = 5              # number of material layers / bins
N_CELLS = 100           # total finite-volume cells (20 per material)

layer_bounds = np.linspace(0.0, L, N_BINS + 1)   # [0, 2, 4, 6, 8, 10]

print("=== 5-layer diffusion problem ===")
print(f"Layer bounds : {layer_bounds}")

# ------------------------------------------------------------------ #
# LHS parameter study                                                #
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

# ------------------------------------------------------------------ #
# Plot                                                               #
# ------------------------------------------------------------------ #
N_EXAMPLES = 5
rng = np.random.default_rng(SEED)
example_indices = rng.choice(M_SAMPLES, size=N_EXAMPLES, replace=False)

fig, ax = plt.subplots(figsize=(9, 5))

colors = plt.get_cmap('tab10').colors
for i, idx in enumerate(example_indices):
    ax.plot(x, phi_all[idx], color=colors[i], lw=1.5, label=f'Sample {i+1}')

# Layer boundaries
for xb in layer_bounds[1:-1]:
    ax.axvline(xb, color='gray', ls=':', lw=0.8)

ax.set_xlabel('x  [cm]', fontsize=12)
ax.set_ylabel('φ(x)  [a.u.]', fontsize=12)
ax.set_title(f'1D diffusion – 5-layer slab  (N_cells={N_CELLS}, M_LHS={M_SAMPLES})', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('output_graphs/5layer_diffusion.png', dpi=150)
print("\nSaved: output_graphs/5layer_diffusion.png")
plt.show()
