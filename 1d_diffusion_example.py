"""
1D diffusion example: 5-bin (5-layer) slab problem.

Uses:
  - lhs_generation.py   : DiffusionParamSampler to draw LHS parameter sets
  - second_order_solver.py : solve_diffusion() to compute the scalar flux

Problem setup
-------------
  Domain : x ∈ [0, L],  L = 10 cm
  Layers : 5 equal-width bins, each of width L/5 = 2 cm
  BCs    : zero-flux Dirichlet on both ends (φ(0) = φ(L) = 0)
  Mesh   : N = 100 cells (20 cells per layer)

A single "nominal" solve is printed and plotted.  A batch of M LHS samples
is also solved and the resulting flux envelopes are shown.
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

# ------------------------------------------------------------------ #
# Nominal (fixed) material properties – one set per bin               #
# ------------------------------------------------------------------ #
D_nominal       = np.array([1.0, 0.5, 1.5, 0.8, 1.2])   # diffusion coeff [cm]
Sigma_a_nominal = np.array([0.1, 0.8, 0.2, 0.5, 0.15])  # absorption [cm^-1]
q_nominal       = np.array([1.0, 0.0, 0.5, 0.0, 1.0])   # source [n cm^-3 s^-1]

# ------------------------------------------------------------------ #
# Nominal solve                                                        #
# ------------------------------------------------------------------ #
x, phi = solve_diffusion(
    L, N_CELLS, layer_bounds,
    D_nominal, Sigma_a_nominal, q_nominal,
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
)

print("=== 5-bin diffusion problem ===")
print(f"Layer bounds : {layer_bounds}")
print(f"D            : {D_nominal}")
print(f"Sigma_a      : {Sigma_a_nominal}")
print(f"q            : {q_nominal}")
print(f"Peak flux    : {phi.max():.4f}  at x = {x[phi.argmax()]:.2f} cm")

# ------------------------------------------------------------------ #
# LHS parameter study                                                  #
# ------------------------------------------------------------------ #
M_SAMPLES = 50
SEED      = 42

D_bounds       = [0.2, 2.0]    # [cm]
sigma_a_bounds = [0.05, 1.0]   # [cm^-1]
q_bounds       = [0.0, 2.0]    # [n cm^-3 s^-1]

sampler = DiffusionParamSampler(N_BINS, D_bounds, sigma_a_bounds, q_bounds)
X = sampler.sample(M_SAMPLES, random_state=SEED)

print(f"\nLHS design: {M_SAMPLES} samples, {sampler.p} parameters")

# Solve for every LHS sample and collect flux profiles
phi_all = np.empty((M_SAMPLES, N_CELLS))
for i, g in enumerate(X):
    D_i, Siga_i, q_i = sampler.unpack(g)
    _, phi_i = solve_diffusion(
        L, N_CELLS, layer_bounds, D_i, Siga_i, q_i,
        bc_left=('dirichlet', 0.0),
        bc_right=('dirichlet', 0.0),
    )
    phi_all[i] = phi_i

phi_mean = phi_all.mean(axis=0)
phi_lo   = phi_all.min(axis=0)
phi_hi   = phi_all.max(axis=0)

# ------------------------------------------------------------------ #
# Plot                                                                 #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(9, 5))

# LHS envelope
ax.fill_between(x, phi_lo, phi_hi, alpha=0.25, color='steelblue', label='LHS range')
ax.plot(x, phi_mean, color='steelblue', lw=1.5, ls='--', label='LHS mean')

# Nominal solution
ax.plot(x, phi, color='crimson', lw=2, label='Nominal')

# Layer boundaries
for xb in layer_bounds[1:-1]:
    ax.axvline(xb, color='gray', ls=':', lw=0.8)

ax.set_xlabel('x  [cm]', fontsize=12)
ax.set_ylabel('φ(x)  [a.u.]', fontsize=12)
ax.set_title(f'1D diffusion – 5-bin slab  (N_cells={N_CELLS}, M_LHS={M_SAMPLES})', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('output_graphs/5bin_diffusion.png', dpi=150)
print("\nSaved: output_graphs/5bin_diffusion.png")
plt.show()
