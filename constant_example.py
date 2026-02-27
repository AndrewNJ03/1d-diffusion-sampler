"""
1D diffusion example: Constant Solution

Solves a uniform slab with:
  Q       = 1.0  [n cm^-3 s^-1]  (uniform source)
  sigma_a = 1.0  [cm^-1]         (uniform absorption)
  D       = 1/3  [cm]            (uniform diffusion coefficient)

Uses:
  - second_order_solver.py : solve_diffusion() to compute the scalar flux
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from second_order_solver import solve_diffusion

# ------------------------------------------------------------------ #
# Problem geometry                                                   #
# ------------------------------------------------------------------ #
L      = 10.0           # slab length [cm]
N_BINS = 5              # number of material layers / bins
N_CELLS = 100           # total finite-volume cells (20 per material)

layer_bounds = np.linspace(0.0, L, N_BINS + 1)   # [0, 2, 4, 6, 8, 10]

print("=== Constant solution: Q=1, sigma_a=1 ===")
print(f"Layer bounds : {layer_bounds}")

# ------------------------------------------------------------------ #
# Constant material properties                                       #
# ------------------------------------------------------------------ #
Q       = 1.0          # uniform source [n cm^-3 s^-1]
SIGMA_A = 1.0          # uniform absorption [cm^-1]
D       = 1.0 / 3.0   # uniform diffusion coefficient [cm]

D_vec      = np.full(N_BINS, D)
siga_vec   = np.full(N_BINS, SIGMA_A)
q_vec      = np.full(N_BINS, Q)

# ------------------------------------------------------------------ #
# Solve                                                              #
# ------------------------------------------------------------------ #
x, phi = solve_diffusion(
    L, N_CELLS, layer_bounds, D_vec, siga_vec, q_vec,
    bc_left=('dirichlet', Q / SIGMA_A),
    bc_right=('dirichlet', Q / SIGMA_A),
)

print(f"\nMax flux: {phi.max():.4f}  at x = {x[phi.argmax()]:.2f} cm")
print(f"Analytic max (Q/sigma_a = {Q/SIGMA_A:.4f})")

# ------------------------------------------------------------------ #
# Plot                                                               #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, phi, lw=2, label=f'Numerical (Q={Q}, σ_a={SIGMA_A}, D={D:.3f})')

# Layer boundaries
for xb in layer_bounds[1:-1]:
    ax.axvline(xb, color='gray', ls=':', lw=0.8)

ax.set_xlabel('x  [cm]', fontsize=12)
ax.set_ylabel('φ(x)  [a.u.]', fontsize=12)
ax.set_title(f'1D diffusion – constant solution  (N_cells={N_CELLS})', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('output_graphs/constant_solution.png', dpi=150)
print("\nSaved: output_graphs/constant_solution.png")
plt.show()
