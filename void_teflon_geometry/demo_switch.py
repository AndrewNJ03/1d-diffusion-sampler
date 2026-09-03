"""
Demo: smooth Void/Teflon switch vs. the hard piecewise-constant reference.

Uses:
  - common/second_order_solver.py : build_mesh(), assemble_system(),
                                     assign_material_properties() (hard reference)
  - void_teflon_geometry/geometry.py : void_teflon_field(), solve_void_teflon()
                                        (smooth switch, this module)

Geometry constants below match data/constantR_void_10/void.py /
case_key.txt (core_radius, shield_thickness, and the fixed void+dielectric
shell width). No CSV data is read here — this only exercises the ROM
sandbox's own solver with the new geometry parameterization.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh, assign_material_properties, assemble_system
from scipy.sparse.linalg import spsolve

from geometry import void_teflon_field, solve_void_teflon

# ------------------------------------------------------------------ #
# Geometry (matches data/constantR_void_10/void.py)                  #
# ------------------------------------------------------------------ #
L                 = 1.0
core_radius       = 0.16921
shield_thickness  = 0.07801
shell_width       = 0.25278            # void_thickness + dielectric_thickness
void_thickness    = 0.05               # exaggerated vs. case 0000 for a visible transition
dielectric_thk    = shell_width - void_thickness

N_CELLS = 2000

# Illustrative (non-physical) material values — placeholders, one per region.
val_D  = dict(core=1.0, void=1.8, teflon=0.6, shield=0.3)
val_Sa = dict(core=0.5, void=0.01, teflon=0.4, shield=0.9)
val_q  = dict(core=1.0, void=0.0, teflon=0.0, shield=0.0)

mat_core   = (val_D['core'],   val_Sa['core'],   val_q['core'])
mat_void   = (val_D['void'],   val_Sa['void'],   val_q['void'])
mat_teflon = (val_D['teflon'], val_Sa['teflon'], val_q['teflon'])
mat_shield = (val_D['shield'], val_Sa['shield'], val_q['shield'])

# ------------------------------------------------------------------ #
# Hard reference profile (existing piecewise-constant assignment)     #
# ------------------------------------------------------------------ #
x_centers, x_faces, dx = build_mesh(L, N_CELLS)

layer_bounds = [
    0.0,
    shield_thickness,
    shield_thickness + dielectric_thk,
    shield_thickness + dielectric_thk + void_thickness,
    L - (shield_thickness + dielectric_thk + void_thickness),
    L - (shield_thickness + dielectric_thk),
    L - shield_thickness,
    L,
]
D_layers  = [val_D['shield'],  val_D['teflon'],  val_D['void'],  val_D['core'],
             val_D['void'],  val_D['teflon'],  val_D['shield']]
Sa_layers = [val_Sa['shield'], val_Sa['teflon'], val_Sa['void'], val_Sa['core'],
             val_Sa['void'], val_Sa['teflon'], val_Sa['shield']]
q_layers  = [val_q['shield'],  val_q['teflon'],  val_q['void'],  val_q['core'],
             val_q['void'],  val_q['teflon'],  val_q['shield']]

D_hard, Sa_hard, q_hard = assign_material_properties(x_centers, layer_bounds, D_layers, Sa_layers, q_layers)
A_hard, rhs_hard = assemble_system(dx, D_hard, Sa_hard, q_hard,
                                    bc_left=('dirichlet', 0.0), bc_right=('dirichlet', 0.0))
phi_hard = spsolve(A_hard, rhs_hard)

# ------------------------------------------------------------------ #
# Smooth switch at increasing steepness                               #
# ------------------------------------------------------------------ #
steepnesses = (50, 200, 1000)

fig, (ax_mat, ax_flux) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

ax_mat.plot(x_centers, D_hard, 'k-', lw=1.5, label='Hard piecewise-constant (D)')
ax_flux.plot(x_centers, phi_hard, 'k-', lw=1.5, label='Hard piecewise-constant')

for steepness in steepnesses:
    D_smooth, ab = void_teflon_field(
        x_centers, L, core_radius, void_thickness, shell_width,
        val_D['core'], val_D['void'], val_D['teflon'], val_D['shield'], steepness,
    )
    ax_mat.plot(x_centers, D_smooth, lw=1.3, label=f'Smooth switch  a={steepness}')

    x_c, phi_smooth, _ = solve_void_teflon(
        L, N_CELLS, core_radius, void_thickness, shell_width,
        mat_core, mat_void, mat_teflon, mat_shield, steepness,
        bc_left=('dirichlet', 0.0), bc_right=('dirichlet', 0.0),
    )
    ax_flux.plot(x_c, phi_smooth, lw=1.3, label=f'Smooth switch  a={steepness}')

for xb in (core_radius, core_radius + void_thickness,
           L - core_radius - void_thickness, L - core_radius,
           shield_thickness, L - shield_thickness):
    ax_mat.axvline(xb, color='gray', ls=':', lw=0.6)
    ax_flux.axvline(xb, color='gray', ls=':', lw=0.6)

ax_mat.set_ylabel('D(x)  [a.u.]', fontsize=11)
ax_mat.set_title('Smooth Void/Teflon switch vs. hard geometry', fontsize=12)
ax_mat.legend(fontsize=9)

ax_flux.set_xlabel('x  [cm]', fontsize=11)
ax_flux.set_ylabel(r'$\phi(x)$', fontsize=11)
ax_flux.set_title('Resulting flux', fontsize=12)
ax_flux.legend(fontsize=9)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), 'output_graphs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'void_teflon_switch_demo.png')
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

max_err = np.max(np.abs(D_smooth - D_hard))
print(f"Max |D_smooth - D_hard| at steepness={steepnesses[-1]}: {max_err:.4e}")
