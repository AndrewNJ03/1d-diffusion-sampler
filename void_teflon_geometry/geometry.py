"""
Smooth Void/Teflon interface parameterization for the symmetric slab geometry.

This module is additive: it reuses the existing mesh/assembly/solve
infrastructure in common/second_order_solver.py unchanged, and only supplies
a new way to build the per-cell material arrays (D, Sigma_a, q) for the
mirrored slab geometry used in the constantR_void_10 study:

    Shield | Dielectric (Teflon) | Void | Core | Core | Void | Dielectric (Teflon) | Shield

The two Core halves are the same material, so materially this collapses to
a single Core region of half-width core_radius centered on the slab; only
the Void/Teflon interface actually moves across the case sweep (void
thickness and dielectric thickness trade off so their sum, the shell width,
stays fixed, as do core_radius and shield_thickness).

Under the existing assign_material_properties() (piecewise-constant, hard
step at layer_bounds), that interface's location is not differentiable:
moving it changes which mesh cell it falls in, not a smooth function of its
own coordinate. This module replaces the hard step, at the Void/Teflon
interface only, with the logistic activation

    sigma(x; a, b) = 1 / (1 + exp(-a*(x - b)))

so the interface location becomes a smooth parameter. `a` is strictly the
steepness of the transition and `b` is strictly the interface's physical
location (sigma(b) = 0.5 exactly, for any a) — there is no separate
conversion step. The Core/Void and Teflon/Shield boundaries are left as
hard steps, since they do not move in this study.

Public API
----------
sigmoid_switch(x, a, b)
    The activation function itself.

void_teflon_field(...)
    Smooth scalar field over a mesh for one material property.

assign_void_teflon_properties(...)
    Builds (D, Sigma_a, q) cell arrays in one call — drop-in compatible
    with assemble_system() from common/second_order_solver.py.

solve_void_teflon(...)
    Full forward solve: builds the mesh and smooth material arrays here,
    then calls assemble_system()/spsolve() from the existing common solver.
"""

import os
import sys

import numpy as np
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from second_order_solver import build_mesh, assemble_system  # noqa: E402


def sigmoid_switch(x, a, b):
    """
    Logistic activation used as a smooth Void/Teflon switch.

        sigma(x; a, b) = 1 / (1 + exp(-a*(x - b)))

    sigma -> 0 as a*(x-b) -> -infinity, sigma -> 1 as a*(x-b) -> +infinity;
    sigma(b) = 0.5 exactly. `a` is strictly the steepness of the
    transition; `b` is strictly the interface's physical center.
    """
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def void_teflon_field(
    x_centers: np.ndarray,
    L: float,
    core_radius: float,
    void_thickness: float,
    shell_width: float,
    val_core: float,
    val_void: float,
    val_teflon: float,
    val_shield: float,
    steepness: float,
):
    """
    Build a smooth material-property field over the mirrored slab

        Shield | Teflon | Void | Core | Core | Void | Teflon | Shield

    using a hard step at the fixed Core/Void and Teflon/Shield boundaries,
    and the logistic sigmoid_switch at the moving Void/Teflon boundary.

    Parameters
    ----------
    x_centers      : (N,) cell-center coordinates, x in [0, L]
    L              : slab length
    core_radius    : half-width of the (fixed) Core region, centered at L/2
    void_thickness : Void shell thickness (the moving geometry parameter);
                     0 <= void_thickness <= shell_width
    shell_width    : fixed combined Void + Teflon shell width
                     (= void_thickness + dielectric_thickness)
    val_core, val_void, val_teflon, val_shield : material property value
                     (e.g. D, Sigma_a, or q) for each region
    steepness      : sigmoid steepness `a`; larger -> closer to a hard step

    Returns
    -------
    field  : (N,) smooth property array
    (a, b) : sigmoid_switch coefficients used for the Void/Teflon interface
    """
    if not (0.0 <= void_thickness <= shell_width):
        raise ValueError(
            f"void_thickness ({void_thickness}) must lie in [0, shell_width={shell_width}]."
        )

    center = 0.5 * L
    r = np.abs(x_centers - center)               # radial distance from slab center
    r_interface = core_radius + void_thickness    # moving Void/Teflon boundary
    a, b = steepness, r_interface

    # Smooth Void -> Teflon blend, valid within the shell [core_radius, core_radius+shell_width]
    s = sigmoid_switch(r, a, b)
    shell_val = val_void + (val_teflon - val_void) * s

    field = np.where(r < core_radius, val_core, shell_val)
    field = np.where(r > core_radius + shell_width, val_shield, field)
    return field, (a, b)


def assign_void_teflon_properties(
    x_centers: np.ndarray,
    L: float,
    core_radius: float,
    void_thickness: float,
    shell_width: float,
    mat_core,
    mat_void,
    mat_teflon,
    mat_shield,
    steepness: float,
):
    """
    Build (D, Sigma_a, q) cell arrays for the mirrored Void/Teflon slab,
    smoothly switching at the moving Void/Teflon interface.

    mat_core, mat_void, mat_teflon, mat_shield : each a (D, Sigma_a, q)
        triple of material property values for that region.

    Returns
    -------
    D, Sigma_a, q : (N,) cell-wise arrays, drop-in compatible with
                    assemble_system() in common/second_order_solver.py
    (a, b)        : sigmoid_switch coefficients used for the Void/Teflon interface
    """
    D_core, Sa_core, q_core = mat_core
    D_void, Sa_void, q_void = mat_void
    D_tef, Sa_tef, q_tef    = mat_teflon
    D_shd, Sa_shd, q_shd    = mat_shield

    D, ab = void_teflon_field(
        x_centers, L, core_radius, void_thickness, shell_width,
        D_core, D_void, D_tef, D_shd, steepness,
    )
    Sigma_a, _ = void_teflon_field(
        x_centers, L, core_radius, void_thickness, shell_width,
        Sa_core, Sa_void, Sa_tef, Sa_shd, steepness,
    )
    q, _ = void_teflon_field(
        x_centers, L, core_radius, void_thickness, shell_width,
        q_core, q_void, q_tef, q_shd, steepness,
    )
    return D, Sigma_a, q, ab


def solve_void_teflon(
    L: float,
    n_cells: int,
    core_radius: float,
    void_thickness: float,
    shell_width: float,
    mat_core,
    mat_void,
    mat_teflon,
    mat_shield,
    steepness: float,
    bc_left=('dirichlet', 0.0),
    bc_right=('dirichlet', 0.0),
):
    """
    Full forward solve on the mirrored Void/Teflon slab.

    Builds the mesh and the smooth per-cell material arrays here, then
    reuses assemble_system()/spsolve() from the existing (unmodified)
    common/second_order_solver.py infrastructure.

    Returns
    -------
    x_centers : (N,) cell-center coordinates
    phi       : (N,) scalar flux solution
    (a, b)    : sigmoid_switch coefficients used for the Void/Teflon interface
    """
    x_centers, x_faces, dx = build_mesh(L, n_cells)
    D, Sigma_a, q, ab = assign_void_teflon_properties(
        x_centers, L, core_radius, void_thickness, shell_width,
        mat_core, mat_void, mat_teflon, mat_shield, steepness,
    )
    A, rhs = assemble_system(dx, D, Sigma_a, q, bc_left, bc_right)
    phi = spsolve(A, rhs)
    return x_centers, phi, ab
