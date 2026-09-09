"""
Shared plotting helper for the void_teflon_geometry scripts.

Collects the one piece of visual logic several scripts here repeat: marking
where the slab's material composition switches (Core/shell, Teflon/Shield,
and -- when relevant to the plot -- the moving Void/Teflon interface) with a
dashed vertical line.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from geometry import material_boundaries  # noqa: E402


def draw_material_boundaries(
    ax, L, core_radius, shell_width, void_thickness=None,
    xlim=None, color='gray', ls='--', lw=0.9, label='Material boundary',
):
    """
    Draw a dashed vertical line at every material boundary that actually
    falls inside xlim (default: the axes' current x-limits, so call this
    after the data is plotted / axes are autoscaled) -- i.e. only the
    switches that are relevant to what this plot is showing. Only the
    first drawn line gets a legend label.

    Returns the x-locations that were drawn (possibly empty).
    """
    lo, hi = xlim if xlim is not None else ax.get_xlim()
    xs = material_boundaries(L, core_radius, shell_width, void_thickness)
    xs = xs[(xs >= lo) & (xs <= hi)]
    for i, x in enumerate(xs):
        ax.axvline(x, color=color, ls=ls, lw=lw, label=label if i == 0 else None)
    return xs
