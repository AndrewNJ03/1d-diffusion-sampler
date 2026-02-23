"""
Masking utility for 1D diffusion solutions.

Public API
----------
mask_solution(x, phi, x_min, x_max)
    Returns the sub-arrays of x and phi restricted to [x_min, x_max].

plot_masked_solution(x, phi, x_min, x_max, **kwargs)
    Plots only the interval of interest from the solution.
"""

import numpy as np
import matplotlib.pyplot as plt


def mask_solution(
    x: np.ndarray,
    phi: np.ndarray,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict a solution to the interval [x_min, x_max].

    Parameters
    ----------
    x     : (N,) cell-center coordinates
    phi   : (N,) scalar flux (or any solution array of the same length)
    x_min : left end of the interval of interest
    x_max : right end of the interval of interest

    Returns
    -------
    x_masked   : (M,) coordinates within [x_min, x_max]
    phi_masked : (M,) solution values at those coordinates
    """
    if x_min >= x_max:
        raise ValueError(f"x_min ({x_min}) must be strictly less than x_max ({x_max}).")

    mask = (x >= x_min) & (x <= x_max)
    return x[mask], phi[mask]


def plot_masked_solution(
    x: np.ndarray,
    phi: np.ndarray,
    x_min: float,
    x_max: float,
    ax=None,
    label: str = "φ(x)",
    title: str | None = None,
    **plot_kwargs,
):
    """
    Plot the solution restricted to [x_min, x_max].

    Parameters
    ----------
    x, phi      : full solution arrays returned by solve_diffusion()
    x_min, x_max: interval of interest
    ax          : existing Axes to draw on; creates a new figure if None
    label       : line label for the legend
    title       : axes title; auto-generated if None
    **plot_kwargs: forwarded to ax.plot()

    Returns
    -------
    ax : the Axes object used for plotting
    """
    x_m, phi_m = mask_solution(x, phi, x_min, x_max)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x_m, phi_m, label=label, **plot_kwargs)
    ax.set_xlabel("x  [cm]")
    ax.set_ylabel("φ(x)  [a.u.]")
    ax.set_title(title or f"Solution on [{x_min}, {x_max}]")
    ax.legend()
    plt.tight_layout()

    return ax
