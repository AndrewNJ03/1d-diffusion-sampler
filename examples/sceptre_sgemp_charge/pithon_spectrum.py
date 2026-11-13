"""
pithon_spectrum.py: sampled PITHON 250-keV flash x-ray spectra on the SCEPTRE
30-group photon structure, one spectrum per case.

Model (fit to Fig. 2 of SAND2014-15576, 250-keV endpoint, unit fluence):

    N(E) = (1 - fK) * n_c(E) + fK * [ (1-r) G(E; E_Ka, sig) + r G(E; E_Kb, sig) ]
    n_c(E) = A * E^-1 * (1 - E/E0)^a * exp(-(Ec/E)^3),   integral of n_c = 1

    E0  endpoint (diode voltage)                  [keV]
    a   bremsstrahlung roll-off exponent          (1 = thick-target Kramers, ~2 thin foil)
    Ec  low-energy cutoff (anode/window/air/filter photoelectric absorption) [keV]
    fK  Ta K-line photon fraction
    r   K-beta share of the line photons

Each case gets one full spectrum drawn from a Latin hypercube over these knobs, so
the 30 group values stay physically consistent with one another. Never sample the
groups independently. Fix a knob by giving it degenerate bounds (v, v), the same
convention as common/lhs_generation.py.

Writes, to --out-dir:
  spectrum_edges_MeV.txt  the 31 group edges, high to low (SCEPTRE order)
  spectrum_params.txt     case_id, E0_keV, a, Ec_keV, fK, r   (the surrogate inputs)
  spectra.txt             case_id + 30 group values, high to low energy
  spectra_samples.png     every sampled spectrum, with the baseline and Fig. 2 (--plot)

Group values (--units):
  fraction  photons in the group per photon in the whole spectrum (rows sum to 1)
  per_mev   group-averaged photons/MeV at unit fluence (fraction / group width)
Multiply by the fluence (about 2.5e12 photons/cm^2 per kRad(CaF2)) for an absolute
source.

Use from a deck generator:
    from pithon_spectrum import load_spectra
    edges, params, spectra = load_spectra("spectra_out")   # spectra[case_id] -> (30,)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from lhs_generation import latin_hypercube_sample  # noqa: E402

# SCEPTRE photon group edges, MeV, high to low: 31 edges, 30 groups.
# The 0.057-0.058 group isolates the Ta K-alpha line at 57.5 keV.
EDGES_MEV = np.array([
    0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08,
    0.07, 0.065, 0.06, 0.058, 0.057, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02,
    0.0175, 0.015, 0.0125, 0.01, 0.0001,
])
N_GROUPS = len(EDGES_MEV) - 1

# Baseline fit to SAND2014-15576 Fig. 2 (energies in keV).
BASE = dict(E0=260.0, a=1.8, Ec=20.0, fK=0.15, r=0.12, E_Ka=57.5, E_Kb=66.0, sig=1.5)
PHOTONS_PER_KRAD_CAF2 = 2.5e12  # photons/cm^2 per kRad(CaF2), 250-keV spectrum

# Sampled knobs and their default bounds: a band around the 250-keV baseline.
# E0 must stay below the top edge (500 keV) so no photons fall outside the groups.
PARAMS = ["E0", "a", "Ec", "fK", "r"]
PARAM_COLUMNS = ["E0_keV", "a", "Ec_keV", "fK", "r"]
DEFAULT_BOUNDS = dict(E0=(240.0, 280.0), a=(1.4, 2.2), Ec=(15.0, 26.0), fK=(0.10, 0.20), r=(0.12, 0.12))

# Fig. 2 of SAND2014-15576, digitized (edges in keV, photons/MeV), for the plot only.
FIG2_EDGES = np.array([10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                       80, 90, 100, 120, 140, 160, 180, 200, 225, 250], float)
FIG2_VALS = np.array([0.5, 2.0, 7.5, 14, 17, 16.5, 14.5, 13, 12, 10.5, 35.5, 8.5,
                      10.5, 5.8, 4.5, 3.8, 2.8, 1.7, 1.0, 0.7, 0.4, 0.2, 0.05])


# ------------------------------------------------------------------ #
# Model                                                              #
# ------------------------------------------------------------------ #

def _gauss(E, mu, sig):
    return np.exp(-0.5 * ((E - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def continuum(E, E0, a, Ec):
    """Unnormalized continuum shape, E in keV."""
    E = np.asarray(E, float)
    out = np.zeros_like(E)
    m = (E > 0) & (E < E0)
    out[m] = (1.0 / E[m]) * (1 - E[m] / E0) ** a * np.exp(-(Ec / E[m]) ** 3)
    return out


def spectrum(E, E0=260.0, a=1.8, Ec=20.0, fK=0.15, r=0.12, E_Ka=57.5, E_Kb=66.0, sig=1.5):
    """Unit-fluence photon number spectrum, photons/keV, E in keV."""
    grid = np.linspace(0.1, E0, 20000)
    A = 1.0 / np.trapezoid(continuum(grid, E0, a, Ec), grid)
    lines = (1 - r) * _gauss(E, E_Ka, sig) + r * _gauss(E, E_Kb, sig)
    return (1 - fK) * A * continuum(E, E0, a, Ec) + fK * lines


def group_fractions(edges_mev=EDGES_MEV, n_quad=400, **p):
    """
    Photons in each group per photon in the spectrum, in the order of edges_mev.
    Integrates the model over each group, then renormalizes so the groups sum to 1
    (the model puts a negligible tail below 0.1 keV and nothing above E0).
    """
    e = np.sort(np.asarray(edges_mev, float)) * 1000.0           # keV, ascending
    t = np.linspace(0.0, 1.0, n_quad)
    g = e[:-1, None] + t[None, :] * np.diff(e)[:, None]          # (groups, n_quad)
    p = dict(BASE, **p)
    frac = np.trapezoid(spectrum(g, **p), g, axis=1)
    frac /= frac.sum()
    return frac if edges_mev[0] < edges_mev[-1] else frac[::-1]


def to_units(fractions, edges_mev=EDGES_MEV, units="fraction"):
    """Convert group fractions to the requested units (see module docstring)."""
    if units == "fraction":
        return fractions
    if units == "per_mev":
        return fractions / np.abs(np.diff(edges_mev))
    raise ValueError(f"unknown units {units!r}; use 'fraction' or 'per_mev'")


# ------------------------------------------------------------------ #
# Sampling                                                           #
# ------------------------------------------------------------------ #

def sample_params(n_cases, bounds=None, seed=1):
    """(n_cases, 5) LHS over PARAMS; degenerate bounds (v, v) fix that knob at v."""
    b = dict(DEFAULT_BOUNDS, **(bounds or {}))
    b = np.array([b[k] for k in PARAMS], float)
    if np.any(b[:, 1] < b[:, 0]):
        raise ValueError("each lower bound must not exceed its upper bound")
    if b[0, 1] >= EDGES_MEV.max() * 1000.0:
        raise ValueError("E0 upper bound must be below the top group edge (500 keV)")
    fixed = b[:, 1] == b[:, 0]
    b_lhs = b.copy()
    b_lhs[fixed, 1] += 1.0                                         # dummy width for the LHS
    X = latin_hypercube_sample(n_cases, b_lhs, random_state=seed)
    X[:, fixed] = b[fixed, 0]
    return X


def sample_spectra(n_cases, bounds=None, seed=1, units="fraction", edges_mev=EDGES_MEV):
    """Returns (params (n_cases, 5), spectra (n_cases, n_groups)) in the order of edges_mev."""
    X = sample_params(n_cases, bounds, seed)
    S = np.array([to_units(group_fractions(edges_mev, **dict(zip(PARAMS, x))), edges_mev, units)
                  for x in X])
    return X, S


# ------------------------------------------------------------------ #
# I/O                                                                #
# ------------------------------------------------------------------ #

def write_spectra(out_dir, X, S, units, edges_mev=EDGES_MEV):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(X))[:, None]
    np.savetxt(out / "spectrum_edges_MeV.txt", edges_mev, fmt="%.6g",
               header="photon group edges, MeV, in the same order as the columns of spectra.txt")
    np.savetxt(out / "spectrum_params.txt", np.hstack([ids, X]), delimiter=",",
               fmt=["%d"] + ["%.8g"] * X.shape[1], header="case_id," + ",".join(PARAM_COLUMNS))
    groups = [f"g{k + 1}_{edges_mev[k]:g}-{edges_mev[k + 1]:g}MeV" for k in range(S.shape[1])]
    np.savetxt(out / "spectra.txt", np.hstack([ids, S]), delimiter=",",
               fmt=["%d"] + ["%.8e"] * S.shape[1],
               header=f"units={units}\ncase_id," + ",".join(groups))


def load_spectra(out_dir):
    """Returns (edges_MeV (31,), params (n, 5), spectra (n, 30)); row i is case i."""
    out = Path(out_dir)
    edges = np.loadtxt(out / "spectrum_edges_MeV.txt")
    params = np.loadtxt(out / "spectrum_params.txt", delimiter=",", ndmin=2)
    spectra = np.loadtxt(out / "spectra.txt", delimiter=",", ndmin=2)
    for name, arr in (("spectrum_params.txt", params), ("spectra.txt", spectra)):
        if not np.array_equal(arr[:, 0], np.arange(len(arr))):
            raise ValueError(f"{name}: case_id column must be 0..n-1 in order")
    return edges, params[:, 1:], spectra[:, 1:]


def plot_spectra(path, X, edges_mev=EDGES_MEV):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.sort(edges_mev) * 1000.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for x in X:   # photons/MeV, so the shapes are comparable to Fig. 2
        frac = group_fractions(e / 1000.0, **dict(zip(PARAMS, x)))
        ax.stairs(frac / np.diff(e) * 1000.0, e, color="C0", lw=0.6, alpha=0.25)
    ax.stairs(group_fractions(e / 1000.0) / np.diff(e) * 1000.0, e, color="C3", lw=1.8,
              label="baseline (E0=260, a=1.8, Ec=20 keV, fK=0.15)")
    ax.stairs(FIG2_VALS, FIG2_EDGES, color="k", lw=1.4, ls="--", label="SAND2014-15576 Fig. 2")
    ax.plot([], [], color="C0", label=f"{len(X)} sampled spectra")
    ax.set_xlim(0, 320)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("photons / MeV (unit fluence, group-averaged)")
    ax.set_title("Sampled PITHON spectra on the SCEPTRE 30-group structure")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cases", type=int, required=True, help="one spectrum per case")
    ap.add_argument("--out-dir", default="spectra_out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--units", choices=["fraction", "per_mev"], default="fraction")
    for k, (lo, hi) in DEFAULT_BOUNDS.items():
        ap.add_argument(f"--{k}", nargs=2, type=float, metavar=("LO", "HI"), default=(lo, hi),
                        help=f"bounds for {k} (default {lo} {hi}; equal values fix it)")
    ap.add_argument("--plot", action="store_true", help="also write spectra_samples.png")
    args = ap.parse_args(argv)

    bounds = {k: tuple(getattr(args, k)) for k in PARAMS}
    X, S = sample_spectra(args.n_cases, bounds, args.seed, args.units)
    write_spectra(args.out_dir, X, S, args.units)
    if args.plot:
        plot_spectra(Path(args.out_dir) / "spectra_samples.png", X)
    print(f"wrote {args.n_cases} spectra ({N_GROUPS} groups, units={args.units}) to {args.out_dir}")


if __name__ == "__main__":
    main()
