"""
generate_samples.py: standalone SCEPTRE case sampler. One joint design over the
geometry thicknesses and a parametric PITHON 250-keV photon spectrum, written as one
text table that a deck/template generator reads row by row.

Needs only numpy (matplotlib for --plot). No other file in this repo is imported.

Spectrum model (fit to Fig. 2 of SAND2014-15576, 250-keV endpoint, unit fluence):

    N(E) = (1 - fK) * n_c(E) + fK * [ (1-r) G(E; E_Ka, sig) + r G(E; E_Kb, sig) ]
    n_c(E) = A * E^-1 * (1 - E/E0)^a * exp(-(Ec/E)^3),   integral of n_c = 1

    E0  endpoint (diode voltage)                  [keV]
    a   bremsstrahlung roll-off exponent          (1 = thick-target Kramers, ~2 thin foil)
    Ec  low-energy cutoff (anode/window/air/filter photoelectric absorption) [keV]
    fK  Ta K-line photon fraction
    r   K-beta share of the line photons

Design
    Every sampled input (thicknesses and spectrum knobs) is one column of a single
    Latin hypercube, so there is one RNG and no chance of two columns sharing a
    permutation. Of --n-candidates random LHS designs, the 10% with the smallest
    largest column correlation are kept, and of those the one with the largest
    minimum pairwise distance (maximin, in the unit cube). A spectrum is
    then computed from each row's knobs and integrated over the 30 SCEPTRE groups,
    so its group values are always one physical spectrum, never independent draws.
    An input with equal bounds (v, v) is fixed at v and takes no LHS dimension.
    Optional held-out test cases are uniform random with seed + 1 (repo convention).

Writes, to --out-dir:
  cases.csv               one row per case:
                            case_id, role, <fixed columns>, <inputs>, g1 ... g30
                          <inputs> are the surrogate inputs; g1..g30 are the group
                          values from high to low energy (same order as the edges).
  spectrum_edges_MeV.txt  the 31 group edges in MeV, high to low
  design_bounds.csv       name, lower, upper, sampled (1/0) for every input
  spectra_samples.png     every sampled spectrum against the baseline and Fig. 2 (--plot)

Group values (--units):
  fraction  photons in the group per photon in the spectrum (each row sums to 1)
  per_mev   group-averaged photons/MeV at unit fluence (fraction / group width)
The spectrum is unit fluence; scale by the fluence in the deck (about 2.5e12
photons/cm^2 per kRad(CaF2) for this spectrum), e.g. with --fixed photon_strength=...

Example
    python pithon_spectrum.py --n-train 150 --n-test 50 --out-dir sceptre_cases \\
        --fixed core_radius=0.5 --plot

Reading it in a template generator (no imports from this repo needed):
    import csv
    edges = [float(x) for x in open("sceptre_cases/spectrum_edges_MeV.txt")
             if not x.startswith("#")]
    for row in csv.DictReader(open("sceptre_cases/cases.csv")):
        groups = [float(row[f"g{k}"]) for k in range(1, 31)]
        render(case_id=int(row["case_id"]), void_thickness=float(row["void_thickness"]),
               ..., spectrum=list(zip(edges[:-1], edges[1:], groups)))
"""

import argparse
import csv
from pathlib import Path

import numpy as np

# SCEPTRE photon group edges, MeV, high to low: 31 edges, 30 groups.
# The 0.057-0.058 group isolates the Ta K-alpha line at 57.5 keV.
EDGES_MEV = np.array([
    0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08,
    0.07, 0.065, 0.06, 0.058, 0.057, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02,
    0.0175, 0.015, 0.0125, 0.01, 0.0001,
])
N_GROUPS = len(EDGES_MEV) - 1
GROUP_COLUMNS = [f"g{k + 1}" for k in range(N_GROUPS)]

# Baseline fit to SAND2014-15576 Fig. 2 (energies in keV).
BASE = dict(E0=260.0, a=1.8, Ec=20.0, fK=0.15, r=0.12, E_Ka=57.5, E_Kb=66.0, sig=1.5)
PHOTONS_PER_KRAD_CAF2 = 2.5e12  # photons/cm^2 per kRad(CaF2), 250-keV spectrum

# Design inputs and default bounds, in column order of cases.csv.
# Thicknesses as in make_geometry_cases.py. Spectrum bounds are a band around the
# 250-keV baseline; E0 must stay below the top group edge (500 keV).
GEOMETRY_BOUNDS = dict(void_thickness=(1e-4, 0.1), dielectric_thickness=(0.25, 0.35),
                       shield_thickness=(0.15, 0.25))
SPECTRUM_BOUNDS = dict(E0=(240.0, 280.0), a=(1.4, 2.2), Ec=(15.0, 26.0), fK=(0.10, 0.20),
                       r=(0.12, 0.12))
SPECTRUM_PARAMS = list(SPECTRUM_BOUNDS)

# Fig. 2 of SAND2014-15576, digitized (edges in keV, photons/MeV), for the plot only.
FIG2_EDGES = np.array([10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                       80, 90, 100, 120, 140, 160, 180, 200, 225, 250], float)
FIG2_VALS = np.array([0.5, 2.0, 7.5, 14, 17, 16.5, 14.5, 13, 12, 10.5, 35.5, 8.5,
                      10.5, 5.8, 4.5, 3.8, 2.8, 1.7, 1.0, 0.7, 0.4, 0.2, 0.05])


# ------------------------------------------------------------------ #
# Spectrum model                                                     #
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
    edges_mev = np.asarray(edges_mev, float)
    e = np.sort(edges_mev) * 1000.0                              # keV, ascending
    t = np.linspace(0.0, 1.0, n_quad)
    g = e[:-1, None] + t[None, :] * np.diff(e)[:, None]          # (groups, n_quad)
    frac = np.trapezoid(spectrum(g, **dict(BASE, **p)), g, axis=1)
    frac /= frac.sum()
    return frac if edges_mev[0] < edges_mev[-1] else frac[::-1]


def group_values(units="fraction", edges_mev=EDGES_MEV, **p):
    """Group values in the requested units (see module docstring)."""
    frac = group_fractions(edges_mev, **p)
    if units == "fraction":
        return frac
    if units == "per_mev":
        return frac / np.abs(np.diff(edges_mev))
    raise ValueError(f"unknown units {units!r}; use 'fraction' or 'per_mev'")


# ------------------------------------------------------------------ #
# Design                                                             #
# ------------------------------------------------------------------ #

def lhs_unit(n, d, rng):
    """One random LHS in [0, 1)^d: each column has one point in each of n strata."""
    U = np.empty((n, d))
    for j in range(d):
        U[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return U


def optimized_lhs(n, d, rng, n_candidates=200, keep_frac=0.1):
    """
    Best of n_candidates random LHS designs: keep the keep_frac with the smallest
    largest |correlation| between columns, then take the one of those with the
    largest minimum pairwise distance (maximin). Maximin alone leaves column
    correlations near 0.2-0.3; correlation alone clusters points.
    """
    cands = [lhs_unit(n, d, rng) for _ in range(max(1, n_candidates))]
    if n < 3 or d < 2:
        return cands[0]
    iu = np.triu_indices(n, 1)
    corr = np.array([np.abs(np.corrcoef(U.T) - np.eye(d)).max() for U in cands])
    keep = np.argsort(corr)[:max(1, int(round(keep_frac * len(cands))))]
    spread = [np.sqrt(((cands[i][:, None] - cands[i][None]) ** 2).sum(-1))[iu].min() for i in keep]
    return cands[keep[int(np.argmax(spread))]]


def build_design(bounds, n_train, n_test=0, seed=1, n_candidates=200):
    """
    bounds: dict name -> (lo, hi), in output column order. Returns (X, roles):
    X (n_train + n_test, len(bounds)), training rows from a maximin LHS over the
    non-degenerate inputs (seed), test rows uniform random (seed + 1).
    """
    names = list(bounds)
    b = np.array([bounds[k] for k in names], float)
    if np.any(b[:, 1] < b[:, 0]):
        bad = [k for k, (lo, hi) in zip(names, b) if hi < lo]
        raise ValueError(f"lower bound exceeds upper bound for {bad}")
    active = b[:, 1] > b[:, 0]
    lo, width = b[:, 0], b[:, 1] - b[:, 0]

    U_tr = np.zeros((n_train, len(names)))
    U_tr[:, active] = optimized_lhs(n_train, int(active.sum()), np.random.default_rng(seed),
                                    n_candidates)
    U_te = np.random.default_rng(seed + 1).uniform(size=(n_test, len(names)))
    X = lo + np.vstack([U_tr, U_te]) * width        # fixed inputs: width 0 -> exactly lo
    return X, ["train"] * n_train + ["test"] * n_test


# ------------------------------------------------------------------ #
# Output                                                             #
# ------------------------------------------------------------------ #

def write_cases(out_dir, X, roles, bounds, fixed, units, edges_mev=EDGES_MEV):
    """Writes cases.csv, spectrum_edges_MeV.txt and design_bounds.csv; returns the spectra."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(bounds)
    spec_idx = [names.index(k) for k in SPECTRUM_PARAMS]
    S = np.array([group_values(units, edges_mev, **dict(zip(SPECTRUM_PARAMS, x[spec_idx])))
                  for x in X])

    with open(out / "cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "role", *fixed, *names, *GROUP_COLUMNS])
        for i, (x, s) in enumerate(zip(X, S)):
            w.writerow([i, roles[i], *(f"{v:.10g}" for v in fixed.values()),
                        *(f"{v:.10g}" for v in x), *(f"{v:.8e}" for v in s)])

    with open(out / "spectrum_edges_MeV.txt", "w") as f:
        f.write(f"# photon group edges (MeV), high to low; group gK of cases.csv spans "
                f"line K to line K+1; units={units}\n")
        f.writelines(f"{e:g}\n" for e in edges_mev)

    with open(out / "design_bounds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lower", "upper", "sampled"])
        for k, (lo, hi) in bounds.items():
            w.writerow([k, f"{lo:.10g}", f"{hi:.10g}", int(hi > lo)])
        for k, v in fixed.items():
            w.writerow([k, f"{v:.10g}", f"{v:.10g}", 0])
    return S


def read_cases(out_dir):
    """Returns (edges_MeV (31,), rows: list of dicts with floats, groups as a (30,) array)."""
    out = Path(out_dir)
    edges = np.loadtxt(out / "spectrum_edges_MeV.txt")
    rows = []
    with open(out / "cases.csv", newline="") as f:
        for r in csv.DictReader(f):
            groups = np.array([float(r.pop(g)) for g in GROUP_COLUMNS])
            row = {k: (v if k == "role" else float(v)) for k, v in r.items()}
            row["case_id"] = int(row["case_id"])
            row["groups"] = groups
            rows.append(row)
    return edges, rows


def plot_spectra(path, X, names, edges_mev=EDGES_MEV):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.sort(edges_mev) * 1000.0
    spec_idx = [names.index(k) for k in SPECTRUM_PARAMS]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for x in X:   # photons/MeV, so the shapes are comparable to Fig. 2
        frac = group_fractions(e / 1000.0, **dict(zip(SPECTRUM_PARAMS, x[spec_idx])))
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

def _fixed_arg(s):
    name, _, value = s.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError("use NAME=VALUE")
    return name, float(value)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, required=True, help="LHS (training) cases")
    ap.add_argument("--n-test", type=int, default=0, help="uniform-random held-out cases")
    ap.add_argument("--out-dir", default="sceptre_cases")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-candidates", type=int, default=200,
                    help="random LHS designs to choose the best one from")
    ap.add_argument("--units", choices=["fraction", "per_mev"], default="fraction")
    for k, (lo, hi) in {**GEOMETRY_BOUNDS, **SPECTRUM_BOUNDS}.items():
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, nargs=2, type=float,
                        metavar=("LO", "HI"), default=(lo, hi),
                        help=f"bounds (default {lo:g} {hi:g}; equal values fix it)")
    ap.add_argument("--fixed", type=_fixed_arg, action="append", default=[],
                    metavar="NAME=VALUE",
                    help="constant column written to every case, e.g. core_radius=0.5 (repeatable)")
    ap.add_argument("--plot", action="store_true", help="also write spectra_samples.png")
    args = ap.parse_args(argv)

    bounds = {k: tuple(getattr(args, k)) for k in {**GEOMETRY_BOUNDS, **SPECTRUM_BOUNDS}}
    if bounds["E0"][1] >= EDGES_MEV.max() * 1000.0:
        ap.error("E0 upper bound must be below the top group edge (500 keV)")
    fixed = dict(args.fixed)
    clash = set(fixed) & set(bounds)
    if clash:
        ap.error(f"--fixed names clash with sampled inputs: {sorted(clash)}; "
                 f"fix those with equal bounds instead")

    X, roles = build_design(bounds, args.n_train, args.n_test, args.seed, args.n_candidates)
    write_cases(args.out_dir, X, roles, bounds, fixed, args.units)
    if args.plot:
        plot_spectra(Path(args.out_dir) / "spectra_samples.png", X, list(bounds))

    sampled = [k for k, (lo, hi) in bounds.items() if hi > lo]
    print(f"wrote {len(X)} cases ({args.n_train} train, {args.n_test} test) to "
          f"{args.out_dir}/cases.csv")
    print(f"  sampled inputs ({len(sampled)}): {', '.join(sampled)}")
    print(f"  spectrum: {N_GROUPS} groups, units={args.units}, high to low energy")


if __name__ == "__main__":
    main()
