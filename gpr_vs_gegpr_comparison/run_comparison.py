"""
run_comparison.py — GE-GPR vs standard GPR, one comparison per active parameter,
across all 5 parameters that exist in the two ROM studies in this repo:

  Material-property parameters (sklearn_implementation/, single-layer slab):
      D        — diffusion coefficient
      Sigma_a  — absorption cross section
      q        — volumetric source

  Geometry parameters (void_teflon_geometry/):
      void_thickness — Void/Teflon interface location
      steepness      — sigmoid switch sharpness

For each parameter, the other four are held fixed at a constant value
(degenerate bounds), the existing single-parameter ROM pipeline is run
end-to-end (LHS sampling -> forward solves -> POD -> adjoint gradients),
and GE-GPR is compared against standard GPR trained on the same subset.

This script is a thin driver: all modeling code is reused unchanged from
sklearn_implementation/ge_gpr.py (material) and
void_teflon_geometry/ge_gpr_geometry.py (geometry). The only change made to
that existing code was relaxing ge_gpr_geometry.main()'s guard, which
previously required *both* geometry parameters to be active — it now allows
one, since the material side has always supported single-parameter studies.

Usage
-----
    python3 run_comparison.py
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, os.path.join(_ROOT, 'common'))
sys.path.insert(0, os.path.join(_ROOT, 'sklearn_implementation'))
sys.path.insert(0, os.path.join(_ROOT, 'void_teflon_geometry'))


def _load_module(rel_path, name):
    """Load a .py file as a named module (mirrors compare.py's pattern)."""
    full = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


svd_material = _load_module('common/svd_analysis.py', 'svd_analysis')
ge_gpr = _load_module('sklearn_implementation/ge_gpr.py', 'ge_gpr')
ge_gpr_geometry = _load_module('void_teflon_geometry/ge_gpr_geometry.py', 'ge_gpr_geometry')


# ──────────────────────────────────────────────────────────────────────────── #
# Study configuration — one entry per parameter                                #
# ──────────────────────────────────────────────────────────────────────────── #

# Shared settings across the 3 material single-parameter studies. n_bins=1
# so the parameter vector is exactly (D, Sigma_a, q) -- one real number each
# -- and "holding the other parameters fixed" means degenerate (v, v) bounds,
# the same convention params.GeometryParamSampler uses for geometry.
_MAT_SHARED = dict(
    L=10.0, n_bins=1, n_cells=200, qoi_xmin=3.0, qoi_xmax=7.0,
    m_train=500, m_test=100, seed=1, energy_threshold=0.99999,
)
_MAT_GE_SHARED = dict(
    L=10.0, n_bins=1, n_cells=200, qoi_xmin=3.0, qoi_xmax=7.0,
    m_ge=250, n_restarts=3, learning_curve_sizes=[10, 20, 30, 50, 75, 100, 150, 200, 250],
)
# Fixed values reused from run_pipeline.py's existing "Material 1" convention.
_D_FIX, _SA_FIX, _Q_FIX = 1.0, 0.5, 1.0

# Geometry studies rescaled x10 (L, core_radius, shield_thickness, shell_width)
# so the void/Teflon slab spans the same L=10 domain as the material studies,
# and the QoI window is forced to the identical [3, 7] used there instead of
# the geometry pipeline's own shell-only default -- so this window now also
# covers part of the fixed Core region, unlike that default. void_thickness
# bounds scale x10 (length units) to stay proportional to the new shell_width;
# steepness scales /10 (1/length units, sigmoid_switch's `a*(x-b)`) to keep
# the same physical transition width in cm.
_GEO_SCALE = 10.0
_GEO_L = 10.0
_GEO_CORE_RADIUS = 0.16921 * _GEO_SCALE
_GEO_SHIELD_THICKNESS = 0.07801 * _GEO_SCALE
_GEO_SHELL_WIDTH = 0.25278 * _GEO_SCALE
_GEO_QOI_XMIN, _GEO_QOI_XMAX = 3.0, 7.0  # same window as the material studies
_VOID_FIX = 0.125 * _GEO_SCALE  # midpoint of the (scaled) void_bounds range

_GEO_GE_SHARED = dict(
    L=_GEO_L, core_radius=_GEO_CORE_RADIUS, shield_thickness=_GEO_SHIELD_THICKNESS,
    shell_width=_GEO_SHELL_WIDTH, qoi_xmin=_GEO_QOI_XMIN, qoi_xmax=_GEO_QOI_XMAX,
    m_ge=250, n_restarts=3, learning_curve_sizes=[10, 20, 30, 50, 75, 100, 150, 200, 250],
    run_svd_stage=True,
)

STUDIES = [
    dict(
        name="D", domain="material", label=r"$D$ (diffusion coeff.)",
        svd_kwargs=dict(_MAT_SHARED, d_bounds=(0.5, 2.0), sigma_a_bounds=(_SA_FIX, _SA_FIX),
                         q_bounds=(_Q_FIX, _Q_FIX), output_dir=os.path.join(_HERE, "D")),
        ge_kwargs=dict(_MAT_GE_SHARED, d_bounds=(0.5, 2.0), sigma_a_bounds=(_SA_FIX, _SA_FIX),
                        q_bounds=(_Q_FIX, _Q_FIX), output_dir=os.path.join(_HERE, "D")),
    ),
    dict(
        name="Sigma_a", domain="material", label=r"$\Sigma_a$ (absorption)",
        svd_kwargs=dict(_MAT_SHARED, d_bounds=(_D_FIX, _D_FIX), sigma_a_bounds=(0.05, 1.5),
                         q_bounds=(_Q_FIX, _Q_FIX), output_dir=os.path.join(_HERE, "Sigma_a")),
        ge_kwargs=dict(_MAT_GE_SHARED, d_bounds=(_D_FIX, _D_FIX), sigma_a_bounds=(0.05, 1.5),
                        q_bounds=(_Q_FIX, _Q_FIX), output_dir=os.path.join(_HERE, "Sigma_a")),
    ),
    dict(
        name="q", domain="material", label=r"$q$ (source)",
        svd_kwargs=dict(_MAT_SHARED, d_bounds=(_D_FIX, _D_FIX), sigma_a_bounds=(_SA_FIX, _SA_FIX),
                         q_bounds=(0.0, 2.0), output_dir=os.path.join(_HERE, "q")),
        ge_kwargs=dict(_MAT_GE_SHARED, d_bounds=(_D_FIX, _D_FIX), sigma_a_bounds=(_SA_FIX, _SA_FIX),
                        q_bounds=(0.0, 2.0), output_dir=os.path.join(_HERE, "q")),
    ),
    dict(
        name="void_thickness", domain="geometry", label="void_thickness",
        svd_kwargs=None,
        ge_kwargs=dict(_GEO_GE_SHARED,
                        void_bounds=(0.001 * _GEO_SCALE, 0.25 * _GEO_SCALE),
                        steepness_bounds=(200.0 / _GEO_SCALE, 200.0 / _GEO_SCALE),
                        output_dir=os.path.join(_HERE, "void_thickness")),
    ),
    dict(
        name="steepness", domain="geometry", label=r"steepness",
        svd_kwargs=None,
        ge_kwargs=dict(_GEO_GE_SHARED,
                        void_bounds=(_VOID_FIX, _VOID_FIX),
                        steepness_bounds=(100.0 / _GEO_SCALE, 400.0 / _GEO_SCALE),
                        output_dir=os.path.join(_HERE, "steepness")),
    ),
]


def _run_study(study):
    name = study["name"]
    print(f"\n{'#' * 70}")
    print(f"# Parameter: {name}  ({study['domain']})")
    print(f"{'#' * 70}")

    os.makedirs(study["ge_kwargs"]["output_dir"], exist_ok=True)

    if study["domain"] == "material":
        svd_material.main(**study["svd_kwargs"])
        result = ge_gpr.main(**study["ge_kwargs"])
    else:
        result = ge_gpr_geometry.main(**study["ge_kwargs"])

    return result


def main():
    results = {}
    for study in STUDIES:
        results[study["name"]] = _run_study(study)

    _write_summary_table(results)
    _plot_median_error_bars(results)
    _plot_combined_learning_curves(results)

    print(f"\n{'=' * 70}")
    print(f"  Comparison complete. Per-parameter plots in "
          f"{_HERE}/<param>/, summary in {_HERE}/summary_table.csv "
          f"and {_HERE}/summary_*.png")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────────────────────── #
# Summary outputs                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _write_summary_table(results):
    path = os.path.join(_HERE, "summary_table.csv")
    rows = []
    for study in STUDIES:
        name = study["name"]
        r = results[name]
        eps_ge, eps_std, eps_proj = r["eps_ge"], r["eps_std"], r["eps_proj"]
        med_ge, med_std = np.median(eps_ge), np.median(eps_std)
        rows.append(dict(
            parameter=name,
            domain=study["domain"],
            median_eps_std_gpr=f"{med_std:.4e}",
            median_eps_ge_gpr=f"{med_ge:.4e}",
            improvement_pct=f"{100.0 * (med_std - med_ge) / med_std:.1f}",
            max_eps_std_gpr=f"{np.max(eps_std):.4e}",
            max_eps_ge_gpr=f"{np.max(eps_ge):.4e}",
            median_eps_proj_pod_floor=f"{np.median(eps_proj):.4e}",
        ))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 78}")
    print(f"{'Param':<15}{'domain':<10}{'median std':>14}{'median GE':>14}{'improve %':>12}")
    print(f"{'=' * 78}")
    for row in rows:
        print(f"{row['parameter']:<15}{row['domain']:<10}"
              f"{row['median_eps_std_gpr']:>14}{row['median_eps_ge_gpr']:>14}"
              f"{row['improvement_pct']:>11}%")
    print(f"{'=' * 78}")
    print(f"Saved: {path}")


def _plot_median_error_bars(results):
    names = [s["name"] for s in STUDIES]
    labels = [s["label"] for s in STUDIES]
    med_std = [np.median(results[n]["eps_std"]) for n in names]
    med_ge = [np.median(results[n]["eps_ge"]) for n in names]
    med_proj = [np.median(results[n]["eps_proj"]) for n in names]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, med_std, width, color='tomato', label='Std GPR')
    ax.bar(x + width / 2, med_ge, width, color='steelblue', label='GE-GPR')
    ax.scatter(x, med_proj, color='gray', marker='_', s=400, linewidths=2,
               label=r'$\varepsilon_{\mathrm{proj}}$ (POD floor)', zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r'Median test-set relative error $\varepsilon_y$', fontsize=12)
    ax.set_title('GE-GPR vs Std GPR — median error, one parameter active at a time', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', axis='y', ls='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(_HERE, "summary_median_error_bars.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def _plot_combined_learning_curves(results):
    fig, axes = plt.subplots(1, len(STUDIES), figsize=(4.2 * len(STUDIES), 4.5), sharey=False)
    for ax, study in zip(axes, STUDIES):
        name = study["name"]
        lc = results[name].get("lc_results")
        if not lc:
            ax.set_title(f"{study['label']}\n(no learning curve)", fontsize=10)
            continue
        ax.semilogy(lc['sizes'], lc['std_med'], 'o-', color='tomato', label='Std GPR')
        ax.semilogy(lc['sizes'], lc['ge_med'], '^--', color='steelblue', label='GE-GPR')
        ax.set_title(study['label'], fontsize=11)
        ax.set_xlabel(r'Training samples $M$', fontsize=10)
        ax.grid(True, which='both', ls='--', alpha=0.4)
    axes[0].set_ylabel(r'Median $\varepsilon_y$ (test)', fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle('Learning curves: GE-GPR vs Std GPR, per active parameter', fontsize=13)
    plt.tight_layout()
    path = os.path.join(_HERE, "summary_learning_curves.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
