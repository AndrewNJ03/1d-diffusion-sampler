"""
run_combined_seeds.py -- repeat the combined 5-parameter learning curve over
several independent LHS designs.

run_combined.py reports one design (seed=1). A single nested LHS design gives
a learning curve whose wiggles could be design noise rather than a property of
the method, so this driver re-runs the identical pipeline over N_SEEDS
independent designs (a fresh LHS train/test pair per seed, each swept over the
same nested prefixes M = 10..80) and reports the across-design spread.

Usage
-----
    python3 run_combined_seeds.py
"""

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import ge_gpr_combined  # noqa: E402
from run_combined import COMBINED_KWARGS  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
OUT_DIR = os.path.join(_HERE, "output_graphs_combined_seeds")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sizes = None
    ge_curves, std_curves = [], []
    med_ge_final, med_std_final = [], []

    for seed in SEEDS:
        print("#" * 70)
        print(f"# Combined 5-parameter study -- LHS design seed={seed}")
        print("#" * 70)

        kwargs = dict(COMBINED_KWARGS)
        kwargs["seed"] = seed
        kwargs["output_dir"] = os.path.join(OUT_DIR, f"seed{seed}")
        result = ge_gpr_combined.main(**kwargs)

        lc = result["lc_results"]
        sizes = lc["sizes"]
        ge_curves.append(lc["ge_med"])
        std_curves.append(lc["std_med"])
        med_ge_final.append(float(np.median(result["eps_ge"])))
        med_std_final.append(float(np.median(result["eps_std"])))

    ge = np.asarray(ge_curves)    # (n_seed, n_size)
    std = np.asarray(std_curves)

    # ---------------- CSV ---------------- #
    path = os.path.join(_HERE, "combined_seed_repeats.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "ge_median_over_seeds", "ge_min", "ge_max",
                    "std_median_over_seeds", "std_min", "std_max",
                    "improvement_pct_median"])
        for j, m in enumerate(sizes):
            imp = 100.0 * (std[:, j] - ge[:, j]) / std[:, j]
            w.writerow([m,
                        f"{np.median(ge[:, j]):.4e}", f"{ge[:, j].min():.4e}", f"{ge[:, j].max():.4e}",
                        f"{np.median(std[:, j]):.4e}", f"{std[:, j].min():.4e}", f"{std[:, j].max():.4e}",
                        f"{np.median(imp):.1f}"])

    imp_final = 100.0 * (np.asarray(med_std_final) - np.asarray(med_ge_final)) \
                / np.asarray(med_std_final)

    print("\n" + "=" * 78)
    print(f"Across {len(SEEDS)} independent LHS designs, at M={sizes[-1]}:")
    print(f"  GE-GPR  median eps_y : {np.median(med_ge_final):.4e} "
          f"[{min(med_ge_final):.4e}, {max(med_ge_final):.4e}]")
    print(f"  Std GPR median eps_y : {np.median(med_std_final):.4e} "
          f"[{min(med_std_final):.4e}, {max(med_std_final):.4e}]")
    print(f"  improvement          : {np.median(imp_final):.1f}% "
          f"[{imp_final.min():.1f}%, {imp_final.max():.1f}%]")
    print("=" * 78)
    print(f"Saved: {path}")

    # ---------------- Figure ---------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(sizes, std.min(axis=0), std.max(axis=0),
                    color='tomato', alpha=0.22, lw=0)
    ax.fill_between(sizes, ge.min(axis=0), ge.max(axis=0),
                    color='steelblue', alpha=0.22, lw=0)
    ax.semilogy(sizes, np.median(std, axis=0), 'o-', color='tomato',
                label='Std GPR (median of designs)')
    ax.semilogy(sizes, np.median(ge, axis=0), '^--', color='steelblue',
                label='GE-GPR (median of designs)')
    ax.set_xlabel('Training samples $M$', fontsize=11)
    ax.set_ylabel(r'Median test error $\varepsilon_y$', fontsize=11)
    ax.set_title(f'Combined 5-parameter case, {len(SEEDS)} independent LHS designs',
                 fontsize=12)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(_HERE, "combined_seed_repeats.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
