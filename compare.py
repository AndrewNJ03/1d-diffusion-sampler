import importlib.util
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'common'))
sys.path.insert(0, os.path.join(_HERE, 'pytorch_implementation'))


def _load_module(rel_path, module_name):
    """Load a .py file as a named module without polluting sys.path."""
    full = os.path.join(_HERE, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_svd    = _load_module('common/svd_analysis.py',              'svd_analysis')
pt_gpr  = _load_module('pytorch_implementation/gpr_pod.py',   'pt_gpr_pod')
sk_gpr  = _load_module('sklearn_implementation/gpr_pod.py',   'sk_gpr_pod')

run_svd = _svd.main


# configuration (mirrors run_pipeline.py defaults) #
CFG = dict(
    L=10.0,
    n_bins=3,
    n_cells=200,
    d_bounds=[[0.5, 2.0], [0.5, 2.0], [0.5, 2.0]],
    sigma_a_bounds=[[0.05, 1.5], [0.05, 1.5], [0.05, 1.5]],
    q_bounds=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    qoi_xmin=3.0,
    qoi_xmax=7.0,
    output_dir="output_graphs_compare",
    # Stage 1
    m_train=500,
    m_test=100,
    seed=1,
    energy_threshold=0.999,
    # Stage 2 — PyTorch
    n_epochs=5000,
    lr=0.05,
    # Stage 2 — sklearn
    n_restarts=5,
    # Fixed materials (same as run_pipeline.py default)
    fixed_mats={1: {"D": 1.0, "sigma_a": 0.5, "q": 1.0}},
)


def _apply_fixed_mats(cfg):
    """Overwrite per-layer bounds for fixed materials in-place."""
    for mat_idx, vals in (cfg.get('fixed_mats') or {}).items():
        i = mat_idx - 1
        for key, param in [("D", "d_bounds"), ("sigma_a", "sigma_a_bounds"), ("q", "q_bounds")]:
            if key in vals:
                v = vals[key]
                cfg[param][i] = [v, v]


def main(cfg=None):
    if cfg is None:
        cfg = CFG.copy()
        cfg['d_bounds']       = [list(b) for b in cfg['d_bounds']]
        cfg['sigma_a_bounds'] = [list(b) for b in cfg['sigma_a_bounds']]
        cfg['q_bounds']       = [list(b) for b in cfg['q_bounds']]

    _apply_fixed_mats(cfg)
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    shared = {k: cfg[k] for k in (
        'L', 'n_bins', 'n_cells',
        'd_bounds', 'sigma_a_bounds', 'q_bounds',
        'qoi_xmin', 'qoi_xmax', 'output_dir',
    )}

    # ------------------------------------------------------------------ #
    # Stage 1 — SVD/POD (run once, shared)                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Stage 1 — SVD / POD  (shared snapshot)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    run_svd(**shared, m_train=cfg['m_train'], m_test=cfg['m_test'],
            seed=cfg['seed'], energy_threshold=cfg['energy_threshold'])
    print(f"  [Stage 1]  DONE  ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # Stage 2a — GPyTorch GPR                                            #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Stage 2a — GPyTorch GPR")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    res_pt = pt_gpr.main(**shared, n_epochs=cfg['n_epochs'], lr=cfg['lr'])
    print(f"  [GPyTorch]  DONE  ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # Stage 2b — sklearn GPR                                             #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Stage 2b — sklearn GPR")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    res_sk = sk_gpr.main(**shared, n_restarts=cfg['n_restarts'])
    print(f"  [sklearn]  DONE  ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # Comparison plots                                                    #
    # ------------------------------------------------------------------ #
    _plot_error_overlay(res_pt, res_sk, output_dir)
    _plot_sorted_errors(res_pt, res_sk, output_dir)
    _plot_parity_comparison(res_pt, res_sk, output_dir)
    _plot_flux_comparison(res_pt, res_sk, output_dir)
    _print_summary_table(res_pt, res_sk)

    print(f"\n{'='*60}")
    print(f"  Comparison plots saved to: {output_dir}/compare_*.png")
    print(f"{'='*60}\n")


# ── plot helpers ──────────────────────────────────────────────────────────── #

def _plot_error_overlay(pt, sk, output_dir):
    """Overlaid ε_y histograms for both methods + POD lower bound."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- histogram ---
    ax = axes[0]
    bins = np.histogram_bin_edges(
        np.concatenate([pt['eps_y'], sk['eps_y'], pt['eps_proj']]), bins=25
    )
    ax.hist(pt['eps_proj'], bins=bins, alpha=0.50, color='tomato',
            label=r'$\varepsilon_{\mathrm{proj}}$ (POD lower bound)')
    ax.hist(pt['eps_y'],    bins=bins, alpha=0.60, color='royalblue',
            label=r'$\varepsilon_y$ — GPyTorch')
    ax.hist(sk['eps_y'],    bins=bins, alpha=0.55, color='darkorange',
            label=r'$\varepsilon_y$ — sklearn', histtype='step', lw=2)
    ax.set_xlabel(r'Relative error $\varepsilon_y$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Field error distribution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)

    # --- ε_alpha histogram ---
    ax = axes[1]
    bins_a = np.histogram_bin_edges(
        np.concatenate([pt['eps_alpha'], sk['eps_alpha']]), bins=25
    )
    ax.hist(pt['eps_alpha'], bins=bins_a, alpha=0.65, color='royalblue',
            label=r'$\varepsilon_\alpha$ — GPyTorch')
    ax.hist(sk['eps_alpha'], bins=bins_a, alpha=0.55, color='darkorange',
            label=r'$\varepsilon_\alpha$ — sklearn', histtype='step', lw=2)
    ax.set_xlabel(r'Relative coefficient error $\varepsilon_\alpha$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Coefficient error distribution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('GPyTorch vs sklearn — Error distributions', fontsize=14)
    plt.tight_layout()
    path = f"{output_dir}/compare_error_distributions.png"
    plt.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    plt.close(fig)


def _plot_sorted_errors(pt, sk, output_dir):
    """Sorted ε_y curves for GPyTorch, sklearn, and POD lower bound."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(np.sort(pt['eps_proj']), 's--', ms=3, lw=1.2, color='tomato',
                label=r'$\varepsilon_{\mathrm{proj}}$ (POD lower bound)')
    ax.semilogy(np.sort(pt['eps_y']),   'o-',  ms=3, lw=1.5, color='royalblue',
                label=r'$\varepsilon_y$ — GPyTorch')
    ax.semilogy(np.sort(sk['eps_y']),   '^-',  ms=3, lw=1.5, color='darkorange',
                label=r'$\varepsilon_y$ — sklearn')
    ax.set_xlabel('Test sample (sorted by error)', fontsize=12)
    ax.set_ylabel(r'Relative error $\varepsilon_y$', fontsize=12)
    ax.set_title(
        f'Sorted field reconstruction error\n'
        f'(M_train={pt["M_tr"]}, R={pt["R"]})',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    path = f"{output_dir}/compare_sorted_errors.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def _plot_parity_comparison(pt, sk, output_dir):
    """
    Side-by-side parity plots for up to 4 leading POD modes.
    Left column = GPyTorch, right column = sklearn.
    """
    R      = pt['R']
    n_show = min(4, R)

    fig, axes = plt.subplots(n_show, 2,
                              figsize=(9, 4.0 * n_show),
                              squeeze=False)

    for k in range(n_show):
        for col, (res, label, color) in enumerate([
            (pt, 'GPyTorch', 'royalblue'),
            (sk, 'sklearn',  'darkorange'),
        ]):
            ax      = axes[k][col]
            true    = res['alpha_te_true'][:, k]
            pred    = res['alpha_pred'][:, k]
            sigma_k = res['alpha_pred_std'][:, k]
            lo = min(true.min(), (pred - 2 * sigma_k).min())
            hi = max(true.max(), (pred + 2 * sigma_k).max())
            ax.errorbar(true, pred, yerr=2 * sigma_k, fmt='o', ms=3, lw=0.7,
                        alpha=0.65, color=color,
                        ecolor='lightgrey', label=r'pred $\pm 2\sigma$')
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='ideal')
            ax.set_xlabel(fr'True $\alpha_{k+1}$', fontsize=10)
            ax.set_ylabel(fr'Pred $\alpha_{k+1}$', fontsize=10)
            ax.set_title(f'Mode {k+1} — {label}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('Parity plots: GPyTorch (left) vs sklearn (right)', fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/compare_parity.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def _plot_flux_comparison(pt, sk, output_dir):
    """
    Per test sample: top row = GPyTorch, bottom row = sklearn.
    Up to 4 test samples shown as columns.
    """
    n_plot   = min(4, pt['M_te'])
    idx_plot = np.arange(n_plot)

    fig, axes = plt.subplots(2, n_plot,
                              figsize=(4.5 * n_plot, 8.0),
                              sharey='row')

    for col, idx in enumerate(idx_plot):
        for row, (res, label, color) in enumerate([
            (pt, 'GPyTorch', 'royalblue'),
            (sk, 'sklearn',  'darkorange'),
        ]):
            ax = axes[row][col]
            x  = res['x_qoi']
            ax.fill_between(
                x,
                res['Y_pred'][:, idx] - 2 * res['Y_pred_std'][:, idx],
                res['Y_pred'][:, idx] + 2 * res['Y_pred_std'][:, idx],
                alpha=0.20, color=color, label=r'$\pm 2\sigma$',
            )
            ax.plot(x, res['Y_te'][:, idx],   'k-',  lw=2.0, label=r'True')
            ax.plot(x, res['Y_pred'][:, idx],  '--',  lw=1.5, color=color,
                    label=fr'$\hat{{y}}$ {label}')
            ax.set_xlabel('x  [cm]', fontsize=10)
            if col == 0:
                ax.set_ylabel(fr'$\phi(x;\,\mu)$ — {label}', fontsize=10)
            ax.set_title(
                f'Sample {idx}  ({label})\n'
                fr'$\varepsilon_y={res["eps_y"][idx]:.1e}$'
                fr'  $\varepsilon_{{\mathrm{{proj}}}}={res["eps_proj"][idx]:.1e}$',
                fontsize=9,
            )
            ax.legend(fontsize=7)
            ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('Flux reconstruction: GPyTorch (top) vs sklearn (bottom)', fontsize=13)
    plt.tight_layout()
    path = f"{output_dir}/compare_flux_reconstruction.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def _print_summary_table(pt, sk):
    """Print a side-by-side summary table."""
    print(f"\n{'='*65}")
    print(f"{'Metric':<30} {'GPyTorch':>15} {'sklearn':>15}")
    print(f"{'='*65}")
    for label, pt_val, sk_val in [
        ('Median ε_y',    np.median(pt['eps_y']),    np.median(sk['eps_y'])),
        ('Mean   ε_y',    np.mean(pt['eps_y']),      np.mean(sk['eps_y'])),
        ('Max    ε_y',    np.max(pt['eps_y']),        np.max(sk['eps_y'])),
        ('Median ε_proj', np.median(pt['eps_proj']), np.median(sk['eps_proj'])),
        ('Median ε_α',    np.median(pt['eps_alpha']),np.median(sk['eps_alpha'])),
        ('Mean   ε_α',    np.mean(pt['eps_alpha']),  np.mean(sk['eps_alpha'])),
    ]:
        print(f"  {label:<28} {pt_val:>15.3e} {sk_val:>15.3e}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
