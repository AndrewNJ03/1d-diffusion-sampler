"""
compare_ge_vs_gp.py: GE-GPR vs standard GP on the same thickness data from the
coupled electron-photon deterministic transport solve.

Inputs: qoi_geometry_adjoint.csv from void_sensitivity.py --parameterization
independent, and case_manifest.csv from make_geometry_cases.py, plus optionally
gradient_check.json from check_gradients_fd.py.

To keep the comparison fair, both models in every study, seed and size M use the same
training cases, the same held-out cases, the same values q, the same kernel family
and hyperparameter ranges, the same diagonal jitter and the same restart count, and
every fit starts cold. The jitter matters: it sets a numerical error floor, so a
smaller one for either model would look like better accuracy.
Inputs and outputs are scaled from the training subset only. Thicknesses that do not
vary in a study are dropped.

Outputs in --out-dir:
  comparison_results.csv   one row per study, seed, M and model
  comparison_summary.csv   medians, quartiles and improvement per study and M
  predictions.npz          held-out means and stds for every fit
  run_info.json            settings, data counts and the gradient-check verdict
  <study>_*.png            learning curves, calibration, parity and band panels

--splits K redraws the train and test cases K times within each study's usable cases,
keeping the test size, so the result does not rest on one test set. Without it, the
manifest's split is used and --seeds only changes the training order.

Solve counts: a GE-GPR training case costs 2 solves (forward + adjoint) and a GP case
costs 1, so the learning curves are also plotted against solves. The synthetic mode
runs the whole pipeline on an analytic response, for testing without transport data:

    python3 compare_ge_vs_gp.py --synthetic --out-dir /tmp/ge_vs_gp
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from band_plots import (GE_COLOR, GE_STYLE, GP_COLOR, GP_STYLE, ge_hyperparameters,
                        gp_hyperparameters, matched_gp_kernel, plot_band_panels, predict_ge,
                        predict_gp, report_bands)
from check_gradients_fd import read_table, GRADIENT_COLUMNS
from make_geometry_cases import DEFAULT_BOUNDS, PARAMS, STUDY_PARAMS, build_design, read_manifest, write_design

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sklearn_implementation"))
from ge_gpr import GEGaussianProcess  # noqa: E402

GE, GP = "ge_gpr", "standard_gp"
SOLVES_PER_CASE = {GE: 2, GP: 1}
MODEL_STYLE = {GE: ("GE-GPR", GE_COLOR, GE_STYLE, "o"), GP: ("Standard GP", GP_COLOR, GP_STYLE, "s")}
NOMINAL_COVERAGE = 0.9545   # Gaussian probability inside +/-2 sigma
MIN_AUTO_SIZE = 5
JITTER = 1e-6       # GEGaussianProcess's default, which its augmented matrix needs


def relative_l2_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


def pool_sizes(pool: int) -> list:
    """Training sizes for a pool of cases, chosen so that many M also appear as 2M."""
    base = {pool // 4, pool // 3, pool // 2}
    if pool // 8 >= MIN_AUTO_SIZE:
        base.add(pool // 8)
    return sorted({m for b in base for m in (b, 2 * b) if 2 <= m <= pool} | {pool})


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple:
    """Approximate 95% binomial interval for a proportion from n cases."""
    if n <= 0 or not np.isfinite(p):
        return np.nan, np.nan
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def fit_pair(X, y, G, seed: int, n_restarts: int, ge_noise: str = "shared"):
    """Fit both models on one training subset, all scaling from that subset."""
    x_scaler = StandardScaler().fit(X)
    X_n = x_scaler.transform(X)
    y_mu, y_sd = float(y.mean()), float(y.std()) or 1.0
    z = (y - y_mu) / y_sd
    G_n = G * x_scaler.scale_ / y_sd      # chain rule, as in ge_gpr_scalar.py

    t0 = time.perf_counter()
    ge = GEGaussianProcess(active_dims=np.arange(X.shape[1]), n_restarts=n_restarts,
                           jitter=JITTER, per_dim_gradient_noise=ge_noise == "per-dim").fit(X_n, z, G_n)
    t1 = time.perf_counter()
    gp = GaussianProcessRegressor(kernel=matched_gp_kernel(X.shape[1]), n_restarts_optimizer=n_restarts,
                                  normalize_y=False, alpha=JITTER, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)   # at-bound flags report this per fit
        gp.fit(X_n, z)
    t2 = time.perf_counter()
    return dict(ge=ge, gp=gp, x_scaler=x_scaler, y_mu=y_mu, y_sd=y_sd,
                times={GE: t1 - t0, GP: t2 - t1})


def metrics(pred: dict, y_test: np.ndarray) -> dict:
    """Accuracy and calibration on held-out cases. NLPD is scaled by the test range to be unit-free."""
    err = pred["mean"] - y_test
    sd = np.maximum(pred["sd_total"], 1e-300)
    span = float(np.ptp(y_test)) or 1.0
    nlpd = 0.5 * (err / sd) ** 2 + np.log(sd / span) + 0.5 * np.log(2 * np.pi)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    y_std = float(np.std(y_test))
    return dict(eps_q=relative_l2_error(pred["mean"], y_test),
                eps_q_std=rmse / y_std if y_std > 0 else np.nan,
                n_test=len(y_test), n_inside=int(np.sum(np.abs(err) <= 2 * pred["sd_total"])),
                rmse=rmse,
                max_abs_error=float(np.max(np.abs(err))),
                nlpd=float(np.mean(nlpd)),
                coverage_2sigma=float(np.mean(np.abs(err) <= 2 * pred["sd_total"])),
                coverage_2sigma_latent=float(np.mean(np.abs(err) <= 2 * pred["sd_latent"])),
                mean_band_width_rel=float(np.mean(4 * pred["sd_total"]) / span))


def run_one(study, seed, m, X, y, G, train_idx, X_test, y_test, n_restarts, ge_noise="shared"):
    """One seed and size. Each model's row records its fit; a failed fit gives NaN metrics."""
    idx = train_idx[:m]
    base = dict(study=study, seed=seed, M=m)
    try:
        fit = fit_pair(X[idx], y[idx], G[idx], seed, n_restarts, ge_noise)
    except np.linalg.LinAlgError as exc:
        return [dict(base, model=model, solves=m * SOLVES_PER_CASE[model], fit_status=f"failed: {exc}")
                for model in (GE, GP)], {}
    X_t = fit["x_scaler"].transform(X_test)
    preds = {GE: predict_ge(fit["ge"], X_t, fit["y_mu"], fit["y_sd"]),
             GP: predict_gp(fit["gp"], X_t, fit["y_mu"], fit["y_sd"])}
    hypers = {GE: ge_hyperparameters(fit["ge"]), GP: gp_hyperparameters(fit["gp"])}
    rows = []
    for model in (GE, GP):
        rows.append(dict(base, model=model, solves=m * SOLVES_PER_CASE[model], fit_status="ok",
                         **metrics(preds[model], y_test), log_mll=hypers[model]["lml"],
                         fit_seconds=fit["times"][model],
                         hyperparameters=json.dumps(hypers[model]["values"]),
                         at_bound=";".join(hypers[model]["at_bound"])))
    return rows, {(study, seed, m): preds}


def load_study_data(table_path, manifest_path, allow_failed=False) -> dict:
    """
    {study: dict(X, y, G, train, test, active)} of usable cases, in table order. Role
    fd cases are left out. Cases without a gradient, or with a failed boundary
    identity unless allow_failed, are dropped from BOTH models, so they share the data.
    """
    table, manifest = read_table(table_path), read_manifest(manifest_path)
    out = {}
    for study in dict.fromkeys(m["study"] for m in manifest.values()):
        ids = [c for c, m in manifest.items() if m["study"] == study and m["role"] != "fd"]
        absent = [c for c in ids if c not in table]
        ids = [c for c in ids if c in table]
        bad_grad = [c for c in ids if not all(np.isfinite(table[c][g]) for g in GRADIENT_COLUMNS.values())]
        bad_status = [c for c in ids if table[c]["status"] == 2 and not allow_failed]
        keep = [c for c in ids if c not in set(bad_grad) | set(bad_status)]
        if absent or bad_grad or bad_status:
            print(f"Study {study}: dropped {len(absent)} absent, {len(bad_grad)} without gradients, "
                  f"{len(bad_status)} with a failed boundary identity")
        X = np.array([[table[c][p] for p in PARAMS] for c in keep]).reshape(-1, len(PARAMS))
        y = np.array([table[c]["q"] for c in keep])
        G = np.array([[table[c][GRADIENT_COLUMNS[p]] for p in PARAMS] for c in keep]).reshape(-1, len(PARAMS))
        roles = np.array([manifest[c]["role"] for c in keep])
        active = [j for j in STUDY_PARAMS.get(study, range(len(PARAMS)))]
        out[study] = dict(X=X[:, active], y=y, G=G[:, active], train=np.where(roles == "train")[0],
                          test=np.where(roles == "test")[0], names=[PARAMS[j] for j in active],
                          case_ids=np.array(keep))
    return out


def split_cases(d: dict, seed: int, redraw: bool) -> tuple:
    """(training order, test indices) of one seed. redraw draws a new split of the usable cases."""
    rng = np.random.default_rng(seed)
    if not redraw:
        return rng.permutation(d["train"]), d["test"]
    shuffled = rng.permutation(np.concatenate([d["train"], d["test"]]))
    n_test = len(d["test"])
    return shuffled[n_test:], np.sort(shuffled[:n_test])


def run_comparison(data: dict, sizes_single, sizes_all, seeds, n_restarts, n_jobs=1,
                   redraw_splits=False, ge_noise="shared") -> tuple:
    tasks = []
    for study, d in data.items():
        if len(d["test"]) == 0:
            print(f"Study {study}: no held-out cases, skipped")
            continue
        sizes = sizes_all if study == "all" else sizes_single
        pool = len(d["train"])
        use = [m for m in (sizes if sizes is not None else pool_sizes(pool)) if 2 <= m <= pool]
        if pool not in use and pool >= 2:
            use.append(pool)
        if not use:
            print(f"Study {study}: only {pool} training case(s), skipped")
            continue
        d["sizes"] = use
        d["split_test"] = {}
        if not redraw_splits and len(seeds) > 1:
            print(f"Study {study}: at M = {pool}, every seed fits the same {pool} cases and all use the "
                  f"same {len(d['test'])} test cases. Use --splits to vary them.")
        for seed in seeds:
            order, test = split_cases(d, seed, redraw_splits)
            d["split_test"][seed] = test
            for m in use:
                tasks.append(delayed(run_one)(study, seed, m, d["X"], d["y"], d["G"], order,
                                              d["X"][test], d["y"][test], n_restarts, ge_noise))
    print(f"Fitting {len(tasks)} model pairs on {n_jobs} process(es)")
    results = Parallel(n_jobs=n_jobs)(tasks)
    rows = [r for res in results for r in res[0]]
    preds = {k: v for res in results for k, v in res[1].items()}
    for r in rows:      # how many different test cases a study's coverage rests on
        r["n_unique_test"] = len(set().union(*map(set, data[r["study"]]["split_test"].values())))
    return rows, preds


def summarize(rows: list) -> list:
    """Medians per study, M and model, with paired per-seed improvement of GE-GPR over GP."""
    ok = [r for r in rows if r["fit_status"] == "ok"]
    out = []
    for study in dict.fromkeys(r["study"] for r in ok):
        for m in sorted({r["M"] for r in ok if r["study"] == study}):
            at = [r for r in ok if r["study"] == study and r["M"] == m]
            ge = {r["seed"]: r for r in at if r["model"] == GE}
            gp = {r["seed"]: r for r in at if r["model"] == GP}
            paired = [1.0 - ge[s]["eps_q"] / gp[s]["eps_q"] for s in ge if s in gp and gp[s]["eps_q"] > 0]
            # GP at twice the cases costs the same number of solves as GE-GPR at M.
            same_cost = {r["seed"]: r["eps_q"] for r in ok if r["study"] == study and r["model"] == GP
                         and r["M"] == 2 * m}
            paired_cost = [1.0 - ge[s]["eps_q"] / same_cost[s] for s in ge if s in same_cost and same_cost[s] > 0]
            row = dict(study=study, M=m, n_seeds=len(ge),
                       median_improvement_same_M=float(np.median(paired)) if paired else np.nan,
                       ge_win_rate_same_M=float(np.mean(np.array(paired) > 0)) if paired else np.nan,
                       gp_eps_q_at_same_solves=float(np.median(list(same_cost.values()))) if same_cost else np.nan,
                       median_improvement_same_solves=float(np.median(paired_cost)) if paired_cost else np.nan,
                       ge_win_rate_same_solves=float(np.mean(np.array(paired_cost) > 0)) if paired_cost else np.nan)
            for model, recs in ((GE, ge), (GP, gp)):
                eps = [r["eps_q"] for r in recs.values()]
                row[f"{model}_eps_q_median"] = float(np.median(eps)) if eps else np.nan
                row[f"{model}_eps_q_q1"] = float(np.percentile(eps, 25)) if eps else np.nan
                row[f"{model}_eps_q_q3"] = float(np.percentile(eps, 75)) if eps else np.nan
                row[f"{model}_eps_q_std_median"] = float(np.median([r["eps_q_std"] for r in recs.values()])) if recs else np.nan
                inside = sum(r["n_inside"] for r in recs.values())
                total = sum(r["n_test"] for r in recs.values())
                pooled = inside / total if total else np.nan
                # The interval uses the different cases tested: repeats of a case are not independent.
                n_unique = max((r["n_unique_test"] for r in recs.values()), default=0)
                row[f"{model}_coverage_pooled"] = pooled
                row[f"{model}_coverage_pooled_low"], row[f"{model}_coverage_pooled_high"] = wilson_interval(pooled, n_unique)
                row[f"{model}_coverage_n_cases"] = n_unique
                for key in ("coverage_2sigma", "nlpd", "mean_band_width_rel"):
                    vals = [r[key] for r in recs.values()]
                    row[f"{model}_{key}_median"] = float(np.median(vals)) if vals else np.nan
                row[f"{model}_runs_at_bound"] = sum(bool(r["at_bound"]) for r in recs.values())
            out.append(row)
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _style_axes(ax):
    ax.grid(True, color="#e4e3df", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _curve(rows, study, model, key, x_key):
    recs = [r for r in rows if r["study"] == study and r["model"] == model and r["fit_status"] == "ok"]
    xs = sorted({r[x_key] for r in recs})
    vals = [[r[key] for r in recs if r[x_key] == x] for x in xs]
    return (np.array(xs), np.array([np.median(v) for v in vals]),
            np.array([np.percentile(v, 25) for v in vals]), np.array([np.percentile(v, 75) for v in vals]))


def plot_learning_curves(rows, study, out_dir):
    """Held-out error against training cases and against solves: median with quartile band."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, x_key, label in zip(axes, ("M", "solves"), ("training cases M", "transport solves")):
        for model in (GE, GP):
            name, color, style, marker = MODEL_STYLE[model]
            x, med, q1, q3 = _curve(rows, study, model, "eps_q", x_key)
            ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)
            ax.plot(x, med, style, color=color, lw=2, marker=marker, ms=6, label=name)
        ax.set_xlabel(label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style_axes(ax)
    axes[0].set_ylabel(r"held-out relative error $\varepsilon_q$")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Learning curves, study: {study} (median and quartiles over seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{study}_learning_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration(rows, study, out_dir):
    """Share of held-out cases inside +/-2 sigma against M; 95.45% is well calibrated."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(NOMINAL_COVERAGE, color="#52514e", lw=1)
    x_all = sorted({r["M"] for r in rows if r["study"] == study})
    ax.annotate("nominal 95.45%", (x_all[-1], NOMINAL_COVERAGE), xytext=(0, -13), ha="right",
                textcoords="offset points", fontsize=8, color="#52514e")
    for model in (GE, GP):
        name, color, style, marker = MODEL_STYLE[model]
        x, med, q1, q3 = _curve(rows, study, model, "coverage_2sigma", "M")
        ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)
        ax.plot(x, med, style, color=color, lw=2, marker=marker, ms=6, label=name)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("training cases M")
    ax.set_ylabel("held-out cases inside ±2σ")
    ax.set_title(f"Band calibration, study: {study}")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{study}_calibration.png", dpi=150)
    plt.close(fig)


def plot_parity(preds, y_test, study, seed, m, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    pair = preds[(study, seed, m)]
    lims = [y_test.min(), y_test.max()]
    for ax, model in zip(axes, (GE, GP)):
        name, color, _, _ = MODEL_STYLE[model]
        ax.errorbar(y_test, pair[model]["mean"], yerr=2 * pair[model]["sd_total"], fmt="o", ms=5,
                    color=color, ecolor=color, elinewidth=1, alpha=0.8)
        ax.plot(lims, lims, color="#52514e", lw=1)
        ax.set_title(f"{name}, ε_q = {relative_l2_error(pair[model]['mean'], y_test):.2e}")
        ax.set_xlabel("true q")
        _style_axes(ax)
    axes[0].set_ylabel("predicted q (±2σ)")
    fig.suptitle(f"Held-out parity, study: {study}, M = {m}, seed {seed}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{study}_parity.png", dpi=150)
    plt.close(fig)


def plot_slices(d, study, seed, m, n_restarts, out_dir, ge_noise="shared"):
    """Band panels along each thickness at the largest M; others held at the training median."""
    order = np.random.default_rng(seed).permutation(d["train"])[:m]
    X, y, G = d["X"][order], d["y"][order], d["G"][order]
    fit = fit_pair(X, y, G, seed, n_restarts, ge_noise)
    X_test, y_test = d["X"][d["test"]], d["y"][d["test"]]
    for j, name in enumerate(d["names"]):
        grid = np.linspace(min(X[:, j].min(), X_test[:, j].min()), max(X[:, j].max(), X_test[:, j].max()), 300)
        Xq = np.tile(np.median(d["X"][d["train"]], axis=0), (len(grid), 1))
        Xq[:, j] = grid
        Xq_n = fit["x_scaler"].transform(Xq)
        ge = predict_ge(fit["ge"], Xq_n, fit["y_mu"], fit["y_sd"])
        gp = predict_gp(fit["gp"], Xq_n, fit["y_mu"], fit["y_sd"])
        print(f"Study {study}, slice along {name}, M = {m}:")
        y_range = float(np.ptp(d["y"])) or 1.0
        report_bands("  GE-GPR", ge, y_range, ge_hyperparameters(fit["ge"]))
        report_bands("  Standard GP", gp, y_range, gp_hyperparameters(fit["gp"]))
        # Measured cases lie on the curve only when this is the one thickness that varies.
        on_slice = X.shape[1] == 1
        title = f"Study {study}: along {name}, M = {m}"
        if not on_slice:
            title += ", others at training median"
        plot_band_panels(grid, ge, gp, X[:, j] if on_slice else None, y if on_slice else None,
                         xlabel=name, ylabel="q", title=title,
                         path=out_dir / f"{study}_bands_{name}.png",
                         X_test=X_test[:, j] if on_slice else None, y_test=y_test if on_slice else None)


# --------------------------------------------------------------------------- #
# Synthetic response, to test the pipeline without transport data             #
# --------------------------------------------------------------------------- #

def synthetic_response(X: np.ndarray, bounds: np.ndarray) -> tuple:
    """q(t_v, t_d, t_s) and its exact gradient: smooth, nonlinear, with interactions."""
    span = bounds[:, 1] - bounds[:, 0]
    u = (X - bounds[:, 0]) / span
    a, b, c = u[:, 0], u[:, 1], u[:, 2]
    q = 4e-8 * (1.0 + 0.3 * np.sin(3.0 * a) + 0.2 * np.exp(-b) * np.cos(2.0 * c) + 0.15 * a * b * c)
    dq = 4e-8 * np.column_stack([
        0.9 * np.cos(3.0 * a) + 0.15 * b * c,
        -0.2 * np.exp(-b) * np.cos(2.0 * c) + 0.15 * a * c,
        -0.4 * np.exp(-b) * np.sin(2.0 * c) + 0.15 * a * b,
    ])
    return q, dq / span


def write_synthetic_table(out_dir, gradient_error="none", seed=1, **design_kw) -> tuple:
    """Case design and a table in void_sensitivity.py's format, with an analytic q."""
    bounds = np.array([DEFAULT_BOUNDS[k] for k in ("void", "dielectric", "shield")], float)
    X, manifest = build_design(bounds, bounds.mean(axis=1), seed=seed, **design_kw)
    _, manifest_path = write_design(out_dir, X, manifest, core_thickness=0.1)
    q, dq = synthetic_response(X, bounds)
    if gradient_error == "wrong":      # right size, unrelated to the response
        dq = np.random.default_rng(seed).permutation(dq, axis=0)
    elif gradient_error != "none":
        raise ValueError("gradient_error must be 'none' or 'wrong'")
    table_path = Path(out_dir) / "qoi_geometry_adjoint.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", *PARAMS, "q", "dq_dtv", "dq_dtd", "dq_dts", "form", "status",
                         "dump_per_key_length", "max_layer_spread"])
        for m, x, qi, g in zip(manifest, X, q, dq):
            grads = g if m["role"] != "fd" else [np.nan] * 3   # as for a perturbed run with no adjoint
            writer.writerow([m["case_id"], *x, qi, *grads, "adjoint", 0, 1.0, 0.0])
    return table_path, manifest_path


def _value(v, fmt):
    return format(v, fmt) if np.isfinite(v) else "-"


def print_summary(summary: list, n_seeds: int, redrawn: bool):
    what = f"{n_seeds} train/test splits" if redrawn else f"{n_seeds} training orders on one test set"
    print(f"\nMedians over {what}. Same-M compares equal cases; same-solves compares GE-GPR at M with "
          "GP at 2M, the same cost.")
    print(f"{'study':11s} {'M':>4s} {'GE eps_q':>10s} {'quartiles':>19s} {'GP eps_q':>10s} {'quartiles':>19s} "
          f"{'same M':>7s} {'win':>5s} {'same solves':>12s} {'win':>5s}")
    for s in summary:
        print(f"{s['study']:11s} {s['M']:4d} {s[GE + '_eps_q_median']:10.3e} "
              f"{s[GE + '_eps_q_q1']:9.2e} {s[GE + '_eps_q_q3']:9.2e} {s[GP + '_eps_q_median']:10.3e} "
              f"{s[GP + '_eps_q_q1']:9.2e} {s[GP + '_eps_q_q3']:9.2e} "
              f"{_value(s['median_improvement_same_M'], '7.1%'):>7s} {_value(s['ge_win_rate_same_M'], '5.0%'):>5s} "
              f"{_value(s['median_improvement_same_solves'], '12.1%'):>12s} "
              f"{_value(s['ge_win_rate_same_solves'], '5.0%'):>5s}")
    print("win: share of paired fits where GE-GPR has the lower error. '-': GP has too few cases.\n")
    print(f"{'study':11s} {'M':>4s} {'GE err/std':>10s} {'GP err/std':>10s} {'GE NLPD':>8s} {'GP NLPD':>8s} "
          f"{'GE coverage':>22s} {'GP coverage':>22s}")
    for s in summary:
        cover = {m: f"{s[m + '_coverage_pooled']:.2f} [{s[m + '_coverage_pooled_low']:.2f}, "
                    f"{s[m + '_coverage_pooled_high']:.2f}]" for m in (GE, GP)}
        print(f"{s['study']:11s} {s['M']:4d} {s[GE + '_eps_q_std_median']:10.3e} {s[GP + '_eps_q_std_median']:10.3e} "
              f"{s[GE + '_nlpd_median']:8.2f} {s[GP + '_nlpd_median']:8.2f} {cover[GE]:>22s} {cover[GP]:>22s}")
    print(f"err/std: RMSE over the test q standard deviation. Coverage: share inside +/-2 sigma (nominal "
          f"{NOMINAL_COVERAGE:.2f}), with an approximate 95% interval from the different test cases.\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="GE-GPR vs standard GP on thickness data")
    ap.add_argument("--table", help="qoi_geometry_adjoint.csv")
    ap.add_argument("--manifest", help="case_manifest.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gradient-check", help="gradient_check.json from check_gradients_fd.py")
    ap.add_argument("--allow-failed", action="store_true", help="keep cases with a failed boundary identity")
    ap.add_argument("--sizes-single", nargs="+", type=int, help="default: chosen from the training pool")
    ap.add_argument("--sizes-all", nargs="+", type=int, help="default: chosen from the training pool")
    ap.add_argument("--seeds", type=int, default=5, help="number of random training orders")
    ap.add_argument("--splits", type=int, help="redraw train and test this many times (replaces --seeds)")
    ap.add_argument("--ge-noise", default="shared", choices=["shared", "per-dim"],
                    help="one gradient noise term for GE-GPR, or one per thickness")
    ap.add_argument("--restarts", type=int, default=5, help="optimizer restarts, same for both models")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true", help="analytic response, no transport data needed")
    ap.add_argument("--synthetic-gradient-error", default="none", choices=["none", "wrong"],
                    help="wrong: shuffled gradients, to test that the comparison notices")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        table_path, manifest_path = write_synthetic_table(out_dir, a.synthetic_gradient_error)
    elif a.table and a.manifest:
        table_path, manifest_path = Path(a.table), Path(a.manifest)
    else:
        ap.error("pass --table and --manifest, or --synthetic")

    verdict = None
    if a.gradient_check:
        verdict = json.loads(Path(a.gradient_check).read_text())
        failed = [p for p in PARAMS if verdict.get(p, {}).get("passed") is not True]
        if failed:
            print(f"WARNING: gradients failed or missed the finite-difference check for: {', '.join(failed)}. "
                  "A poor GE-GPR result would then say nothing about gradient enhancement.")
    elif not a.synthetic:
        print("WARNING: no --gradient-check given, so these gradients are not validated against "
              "perturbed runs. Run check_gradients_fd.py before trusting this comparison.")

    data = load_study_data(table_path, manifest_path, a.allow_failed)
    seeds = list(range(1, (a.splits or a.seeds) + 1))
    rows, preds = run_comparison(data, a.sizes_single, a.sizes_all, seeds, a.restarts, a.n_jobs,
                                 redraw_splits=bool(a.splits), ge_noise=a.ge_noise)
    if not rows:
        raise SystemExit("Nothing to compare: no study has both training and held-out cases.")
    summary = summarize(rows)
    write_csv(out_dir / "comparison_results.csv", rows)
    write_csv(out_dir / "comparison_summary.csv", summary)
    arrays = {f"{study}__seed{seed}__M{m}__{model}__{key}": val
              for (study, seed, m), pair in preds.items() for model, pred in pair.items()
              for key, val in pred.items()}
    for study, d in data.items():
        for seed, test in d.get("split_test", {}).items():
            arrays[f"{study}__seed{seed}__test_case_ids"] = d["case_ids"][test]
            arrays[f"{study}__seed{seed}__test_y"] = d["y"][test]
    np.savez(out_dir / "predictions.npz", **arrays)
    n_failed = sum(r["fit_status"] != "ok" for r in rows) // 2
    if n_failed:
        print(f"WARNING: {n_failed} model pair(s) failed to fit; see fit_status in comparison_results.csv")
    (out_dir / "run_info.json").write_text(json.dumps(dict(
        table=str(table_path), manifest=str(manifest_path), seeds=seeds, restarts=a.restarts,
        splits_redrawn=bool(a.splits), ge_noise=a.ge_noise,
        synthetic=a.synthetic, synthetic_gradient_error=a.synthetic_gradient_error if a.synthetic else None,
        gradient_check=verdict, failed_fits=n_failed,
        cases={s: dict(train=len(d["train"]), test=len(d["test"])) for s, d in data.items()}), indent=2))

    print_summary(summary, len(seeds), bool(a.splits))

    for study, d in data.items():
        if "sizes" not in d:
            continue
        plot_learning_curves(rows, study, out_dir)
        plot_calibration(rows, study, out_dir)
        m_max = d["sizes"][-1]
        if (study, seeds[0], m_max) in preds:
            plot_parity(preds, d["y"][d["split_test"][seeds[0]]], study, seeds[0], m_max, out_dir)
            plot_slices(d, study, seeds[0], m_max, a.restarts, out_dir, a.ge_noise)
    print(f"\nSaved results and plots to {out_dir}")
    return rows, summary


if __name__ == "__main__":
    main()
