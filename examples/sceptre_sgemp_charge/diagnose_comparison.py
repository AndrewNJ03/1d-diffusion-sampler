"""
diagnose_comparison.py: what limits GE-GPR in a comparison, using only the runs on disk.

Checks, per study. All of them use a standard GP fitted to q alone on every usable case,
so it has not seen the adjoint gradients:
  gradients   adjoint gradients against that GP's derivatives. This can show a scale error
              or a missing term. It is coarser than the perturbed-run check in
              check_gradients_fd.py and does not replace it.
  noise       the GP's fitted noise, as an estimate of any noise floor in q, against the
              error both models reach, and how much of the residuals the mesh explains.
  outliers    standardized leave-one-out residuals, which flag a case the others disagree
              with, whether it is in training or test.
  test cases  each test case's share of the squared error, and how often it leaves the band.
  fits        fitted noise and length scales against M.
  failed      the cases dropped for a failed boundary identity, and what they share.

Writes diagnosis_gradients_<study>.csv, diagnosis_loo_<study>.csv, diagnosis_cases_<study>.csv,
diagnosis_failed_<study>.csv and diagnosis.json to the comparison directory. Findings are
prompts for inspection, not proof.

    python3 diagnose_comparison.py --results RESULTS --data-dir DATA_DIR
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

from band_plots import matched_gp_kernel

from check_gradients_fd import read_table
from compare_ge_vs_gp import GE, GP, JITTER, load_study_data, write_csv
from make_geometry_cases import PARAMS, STUDY_PARAMS, read_manifest

SLOPE_TOL = 0.05         # an adjoint/GP slope further than this from 1 is flagged, if subsets agree
SLOPE_SPREADS = 2.0      # and further than this many subset standard deviations
CORRELATION_MIN = 0.95
OUTLIER_SHARE = 0.5      # one test case holding this much of the squared error
OUTLIER_Z = 6.0          # leave-one-out residual, in robust standard deviations of the others
OUTLIER_MIN_REL = 1e-3   # and at least this share of the case's q, to matter in practice
OUTLIER_MIN_TESTS = 8    # with fewer test cases, one holding most of the error is unremarkable
TEST_MIN_REL = 1e-4      # rms error of that test case relative to q, to matter in practice
NOISE_AT_FLOOR = 2.0     # fitted noise within this factor of its lower bound is no evidence of noise
NOISE_RATIO = 0.5        # noise at least this share of a model's final error can limit it
PLATEAU_GAIN = 0.3       # error falling by less than this share from half the pool to all of it
SUBSETS, SUBSET_SHARE = 10, 0.8
NOISE_FLOOR = np.exp(-10.0)
GRADIENT_NOISE_FLAG = 1e-2


# --------------------------------------------------------------------------- #
# A standard GP on q alone, as a reference that has not seen the gradients    #
# --------------------------------------------------------------------------- #

def value_gp(X: np.ndarray, y: np.ndarray, restarts: int = 3, seed: int = 0) -> dict:
    """Fit on standardized data. Returns the q gradients at X, noise in q units and LOO residuals."""
    x_mu, x_sd = X.mean(axis=0), X.std(axis=0)
    y_mu, y_sd = float(y.mean()), float(y.std()) or 1.0
    U, z = (X - x_mu) / x_sd, (y - y_mu) / y_sd
    gp = GaussianProcessRegressor(kernel=matched_gp_kernel(X.shape[1]), n_restarts_optimizer=restarts,
                                  alpha=JITTER, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(U, z)
    ell = np.atleast_1d(gp.kernel_.k1.k2.length_scale)
    sf2 = gp.kernel_.k1.k1.constant_value
    R = U[:, None, :] - U[None, :, :]
    K = sf2 * np.exp(-0.5 * np.sum((R / ell) ** 2, axis=2))
    grads = -np.einsum("ijm,j->im", K[:, :, None] * R / ell ** 2, gp.alpha_) * y_sd / x_sd
    # Leave-one-out with the hyperparameters fixed.
    K_inv = np.linalg.inv(gp.kernel_(U) + JITTER * np.eye(len(y)))
    diag = np.diag(K_inv)
    loo = (K_inv @ z) / diag
    noise_z = float(np.sqrt(gp.kernel_.k2.noise_level))
    return dict(grads=grads, noise=noise_z * y_sd, noise_at_floor=noise_z <= NOISE_AT_FLOOR * NOISE_FLOOR,
                loo=loo * y_sd, loo_z=loo * np.sqrt(diag))


def flag_outlier(loo_z: np.ndarray, loo: np.ndarray, y: np.ndarray) -> int:
    """Index of a case far outside the others' standardized residuals and materially off, or -1."""
    worst = int(np.argmax(np.abs(loo_z)))
    others = np.delete(np.abs(loo_z), worst)
    scale = 1.4826 * float(np.median(others))
    far = len(others) >= 5 and scale > 0 and abs(loo_z[worst]) > OUTLIER_Z * scale
    return worst if far and abs(loo[worst]) >= OUTLIER_MIN_REL * abs(y[worst]) else -1


def _slope(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(b @ b)
    return float(a @ b / denom) if denom > 0 else np.nan


def gradient_rows(X: np.ndarray, y: np.ndarray, G: np.ndarray, names: list, grads: np.ndarray) -> list:
    """Adjoint against the value GP's derivative per thickness, with the spread over data subsets."""
    rng = np.random.default_rng(0)
    slopes = []
    for _ in range(SUBSETS):
        keep = rng.permutation(len(y))[: int(SUBSET_SHARE * len(y))]
        sub = value_gp(X[keep], y[keep], restarts=1)["grads"]
        slopes.append([_slope(G[keep, j], sub[:, j]) for j in range(len(names))])
    spread = np.std(slopes, axis=0)
    y_sd = float(np.std(y)) or 1.0
    rows = []
    for j, name in enumerate(names):
        a, b = G[:, j], grads[:, j]
        rows.append(dict(
            thickness=name, slope=_slope(a, b), slope_spread=float(spread[j]),
            correlation=float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan,
            median_rel_difference=float(np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))),
            # Change in q across the thickness range, relative to q's spread: small means a
            # wrong gradient here matters little.
            influence=float(np.sqrt(np.mean(b ** 2)) * np.ptp(X[:, j]) / y_sd)))
    return rows


def judge_gradients(rows: list) -> list:
    """Flag a thickness whose slope is off by more than the tolerance and the subset spread."""
    flagged = []
    for r in rows:
        off = abs(r["slope"] - 1) > max(SLOPE_TOL, SLOPE_SPREADS * r["slope_spread"])
        if off or r["correlation"] < CORRELATION_MIN:
            flagged.append(r)
    return flagged


# --------------------------------------------------------------------------- #
# Mesh and report                                                             #
# --------------------------------------------------------------------------- #

def read_report(path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {int(r["case_id"]): r for r in csv.DictReader(f)}


def mesh_explained(residual: np.ndarray, case_ids: np.ndarray, X: np.ndarray, names: list, report: dict) -> dict:
    """Adjusted R^2 of the residuals on element sizes, and residual size with a mirrored-count note."""
    if not report or any(int(c) not in report or not report[int(c)]["n_shield"] for c in case_ids):
        return {}
    columns = {"void_thickness": "n_void", "dielectric_thickness": "n_dielectric", "shield_thickness": "n_shield"}
    features = []
    for j, name in enumerate(names):
        counts = np.array([float(report[int(c)][columns[name]]) for c in case_ids])
        features.append(X[:, j] / counts)
    noted = np.array(["element count" in report[int(c)]["message"] for c in case_ids])
    features = [f for f in features if np.ptp(f) > 0]
    out = dict(cases_with_count_note=int(noted.sum()))
    if noted.any() and (~noted).any():
        out["median_abs_residual_noted"] = float(np.median(np.abs(residual[noted])))
        out["median_abs_residual_other"] = float(np.median(np.abs(residual[~noted])))
    n, k = len(residual), len(features)
    if k and n > k + 2:
        A = np.column_stack([np.ones(n)] + [(f - f.mean()) / f.std() for f in features])
        fit = A @ np.linalg.lstsq(A, residual, rcond=None)[0]
        ss_res, ss_tot = np.sum((residual - fit) ** 2), np.sum((residual - residual.mean()) ** 2)
        if ss_tot > 0:
            out["mesh_adjusted_r2"] = float(1 - (ss_res / (n - k - 1)) / (ss_tot / (n - 1)))
    return out


# --------------------------------------------------------------------------- #
# Comparison output                                                           #
# --------------------------------------------------------------------------- #

def read_csv_rows(path) -> list:
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def case_errors(preds, study: str, d: dict, m: int) -> tuple:
    """Per-test-case rows at size m, pooled over seeds, and the error without the worst case."""
    rows, without_worst = [], {}
    seeds = sorted({int(k.split("__")[1][4:]) for k in preds.files
                    if k.startswith(f"{study}__seed") and f"__M{m}__" in k})
    for model in (GE, GP):
        per_case, per_seed = {}, []
        for seed in seeds:
            prefix = f"{study}__seed{seed}__"
            mean_key = f"{prefix}M{m}__{model}__mean"
            if mean_key not in preds.files:
                continue
            if f"{prefix}test_case_ids" in preds.files:   # older runs used only the manifest split
                ids, y = preds[f"{prefix}test_case_ids"], preds[f"{prefix}test_y"]
            else:
                ids, y = d["case_ids"][d["test"]], d["y"][d["test"]]
            err = preds[mean_key] - y
            sd = preds[f"{prefix}M{m}__{model}__sd_total"]
            for c, e, s in zip(ids, err, sd):
                per_case.setdefault(int(c), []).append((e, abs(e) <= 2 * s))
            keep = np.abs(err) < np.abs(err).max()
            per_seed.append(np.linalg.norm(err[keep]) / max(np.linalg.norm(y[keep]), 1e-300))
        total = sum(e ** 2 for v in per_case.values() for e, _ in v) or 1.0
        for c, v in per_case.items():
            errs = np.array([e for e, _ in v])
            rows.append(dict(model=model, M=m, case_id=c, n_tests=len(v), rms_error=float(np.sqrt(np.mean(errs ** 2))),
                             share_of_squared_error=float(np.sum(errs ** 2) / total),
                             times_outside_band=sum(not inside for _, inside in v)))
        without_worst[model] = float(np.median(per_seed)) if per_seed else np.nan
    rows.sort(key=lambda r: (r["model"], -r["share_of_squared_error"]))
    return rows, without_worst


def hyperparameter_rows(results: list, study: str) -> list:
    out = []
    at = [r for r in results if r["study"] == study and r.get("fit_status") == "ok"]
    for model in (GE, GP):
        for m in sorted({int(r["M"]) for r in at}):
            recs = [r for r in at if r["model"] == model and int(r["M"]) == m]
            if not recs:
                continue
            values = [json.loads(r["hyperparameters"]) for r in recs]
            row = dict(model=model, M=m, runs=len(recs), runs_at_bound=sum(bool(r["at_bound"]) for r in recs))
            row.update({k: float(np.median([v[k] for v in values])) for k in values[0]})
            out.append(row)
    return out


def diagnose_study(study: str, d: dict, table: dict, manifest: dict, report: dict, comparison_dir: Path,
                   results: list, summary: list, preds) -> dict:
    out = dict(study=study, findings=[])
    names, X, y, G = d["names"], d["X"], d["y"], d["G"]
    print(f"\n{'=' * 78}\nStudy {study}: {len(y)} usable cases, {len(d['train'])} training, {len(d['test'])} test\n{'=' * 78}")

    # Error against size, over the larger sizes, where a plateau is not hidden by small M.
    at = sorted((s for s in summary if s["study"] == study), key=lambda s: int(s["M"]))
    plateau, ge_end, gp_end = False, np.nan, np.nan
    if at:
        last = at[-1]
        half = min(at, key=lambda s: abs(int(s["M"]) - int(last["M"]) / 2))
        ge_end, gp_end = float(last[f"{GE}_eps_q_median"]), float(last[f"{GP}_eps_q_median"])
        ge_half, gp_half = float(half[f"{GE}_eps_q_median"]), float(half[f"{GP}_eps_q_median"])
        if int(half["M"]) < int(last["M"]):
            gain = 1 - ge_end / ge_half
            plateau = gain < PLATEAU_GAIN
            print(f"Error from M = {half['M']} to {last['M']}: GE-GPR {ge_half:.3e} -> {ge_end:.3e} ({gain:.0%} lower), "
                  f"GP {gp_half:.3e} -> {gp_end:.3e} ({1 - gp_end / gp_half:.0%} lower)")
            if plateau:
                out["findings"].append(f"plateau: GE-GPR error falls {gain:.0%} from M = {half['M']} to {last['M']}")
    out.update(plateau=plateau, ge_final_eps_q=ge_end, gp_final_eps_q=gp_end)

    # Reference GP on q alone.
    if len(y) < 2 * X.shape[1] + 3:
        print("Checks on q alone skipped: too few cases.")
        ref = None
    else:
        ref = value_gp(X, y)
    if ref is not None:
        rows = gradient_rows(X, y, G, names, ref["grads"])
        write_csv(comparison_dir / f"diagnosis_gradients_{study}.csv", rows)
        print("\nAdjoint gradients vs derivatives of a GP fitted to q alone (slope 1 means they agree):")
        print(f"  {'thickness':22s} {'slope':>7s} {'subset sd':>9s} {'corr':>7s} {'med diff':>9s} {'influence':>9s}")
        for r in rows:
            print(f"  {r['thickness']:22s} {r['slope']:7.3f} {r['slope_spread']:9.3f} {r['correlation']:7.3f} "
                  f"{r['median_rel_difference']:9.2%} {r['influence']:9.2f}")
        flagged = judge_gradients(rows)
        out["gradient_flags"] = flagged
        for f in flagged:
            out["findings"].append(f"gradient mismatch in {f['thickness']}: slope {f['slope']:.3f} (subset sd "
                                   f"{f['slope_spread']:.3f}), correlation {f['correlation']:.3f}, influence {f['influence']:.2f}")

        rms_q = float(np.sqrt(np.mean(y ** 2)))
        worst = flag_outlier(ref["loo_z"], ref["loo"], y)
        if worst >= 0:
            cid = int(d["case_ids"][worst])
            out["loo_outlier"] = cid
            out["findings"].append(f"outlier: case {cid} disagrees with the others (leave-one-out residual "
                                   f"{ref['loo'][worst]:.2e}, {abs(ref['loo'][worst]) / abs(y[worst]):.2%} of its q)")
            keep = np.arange(len(y)) != worst
            without = value_gp(X[keep], y[keep])
            noise_eps, at_floor = without["noise"] / float(np.sqrt(np.mean(y[keep] ** 2))), without["noise_at_floor"]
            print(f"\nCase {cid} is a leave-one-out outlier; noise below is estimated without it.")
        else:
            noise_eps, at_floor = ref["noise"] / rms_q, ref["noise_at_floor"]
        out.update(noise_eps_q=noise_eps, noise_at_floor=at_floor)
        write_csv(comparison_dir / f"diagnosis_loo_{study}.csv",
                  [dict(case_id=int(c), loo_residual=float(r), loo_standardized=float(z), q=float(q))
                   for c, r, z, q in zip(d["case_ids"], ref["loo"], ref["loo_z"], y)])
        print(f"\nEstimated noise in q, relative to its size: "
              + (f"at the fit's lower bound ({noise_eps:.3e}), so none detected." if at_floor else f"{noise_eps:.3e}.")
              + f" Final errors: GE-GPR {ge_end:.3e}, GP {gp_end:.3e}.")
        ends = [e for e in (ge_end, gp_end) if np.isfinite(e)]
        if ends and not at_floor and noise_eps >= NOISE_RATIO * min(ends):
            out["findings"].append(f"possible noise floor in q: estimated noise {noise_eps:.2e} is close to the final "
                                   f"errors ({ge_end:.2e}, {gp_end:.2e})")
        mesh = mesh_explained(ref["loo"], d["case_ids"], X, names, report)
        out["mesh"] = mesh
        if "mesh_adjusted_r2" in mesh:
            print(f"Element sizes explain {max(mesh['mesh_adjusted_r2'], 0):.0%} of the leave-one-out residuals (adjusted R^2).")
        if "median_abs_residual_noted" in mesh:
            print(f"Median |residual|: {mesh['median_abs_residual_noted']:.3e} in the {mesh['cases_with_count_note']} cases "
                  f"with a mirrored-count note, {mesh['median_abs_residual_other']:.3e} in the others.")

    # Test cases.
    if preds is not None and at:
        m = int(at[-1]["M"])
        rows, without = case_errors(preds, study, d, m)
        write_csv(comparison_dir / f"diagnosis_cases_{study}.csv", rows or [dict(note="no predictions")])
        if rows:
            print(f"\nLargest test-case errors at M = {m}:")
            print(f"  {'model':12s} {'case':>5s} {'rms error':>10s} {'share':>7s} {'outside band':>13s}")
            for model in (GE, GP):
                for r in [r for r in rows if r["model"] == model][:3]:
                    print(f"  {model:12s} {r['case_id']:5d} {r['rms_error']:10.3e} {r['share_of_squared_error']:7.1%} "
                          f"{r['times_outside_band']:6d} of {r['n_tests']:d}")
            print(f"  Median error without each fit's worst test case: GE-GPR {without[GE]:.3e}, GP {without[GP]:.3e}")
            out["without_worst"] = without
            for model in (GE, GP):
                top = next((r for r in rows if r["model"] == model), None)
                # Only meaningful when other test cases exist to share the error.
                if (top and top["share_of_squared_error"] >= OUTLIER_SHARE
                        and len({r["case_id"] for r in rows}) >= OUTLIER_MIN_TESTS
                        and top["rms_error"] >= TEST_MIN_REL * float(np.sqrt(np.mean(y ** 2)))):
                    out.setdefault("test_outliers", []).append(dict(model=model, case_id=top["case_id"],
                                                                    share=top["share_of_squared_error"]))
                    out["findings"].append(f"test case {top['case_id']} holds {top['share_of_squared_error']:.0%} of "
                                           f"the {model} squared error")

    # Fits.
    hyper = hyperparameter_rows(results, study)
    if hyper:
        print("\nMedian fitted hyperparameters (standardized units; noise floor "
              f"{NOISE_FLOOR:.1e}):")
        for row in hyper:
            values = "  ".join(f"{k}={v:.3g}" for k, v in row.items() if k not in ("model", "M", "runs", "runs_at_bound"))
            print(f"  {row['model']:12s} M={row['M']:<4d} at bound {row['runs_at_bound']}/{row['runs']}  {values}")
        final = [r for r in hyper if r["model"] == GE][-1:]
        for row in final:
            noises = {k: v for k, v in row.items() if k.startswith("noise_gradient")}
            high = {k: v for k, v in noises.items() if v > GRADIENT_NOISE_FLAG}
            out["gradient_noise"] = noises
            if high:
                out["findings"].append("GE-GPR discounts gradients: " + ", ".join(f"{k}={v:.3g}" for k, v in high.items()))
            if len(noises) > 1 and max(noises.values()) > 10 * min(noises.values()):
                out["findings"].append("gradient noise differs by thickness: "
                                       + ", ".join(f"{k}={v:.3g}" for k, v in noises.items()))

    # Failed cases.
    ids = [c for c, m in manifest.items() if m["study"] == study and m["role"] != "fd" and c in table and table[c]["status"] == 2]
    failed_rows = []
    all_ids = [c for c, m in manifest.items() if m["study"] == study and m["role"] != "fd" and c in table]
    active = STUDY_PARAMS.get(study, range(len(PARAMS)))
    if ids:
        print(f"\n{len(ids)} case(s) with a failed boundary identity (fraction: 0 at the low end of the range, 1 at the high end):")
        for c in ids:
            row = dict(case_id=c, identity_error=table[c].get("identity_error", np.nan))
            for j in active:
                values = [table[k][PARAMS[j]] for k in all_ids]
                span = max(values) - min(values)
                row[PARAMS[j]] = table[c][PARAMS[j]]
                row[f"{PARAMS[j]}_fraction"] = (table[c][PARAMS[j]] - min(values)) / span if span > 0 else np.nan
            rec = report.get(c, {})
            row.update(n_void=rec.get("n_void", ""), n_dielectric=rec.get("n_dielectric", ""), n_shield=rec.get("n_shield", ""),
                       report_note=rec.get("message", ""))
            failed_rows.append(row)
            parts = ", ".join(f"{PARAMS[j].split('_')[0]} {row[PARAMS[j]]:.5g} ({row[f'{PARAMS[j]}_fraction']:.2f})" for j in active)
            print(f"  case {c:4d}: identity miss {row['identity_error']:.1%}, {parts}"
                  + (f"; {row['report_note']}" if row["report_note"] else ""))
        write_csv(comparison_dir / f"diagnosis_failed_{study}.csv", failed_rows)
        out["findings"].append(f"{len(ids)} cases failed the boundary identity: {ids}")
    out["failed_cases"] = ids
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="What limits GE-GPR in a comparison")
    ap.add_argument("--results", required=True, help="directory with qoi_geometry_adjoint.csv")
    ap.add_argument("--data-dir", required=True, help="directory with case_manifest.csv and collection_report.csv")
    ap.add_argument("--comparison", help="comparison output (default: RESULTS/comparison)")
    a = ap.parse_args(argv)

    results_dir, data_dir = Path(a.results), Path(a.data_dir)
    comparison_dir = Path(a.comparison) if a.comparison else results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    table_path, manifest_path = results_dir / "qoi_geometry_adjoint.csv", data_dir / "case_manifest.csv"
    table, manifest = read_table(table_path), read_manifest(manifest_path)
    report = read_report(data_dir / "collection_report.csv")
    results = read_csv_rows(comparison_dir / "comparison_results.csv")
    summary = read_csv_rows(comparison_dir / "comparison_summary.csv")
    preds_path = comparison_dir / "predictions.npz"
    preds = np.load(preds_path) if preds_path.exists() else None

    data = load_study_data(table_path, manifest_path)
    diagnoses = {study: diagnose_study(study, d, table, manifest, report, comparison_dir, results, summary, preds)
                 for study, d in data.items() if len(d["y"])}
    has_fd = any(m["role"] == "fd" for m in manifest.values())

    print(f"\n{'=' * 78}\nFindings\n{'=' * 78}")
    for study, diag in diagnoses.items():
        print(f"{study}:")
        for f in diag["findings"] or ["no flags from these checks"]:
            print(f"  - {f}")
    if not has_fd:
        print("No perturbed cases: the gradients have no direct check. make_fd_cases.py can add them.")
    (comparison_dir / "diagnosis.json").write_text(json.dumps(diagnoses, indent=2, default=float))
    print(f"\nSaved diagnosis files to {comparison_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
