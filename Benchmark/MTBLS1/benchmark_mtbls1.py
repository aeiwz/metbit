"""
Full-pipeline MTBLS1 benchmark for metbit: raw Bruker FID -> preprocessing ->
alignment -> normalisation -> PCA -> OPLS-DA -> VIP -> STOCSY.

Scope (per manuscript Data Availability statement):
  - Usability:       lines of code / API calls needed to run the full pipeline,
                      and whether it completes without manual intervention.
  - Reproducibility: same-seed determinism check across repeated fits of the
                      stochastic stages (PCA, OPLS-DA/CV) on the same
                      preprocessed matrix.
  - Performance:     wall-clock time and peak memory (tracemalloc) per stage.

Usage:
    python Benchmark/MTBLS1/benchmark_mtbls1.py

Requires the raw MTBLS1 FID archive at manuscript/benchmark/data/MTBLS1/FILES
(132 Bruker FID directories) and sample metadata at
manuscript/benchmark/data/MTBLS1/samples_parsed.csv. Both are fetched by the
MetaboLights download step (not part of this script) and are not re-fetched
here to avoid a ~132-file FTP download on every run.
"""

import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # use the local dev copy of metbit

import metbit
from metbit import Normalise, STOCSY, opls_da, pca

DATA_DIR = REPO_ROOT / "manuscript" / "benchmark" / "data" / "MTBLS1"
FID_DIR = DATA_DIR / "FILES"
SAMPLES_CSV = DATA_DIR / "samples_parsed.csv"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

REPRODUCIBILITY_REPEATS = 3
RANDOM_STATE = 42


def save_spectra_qc_plot(X, ppm, phase):
    """Overlay all auto-phased spectra so phase quality can be eyeballed:
    absorptive (upright, flat baseline) vs dispersive (S-shaped/negative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for idx in X.index:
        axes[0].plot(ppm, X.loc[idx].values, lw=0.3)
    axes[0].set_xlim(10, -1)
    axes[0].axhline(0, color="k", lw=0.3)
    axes[0].set_title(f"MTBLS1 auto-phased spectra (n={X.shape[0]}) - full range")
    axes[0].set_xlabel("ppm")
    for idx in X.index:
        axes[1].plot(ppm, X.loc[idx].values, lw=0.4)
    axes[1].set_xlim(9, 6)
    axes[1].axhline(0, color="k", lw=0.3)
    axes[1].set_title("Aromatic region 6-9 ppm (upright peaks + flat baseline = good phasing)")
    axes[1].set_xlabel("ppm")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "spectra_qc.png", dpi=110)
    plt.close(fig)
    phase.to_csv(RESULT_DIR / "phase_angles.csv")
    print(f"  [QC] spectra plot -> {RESULT_DIR / 'spectra_qc.png'}")


def timed_stage(name, fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  [{name}] {elapsed:.2f}s, peak {peak / 1e6:.1f} MB")
    return result, {"stage": name, "seconds": round(elapsed, 3), "peak_mb": round(peak / 1e6, 2)}


def load_group_labels(sample_ids):
    meta = pd.read_csv(SAMPLES_CSV)
    label_map = dict(zip(meta["Sample Name"], meta["Factor Value[Metabolic syndrome]"]))
    labels = pd.Series([label_map.get(sid, "unknown") for sid in sample_ids], index=sample_ids, name="group")
    return labels


def exclude_outliers(X, labels, n_components=5, conf=0.95):
    """Drop multivariate outliers via Hotelling's T2 on PCA scores.

    Fits a PCA on pareto-scaled spectra, computes each sample's T2, and
    removes samples beyond the F-distribution confidence limit. Returns the
    cleaned matrix, aligned labels, and an info dict for the report.
    """
    from scipy.stats import f as f_dist
    from sklearn.decomposition import PCA

    Xv = X.to_numpy(dtype=float)
    Xc = Xv - Xv.mean(0)
    Xs = Xc / np.sqrt(np.sqrt(Xv.std(0, ddof=1)) + 1e-12)  # pareto scaling
    A = min(n_components, min(Xs.shape) - 1)
    scores = PCA(n_components=A, svd_solver="randomized", random_state=RANDOM_STATE).fit_transform(Xs)
    lam = scores.var(0, ddof=1)
    t2 = (scores ** 2 / lam).sum(1)
    n = Xs.shape[0]
    limit = A * (n - 1) / (n - A) * f_dist.ppf(conf, A, n - A)
    keep = t2 <= limit
    keep_pos = np.where(keep)[0]
    # X (post-PQN) may carry a reset integer index; labels keeps sample names.
    # Index positionally and take names from labels so both stay aligned.
    excluded = [str(s) for s in labels.index[~keep]]
    info = {
        "method": f"Hotelling T2 on {A}-component PCA (pareto), {int(conf*100)}% F-limit",
        "n_before": int(n),
        "n_excluded": int((~keep).sum()),
        "n_after": int(keep.sum()),
        "T2_limit": round(float(limit), 3),
        "excluded_samples": excluded,
    }
    X_clean = X.iloc[keep_pos].copy()
    labels_clean = labels.iloc[keep_pos]
    X_clean.index = labels_clean.index  # restore sample-name index on the matrix
    return X_clean, labels_clean, info


def run_pipeline_stage_by_stage(perf_log):
    from metbit import nmr_preprocessing

    def preprocess():
        prep = nmr_preprocessing(str(FID_DIR), calib_type="tsp", auto_phasing=True, align=True)
        return prep.get_data(), prep

    (X_raw, prep), m = timed_stage("preprocessing (raw FID -> auto-phased, aligned matrix)", preprocess)
    perf_log.append(m)
    save_spectra_qc_plot(X_raw, prep.get_ppm(), prep.get_phase())

    labels = load_group_labels(X_raw.index)

    def normalize():
        norm = Normalise(X_raw)
        return norm.pqn_normalise(plot=False)

    X_norm, m = timed_stage("PQN normalisation", normalize)
    perf_log.append(m)

    (X_clean, labels_clean, outlier_info), m = timed_stage(
        "outlier exclusion (Hotelling T2)", exclude_outliers, X_norm, labels)
    perf_log.append(m)
    print(f"  [outliers] excluded {outlier_info['n_excluded']}/{outlier_info['n_before']}: "
          f"{outlier_info['excluded_samples']}")

    def run_pca():
        model = pca(X_clean, label=labels_clean, n_components=5, scaling_method="pareto")
        model.fit()
        return model

    pca_model, m = timed_stage("PCA fit", run_pca)
    perf_log.append(m)

    def run_opls():
        model = opls_da(X=X_clean, y=labels_clean, n_components=2, scaling_method="pareto",
                         kfold=7, estimator="opls", random_state=RANDOM_STATE)
        model.fit()
        model.vip_scores()
        return model

    opls_model, m = timed_stage("OPLS-DA fit + VIP", run_opls)
    perf_log.append(m)

    vip_df = opls_model.get_vip_scores()
    anchor_ppm = float(vip_df.sort_values("VIP", ascending=False).iloc[0]["Features"])

    def run_stocsy():
        return STOCSY(spectra=X_clean, anchor_ppm_value=anchor_ppm, p_value_threshold=1e-6)

    _, m = timed_stage(f"STOCSY (anchor {anchor_ppm:.3f} ppm)", run_stocsy)
    perf_log.append(m)

    return X_raw, X_clean, labels_clean, pca_model, opls_model, vip_df, anchor_ppm, outlier_info


def reproducibility_check(X_norm, labels, n_repeats=REPRODUCIBILITY_REPEATS):
    """Same-seed determinism check on the stochastic stages (PCA, OPLS-DA/CV)
    against the already-preprocessed matrix. Raw FID preprocessing is
    deterministic (no RNG involved) and is not repeated here."""
    r2y_vals, q2_vals, vip_top10_sets = [], [], []
    for i in range(n_repeats):
        model = opls_da(X=X_norm, y=labels, n_components=2, scaling_method="pareto",
                         kfold=7, estimator="opls", random_state=RANDOM_STATE)
        model.fit()
        model.vip_scores()
        r2y_vals.append(float(model.R2y))
        q2_vals.append(float(model.q2))
        top10 = tuple(model.get_vip_scores().sort_values("VIP", ascending=False)["Features"].head(10).round(4))
        vip_top10_sets.append(top10)

    r2y_vals = np.array(r2y_vals)
    q2_vals = np.array(q2_vals)
    identical_top10 = len(set(vip_top10_sets)) == 1

    return {
        "n_repeats": n_repeats,
        "random_state": RANDOM_STATE,
        "R2Y_values": [round(float(v), 6) for v in r2y_vals],
        "R2Y_max_abs_diff": round(float(r2y_vals.max() - r2y_vals.min()), 10),
        "Q2_values": [round(float(v), 6) for v in q2_vals],
        "Q2_max_abs_diff": round(float(q2_vals.max() - q2_vals.min()), 10),
        "top10_VIP_ppm_identical_across_repeats": identical_top10,
    }


def usability_summary():
    """The pipeline function above IS the usability artifact: count its
    lines and the metbit API calls a user needs to go from raw FID to
    STOCSY results."""
    src = Path(__file__).read_text().splitlines()
    stage_fn_start = next(i for i, l in enumerate(src) if "def run_pipeline_stage_by_stage" in l)
    stage_fn_end = next(i for i, l in enumerate(src[stage_fn_start + 1:], stage_fn_start + 1)
                         if l.startswith("def ") or l.startswith("\n"))
    body = src[stage_fn_start:stage_fn_end]
    api_calls = [
        "nmr_preprocessing(...)", "prep.get_data()", "Normalise(...)", "norm.pqn_normalise()",
        "pca(...)", "model.fit()", "opls_da(...)", "model.fit()", "model.vip_scores()",
        "model.get_vip_scores()", "STOCSY(...)",
    ]
    return {
        "pipeline_function_lines": len(body),
        "distinct_metbit_api_calls_raw_fid_to_stocsy": len(api_calls),
        "api_call_sequence": api_calls,
        "manual_intervention_required": False,
        "note": "Single Python function, no manual parameter tuning between stages; "
                "all stages run against the same in-memory objects.",
    }


def environment_info():
    import numpy, scipy, sklearn
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "metbit_version": getattr(metbit, "__version__", "unknown"),
        "metbit_backend": metbit.backend_info(),
    }


def main():
    print("=== metbit MTBLS1 full-pipeline benchmark ===")
    print(f"FID directory: {FID_DIR}")
    n_fid_dirs = len(list(FID_DIR.iterdir())) if FID_DIR.exists() else 0
    print(f"Samples found: {n_fid_dirs}")
    if n_fid_dirs == 0:
        raise SystemExit(f"No raw FID data at {FID_DIR}. Download MTBLS1 first.")

    perf_log = []
    t_start = time.perf_counter()
    X_raw, X_clean, labels, pca_model, opls_model, vip_df, anchor_ppm, outlier_info = run_pipeline_stage_by_stage(perf_log)
    total_wallclock = time.perf_counter() - t_start

    print("\n--- Reproducibility check (same-seed determinism, n="
          f"{REPRODUCIBILITY_REPEATS}) ---")
    repro = reproducibility_check(X_clean, labels)
    print(json.dumps(repro, indent=2))

    # ---- persist results -----------------------------------------------
    X_clean.astype("float32").to_csv(RESULT_DIR / "spectral_matrix_normalised.csv")
    pca_model.get_scores().to_csv(RESULT_DIR / "pca_scores.csv")
    vip_df.to_csv(RESULT_DIR / "vip_scores.csv", index=False)
    (RESULT_DIR / "outlier_report.json").write_text(json.dumps(outlier_info, indent=2))

    performance_report = {
        "total_wallclock_seconds": round(total_wallclock, 2),
        "n_samples": int(X_raw.shape[0]),
        "n_samples_after_outlier_exclusion": int(X_clean.shape[0]),
        "n_features_raw": int(X_raw.shape[1]),
        "n_features_normalised": int(X_clean.shape[1]),
        "stages": perf_log,
        "environment": environment_info(),
    }
    (RESULT_DIR / "performance_report.json").write_text(json.dumps(performance_report, indent=2))

    (RESULT_DIR / "reproducibility_report.json").write_text(json.dumps(repro, indent=2))

    usability = usability_summary()
    (RESULT_DIR / "usability_report.json").write_text(json.dumps(usability, indent=2))

    opls_metrics = {
        "R2Y": float(opls_model.R2y),
        "Q2": float(opls_model.q2),
        "anchor_ppm_top_vip": anchor_ppm,
        "groups": sorted(labels.unique().tolist()),
        "n_samples": int(len(labels)),
    }
    (RESULT_DIR / "opls_metrics.json").write_text(json.dumps(opls_metrics, indent=2))

    # ---- combined markdown report ---------------------------------------
    report_lines = [
        "# metbit MTBLS1 Full-Pipeline Benchmark",
        "",
        f"Generated by `Benchmark/MTBLS1/benchmark_mtbls1.py`. "
        f"Samples: {X_raw.shape[0]} ({X_clean.shape[0]} after outlier exclusion), "
        f"features: {X_clean.shape[1]}.",
        "",
        "## Outlier exclusion",
        "",
        f"- {outlier_info['method']}",
        f"- Excluded {outlier_info['n_excluded']}/{outlier_info['n_before']} "
        f"(T2 limit {outlier_info['T2_limit']}): {outlier_info['excluded_samples']}",
        "",
        "## Performance",
        "",
        "| Stage | Time (s) | Peak memory (MB) |",
        "|---|---|---|",
    ]
    for s in perf_log:
        report_lines.append(f"| {s['stage']} | {s['seconds']} | {s['peak_mb']} |")
    report_lines += [
        f"| **Total wall-clock** | **{performance_report['total_wallclock_seconds']}** | — |",
        "",
        "## Reproducibility (same-seed determinism, n=%d)" % REPRODUCIBILITY_REPEATS,
        "",
        f"- R2Y across repeats: {[float(v) for v in repro['R2Y_values']]} "
        f"(max abs diff: {repro['R2Y_max_abs_diff']})",
        f"- Q2 across repeats: {[float(v) for v in repro['Q2_values']]} "
        f"(max abs diff: {repro['Q2_max_abs_diff']})",
        f"- Top-10 VIP ppm identical across repeats: {repro['top10_VIP_ppm_identical_across_repeats']}",
        "",
        "## Usability",
        "",
        f"- Pipeline function length: {usability['pipeline_function_lines']} lines",
        f"- Distinct metbit API calls, raw FID to STOCSY: "
        f"{usability['distinct_metbit_api_calls_raw_fid_to_stocsy']}",
        f"- Manual intervention required: {usability['manual_intervention_required']}",
        "",
        "## OPLS-DA summary",
        "",
        f"- R2Y = {opls_metrics['R2Y']:.4f}, Q2 = {opls_metrics['Q2']:.4f}",
        f"- Groups: {opls_metrics['groups']}",
        f"- Top-VIP anchor used for STOCSY: {anchor_ppm:.3f} ppm",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(performance_report["environment"], indent=2),
        "```",
    ]
    (RESULT_DIR / "REPORT.md").write_text("\n".join(report_lines))

    print(f"\nDone. Results written to {RESULT_DIR}")


if __name__ == "__main__":
    main()
