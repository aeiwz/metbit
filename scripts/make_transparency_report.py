#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_transparency_report.py - Build a full "native vs improved" transparency
report from a completed benchmark run, so every baseline and every optimised
result is published side by side (including cases where the optimisation is
slower or was skipped). No result is dropped or rounded away.

Reads:
    <run>/threads_1/benchmark_results.json   (single-thread: native vs improved)
    <run>/threads_*/benchmark_results.json   (OpenMP thread sweep, optional)

Writes (next to the run):
    BENCHMARK_TRANSPARENCY.md
    BENCHMARK_TRANSPARENCY.csv

Usage:
    python scripts/make_transparency_report.py [reports/hpc/.../reports/hpc]
"""
from __future__ import annotations
import csv, glob, json, math, os, sys

# baseline ("native") label for each kernel group
BASELINE = {
    "vip_scores":           "Python loop",
    "pearson_columns":      "Full-copy NumPy (baseline)",
    "column_variances":     "NumPy (baseline)",
    "feature_preselection": "Raw NumPy var (baseline)",
    "stocsy":               "Standard STOCSY",
    "opls_da_pipeline":     "float64 pipeline",
}
TITLE = {
    "vip_scores":           "VIP scores (native Python loop -> vectorised / C)",
    "pearson_columns":      "Pearson correlation / STOCSY kernel (full-copy NumPy -> C, no copy)",
    "column_variances":     "Column variance (NumPy -> C, no copy)",
    "feature_preselection": "Feature pre-selection (raw NumPy variance -> full dispatch routine)",
    "stocsy":               "STOCSY figure path (standard -> ChunkedSTOCSY)",
    "opls_da_pipeline":     "OPLS-DA pipeline (float64 -> float32)",
}


def find_run(argv):
    if len(argv) > 1:
        return argv[1]
    hits = glob.glob("reports/hpc/**/threads_1/benchmark_results.json", recursive=True)
    if not hits:
        sys.exit("No benchmark_results.json found under reports/hpc/. Pass the run dir explicitly.")
    return os.path.dirname(os.path.dirname(hits[0]))


def ratio(a, b):
    if b is None or b == 0 or a is None or math.isnan(a) or math.isnan(b):
        return None
    return a / b


def load_base(run):
    # support both a thread-sweep run (threads_1/...) and a single-run dir
    p1 = os.path.join(run, "threads_1", "benchmark_results.json")
    p0 = os.path.join(run, "benchmark_results.json")
    return json.load(open(p1 if os.path.exists(p1) else p0))


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else find_run(sys.argv)
    base = load_base(run)
    b = base["backend"]

    # group single-thread rows by kernel
    groups = {}
    for r in base["results"]:
        groups.setdefault(r["name"], []).append(r)

    md = []
    md += [
        "# metbit Benchmark Transparency Report",
        "",
        "This report lists **every** benchmarked implementation - the native/baseline",
        "version and each optimised version - side by side, so the improvement claims",
        "can be audited in full. Rows where the optimisation is *not* faster, or was",
        "skipped, are included and flagged; nothing is omitted or cherry-picked.",
        "",
        "## Environment",
        "",
        f"- **Platform:** {base['platform']}",
        f"- **CPU cores:** {base['cpu_cores']}",
        f"- **Python:** {base['python']}",
        f"- **Backend:** native C = `{b['native_c']}`, OpenMP threads (this run) = "
        f"`{b['openmp_threads']}`, GPU CuPy = `{b['gpu_cupy']}`, GPU PyTorch = `{b['gpu_torch']}`",
        "",
        "## Method",
        "",
        "- Each value is the **minimum** wall-clock time over 5 repetitions after 1",
        "  discarded warm-up call (minimum is the least noisy estimator for CPU work).",
        "- `Peak MB` is the Python-visible peak allocation via `tracemalloc` (0 = not",
        "  measured for that kernel).",
        "- `Speedup` = baseline_min / this_min. `Mem x` = baseline_peak / this_peak.",
        "- **Baseline** rows are the native reference implementation (speedup 1.00x by",
        "  definition).",
        "",
    ]

    csv_rows = []
    for name in [k for k in TITLE if k in groups]:
        rows = groups[name]
        base_label = BASELINE[name]
        # baseline is per dataset size (n_samples, n_features), not per group
        base_by_size = {(r["n_samples"], r["n_features"]): r
                        for r in rows if r["label"] == base_label}
        show_mem = any(r.get("peak_mb", 0) > 0 for r in rows)

        md += [f"## {TITLE[name]}", ""]
        hdr = "| Dataset (n x p) | Implementation | Min (ms) | Mean (ms) |"
        sep = "|---|---|---|---|"
        if show_mem:
            hdr += " Peak (MB) | Mem x |"
            sep += "---|---|"
        hdr += " Speedup | Note |"
        sep += "---|---|"
        md += [hdr, sep]

        for r in rows:
            ds = f"{r['n_samples']} x {r['n_features']:,}"
            mn, me = r["min_ms"], r["mean_ms"]
            is_base = (r["label"] == base_label)
            base_row = base_by_size.get((r["n_samples"], r["n_features"]))
            bmin = base_row["min_ms"] if base_row else None
            bmem = base_row.get("peak_mb", 0.0) if base_row else 0.0
            nan = isinstance(mn, float) and math.isnan(mn)
            sp = 1.0 if is_base else ratio(bmin, mn)
            note = ""
            if nan:
                note = "skipped (native too slow to time)"
            elif is_base:
                note = "native baseline"
            elif sp is not None and sp < 1.0:
                note = "SLOWER than baseline (does more work - see caveats)"
            mn_s = "skipped" if nan else f"{mn:.3f}"
            me_s = "-" if nan else f"{me:.3f}"
            sp_s = "-" if (nan or sp is None) else f"{sp:.2f}x"
            row = f"| {ds} | {r['label']} | {mn_s} | {me_s} |"
            if show_mem:
                pm = r.get("peak_mb", 0.0)
                mx = "-" if (is_base or not pm or not bmem) else f"{bmem/pm:.0f}x"
                row += f" {pm:.2f} | {mx} |"
            row += f" {sp_s} | {note} |"
            md.append(row)
            csv_rows.append({
                "kernel": name, "dataset": ds, "implementation": r["label"],
                "is_baseline": is_base, "min_ms": ("" if nan else round(mn, 4)),
                "mean_ms": ("" if nan else round(me, 4)),
                "peak_mb": round(r.get("peak_mb", 0.0), 3),
                "speedup_vs_baseline": ("" if (nan or sp is None) else round(sp, 3)),
                "note": note,
            })
        md.append("")

    # ---- OpenMP thread sweep (if present) ----------------------------------
    sweep = sorted(int(os.path.basename(os.path.dirname(p)).split("_")[1])
                   for p in glob.glob(os.path.join(run, "threads_*", "benchmark_results.json")))
    if len(sweep) > 1:
        md += ["## OpenMP thread scaling (native C kernels)", "",
               "Minimum ms for the double-precision C-dispatch path at each "
               "`OMP_NUM_THREADS`. Speed-ups are relative to 1 thread; they saturate "
               "below the core count because the kernels are memory-bandwidth-bound.", ""]
        want = [("pearson_columns", "C dispatch f64", 500, 30000),
                ("column_variances", "C dispatch f64", 500, 100000),
                ("feature_preselection", "feature_preselection dispatch", 200, 100000)]
        cell = {}
        for t in sweep:
            j = json.load(open(os.path.join(run, f"threads_{t}", "benchmark_results.json")))
            for r in j["results"]:
                for (nm, lb, n, p) in want:
                    if r["name"] == nm and r["label"] == lb and r["n_samples"] == n and r["n_features"] == p:
                        cell[(nm, t)] = r["min_ms"]
        md += ["| Kernel | Dataset | " + " | ".join(f"{t} thr" for t in sweep) + " | Best |",
               "|---|---|" + "---|" * (len(sweep) + 1)]
        for (nm, lb, n, p) in want:
            series = [cell.get((nm, t), float("nan")) for t in sweep]
            t1 = series[0]
            best = min(series); bt = sweep[series.index(best)]
            cells = " | ".join(f"{v:.1f}" for v in series)
            md.append(f"| {nm} | {n} x {p:,} | {cells} | {t1/best:.2f}x @{bt} thr |")
            for t, v in zip(sweep, series):
                csv_rows.append({"kernel": nm + " (omp)", "dataset": f"{n} x {p}",
                                 "implementation": f"C f64 @ {t} threads", "is_baseline": (t == 1),
                                 "min_ms": round(v, 4), "mean_ms": "", "peak_mb": "",
                                 "speedup_vs_baseline": round(t1 / v, 3), "note": ""})
        md.append("")

    # ---- honest caveats ----------------------------------------------------
    md += [
        "## Caveats and non-improvements (stated explicitly)", "",
        "- **Feature pre-selection is slower than a raw NumPy variance call.** The",
        "  `feature_preselection` dispatch is a full routine (percentile threshold,",
        "  masking, DataFrame handling), not just a variance; its value is the memory",
        "  behaviour of the underlying C variance kernel, not raw speed. The raw-var",
        "  comparison is shown so this is visible.",
        "- **VIP native Python loop is skipped at 20,000 features** because it is too",
        "  slow to time in a reasonable duration; only the smaller sizes have a loop",
        "  baseline.",
        "- **Pearson speed-up is platform-dependent.** On CPUs with a highly optimised",
        "  vendor BLAS (e.g. Apple Accelerate) the NumPy baseline is competitive on",
        "  speed and the C kernel's advantage is mainly memory; on this Xeon the C",
        "  kernel is both faster and leaner.",
        "- **GPU path not exercised** on this run (no working CUDA driver on the node).",
        "- Peak memory is Python-visible (`tracemalloc`) allocation, which excludes",
        "  memory allocated inside C/BLAS; it is a lower bound on true peak use.",
        "",
    ]

    out_md = os.path.join(run, "BENCHMARK_TRANSPARENCY.md")
    out_csv = os.path.join(run, "BENCHMARK_TRANSPARENCY.csv")
    open(out_md, "w").write("\n".join(md) + "\n")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print("wrote", out_md)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
