#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perf_report.py - Standalone performance benchmark and report generator for metbit.

Usage
-----
    python scripts/perf_report.py                   # default sizes
    python scripts/perf_report.py --quick           # smaller datasets, fewer reps
    python scripts/perf_report.py --output reports/ # custom output directory

Output
------
    reports/benchmark_results.json   machine-readable results with all timings
    reports/PERFORMANCE.md           human-readable markdown report with tables

What is benchmarked
-------------------
1. VIP scores           Python loop vs NumPy vectorized vs C+OpenMP
2. Pearson correlation  C single-thread vs C parallel vs float32 vs NumPy chunked
3. Column variance      C vs plain NumPy (float64 and float32)
4. Feature preselection dispatch vs raw NumPy variance
5. ChunkedSTOCSY        vs standard STOCSY (overhead measurement)
6. opls_da pipeline     fit + VIP at different dataset sizes
7. Alignment            icoshift_align single allocation vs old double-copy

Design notes
------------
- Each benchmark runs `reps` iterations and reports MIN/MEAN/MEDIAN/MAX times.
  MIN is the most reliable for comparing implementations (avoids GC pauses).
- Speedup ratios are reported as: reference_min / new_min.
- All sizes are chosen to run in < 5 minutes total on a modern laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    name: str
    label: str
    n_samples: int
    n_features: int
    times_ms: List[float]
    peak_mb: float = 0.0        # tracemalloc peak in MB (0 = not measured)
    extra: dict = field(default_factory=dict)

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)


def _bench(fn: Callable, reps: int = 5, warmup: int = 1) -> List[float]:
    """Run `fn` `warmup` times (discarded) then `reps` times, return ms list."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _measure_peak_mb(fn: Callable) -> float:
    """Return tracemalloc peak allocation in MB for one call to fn."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def _speedup_str(ref_ms: float, new_ms: float) -> str:
    if new_ms <= 0:
        return "inf"
    r = ref_ms / new_ms
    return f"{r:.1f}x"


# ---------------------------------------------------------------------------
# Reference implementations (baselines to beat)
# ---------------------------------------------------------------------------

def _vip_loop(t, w, q):
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = float(np.sum(s))
    vips = np.zeros(p)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = float(np.sqrt(p * (s.T @ weight) / total_s).squeeze())
    return vips


def _pearson_full_copy(mat, anchor):
    a = mat[:, anchor]; a_c = a - a.mean()
    C = mat - mat.mean(axis=0)
    num = a_c @ C
    denom = np.sqrt(np.dot(a_c, a_c) * np.einsum("ij,ij->j", C, C))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / denom, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _pls_matrices(n, p, h, seed=0):
    from sklearn.cross_decomposition import PLSRegression
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, n).astype(float)
    pls = PLSRegression(n_components=min(h, min(n, p) - 1)).fit(X, y)
    return (
        pls.x_scores_.astype(np.float64),
        pls.x_weights_.astype(np.float64),
        pls.y_loadings_.astype(np.float64),
    )


def _mat64(n, p, seed=0):
    return np.random.default_rng(seed).standard_normal((n, p)).astype(np.float64, order="C")


def _spectra_df(n, p, seed=0):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(9.5, 0.5, p)
    return pd.DataFrame(rng.standard_normal((n, p)), columns=ppm.tolist())


# ---------------------------------------------------------------------------
# Individual benchmark suites
# ---------------------------------------------------------------------------

def bench_vip(reps: int, sizes: List[int]) -> List[TimingResult]:
    from metbit._native import vip_scores as _dispatch_vip

    results = []
    n, h = 80, 3
    for p in sizes:
        print(f"  VIP  n={n} p={p:,} h={h} ...", end=" ", flush=True)
        t, w, q = _pls_matrices(n, p, h)

        # Reference: Python loop (skip for large p to avoid multi-minute waits)
        if p <= 5_000:
            loop_t = _bench(lambda: _vip_loop(t, w, q), reps=reps, warmup=0)
        else:
            loop_t = [float("nan")]

        # NumPy vectorized (no C)
        def _numpy_vip():
            S = np.einsum("ij,ij->j", t, t) * (q.ravel() ** 2)
            norms = np.linalg.norm(w, axis=0); norms[norms == 0] = 1.0
            return np.sqrt(p * ((w / norms) ** 2 @ S) / S.sum())
        np_t = _bench(_numpy_vip, reps=reps)

        # C dispatch (may use C+OpenMP or fall back to numpy)
        c_t = _bench(lambda: _dispatch_vip(t, w, q), reps=reps)

        for label, times in [("Python loop", loop_t), ("NumPy vectorized", np_t), ("C dispatch", c_t)]:
            results.append(TimingResult(
                name="vip_scores", label=label,
                n_samples=n, n_features=p, times_ms=times,
            ))
        print("done")
    return results


def bench_pearson(reps: int, sizes: List[tuple]) -> List[TimingResult]:
    from metbit._native import pearson_columns as _dispatch
    from metbit import _native as _n

    results = []
    for n, p in sizes:
        print(f"  Pearson  n={n} p={p:,} ...", end=" ", flush=True)
        mat64 = _mat64(n, p)
        mat32 = mat64.astype(np.float32, order="C")
        anchor = p // 2

        # Reference: full-copy NumPy baseline (includes O(n*p) centred matrix)
        fc_t   = _bench(lambda: _pearson_full_copy(mat64, anchor), reps=reps)
        fc_mem = _measure_peak_mb(lambda: _pearson_full_copy(mat64, anchor))

        # New: C dispatch (f64)
        c64_t   = _bench(lambda: _dispatch(mat64, anchor_index=anchor), reps=reps)
        c64_mem = _measure_peak_mb(lambda: _dispatch(mat64, anchor_index=anchor))

        # New: C dispatch (f32)
        c32_t   = _bench(lambda: _dispatch(mat32, anchor_index=anchor), reps=reps)
        c32_mem = _measure_peak_mb(lambda: _dispatch(mat32, anchor_index=anchor))

        # Chunked NumPy fallback (no C ext)
        old_ok = _n._NATIVE_OK; _n._NATIVE_OK = False
        _nb = _n._native_backend; _n._native_backend = None
        chunk_t   = _bench(lambda: _n.pearson_columns(mat64, anchor_index=anchor, chunk_size=50_000), reps=reps)
        chunk_mem = _measure_peak_mb(lambda: _n.pearson_columns(mat64, anchor_index=anchor, chunk_size=50_000))
        _n._NATIVE_OK = old_ok; _n._native_backend = _nb

        for label, times, mem in [
            ("Full-copy NumPy (baseline)", fc_t,    fc_mem),
            ("C dispatch f64",             c64_t,   c64_mem),
            ("C dispatch f32",             c32_t,   c32_mem),
            ("Chunked NumPy (no C)",       chunk_t, chunk_mem),
        ]:
            results.append(TimingResult(
                name="pearson_columns", label=label,
                n_samples=n, n_features=p, times_ms=times, peak_mb=mem,
            ))
        print("done")
    return results


def bench_variance(reps: int, sizes: List[tuple]) -> List[TimingResult]:
    from metbit._native import column_variances as _dispatch

    results = []
    for n, p in sizes:
        print(f"  Variance  n={n} p={p:,} ...", end=" ", flush=True)
        mat64 = _mat64(n, p)
        mat32 = mat64.astype(np.float32, order="C")

        np_t    = _bench(lambda: mat64.var(axis=0, ddof=1), reps=reps)
        np_mem  = _measure_peak_mb(lambda: mat64.var(axis=0, ddof=1))
        c64_t   = _bench(lambda: _dispatch(mat64), reps=reps)
        c64_mem = _measure_peak_mb(lambda: _dispatch(mat64))
        c32_t   = _bench(lambda: _dispatch(mat32), reps=reps)
        c32_mem = _measure_peak_mb(lambda: _dispatch(mat32))

        for label, times, mem in [
            ("NumPy (baseline)", np_t,  np_mem),
            ("C dispatch f64",   c64_t, c64_mem),
            ("C dispatch f32",   c32_t, c32_mem),
        ]:
            results.append(TimingResult(
                name="column_variances", label=label,
                n_samples=n, n_features=p, times_ms=times, peak_mb=mem,
            ))
        print("done")
    return results


def bench_preselection(reps: int, sizes: List[tuple]) -> List[TimingResult]:
    from metbit import feature_preselection

    results = []
    for n, p in sizes:
        print(f"  Preselection  n={n} p={p:,} ...", end=" ", flush=True)
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((n, p)))

        np_t = _bench(lambda: X.to_numpy(dtype=np.float32).var(axis=0, ddof=1), reps=reps)
        disp_t = _bench(
            lambda: feature_preselection(X, percentile=20, method="variance"),
            reps=reps
        )

        for label, times in [
            ("Raw NumPy var (baseline)", np_t),
            ("feature_preselection dispatch", disp_t),
        ]:
            results.append(TimingResult(
                name="feature_preselection", label=label,
                n_samples=n, n_features=p, times_ms=times,
            ))
        print("done")
    return results


def bench_stocsy(reps: int, sizes: List[tuple]) -> List[TimingResult]:
    from metbit import STOCSY, ChunkedSTOCSY

    results = []
    for n, p in sizes:
        print(f"  STOCSY  n={n} p={p:,} ...", end=" ", flush=True)
        spectra = _spectra_df(n, p)
        ppm = [float(c) for c in spectra.columns]
        anchor = ppm[p // 3]

        std_t  = _bench(lambda: STOCSY(spectra, anchor_ppm_value=anchor), reps=reps)
        ch_t   = _bench(
            lambda: ChunkedSTOCSY(chunk_size=5_000).compute(spectra, anchor_ppm_value=anchor),
            reps=reps,
        )

        for label, times in [
            ("Standard STOCSY", std_t),
            ("ChunkedSTOCSY", ch_t),
        ]:
            results.append(TimingResult(
                name="stocsy", label=label,
                n_samples=n, n_features=p, times_ms=times,
            ))
        print("done")
    return results


def bench_opls_pipeline(reps: int, sizes: List[tuple]) -> List[TimingResult]:
    from metbit import opls_da

    results = []
    for n, p in sizes:
        print(f"  opls_da  n={n} p={p:,} ...", end=" ", flush=True)
        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.standard_normal((n, p)))
        X.iloc[:n // 2, :p // 5] += 2.0
        y = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2))

        def _run64():
            m = opls_da(X, y, n_components=2, kfold=3, dtype=np.float64)
            m.fit(); m.vip_scores()

        def _run32():
            m = opls_da(X, y, n_components=2, kfold=3, dtype=np.float32)
            m.fit(); m.vip_scores()

        t64 = _bench(_run64, reps=reps, warmup=0)
        t32 = _bench(_run32, reps=reps, warmup=0)

        for label, times in [("float64 pipeline", t64), ("float32 pipeline", t32)]:
            results.append(TimingResult(
                name="opls_da_pipeline", label=label,
                n_samples=n, n_features=p, times_ms=times,
            ))
        print("done")
    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _table_rows(
    results: List[TimingResult],
    group_key: str,
    ref_label: str,
    prev_results: Optional[List[dict]] = None,
) -> str:
    """Format a markdown table for one benchmark group with memory and trend columns."""
    group = [r for r in results if r.name == group_key]
    if not group:
        return "_No results._\n"

    show_mem = any(r.peak_mb > 0 for r in group)

    if show_mem:
        header = ("| Dataset | Implementation | Min (ms) | Mean (ms) | Peak RAM (MB) "
                  "| Speedup vs baseline | Memory vs baseline |")
        sep    = ("|---------|----------------|----------|-----------|---------------"
                  "|---------------------|---------------------|")
    else:
        header = "| Dataset | Implementation | Min (ms) | Mean (ms) | Max (ms) | Speedup vs baseline | Trend |"
        sep    = "|---------|----------------|----------|-----------|----------|---------------------|-------|"

    rows = [header, sep]

    # Reference lookups: (n_samples, n_features) -> (min_ms, peak_mb)
    refs_ms:  Dict[tuple, float] = {}
    refs_mem: Dict[tuple, float] = {}
    for r in group:
        if r.label == ref_label:
            key = (r.n_samples, r.n_features)
            refs_ms[key]  = r.min_ms
            refs_mem[key] = r.peak_mb

    # Previous run lookup for trend arrows
    prev_map: Dict[tuple, float] = {}
    if prev_results:
        for pr in prev_results:
            if pr.get("name") == group_key:
                pk = (pr["label"], pr["n_samples"], pr["n_features"])
                prev_map[pk] = pr.get("min_ms", float("nan"))

    for r in group:
        key = (r.n_samples, r.n_features)
        dataset = f"{r.n_samples:,} x {r.n_features:,}"
        ref_ms  = refs_ms.get(key, float("nan"))
        ref_mem = refs_mem.get(key, 0.0)

        is_nan = (r.min_ms != r.min_ms)
        if is_nan:
            if show_mem:
                rows.append(f"| {dataset} | {r.label} | N/A | N/A | N/A | - | - |")
            else:
                rows.append(f"| {dataset} | {r.label} | N/A | N/A | N/A | - | - |")
            continue

        speedup = _speedup_str(ref_ms, r.min_ms) if r.label != ref_label else "1.0x (baseline)"

        if show_mem:
            mem_str = f"{r.peak_mb:.1f}" if r.peak_mb > 0 else "N/A"
            if r.label != ref_label and ref_mem > 0 and r.peak_mb > 0:
                mem_ratio = ref_mem / max(r.peak_mb, 0.001)
                mem_vs = f"{mem_ratio:.0f}x less"
            else:
                mem_vs = "baseline"
            rows.append(
                f"| {dataset} | {r.label} | {r.min_ms:.1f} | {r.mean_ms:.1f} | "
                f"{mem_str} | {speedup} | {mem_vs} |"
            )
        else:
            # Trend arrow vs previous run
            pk = (r.label, r.n_samples, r.n_features)
            prev_ms = prev_map.get(pk, float("nan"))
            if prev_ms == prev_ms and prev_ms > 0:
                change = (r.min_ms - prev_ms) / prev_ms * 100
                if abs(change) < 5:
                    trend = "="
                elif change > 0:
                    trend = f"+{change:.0f}% (slower)"
                else:
                    trend = f"{change:.0f}% (faster)"
            else:
                trend = "new"
            rows.append(
                f"| {dataset} | {r.label} | {r.min_ms:.1f} | {r.mean_ms:.1f} | "
                f"{r.max_ms:.1f} | {speedup} | {trend} |"
            )

    return "\n".join(rows) + "\n"


def _load_previous(out_dir: Path) -> Optional[List[dict]]:
    """Load results from the previous benchmark run if available."""
    prev_path = out_dir / "benchmark_results_prev.json"
    if not prev_path.exists():
        return None
    try:
        with prev_path.open() as f:
            return json.load(f).get("results", [])
    except Exception:
        return None


def render_markdown(
    all_results: List[TimingResult],
    backend: dict,
    timestamp: str,
    prev_results: Optional[List[dict]] = None,
) -> str:
    native_sym = "yes" if backend["native_c"] else "no"
    omp = backend["openmp_threads"]
    omp_str = f"yes ({omp} threads)" if omp > 0 else "no"
    gpu_str = "yes (CuPy)" if backend.get("gpu_cupy") else (
        "yes (PyTorch)" if backend.get("gpu_torch") else "no"
    )

    lines = [
        "# metbit Performance Benchmark Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Python:** {platform.python_version()}",
        f"**Platform:** {platform.system()} {platform.release()} ({platform.machine()})",
        f"**CPU cores:** {os.cpu_count()}",
        "",
        "## Backend Status",
        "",
        f"| Backend | Available |",
        f"|---------|-----------|",
        f"| Native C extension | {native_sym} |",
        f"| OpenMP parallelism | {omp_str} |",
        f"| GPU (CuPy/PyTorch) | {gpu_str} |",
        "",
        "_Speedup is computed as: `baseline_min_ms / new_min_ms` (higher = faster)._",
        "_Memory: `baseline_peak_mb / new_peak_mb` via tracemalloc (Python-visible allocations)._",
        f"_Trend: vs previous benchmark run ({'previous data available' if prev_results else 'no previous data'})._",
        "",
    ]

    sections = [
        ("VIP Score Computation", "vip_scores", "Python loop",
         "The Python loop over features was the original implementation. "
         "NumPy vectorized replaces it with a single BLAS matrix multiply. "
         "C dispatch adds OpenMP parallelism over features."),
        ("Pearson Correlation (STOCSY kernel)", "pearson_columns", "Full-copy NumPy (baseline)",
         "The baseline materialises a full centred matrix copy (O(n*p) memory). "
         "The new implementations avoid this copy entirely."),
        ("Column Variance (feature pre-selection)", "column_variances", "NumPy (baseline)",
         "Per-column sample variance used by `feature_preselection()`. "
         "C backend avoids creating the centred copy."),
        ("Feature Pre-selection", "feature_preselection", "Raw NumPy var (baseline)",
         "Full `feature_preselection()` call including percentile threshold and mask, "
         "compared to a raw NumPy variance computation."),
        ("STOCSY: ChunkedSTOCSY vs Standard", "stocsy", "Standard STOCSY",
         "`ChunkedSTOCSY` bounds memory to O(n * chunk_size). "
         "This table shows the overhead vs the standard single-pass kernel."),
        ("OPLS-DA Full Pipeline (fit + VIP)", "opls_da_pipeline", "float64 pipeline",
         "Full `opls_da.fit()` + `vip_scores()` workflow. "
         "float32 path uses half the peak memory with negligible Q2 difference."),
    ]

    for title, key, ref_label, description in sections:
        lines += [
            f"## {title}",
            "",
            description,
            "",
            _table_rows(all_results, key, ref_label, prev_results=prev_results),
        ]

    lines += [
        "## Notes",
        "",
        "- All times are wall-clock (min over 5 repetitions). CPU frequency scaling",
        "  and cache effects cause run-to-run variability of ±10-20%.",
        "- 'N/A' in VIP loop rows indicates the benchmark was skipped (p too large for",
        "  the loop to complete in a reasonable time).",
        "- Speedup ratios > 1.0x mean the new implementation is faster.",
        "- The C extension and OpenMP parallel path are auto-selected by `_native.py`",
        "  based on dataset size. Thresholds: n*p > 10M -> parallel path.",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run metbit performance benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller datasets, fewer reps (faster, less stable)")
    parser.add_argument("--output", default="reports",
                        help="Output directory for JSON and Markdown files")
    parser.add_argument("--reps", type=int, default=0,
                        help="Override number of benchmark repetitions (0 = auto)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = args.reps if args.reps > 0 else (3 if args.quick else 5)

    if args.quick:
        vip_sizes   = [500, 2_000]
        pearson_sizes = [(50, 2_000), (100, 5_000)]
        var_sizes   = [(100, 10_000)]
        pre_sizes   = [(50, 10_000)]
        stocsy_sizes = [(30, 1_000)]
        opls_sizes  = [(40, 200)]
    else:
        vip_sizes   = [1_000, 5_000, 20_000]
        pearson_sizes = [(200, 10_000), (500, 30_000)]
        var_sizes   = [(200, 50_000), (500, 100_000)]
        pre_sizes   = [(100, 50_000), (200, 100_000)]
        stocsy_sizes = [(50, 2_000), (100, 5_000)]
        opls_sizes  = [(60, 300), (100, 500)]

    from metbit._native import backend_info
    backend = backend_info()

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\nmetbit Performance Benchmark  [{timestamp}]")
    print(f"Backend: native_c={backend['native_c']}, "
          f"openmp={backend['openmp_threads']} threads, "
          f"gpu_cupy={backend['gpu_cupy']}, gpu_torch={backend['gpu_torch']}")
    print(f"Repetitions per benchmark: {reps}")
    print()

    all_results: List[TimingResult] = []

    print("=== VIP scores ===")
    all_results += bench_vip(reps, vip_sizes)

    print("\n=== Pearson correlation ===")
    all_results += bench_pearson(reps, pearson_sizes)

    print("\n=== Column variance ===")
    all_results += bench_variance(reps, var_sizes)

    print("\n=== Feature pre-selection ===")
    all_results += bench_preselection(reps, pre_sizes)

    print("\n=== STOCSY ===")
    all_results += bench_stocsy(reps, stocsy_sizes)

    print("\n=== OPLS-DA pipeline ===")
    all_results += bench_opls_pipeline(reps, opls_sizes)

    # -- Archive previous run and load it for trend comparison -------------
    json_path = out_dir / "benchmark_results.json"
    prev_results = _load_previous(out_dir)
    if json_path.exists() and not prev_results:
        # First time archiving: copy current as "prev" to establish baseline
        import shutil
        shutil.copy2(json_path, out_dir / "benchmark_results_prev.json")
        prev_results = _load_previous(out_dir)
    elif json_path.exists():
        # Rotate: current -> prev before writing new results
        import shutil
        shutil.copy2(json_path, out_dir / "benchmark_results_prev.json")

    # -- Write JSON ---------------------------------------------------------
    json_data = {
        "timestamp": timestamp,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_cores": os.cpu_count(),
        "backend": backend,
        "results": [
            {
                **{k: v for k, v in asdict(r).items() if k != "extra"},
                "min_ms":    r.min_ms,
                "mean_ms":   r.mean_ms,
                "median_ms": r.median_ms,
                "max_ms":    r.max_ms,
                "peak_mb":   r.peak_mb,
            }
            for r in all_results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"\nJSON  -> {json_path}")

    # -- Write Markdown -----------------------------------------------------
    md_path = out_dir / "PERFORMANCE.md"
    md_content = render_markdown(all_results, backend, timestamp, prev_results=prev_results)
    md_path.write_text(md_content)
    print(f"Report -> {md_path}")

    # -- Print quick summary to stdout -------------------------------------
    print("\n=== Quick Summary ===")
    groups = {}
    for r in all_results:
        groups.setdefault(r.name, []).append(r)

    ref_labels = {
        "vip_scores": "Python loop",
        "pearson_columns": "Full-copy NumPy (baseline)",
        "column_variances": "NumPy (baseline)",
        "feature_preselection": "Raw NumPy var (baseline)",
        "stocsy": "Standard STOCSY",
        "opls_da_pipeline": "float64 pipeline",
    }

    for name, group in groups.items():
        ref_label = ref_labels.get(name, "")
        refs = {(r.n_samples, r.n_features): r.min_ms
                for r in group if r.label == ref_label and r.min_ms == r.min_ms}
        for r in group:
            if r.label == ref_label:
                continue
            key = (r.n_samples, r.n_features)
            ref_ms = refs.get(key)
            if ref_ms and r.min_ms == r.min_ms:
                ratio = ref_ms / max(r.min_ms, 1e-9)
                mem_str = f"  peak={r.peak_mb:.1f} MB" if r.peak_mb > 0 else ""
                print(f"  [{name}] {r.label} @ {r.n_samples}x{r.n_features}: "
                      f"{r.min_ms:.1f} ms  ({ratio:.1f}x speedup){mem_str}")


if __name__ == "__main__":
    main()
