#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_benchmark.py - CPU (C/OpenMP) vs GPU (CuPy/PyTorch CUDA) comparison for metbit.

The general perf_report.py benchmark silently routes only the very largest
matrices to the GPU and labels them "C dispatch". This script instead compares
the CPU and GPU paths of each GPU-enabled kernel *explicitly*, at sizes large
enough for the GPU transfer overhead to pay off, and verifies that the two
paths agree numerically.

Kernels covered (these are the ones with a real GPU path in metbit._native):
    - pearson_columns    (STOCSY correlation)
    - column_variances   (feature pre-selection)
    - vip_scores         (VIP ranking)

Usage (on a CUDA node, from the repo root):
    python scripts/gpu_benchmark.py                 # default sizes
    python scripts/gpu_benchmark.py --quick         # smaller/faster
    python scripts/gpu_benchmark.py --output reports/gpu

Requires one of:
    pip install cupy-cuda12x     # match your CUDA toolkit (…-cuda11x, etc.)
    pip install torch            # a CUDA-enabled build

Without a GPU it still runs and reports the CPU path only, so it is safe to
smoke-test anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import metbit._native as N


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

def _bench(fn, reps=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _min(times):
    return min(times) if times else float("nan")


def _force_cpu(fn):
    """Run fn with GPU backends disabled so the CPU (C/OpenMP) path is taken."""
    cu, to = N._cupy, N._torch
    N._cupy, N._torch = None, None
    try:
        return fn()
    finally:
        N._cupy, N._torch = cu, to


# ---------------------------------------------------------------------------
# GPU device description
# ---------------------------------------------------------------------------

def _gpu_info():
    info = {"available": N.gpu_available(), "cupy": N._cupy is not None,
            "torch": N._torch is not None, "device": None}
    try:
        if N._cupy is not None:
            props = N._cupy.cuda.runtime.getDeviceProperties(0)
            info["device"] = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        elif N._torch is not None:
            info["device"] = N._torch.cuda.get_device_name(0)
    except Exception:
        pass
    if info["device"] is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            ).decode().strip().splitlines()
            if out:
                info["device"] = out[0]
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# per-kernel benchmarks: each returns list of result dicts
# ---------------------------------------------------------------------------

def _rng_mat(n, p, seed=0, dtype=np.float64):
    return np.random.default_rng(seed).standard_normal((n, p)).astype(dtype, order="C")


def bench_pearson(sizes, reps):
    rows = []
    for n, p in sizes:
        mat = _rng_mat(n, p, dtype=np.float32)
        anchor = p // 2
        cpu_t = _bench(lambda: _force_cpu(lambda: N.pearson_columns(mat, anchor_index=anchor)), reps=reps)
        row = {"kernel": "pearson_columns", "n": n, "p": p,
               "elements": n * p, "cpu_min_ms": _min(cpu_t)}
        if N.gpu_available():
            gpu_out = N._pearson_gpu(mat, anchor)
            gpu_t = _bench(lambda: N._pearson_gpu(mat, anchor), reps=reps)
            cpu_out = _force_cpu(lambda: N.pearson_columns(mat, anchor_index=anchor))
            row["gpu_min_ms"] = _min(gpu_t)
            row["speedup"] = row["cpu_min_ms"] / row["gpu_min_ms"] if row["gpu_min_ms"] else float("nan")
            row["max_abs_diff"] = float(np.nanmax(np.abs(np.asarray(gpu_out) - np.asarray(cpu_out))))
        rows.append(row)
        _log(row)
    return rows


def bench_variance(sizes, reps):
    rows = []
    for n, p in sizes:
        mat = _rng_mat(n, p, dtype=np.float32)
        cpu_t = _bench(lambda: _force_cpu(lambda: N.column_variances(mat)), reps=reps)
        row = {"kernel": "column_variances", "n": n, "p": p,
               "elements": n * p, "cpu_min_ms": _min(cpu_t)}
        if N.gpu_available():
            gpu_out = N._column_variances_gpu(mat)
            gpu_t = _bench(lambda: N._column_variances_gpu(mat), reps=reps)
            cpu_out = _force_cpu(lambda: N.column_variances(mat))
            row["gpu_min_ms"] = _min(gpu_t)
            row["speedup"] = row["cpu_min_ms"] / row["gpu_min_ms"] if row["gpu_min_ms"] else float("nan")
            row["max_abs_diff"] = float(np.nanmax(np.abs(np.asarray(gpu_out) - np.asarray(cpu_out))))
        rows.append(row)
        _log(row)
    return rows


def bench_vip(sizes, reps):
    from sklearn.cross_decomposition import PLSRegression
    rows = []
    for n, p in sizes:
        h = 3
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, p)); y = rng.integers(0, 2, n).astype(float)
        pls = PLSRegression(n_components=min(h, min(n, p) - 1)).fit(X, y)
        t = pls.x_scores_.astype(np.float64)
        w = pls.x_weights_.astype(np.float64)
        q = pls.y_loadings_.astype(np.float64)
        cpu_t = _bench(lambda: _force_cpu(lambda: N.vip_scores(t, w, q)), reps=reps)
        row = {"kernel": "vip_scores", "n": n, "p": p,
               "elements": n * p, "cpu_min_ms": _min(cpu_t)}
        if N.gpu_available():
            gpu_out = N._vip_scores_gpu(t, w, q)
            gpu_t = _bench(lambda: N._vip_scores_gpu(t, w, q), reps=reps)
            cpu_out = _force_cpu(lambda: N.vip_scores(t, w, q))
            row["gpu_min_ms"] = _min(gpu_t)
            row["speedup"] = row["cpu_min_ms"] / row["gpu_min_ms"] if row["gpu_min_ms"] else float("nan")
            row["max_abs_diff"] = float(np.nanmax(np.abs(np.asarray(gpu_out) - np.asarray(cpu_out))))
        rows.append(row)
        _log(row)
    return rows


def _log(row):
    base = f"  {row['kernel']:<18} {row['n']}x{row['p']:<8} ({row['elements']/1e6:.0f}M)  cpu={row['cpu_min_ms']:.2f}ms"
    if "gpu_min_ms" in row:
        base += f"  gpu={row['gpu_min_ms']:.2f}ms  {row['speedup']:.2f}x  maxdiff={row['max_abs_diff']:.2e}"
    else:
        base += "  (no GPU)"
    print(base, flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="metbit CPU vs GPU (CUDA) benchmark")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output", default="reports/gpu")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        pearson_sizes = [(200, 20_000), (500, 50_000)]
        var_sizes     = [(200, 50_000), (500, 100_000)]
        vip_sizes     = [(80, 50_000), (80, 200_000)]
    else:
        # Large enough for GPU transfer overhead to pay off; skip-on-OOM below.
        pearson_sizes = [(500, 50_000), (1000, 200_000), (2000, 500_000)]
        var_sizes     = [(500, 100_000), (1000, 500_000), (2000, 1_000_000)]
        vip_sizes     = [(80, 100_000), (80, 500_000), (80, 1_000_000)]

    ginfo = _gpu_info()
    binfo = N.backend_info()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\nmetbit GPU benchmark  [{ts}]")
    print(f"Backend: native_c={binfo['native_c']}, openmp_threads={binfo['openmp_threads']}, "
          f"gpu_cupy={binfo['gpu_cupy']}, gpu_torch={binfo['gpu_torch']}")
    print(f"GPU device: {ginfo['device'] or 'none detected'}")
    if not N.gpu_available():
        print("WARNING: no CUDA backend active. Install cupy-cuda12x or a CUDA-enabled "
              "torch, and ensure a GPU is visible. Reporting CPU path only.\n")
    print()

    results = []
    def guarded(fn, sizes):
        collected = []
        for sz in sizes:
            try:
                collected += fn([sz], args.reps)
            except Exception as e:  # e.g. CUDA out of memory at the largest size
                print(f"  skipped {sz}: {type(e).__name__}: {e}")
        return collected

    print("=== Pearson (STOCSY) ===")
    results += guarded(bench_pearson, pearson_sizes)
    print("=== Column variance ===")
    results += guarded(bench_variance, var_sizes)
    print("=== VIP scores ===")
    results += guarded(bench_vip, vip_sizes)

    data = {
        "timestamp": ts,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_cores": os.cpu_count(),
        "backend": binfo,
        "gpu": ginfo,
        "results": results,
    }
    (out / "gpu_benchmark_results.json").write_text(json.dumps(data, indent=2))

    # markdown table
    lines = [
        "# metbit CPU vs GPU (CUDA) Benchmark", "",
        f"**Generated:** {ts}",
        f"**GPU:** {ginfo['device'] or 'none'}  "
        f"(cupy={ginfo['cupy']}, torch={ginfo['torch']})",
        f"**CPU:** {platform.machine()}, {os.cpu_count()} cores, "
        f"OpenMP={binfo['openmp_threads']} threads", "",
        "GPU timings include host<->device transfer and synchronisation "
        "(end-to-end wall clock). `max_abs_diff` is CPU vs GPU numerical agreement.", "",
        "| Kernel | Size (n x p) | Elements | CPU min (ms) | GPU min (ms) | GPU speedup | max_abs_diff |",
        "|--------|--------------|----------|--------------|--------------|-------------|--------------|",
    ]
    for r in results:
        g = f"{r['gpu_min_ms']:.2f}" if "gpu_min_ms" in r else "N/A"
        sp = f"{r['speedup']:.2f}x" if "speedup" in r else "N/A"
        md = f"{r['max_abs_diff']:.2e}" if "max_abs_diff" in r else "N/A"
        lines.append(f"| {r['kernel']} | {r['n']} x {r['p']:,} | {r['elements']/1e6:.0f}M "
                     f"| {r['cpu_min_ms']:.2f} | {g} | {sp} | {md} |")
    lines += ["", "_Speedup > 1 means the GPU path is faster end-to-end. Values < 1 are "
              "expected at small sizes where transfer overhead dominates._"]
    (out / "GPU_PERFORMANCE.md").write_text("\n".join(lines) + "\n")

    print(f"\nJSON   -> {out/'gpu_benchmark_results.json'}")
    print(f"Report -> {out/'GPU_PERFORMANCE.md'}")


if __name__ == "__main__":
    main()
