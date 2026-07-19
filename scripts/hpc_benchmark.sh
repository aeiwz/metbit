#!/usr/bin/env bash
# hpc_benchmark.sh - Reproducible metbit benchmark for Linux HPC nodes.
#
# Exercises the OpenMP (and, if present, GPU) compute paths that the laptop
# benchmark in the manuscript could not, by rebuilding the C extension from
# source and running scripts/perf_report.py across a sweep of OpenMP thread
# counts. Produces a self-contained tarball to bring back for the paper.
#
# Usage (run from the repository root, ideally on the COMPUTE node):
#     bash scripts/hpc_benchmark.sh
#
# Optional environment overrides:
#     PYTHON=python3.11   interpreter to use            (default: python3)
#     THREADS="1 2 4 8"   OpenMP thread counts to sweep (default: 1 2 4 8 16, capped at nproc)
#     REPS=7              repetitions per benchmark     (default: perf_report default = 5)
#     QUICK=1             use small datasets (smoke test only)
#     PORTABLE=1          drop -march=native (build on login node / heterogeneous cluster)
#     REBUILD=0           skip the source rebuild (use an already-built extension)
#
# ponytail: thin orchestrator around the existing perf_report.py - no benchmark
# logic is duplicated here; this only builds, sweeps threads, and packages.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
REPS="${REPS:-0}"                 # 0 => let perf_report pick its default
QUICK="${QUICK:-0}"
PORTABLE="${PORTABLE:-0}"
REBUILD="${REBUILD:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
# Default thread sweep, capped at available cores; always include 1 and nproc.
if [ -z "${THREADS:-}" ]; then
    THREADS=""
    for t in 1 2 4 8 16 32; do
        [ "$t" -le "$NPROC" ] && THREADS="$THREADS $t"
    done
    # ensure the full core count is represented
    case " $THREADS " in *" $NPROC "*) : ;; *) THREADS="$THREADS $NPROC" ;; esac
fi

OUT_ROOT="reports/hpc"
HOST="$(hostname -s 2>/dev/null || echo node)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE="metbit_hpc_benchmark_${HOST}_${STAMP}.tar.gz"

mkdir -p "$OUT_ROOT"
ENV_FILE="$OUT_ROOT/environment.txt"

echo "=============================================================="
echo " metbit HPC benchmark"
echo " host=$HOST  nproc=$NPROC  python=$PYTHON"
echo " thread sweep:$THREADS"
echo "=============================================================="

# --------------------------------------------------------------------------
# 1. Capture the environment (goes into the bundle for the methods section)
# --------------------------------------------------------------------------
{
    echo "# metbit HPC benchmark environment"
    echo "timestamp_utc: $STAMP"
    echo "hostname: $(hostname -f 2>/dev/null || hostname)"
    echo "nproc: $NPROC"
    echo
    echo "## uname"; uname -a
    echo
    echo "## CPU"; (lscpu 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | head -30) || true
    echo
    echo "## Compiler"; (${CC:-cc} --version 2>/dev/null || gcc --version 2>/dev/null | head -1) || true
    echo
    echo "## Python"; "$PYTHON" --version 2>&1; "$PYTHON" -c "import sys;print(sys.executable)"
    echo
    echo "## GPU (nvidia-smi)"; (nvidia-smi 2>/dev/null || echo "no nvidia-smi / no GPU") || true
} > "$ENV_FILE"
echo "[env]   wrote $ENV_FILE"

# --------------------------------------------------------------------------
# 2. Rebuild the native C extension from source (enables OpenMP on Linux)
# --------------------------------------------------------------------------
if [ "$REBUILD" = "1" ]; then
    echo "[build] removing stale build artifacts and non-Linux extensions"
    rm -rf build/ 2>/dev/null || true
    # remove any prebuilt extension modules (e.g. macOS .so shipped in the repo)
    find metbit -maxdepth 1 -name '_native_backend*.so' -delete 2>/dev/null || true
    BUILD_ENV=""
    [ "$PORTABLE" = "1" ] && BUILD_ENV="METBIT_PORTABLE_BUILD=1"
    echo "[build] $BUILD_ENV $PYTHON -m pip install -e . (compiling OpenMP extension)"
    env $BUILD_ENV "$PYTHON" -m pip install -e . --no-build-isolation --force-reinstall \
        || env $BUILD_ENV "$PYTHON" -m pip install -e .
fi

# --------------------------------------------------------------------------
# 3. Verify the backend actually has the native + OpenMP path active
# --------------------------------------------------------------------------
echo "[check] backend_info:"
"$PYTHON" - <<'PY'
from metbit._native import backend_info
info = backend_info()
for k, v in info.items():
    print(f"        {k} = {v}")
if not info["native_c"]:
    raise SystemExit("ERROR: native C extension NOT active - benchmark would only "
                     "measure the NumPy fallback. Check the build output above.")
if info["openmp_threads"] < 1:
    print("        WARNING: OpenMP threads report 0 - the extension was built "
          "single-threaded. The thread sweep will show no scaling.")
PY

# --------------------------------------------------------------------------
# 4. Run perf_report.py across the OpenMP thread sweep
# --------------------------------------------------------------------------
PERF_ARGS=""
[ "$QUICK" = "1" ] && PERF_ARGS="$PERF_ARGS --quick"
[ "$REPS" != "0" ] && PERF_ARGS="$PERF_ARGS --reps $REPS"

for T in $THREADS; do
    OUT="$OUT_ROOT/threads_${T}"
    echo "--------------------------------------------------------------"
    echo "[run]   OMP_NUM_THREADS=$T -> $OUT"
    echo "--------------------------------------------------------------"
    OMP_NUM_THREADS="$T" METBIT_N_JOBS="$T" \
        "$PYTHON" scripts/perf_report.py --output "$OUT" $PERF_ARGS
done

# --------------------------------------------------------------------------
# 5. Optional GPU run (auto-dispatch picks CuPy/PyTorch CUDA if present)
# --------------------------------------------------------------------------
GPU_PRESENT="$("$PYTHON" -c 'from metbit._native import gpu_available; print(1 if gpu_available() else 0)' 2>/dev/null || echo 0)"
if [ "$GPU_PRESENT" = "1" ]; then
    echo "[gpu]   CUDA backend detected - running explicit CPU-vs-GPU comparison"
    "$PYTHON" scripts/gpu_benchmark.py --output "$OUT_ROOT/gpu" $PERF_ARGS
else
    echo "[gpu]   no CuPy/PyTorch CUDA backend active - skipping GPU pass."
    echo "[gpu]   To enable: pip install cupy-cuda12x  (match your CUDA toolkit),"
    echo "[gpu]   or a CUDA-enabled torch build, then re-run."
fi

# --------------------------------------------------------------------------
# 6. Aggregate the thread sweep into one OpenMP-scaling CSV
# --------------------------------------------------------------------------
"$PYTHON" - "$OUT_ROOT" <<'PY'
import json, sys, glob, os, csv
root = sys.argv[1]
rows = []
for jf in sorted(glob.glob(os.path.join(root, "threads_*", "benchmark_results.json"))):
    threads = int(os.path.basename(os.path.dirname(jf)).split("_")[1])
    data = json.load(open(jf))
    omp = data.get("backend", {}).get("openmp_threads", "")
    for r in data["results"]:
        rows.append({
            "kernel": r["name"], "label": r["label"],
            "n_samples": r["n_samples"], "n_features": r["n_features"],
            "omp_num_threads": threads, "openmp_threads_reported": omp,
            "min_ms": round(r["min_ms"], 4), "peak_mb": round(r.get("peak_mb", 0.0), 3),
        })
out = os.path.join(root, "openmp_scaling.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"[agg]   wrote {out} ({len(rows)} rows)")

# quick scaling preview: C-dispatch min time per thread count for the largest sizes
print("[agg]   OpenMP scaling preview (C-dispatch, largest dataset per kernel):")
by = {}
for r in rows:
    if "C dispatch" not in r["label"] and "dispatch" not in r["label"]:
        continue
    key = (r["kernel"], r["n_samples"], r["n_features"])
    by.setdefault(key, {})[r["omp_num_threads"]] = r["min_ms"]
for kernel in sorted({k[0] for k in by}):
    ks = [k for k in by if k[0] == kernel]
    if not ks:
        continue
    k = max(ks, key=lambda x: x[1] * x[2])  # largest n*p
    series = by[k]
    t1 = series.get(1) or next(iter(series.values()))
    parts = []
    for th in sorted(series):
        sp = (t1 / series[th]) if series[th] else float("nan")
        parts.append(f"{th}t={series[th]:.2f}ms({sp:.2f}x)")
    print(f"        {kernel} {k[1]}x{k[2]}: " + "  ".join(parts))
PY

# --------------------------------------------------------------------------
# 7. Bundle everything to bring back
# --------------------------------------------------------------------------
tar czf "$BUNDLE" "$OUT_ROOT"
echo "=============================================================="
echo " DONE. Bundle -> $BUNDLE"
echo " Contains: environment.txt, threads_*/ (JSON + PERFORMANCE.md),"
echo "           openmp_scaling.csv, and gpu/ if a GPU was present."
echo " Send that tarball back and I will fold the OpenMP/GPU results"
echo " into Table 1 and the Discussion."
echo "=============================================================="
