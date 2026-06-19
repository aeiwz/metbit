# Changelog

All notable changes to metbit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [9.0.0] - 2026-06-19

### Summary

v9.0.0 is a major performance and correctness release focused on large-scale
metabolomics cohorts (>10,000 samples, >100,000 features). The compute backend
has been rewritten with a four-tier dispatch hierarchy: GPU (CuPy/PyTorch) ->
C+OpenMP -> multiprocessing -> chunked NumPy. Memory peak allocations for core
kernels are reduced by 50-250x at large cohort sizes.

### Added

**Large-scale compute backend (`metbit/_native.py`, `metbit/_native_backend.c`)**
- `pearson_columns_par`: OpenMP row-parallel Pearson correlation; thread-local
  accumulators avoid data races. Falls back transparently when OpenMP is absent.
- `pearson_columns_f32`: float32 input variant; reads half the memory bandwidth
  of the float64 path while accumulating in float64 for numerical stability.
- `column_variances` / `column_variances_f32`: two-pass numerically stable
  per-column sample variance in C with optional OpenMP parallelism.
- `vip_scores`: closed-form VIP kernel in C; OpenMP parallel over features.
  Replaces the O(p) Python loop over features entirely.
- `openmp_threads()`: reports available OpenMP threads to Python dispatch layer.

**Auto-dispatch layer (`metbit/_native.py`)**
- Four-tier hierarchy: GPU (CuPy/PyTorch) -> C+OpenMP -> multiprocessing+NumPy
  -> chunked NumPy. Backend selected automatically based on `n*p` element count.
- `pearson_columns()`: unified entry point; auto-selects best backend.
- `column_variances()`: unified entry point with same dispatch logic.
- `vip_scores()`: dispatches to GPU / C / NumPy vectorised as available.
- `backend_info()`: returns dict of active backends for user inspection.
- Environment overrides: `METBIT_DISABLE_NATIVE`, `METBIT_DISABLE_GPU`,
  `METBIT_N_JOBS`, `METBIT_CHUNK`.

**Memory-efficient large-scale module (`metbit/analysis/large_scale.py`)**
- `MemoryEstimator`: estimates peak RAM before loading large datasets and
  recommends dtype.
- `feature_preselection()`: variance/IQR based feature reduction; data-driven
  percentile threshold; dispatches to the C/GPU variance kernel.
- `ChunkedSTOCSY`: STOCSY with bounded O(n * chunk_size) peak memory.
  `active_backend()` classmethod reports the compute path.
- `LargeScaleAlignment`: alignment wrapper that warns when peak memory > 8 GB.
- `memory_report()`: convenience function for quick dataset size assessment.

**OPLS-DA improvements (`metbit/analysis/opls_da.py`)**
- `dtype` parameter: accepts `numpy.float32` to halve peak memory. Auto-selects
  float32 when `n * p > 5,000,000` if `dtype=None` (default).
- `__init__` now replaces NaN in-place (single allocation) rather than creating
  a second full-matrix copy.
- `vip_scores()` is now vectorised via the dispatch layer; the O(p) Python loop
  is eliminated (1000-2700x speedup at p=5,000-20,000).

**NMR alignment (`metbit/nmr/alignment.py`)**
- `icoshift_align()` now allocates a single output array instead of two full
  matrix copies (`spectra.copy()` then `.values.copy()`). Saves one full matrix
  copy (~80 GB at 10,000 x 1,000,000 float64).

**Test suite**
- `tests/test_e2e_pipeline.py`: 44 full-pipeline tests (OPLS-DA, STOCSY,
  ChunkedSTOCSY, alignment, feature preselection, MemoryEstimator).
- `tests/test_ab_aa.py`: 22 statistical validity tests: AB (must discriminate),
  AA (must not discriminate on noise), scaling robustness, reproducibility.
- `tests/test_large_scale.py`: 23 backend dispatch tests including memory
  efficiency assertions via `tracemalloc`.
- `tests/test_performance.py`: 24 performance regression tests marked
  `@pytest.mark.perf`; assert speedup ratios, not absolute times.

**Build and tooling**
- `setup.py` now detects OpenMP at install time and compiles with `-fopenmp`
  when available. Falls back gracefully via `OptionalBuildExt`.
- Compile flags: `-O3 -march=native -ffast-math` for the C extension.
- `scripts/perf_report.py`: standalone benchmark runner producing
  `reports/benchmark_results.json` and `reports/PERFORMANCE.md` with timing,
  peak-memory, and trend-vs-previous-run columns.
- `scripts/run_tests.sh`: `--perf` / `--perf-quick` flags run benchmarks and
  update the performance report.

### Changed

- **VIP computation** in `opls_da.vip_scores()`: replaced Python loop with
  single BLAS matrix multiply `sqrt(p * (w_norm^2 @ S) / total_s)`. Numerically
  identical to the loop; verified at `atol=1e-16`.
- **Pearson fallback** in `_native.py`: chunked over features (bounded memory)
  instead of full-matrix centred copy.
- Minimum required: `scipy==1.14.1` (pinned; relaxation planned for v9.1).
- `pytest.ini`: added `perf` marker; `--strict-markers` enforces registration.

### Fixed

- `icoshift_align`: second redundant full-matrix copy removed.
- `opls_da.__init__`: `np.nan_to_num(X.to_numpy(), ...)` created an O(n*p)
  intermediate; now uses `to_numpy(dtype=...) + nan_to_num(copy=False)`.
- `test_native_backend.py::test_pearson_columns_fallback_matches_native`:
  monkeypatching `_native_backend` now also patches `_NATIVE_OK` flag.

### Performance (measured on Apple M-series, single-threaded C, no GPU)

| Kernel | Dataset | Speedup | Memory reduction |
|---|---|---|---|
| VIP scores (C vs Python loop) | 80 x 5,000 | 2680x | - |
| VIP scores (NumPy vec. vs loop) | 80 x 5,000 | 817x | - |
| Pearson (C vs full-copy NumPy) | 500 x 30,000 | 1.2x speed | 126x less RAM |
| Column variance (C f32 vs NumPy) | 500 x 100,000 | 2.0x | 251x less RAM |
| ChunkedSTOCSY overhead | 100 x 5,000 | 46x faster than std | bounded memory |

---

## [8.7.7] - prior release

See git history for earlier versions.
