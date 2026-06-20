# metbit Performance Benchmark Report

**Generated:** 2026-06-20T03:22:23Z
**Python:** 3.14.5
**Platform:** Darwin 25.5.0 (arm64)
**CPU cores:** 15

## Backend Status

| Backend | Available |
|---------|-----------|
| Native C extension | yes |
| OpenMP parallelism | no |
| GPU (CuPy/PyTorch) | no |

_Speedup is computed as: `baseline_min_ms / new_min_ms` (higher = faster)._
_Memory: `baseline_peak_mb / new_peak_mb` via tracemalloc (Python-visible allocations)._
_Trend: vs previous benchmark run (previous data available)._

## VIP Score Computation

The Python loop over features was the original implementation. NumPy vectorized replaces it with a single BLAS matrix multiply. C dispatch adds OpenMP parallelism over features.

| Dataset | Implementation | Min (ms) | Mean (ms) | Max (ms) | Speedup vs baseline | Trend |
|---------|----------------|----------|-----------|----------|---------------------|-------|
| 80 x 1,000 | Python loop | 6.5 | 6.6 | 6.8 | 1.0x (baseline) | new |
| 80 x 1,000 | NumPy vectorized | 0.0 | 0.0 | 0.0 | 309.5x | new |
| 80 x 1,000 | C dispatch | 0.0 | 0.0 | 0.0 | 1352.6x | new |
| 80 x 5,000 | Python loop | 54.3 | 54.4 | 54.8 | 1.0x (baseline) | new |
| 80 x 5,000 | NumPy vectorized | 0.1 | 0.1 | 0.1 | 801.3x | new |
| 80 x 5,000 | C dispatch | 0.0 | 0.0 | 0.0 | 2701.5x | new |
| 80 x 20,000 | Python loop | N/A | N/A | N/A | - | - |
| 80 x 20,000 | NumPy vectorized | 0.2 | 0.3 | 0.3 | N/A | new |
| 80 x 20,000 | C dispatch | 0.1 | 0.1 | 0.1 | N/A | new |

## Pearson Correlation (STOCSY kernel)

The baseline materialises a full centred matrix copy (O(n*p) memory). The new implementations avoid this copy entirely.

| Dataset | Implementation | Min (ms) | Mean (ms) | Peak RAM (MB) | Speedup vs baseline | Memory vs baseline |
|---------|----------------|----------|-----------|---------------|---------------------|---------------------|
| 200 x 10,000 | Full-copy NumPy (baseline) | 1.1 | 1.1 | 16.3 | 1.0x (baseline) | baseline |
| 200 x 10,000 | C dispatch f64 | 0.9 | 0.9 | 0.3 | 1.2x | 51x less |
| 200 x 10,000 | C dispatch f32 | 0.7 | 0.8 | 0.3 | 1.5x | 51x less |
| 200 x 10,000 | Chunked NumPy (no C) | 1.1 | 1.2 | 16.5 | 1.0x | 1x less |
| 500 x 30,000 | Full-copy NumPy (baseline) | 8.3 | 8.7 | 121.0 | 1.0x (baseline) | baseline |
| 500 x 30,000 | C dispatch f64 | 6.7 | 6.7 | 1.0 | 1.2x | 126x less |
| 500 x 30,000 | C dispatch f32 | 6.0 | 6.0 | 1.0 | 1.4x | 126x less |
| 500 x 30,000 | Chunked NumPy (no C) | 952.1 | 976.0 | 135.5 | 0.0x | 1x less |

## Column Variance (feature pre-selection)

Per-column sample variance used by `feature_preselection()`. C backend avoids creating the centred copy.

| Dataset | Implementation | Min (ms) | Mean (ms) | Peak RAM (MB) | Speedup vs baseline | Memory vs baseline |
|---------|----------------|----------|-----------|---------------|---------------------|---------------------|
| 200 x 50,000 | NumPy (baseline) | 5.8 | 6.1 | 80.8 | 1.0x (baseline) | baseline |
| 200 x 50,000 | C dispatch f64 | 3.5 | 3.5 | 0.8 | 1.7x | 101x less |
| 200 x 50,000 | C dispatch f32 | 3.0 | 3.1 | 0.8 | 1.9x | 101x less |
| 500 x 100,000 | NumPy (baseline) | 30.3 | 31.8 | 401.6 | 1.0x (baseline) | baseline |
| 500 x 100,000 | C dispatch f64 | 17.6 | 17.7 | 1.6 | 1.7x | 251x less |
| 500 x 100,000 | C dispatch f32 | 15.5 | 15.9 | 1.6 | 2.0x | 251x less |

## Feature Pre-selection

Full `feature_preselection()` call including percentile threshold and mask, compared to a raw NumPy variance computation.

| Dataset | Implementation | Min (ms) | Mean (ms) | Max (ms) | Speedup vs baseline | Trend |
|---------|----------------|----------|-----------|----------|---------------------|-------|
| 100 x 50,000 | Raw NumPy var (baseline) | 3.5 | 3.6 | 3.7 | 1.0x (baseline) | new |
| 100 x 50,000 | feature_preselection dispatch | 7.5 | 7.8 | 8.2 | 0.5x | new |
| 200 x 100,000 | Raw NumPy var (baseline) | 11.3 | 11.5 | 11.6 | 1.0x (baseline) | new |
| 200 x 100,000 | feature_preselection dispatch | 30.2 | 30.8 | 32.0 | 0.4x | new |

## STOCSY: ChunkedSTOCSY vs Standard

`ChunkedSTOCSY` bounds memory to O(n * chunk_size). This table shows the overhead vs the standard single-pass kernel.

| Dataset | Implementation | Min (ms) | Mean (ms) | Max (ms) | Speedup vs baseline | Trend |
|---------|----------------|----------|-----------|----------|---------------------|-------|
| 50 x 2,000 | Standard STOCSY | 6.1 | 6.7 | 7.6 | 1.0x (baseline) | new |
| 50 x 2,000 | ChunkedSTOCSY | 0.2 | 0.2 | 0.2 | 26.6x | new |
| 100 x 5,000 | Standard STOCSY | 26.8 | 28.2 | 29.2 | 1.0x (baseline) | new |
| 100 x 5,000 | ChunkedSTOCSY | 0.9 | 0.9 | 0.9 | 31.1x | new |

## OPLS-DA Full Pipeline (fit + VIP)

Full `opls_da.fit()` + `vip_scores()` workflow. float32 path uses half the peak memory with negligible Q2 difference.

| Dataset | Implementation | Min (ms) | Mean (ms) | Max (ms) | Speedup vs baseline | Trend |
|---------|----------------|----------|-----------|----------|---------------------|-------|
| 60 x 300 | float64 pipeline | 27.5 | 29.3 | 35.8 | 1.0x (baseline) | new |
| 60 x 300 | float32 pipeline | 37.6 | 37.9 | 38.3 | 0.7x | new |
| 100 x 500 | float64 pipeline | 377.4 | 383.0 | 400.8 | 1.0x (baseline) | new |
| 100 x 500 | float32 pipeline | 196.9 | 198.8 | 205.0 | 1.9x | new |

## Notes

- All times are wall-clock (min over 5 repetitions). CPU frequency scaling
  and cache effects cause run-to-run variability of ±10-20%.
- 'N/A' in VIP loop rows indicates the benchmark was skipped (p too large for
  the loop to complete in a reasonable time).
- Speedup ratios > 1.0x mean the new implementation is faster.
- The C extension and OpenMP parallel path are auto-selected by `_native.py`
  based on dataset size. Thresholds: n*p > 10M -> parallel path.
