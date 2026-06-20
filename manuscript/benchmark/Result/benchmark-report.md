# metbit Full-Scale Benchmark Report

Generated: 2026-06-20T03:22:23Z
Python: 3.14.5
Platform: Darwin 25.5.0 (arm64)
CPU cores: 15

## Backend

- Native C extension: yes
- OpenMP threads: 0
- GPU CuPy: no
- GPU Torch: no

## Highlights

- `vip_scores` `C dispatch` at 80x5,000: 2701.5x speedup (0.020 ms vs 54.256 ms baseline).
- `vip_scores` `C dispatch` at 80x1,000: 1352.6x speedup (0.005 ms vs 6.538 ms baseline).
- `vip_scores` `NumPy vectorized` at 80x5,000: 801.3x speedup (0.068 ms vs 54.256 ms baseline).
- `vip_scores` `NumPy vectorized` at 80x1,000: 309.5x speedup (0.021 ms vs 6.538 ms baseline).
- `stocsy` `ChunkedSTOCSY` at 100x5,000: 31.1x speedup (0.862 ms vs 26.795 ms baseline).
- `stocsy` `ChunkedSTOCSY` at 50x2,000: 26.6x speedup (0.230 ms vs 6.134 ms baseline).
- `column_variances` `C dispatch f32` at 500x100,000: 2.0x speedup (15.470 ms vs 30.326 ms baseline, peak 1.6 MB).
- `column_variances` `C dispatch f32` at 200x50,000: 1.9x speedup (3.024 ms vs 5.846 ms baseline, peak 0.8 MB).
- `opls_da_pipeline` `float32 pipeline` at 100x500: 1.9x speedup (196.924 ms vs 377.403 ms baseline).
- `column_variances` `C dispatch f64` at 500x100,000: 1.7x speedup (17.597 ms vs 30.326 ms baseline, peak 1.6 MB).

## Result Table

| Benchmark | Dataset | Implementation | Min ms | Mean ms | Peak MB | Speedup |
|---|---:|---|---:|---:|---:|---:|
| vip_scores | 80x1,000 | Python loop | 6.538 | 6.637 | 0.000 | baseline |
| vip_scores | 80x1,000 | NumPy vectorized | 0.021 | 0.026 | 0.000 | 309.5x |
| vip_scores | 80x1,000 | C dispatch | 0.005 | 0.006 | 0.000 | 1352.6x |
| vip_scores | 80x5,000 | Python loop | 54.256 | 54.422 | 0.000 | baseline |
| vip_scores | 80x5,000 | NumPy vectorized | 0.068 | 0.072 | 0.000 | 801.3x |
| vip_scores | 80x5,000 | C dispatch | 0.020 | 0.020 | 0.000 | 2701.5x |
| vip_scores | 80x20,000 | Python loop | N/A | N/A | 0.000 | baseline |
| vip_scores | 80x20,000 | NumPy vectorized | 0.246 | 0.257 | 0.000 | N/A |
| vip_scores | 80x20,000 | C dispatch | 0.077 | 0.081 | 0.000 | N/A |
| pearson_columns | 200x10,000 | Full-copy NumPy (baseline) | 1.091 | 1.147 | 16.323 | baseline |
| pearson_columns | 200x10,000 | C dispatch f64 | 0.901 | 0.924 | 0.320 | 1.2x |
| pearson_columns | 200x10,000 | C dispatch f32 | 0.729 | 0.756 | 0.320 | 1.5x |
| pearson_columns | 200x10,000 | Chunked NumPy (no C) | 1.140 | 1.157 | 16.485 | 1.0x |
| pearson_columns | 500x30,000 | Full-copy NumPy (baseline) | 8.335 | 8.659 | 120.965 | baseline |
| pearson_columns | 500x30,000 | C dispatch f64 | 6.711 | 6.739 | 0.960 | 1.2x |
| pearson_columns | 500x30,000 | C dispatch f32 | 5.964 | 6.024 | 0.960 | 1.4x |
| pearson_columns | 500x30,000 | Chunked NumPy (no C) | 952.100 | 975.994 | 135.533 | 0.0x |
| column_variances | 200x50,000 | NumPy (baseline) | 5.846 | 6.110 | 80.801 | baseline |
| column_variances | 200x50,000 | C dispatch f64 | 3.494 | 3.535 | 0.800 | 1.7x |
| column_variances | 200x50,000 | C dispatch f32 | 3.024 | 3.100 | 0.800 | 1.9x |
| column_variances | 500x100,000 | NumPy (baseline) | 30.326 | 31.775 | 401.601 | baseline |
| column_variances | 500x100,000 | C dispatch f64 | 17.597 | 17.717 | 1.600 | 1.7x |
| column_variances | 500x100,000 | C dispatch f32 | 15.470 | 15.854 | 1.600 | 2.0x |
| feature_preselection | 100x50,000 | Raw NumPy var (baseline) | 3.452 | 3.571 | 0.000 | baseline |
| feature_preselection | 100x50,000 | feature_preselection dispatch | 7.515 | 7.799 | 0.000 | 0.5x |
| feature_preselection | 200x100,000 | Raw NumPy var (baseline) | 11.346 | 11.477 | 0.000 | baseline |
| feature_preselection | 200x100,000 | feature_preselection dispatch | 30.164 | 30.794 | 0.000 | 0.4x |
| stocsy | 50x2,000 | Standard STOCSY | 6.134 | 6.688 | 0.000 | baseline |
| stocsy | 50x2,000 | ChunkedSTOCSY | 0.230 | 0.239 | 0.000 | 26.6x |
| stocsy | 100x5,000 | Standard STOCSY | 26.795 | 28.217 | 0.000 | baseline |
| stocsy | 100x5,000 | ChunkedSTOCSY | 0.862 | 0.885 | 0.000 | 31.1x |
| opls_da_pipeline | 60x300 | float64 pipeline | 27.506 | 29.302 | 0.000 | baseline |
| opls_da_pipeline | 60x300 | float32 pipeline | 37.597 | 37.878 | 0.000 | 0.7x |
| opls_da_pipeline | 100x500 | float64 pipeline | 377.403 | 382.985 | 0.000 | baseline |
| opls_da_pipeline | 100x500 | float32 pipeline | 196.924 | 198.850 | 0.000 | 1.9x |

## Notes

- `N/A` means the baseline was skipped, usually because the large Python-loop baseline would take too long.

## Artifacts

- Detailed report: `reports/PERFORMANCE.md`
- Raw data: `reports/benchmark_results.json`
- Previous run snapshot: `reports/benchmark_results_prev.json`
