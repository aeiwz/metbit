# Test Summary (2026-06-19T04:33:36.739654Z)

- Python: 3.14.5
- OS: Darwin 25.5.0
- Package version: 9.0.0-dev
- Total: 0, Passed: 0, Failed: 0, Skipped: 0
- Duration: 0.00s
- Coverage: 0.00%

## Lowest coverage modules
- N/A

## Performance
- Benchmark run: 2026-06-19T04:33:30Z
- Backend: native_c=yes, openmp_threads=0, gpu=no
- `vip_scores` NumPy vectorized @ 80x500: **143.8x** speedup (0.0 ms vs 2.0 ms baseline)
- `vip_scores` C dispatch @ 80x500: **596.5x** speedup (0.0 ms vs 2.0 ms baseline)
- `vip_scores` NumPy vectorized @ 80x2,000: **373.2x** speedup (0.0 ms vs 15.2 ms baseline)
- `vip_scores` C dispatch @ 80x2,000: **1716.1x** speedup (0.0 ms vs 15.2 ms baseline)
- `pearson_columns` C dispatch f64 @ 50x2,000: **2.3x** speedup (0.0 ms vs 0.1 ms baseline)
- `pearson_columns` C dispatch f32 @ 50x2,000: **2.5x** speedup (0.0 ms vs 0.1 ms baseline)
- `pearson_columns` Chunked NumPy (no C) @ 50x2,000: **1.0x** speedup (0.1 ms vs 0.1 ms baseline)
- `pearson_columns` C dispatch f64 @ 100x5,000: **1.1x** speedup (0.2 ms vs 0.2 ms baseline)
- `pearson_columns` C dispatch f32 @ 100x5,000: **1.3x** speedup (0.2 ms vs 0.2 ms baseline)
- `pearson_columns` Chunked NumPy (no C) @ 100x5,000: **1.0x** speedup (0.2 ms vs 0.2 ms baseline)
- `column_variances` C dispatch f64 @ 100x10,000: **1.9x** speedup (0.3 ms vs 0.7 ms baseline)
- `column_variances` C dispatch f32 @ 100x10,000: **2.4x** speedup (0.3 ms vs 0.7 ms baseline)
- `feature_preselection` feature_preselection dispatch @ 50x10,000: **0.4x** speedup (0.4 ms vs 0.2 ms baseline)
- `stocsy` ChunkedSTOCSY @ 30x1,000: **28.6x** speedup (0.1 ms vs 3.1 ms baseline)
- `opls_da_pipeline` float32 pipeline @ 40x200: **0.8x** speedup (12.2 ms vs 10.2 ms baseline)
- Full report: reports/PERFORMANCE.md

## Failures
- None