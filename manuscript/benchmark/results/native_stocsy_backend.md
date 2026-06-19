# Native STOCSY Backend Benchmark

Test date: 2026-06-19

## Implementation

`metbit._native_backend` is an optional CPython C extension that computes
column-wise Pearson correlations for STOCSY. It uses two row-major passes over
the spectral matrix, releases the Python GIL during computation, and returns
the correlation vector to Python. P-value calculation remains vectorized in
SciPy, and the public `STOCSY(...)` API is unchanged.

If compilation is unavailable, `metbit._native` automatically uses a NumPy
implementation. Set `METBIT_DISABLE_NATIVE=1` to force this fallback.

## Benchmark

Environment:

- Apple M5 Pro
- Python 3.14.5
- NumPy 2.4.6
- Matrix: 132 samples x 50,029 variables
- Native median based on 11 repetitions
- Scalar SciPy baseline based on one repetition

| Implementation | Median time |
|---|---:|
| Native C backend | 0.00325 s |
| NumPy fallback | 0.00410 s |
| Previous scalar `scipy.stats.pearsonr` loop | 3.25077 s |

The native backend was approximately 17% faster than the NumPy fallback and
999.1 times faster than the previous scalar SciPy loop. Maximum absolute
correlation error against SciPy was `4.44e-16`.

Run the benchmark with:

```bash
python manuscript/benchmark/06_benchmark_native_stocsy.py
```

Force the fallback comparison with:

```bash
METBIT_DISABLE_NATIVE=1 \
python manuscript/benchmark/06_benchmark_native_stocsy.py
```
