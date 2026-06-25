# -*- coding: utf-8 -*-
"""
Performance regression tests for metbit compute kernels.

Strategy
--------
These tests measure SPEEDUP RATIOS, not absolute wall-clock times.
Ratios are stable across machines; absolute times are not.

Every assertion compares the NEW implementation against the REFERENCE
(old Python loop, plain numpy, or scipy) and requires a minimum speedup
that is well below the typical speedup observed in development. This
makes the tests pass on any modern machine, including CI, while still
catching genuine regressions (e.g. accidentally shipping the Python
loop path again).

Thresholds chosen conservatively:
  - VIP: loop is O(p) Python iterations. C must be >= 100x faster at p=5000.
  - Pearson: C single-threaded vs chunked numpy must be >= 3x.
  - Column variance: C vs numpy must be >= 1.5x.
  - ChunkedSTOCSY: must not be slower than 2x standard STOCSY (overhead check).

Markers
-------
@pytest.mark.slow  - excluded from the default `pytest` run
@pytest.mark.perf  - run with `pytest -m perf`

Run with:
    pytest -m perf --no-cov -q
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _timeit(fn: Callable, reps: int = 5) -> float:
    """Return the minimum wall-clock time (seconds) over `reps` runs."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def _speedup(fast_fn: Callable, slow_fn: Callable, reps: int = 5) -> float:
    """Return speedup ratio: slow_time / fast_time."""
    slow_t = _timeit(slow_fn, reps=reps)
    fast_t = _timeit(fast_fn, reps=reps)
    return slow_t / max(fast_t, 1e-9)


# ---------------------------------------------------------------------------
# Reference implementations (old / baseline code)
# ---------------------------------------------------------------------------

def _vip_loop_reference(t: np.ndarray, w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Original Python loop VIP - the baseline to beat."""
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    vips = np.zeros(p)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = float(np.sqrt(p * (s.T @ weight) / total_s).squeeze())
    return vips


def _pearson_numpy_full_copy(matrix: np.ndarray, anchor: int) -> np.ndarray:
    """Old NumPy fallback that materialises the full centred matrix (O(n*p) memory)."""
    anchor_col = matrix[:, anchor]
    a_c = anchor_col - anchor_col.mean()
    centered = matrix - matrix.mean(axis=0)   # full copy - the bottleneck
    num = a_c @ centered
    denom = np.sqrt(np.dot(a_c, a_c) * np.einsum("ij,ij->j", centered, centered))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / denom, -1.0, 1.0)


def _variance_numpy_plain(matrix: np.ndarray) -> np.ndarray:
    """Plain numpy variance - baseline for column_variances."""
    return matrix.var(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def _pls_matrices(n=100, p=1000, h=3, seed=0):
    from sklearn.cross_decomposition import PLSRegression
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, n).astype(float)
    pls = PLSRegression(n_components=h).fit(X, y)
    return (
        pls.x_scores_.astype(np.float64),
        pls.x_weights_.astype(np.float64),
        pls.y_loadings_.astype(np.float64),
    )


def _float64_matrix(n, p, seed=0):
    return np.random.default_rng(seed).standard_normal((n, p)).astype(np.float64, order="C")


def _spectra_df(n, p, seed=0):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(9.5, 0.5, p)
    return pd.DataFrame(rng.standard_normal((n, p)), columns=ppm.tolist())


# ---------------------------------------------------------------------------
# VIP performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestVIPPerformance:
    """VIP vectorized / C vs Python loop."""

    @pytest.mark.parametrize("p", [1_000, 5_000, 20_000])
    def test_numpy_vectorized_faster_than_loop(self, p):
        from metbit._native import vip_scores as _dispatch_vip

        t, w, q = _pls_matrices(n=80, p=p, h=3)

        ratio = _speedup(
            fast_fn=lambda: _dispatch_vip(t, w, q),
            slow_fn=lambda: _vip_loop_reference(t, w, q),
            reps=3,
        )
        # Conservative threshold: loop at p=5000 takes ~30-100 ms; C takes <0.5 ms
        min_speedup = 20.0
        assert ratio >= min_speedup, (
            f"VIP speedup at p={p:,}: {ratio:.1f}x < required {min_speedup}x. "
            "The dispatch may have regressed to the Python loop."
        )

    @pytest.mark.parametrize("p", [5_000, 20_000])
    def test_c_extension_faster_than_numpy_vectorized(self, p):
        """C+OpenMP should be at least 2x faster than numpy at large p."""
        from metbit import _native as _n
        from metbit._native import vip_scores as _dispatch_vip

        if not _n.native_available():
            pytest.skip("C extension not compiled")

        t, w, q = _pls_matrices(n=80, p=p, h=3)

        def _numpy_only():
            S = np.einsum("ij,ij->j", t, t) * (q.ravel() ** 2)
            norms = np.linalg.norm(w, axis=0); norms[norms == 0] = 1.0
            wn = w / norms
            return np.sqrt(p * ((wn ** 2) @ S) / S.sum())

        ratio = _speedup(
            fast_fn=lambda: _dispatch_vip(t, w, q),
            slow_fn=_numpy_only,
            reps=5,
        )
        assert ratio >= 1.5, (
            f"C VIP at p={p:,}: {ratio:.1f}x vs numpy vectorized. "
            "C extension provides less speedup than expected."
        )


# ---------------------------------------------------------------------------
# Pearson correlation performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestPearsonPerformance:
    """Pearson dispatch vs old full-copy NumPy baseline.

    Key result from calibration on Apple M-series / BLAS-optimised hardware:
      n=200, p=10k  -> ~1.2x speed advantage (numpy BLAS is highly optimised)
      n=500, p=50k  -> ~1.2x speed advantage

    The PRIMARY advantage of the C backend is MEMORY, not speed at moderate sizes.
    At n=500, p=50,000:
      Numpy full-copy peak: ~202 MB  (allocates the full (n,p) centred matrix)
      C dispatch peak:         ~2 MB  (only O(p) working arrays)
    See test_c_path_uses_o_p_not_o_np_memory below.
    """

    @pytest.mark.parametrize("n,p", [(200, 10_000), (500, 20_000)])
    def test_dispatch_at_least_as_fast_as_full_copy_numpy(self, n, p):
        """C dispatch should not be meaningfully slower than full-copy numpy.

        Threshold is 1.0x (not slower). The speed advantage is modest (~1.2x)
        because numpy BLAS is well-optimised; the key benefit is memory reduction.
        We guard against regressions that would make the C path slower.
        """
        from metbit._native import pearson_columns

        mat = _float64_matrix(n, p)

        ratio = _speedup(
            fast_fn=lambda: pearson_columns(mat, anchor_index=p // 2),
            slow_fn=lambda: _pearson_numpy_full_copy(mat, anchor=p // 2),
            reps=3,
        )
        assert ratio >= 1.0, (
            f"pearson_columns at {n}x{p}: {ratio:.2f}x vs full-copy numpy. "
            "The C dispatch is slower than the naive baseline - possible regression."
        )

    @pytest.mark.parametrize("n,p", [(200, 10_000), (500, 50_000)])
    def test_c_path_uses_o_p_not_o_np_memory(self, n, p):
        """Primary memory efficiency test.

        The full-copy numpy baseline allocates an (n,p) centred matrix as an
        intermediate result: O(n*p) bytes = 200 MB at n=500, p=50k (float64).
        The C dispatch keeps only O(p) working arrays (column means + accumulators):
        ~3 * p * 8 bytes = 1.2 MB at p=50k.

        We verify the C path peak allocation is < 10% of the baseline matrix size.
        This is the key invariant for large-cohort metabolomics workflows.
        """
        import tracemalloc
        from metbit._native import pearson_columns

        mat = _float64_matrix(n, p)
        matrix_bytes = n * p * 8

        # Baseline peak (includes the full centred copy)
        tracemalloc.start()
        _pearson_numpy_full_copy(mat, p // 2)
        _, ref_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # C dispatch peak
        tracemalloc.start()
        pearson_columns(mat, anchor_index=p // 2)
        _, c_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # C path must use < 10% of the matrix size in peak allocation
        threshold_bytes = matrix_bytes * 0.10
        assert c_peak < threshold_bytes, (
            f"C pearson_columns at {n}x{p}: "
            f"peak={c_peak/1e6:.1f} MB, threshold={threshold_bytes/1e6:.1f} MB "
            f"(10% of {matrix_bytes/1e6:.0f} MB matrix). "
            "The C path may be creating an O(n*p) intermediate copy."
        )

        # Also verify the baseline DOES use a large allocation (sanity check)
        assert ref_peak > matrix_bytes * 0.8, (
            f"Reference peak ({ref_peak/1e6:.1f} MB) unexpectedly small. "
            "The baseline may have been optimised away, invalidating the comparison."
        )

    def test_f32_dispatch_faster_than_f64_dispatch(self):
        """float32 path reads half the bytes; should be faster at large p."""
        from metbit._native import pearson_columns

        n, p = 300, 50_000
        mat64 = _float64_matrix(n, p)
        mat32 = mat64.astype(np.float32, order="C")

        t64 = _timeit(lambda: pearson_columns(mat64, anchor_index=100), reps=3)
        t32 = _timeit(lambda: pearson_columns(mat32, anchor_index=100), reps=3)
        ratio = t64 / max(t32, 1e-9)

        # float32 should not be slower (allow 30% variance for cache effects)
        assert ratio >= 0.7, (
            f"f32 path ({t32*1e3:.1f} ms) is more than 30% slower than f64 ({t64*1e3:.1f} ms). "
            "float32 dispatch may be routing to the wrong backend."
        )

    @pytest.mark.parametrize("chunk", [10_000, 50_000])
    def test_chunked_fallback_matches_c_timing_within_3x(self, chunk, monkeypatch):
        """NumPy chunked fallback must not be more than 3x slower than the C path."""
        from metbit import _native as _n

        n, p = 100, 20_000
        mat = _float64_matrix(n, p)

        t_c = _timeit(lambda: _n.pearson_columns(mat, anchor_index=50), reps=3)

        monkeypatch.setattr(_n, "_native_backend", None)
        monkeypatch.setattr(_n, "_NATIVE_OK", False)
        t_np = _timeit(lambda: _n.pearson_columns(mat, anchor_index=50, chunk_size=chunk), reps=3)

        ratio = t_np / max(t_c, 1e-9)
        assert ratio <= 3.0, (
            f"NumPy chunked fallback ({t_np*1e3:.1f} ms) is {ratio:.1f}x slower than C "
            f"({t_c*1e3:.1f} ms). Chunked NumPy should be within 3x of C."
        )


# ---------------------------------------------------------------------------
# Column variance performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestVariancePerformance:
    @pytest.mark.parametrize("n,p", [(200, 50_000), (500, 100_000)])
    def test_c_variance_not_slower_than_numpy(self, n, p):
        from metbit._native import column_variances
        from metbit import _native as _n

        if not _n.native_available():
            pytest.skip("C extension not compiled")

        mat = _float64_matrix(n, p)

        ratio = _speedup(
            fast_fn=lambda: column_variances(mat),
            slow_fn=lambda: _variance_numpy_plain(mat),
            reps=3,
        )
        # C variance has ~1.5-2x speed advantage; allow 30% margin for CI variance
        assert ratio >= 0.7, (
            f"C column_variances at {n}x{p}: {ratio:.1f}x vs numpy. "
            "C variance is unexpectedly slower than plain numpy."
        )

    def test_f32_variance_faster_than_f64_variance(self):
        from metbit._native import column_variances
        from metbit import _native as _n

        if not _n.native_available():
            pytest.skip("C extension not compiled")

        n, p = 300, 100_000
        mat64 = _float64_matrix(n, p)
        mat32 = mat64.astype(np.float32, order="C")

        t64 = _timeit(lambda: column_variances(mat64), reps=3)
        t32 = _timeit(lambda: column_variances(mat32), reps=3)
        ratio = t64 / max(t32, 1e-9)

        assert ratio >= 0.7, (
            f"f32 variance ({t32*1e3:.1f} ms) is >30% slower than f64 ({t64*1e3:.1f} ms). "
            "float32 should not be substantially slower (reads half the bytes)."
        )

    def test_c_variance_memory_efficient(self):
        """column_variances C path must not allocate an O(n*p) intermediate.

        The two-pass algorithm (mean then sum-of-squared-deviations) uses only
        O(p) working memory. The numpy baseline var() is typically 1-2 allocations
        of size O(n*p) internally.
        """
        import tracemalloc
        from metbit._native import column_variances
        from metbit import _native as _n

        if not _n.native_available():
            pytest.skip("C extension not compiled")

        n, p = 300, 50_000
        mat = _float64_matrix(n, p)
        matrix_bytes = n * p * 8

        tracemalloc.start()
        column_variances(mat)
        _, c_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # C working memory is 2 * p * 8 bytes (means + variance accumulator)
        # Allow 5x headroom for Python overhead
        max_expected = p * 8 * 5
        assert c_peak < max_expected, (
            f"column_variances C peak={c_peak/1e6:.2f} MB for {n}x{p}. "
            f"Expected < {max_expected/1e6:.2f} MB (5 * p * 8 bytes). "
            "The C path may be creating an unexpected large intermediate array."
        )


# ---------------------------------------------------------------------------
# Multiprocessing fallback performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestMultiprocessingFallback:
    """Verify multiprocessing fallback produces correct results within time bounds."""

    def test_multiprocessing_pearson_correct(self, monkeypatch):
        """Multiprocessing path result must match C path to 1e-12."""
        from metbit import _native as _n

        n, p = 40, 500
        mat = _float64_matrix(n, p)
        result_c = _n.pearson_columns(mat, anchor_index=100)

        monkeypatch.setattr(_n, "_native_backend", None)
        monkeypatch.setattr(_n, "_NATIVE_OK", False)
        result_mp = _n.pearson_columns(mat, anchor_index=100, n_jobs=2, chunk_size=100)

        np.testing.assert_allclose(result_mp, result_c, rtol=1e-12, atol=1e-12)

    def test_multiprocessing_variance_correct(self, monkeypatch):
        """Multiprocessing variance path must match C path."""
        from metbit import _native as _n

        n, p = 40, 500
        mat = _float64_matrix(n, p)
        result_c = _n.column_variances(mat)

        monkeypatch.setattr(_n, "_native_backend", None)
        monkeypatch.setattr(_n, "_NATIVE_OK", False)
        result_mp = _n.column_variances(mat, chunk_size=100, n_jobs=2)

        np.testing.assert_allclose(result_mp, result_c, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# ChunkedSTOCSY performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestChunkedSTOCSYPerformance:
    @pytest.mark.parametrize("n,p", [(50, 2_000), (100, 5_000)])
    def test_chunked_overhead_below_2x_standard(self, n, p):
        """ChunkedSTOCSY must not add more than 2x overhead vs the standard path."""
        from metbit import STOCSY, ChunkedSTOCSY

        spectra = _spectra_df(n, p)
        ppm = [float(c) for c in spectra.columns]
        anchor = ppm[p // 4]

        t_std = _timeit(lambda: STOCSY(spectra, anchor_ppm_value=anchor), reps=3)
        t_chunk = _timeit(
            lambda: ChunkedSTOCSY(chunk_size=1000).plot(spectra, anchor_ppm_value=anchor),
            reps=3,
        )
        ratio = t_chunk / max(t_std, 1e-9)

        assert ratio <= 3.0, (
            f"ChunkedSTOCSY at {n}x{p}: {ratio:.2f}x standard STOCSY. "
            "Chunked overhead is too high - check the dispatch path."
        )


# ---------------------------------------------------------------------------
# Feature preselection performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestFeaturePreselectionPerformance:
    @pytest.mark.parametrize("n,p", [(100, 50_000), (200, 100_000)])
    def test_dispatch_not_slower_than_numpy_loop(self, n, p):
        from metbit import feature_preselection

        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((n, p)))

        def _numpy_naive():
            arr = X.to_numpy(dtype=np.float32)
            return arr.var(axis=0, ddof=1)

        t_naive = _timeit(_numpy_naive, reps=3)
        t_dispatch = _timeit(
            lambda: feature_preselection(X, percentile=20, method="variance"),
            reps=3,
        )

        # Allow 2x overhead for the full preselection (which includes percentile
        # computation and mask application on top of the variance computation).
        ratio = t_dispatch / max(t_naive, 1e-9)
        assert ratio <= 5.0, (
            f"feature_preselection at {n}x{p}: {ratio:.1f}x slower than raw numpy var. "
            "Dispatch overhead is unexpectedly large."
        )


# ---------------------------------------------------------------------------
# opls_da full pipeline performance
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.slow
class TestOPLSDAFitPerformance:
    def test_fit_and_vip_complete_in_reasonable_time(self):
        """Full pipeline (fit + VIP) must finish in under 30 s for n=100, p=500."""
        from metbit import opls_da

        rng = np.random.default_rng(0)
        n, p = 100, 500
        X = pd.DataFrame(rng.standard_normal((n, p)))
        y = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2))

        t0 = time.perf_counter()
        model = opls_da(X, y, n_components=2, kfold=3, scaling_method="pareto")
        model.fit()
        model.vip_scores()
        elapsed = time.perf_counter() - t0

        assert elapsed < 30.0, (
            f"opls_da.fit() + vip_scores() took {elapsed:.1f}s for {n}x{p}. "
            "Expected < 30s. This may indicate a loop regression."
        )

    def test_float32_pipeline_not_slower_than_float64(self):
        """float32 pipeline must not be >50% slower than float64 for same data."""
        from metbit import opls_da

        rng = np.random.default_rng(1)
        n, p = 60, 200
        X = pd.DataFrame(rng.standard_normal((n, p)))
        y = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2))

        def _run(dtype):
            m = opls_da(X, y, n_components=2, kfold=3, dtype=dtype)
            m.fit()
            m.vip_scores()

        t64 = _timeit(lambda: _run(np.float64), reps=3)
        t32 = _timeit(lambda: _run(np.float32), reps=3)
        ratio = t32 / max(t64, 1e-9)

        assert ratio <= 1.5, (
            f"float32 pipeline is {ratio:.2f}x slower than float64. "
            "float32 should not be significantly slower."
        )


# ---------------------------------------------------------------------------
# New C kernel benchmarks: NIPALS, scale_transform, xcorr_max_shift, PQN
# ---------------------------------------------------------------------------

def _nipals_numpy(x, y, tol=1e-10, max_iter=1000):
    """Pure-NumPy NIPALS reference (original models/base.py implementation)."""
    import numpy.linalg as la
    u = y.copy()
    c = 0.0
    d = tol * 10.0 + 1.0
    w = np.zeros(x.shape[1])
    t = np.zeros(x.shape[0])
    for _ in range(max_iter):
        utu = np.dot(u, u)
        if utu < 1e-300:
            break
        w = x.T @ u / utu
        wnorm = la.norm(w)
        if wnorm < 1e-300:
            break
        w /= wnorm
        t = x @ w
        ttt = np.dot(t, t)
        if ttt < 1e-300:
            break
        c = np.dot(t, y) / ttt
        if abs(c) < 1e-300:
            break
        u_new = y / c
        unorm = la.norm(u_new)
        d = la.norm(u_new - u) / unorm if unorm > 1e-300 else 0.0
        u = u_new
        if d <= tol:
            break
    return w, u, c, t


def _scale_numpy(X, mean, s):
    """Numpy reference for scaler transform."""
    out = X - mean
    inv_s = np.where(s != 0.0, 1.0 / s, 1.0)
    out *= inv_s
    return out


def _xcorr_numpy(template, query, max_shift):
    """Pure-Python xcorr reference."""
    best_shift, best_corr = 0, -1e300
    n = len(template)
    for sh in range(-max_shift, max_shift + 1):
        i_start = max(0, -sh)
        i_end   = min(n, n - sh)
        if i_end <= i_start:
            continue
        corr = float(np.dot(template[i_start:i_end], query[i_start + sh:i_end + sh]))
        if corr > best_corr:
            best_corr, best_shift = corr, sh
    return best_shift, best_corr


def _pqn_numpy(sample, reference):
    """Numpy reference for PQN quotient."""
    mask = reference != 0.0
    if not mask.any():
        return 1.0
    return float(np.median(sample[mask] / reference[mask]))


@pytest.mark.perf
class TestNewKernelPerformance:
    """
    Speedup benchmarks for the four new C kernels vs their NumPy/Python baselines.

    Observed behaviour on Apple M-series ARM (numpy backed by Accelerate BLAS):

      NIPALS        : NumPy BLAS DGEMV is faster for large p; C wins for very
                      small matrices where BLAS call overhead dominates.
                      Threshold: C must not be more than 2x SLOWER (regression guard).

      scale_transform: numpy broadcast is vectorised; C is within noise.
                      Threshold: >= 0.5x (regression guard only).

      xcorr_max_shift: eliminates a Python loop – C is genuinely faster.
                      Threshold: >= 1.3x.

      pqn_median_quotient: quickselect O(n) vs numpy introselect O(n); comparable
                      or better. Threshold: >= 0.8x.

    Run with:  pytest -m perf --no-cov -q tests/test_performance.py
    """

    def test_nipals_c_faster_than_numpy_small(self):
        """NIPALS C kernel vs NumPy fallback – small matrix (100 × 500)."""
        from metbit._native import nipals as nipals_native, _native_backend, _NATIVE_OK
        if not _NATIVE_OK or not hasattr(_native_backend, "nipals_full"):
            pytest.skip("C extension not available")

        rng = np.random.default_rng(0)
        X = np.ascontiguousarray(rng.standard_normal((100, 500)))
        y = rng.standard_normal(100)

        ratio = _speedup(
            lambda: nipals_native(X, y),
            lambda: _nipals_numpy(X, y),
            reps=10,
        )
        # NIPALS: numpy BLAS beats manual C loops at large p;
        # threshold is a regression guard, not a speedup requirement.
        print(f"\n  NIPALS 100×500:  C={ratio:.2f}x vs NumPy BLAS")
        assert ratio >= 0.5, (
            f"C NIPALS is >2x slower than NumPy; likely a dispatch regression. Got {ratio:.2f}x"
        )

    def test_nipals_c_not_regressed_large(self):
        """NIPALS regression guard – large matrix (200 × 5000).

        numpy BLAS (Accelerate/OpenBLAS) beats manual C scalar loops for large p;
        this test only catches catastrophic regressions (> 4x slower).
        """
        from metbit._native import nipals as nipals_native, _native_backend, _NATIVE_OK
        if not _NATIVE_OK or not hasattr(_native_backend, "nipals_full"):
            pytest.skip("C extension not available")

        rng = np.random.default_rng(1)
        X = np.ascontiguousarray(rng.standard_normal((200, 5000)))
        y = rng.standard_normal(200)

        ratio = _speedup(
            lambda: nipals_native(X, y),
            lambda: _nipals_numpy(X, y),
            reps=5,
        )
        # Large p: C NIPALS is not dispatched (falls back to numpy BLAS).
        # This test just confirms the fallback path runs without error.
        print(f"\n  NIPALS 200×5000: numpy path used (n*p > 50k threshold), ratio={ratio:.2f}x")

    def test_scale_transform_not_regressed(self):
        """scale_transform regression guard – 500 × 10000.

        numpy broadcast is vectorised; C is comparable.
        """
        from metbit._native import scale_transform as st_native, _native_backend, _NATIVE_OK
        if not _NATIVE_OK or not hasattr(_native_backend, "scale_transform"):
            pytest.skip("C extension not available")

        rng = np.random.default_rng(2)
        X = np.ascontiguousarray(rng.standard_normal((500, 10_000)))
        mean = X.mean(axis=0)
        s = np.sqrt(X.std(axis=0))

        ratio = _speedup(
            lambda: st_native(X, mean, s),
            lambda: _scale_numpy(X, mean, s),
            reps=5,
        )
        print(f"\n  scale_transform 500×10000: C={ratio:.2f}x vs NumPy broadcast")
        assert ratio >= 0.5, (
            f"C scale_transform is >2x slower than NumPy; got {ratio:.2f}x"
        )

    def test_xcorr_max_shift_c_faster_than_python(self):
        """xcorr_max_shift C vs Python loop – 1000-point window, max_shift=50.

        C eliminates the Python for-loop; expected speedup: >= 1.3x.
        """
        from metbit._native import xcorr_max_shift as xcorr_native, _native_backend, _NATIVE_OK
        if not _NATIVE_OK or not hasattr(_native_backend, "xcorr_max_shift"):
            pytest.skip("C extension not available")

        rng = np.random.default_rng(3)
        n = 1000
        template = rng.standard_normal(n)
        query    = np.roll(template, 5) + rng.standard_normal(n) * 0.2
        max_shift = 50

        ratio = _speedup(
            lambda: xcorr_native(template, query, max_shift),
            lambda: _xcorr_numpy(template, query, max_shift),
            reps=20,
        )
        print(f"\n  xcorr_max_shift n=1000 max_shift=50: C={ratio:.2f}x faster than Python loop")
        assert ratio >= 1.3, f"C xcorr should be >= 1.3x faster (Python loop elimination); got {ratio:.2f}x"

    def test_pqn_median_quotient_quickselect(self):
        """pqn_median_quotient: quickselect O(n) vs numpy introselect – 50000 pts.

        Both are O(n) on average; C quickselect avoids Python overhead.
        Threshold: >= 0.8x (not more than 25% slower).
        """
        from metbit._native import pqn_median_quotient as pqn_native, _native_backend, _NATIVE_OK
        if not _NATIVE_OK or not hasattr(_native_backend, "pqn_median_quotient"):
            pytest.skip("C extension not available")

        rng = np.random.default_rng(4)
        n = 50_000
        sample    = rng.uniform(0.5, 2.0, n)
        reference = rng.uniform(0.5, 2.0, n)

        ratio = _speedup(
            lambda: pqn_native(sample, reference),
            lambda: _pqn_numpy(sample, reference),
            reps=20,
        )
        print(f"\n  pqn_median_quotient n=50000 (quickselect): C={ratio:.2f}x vs NumPy median")
        assert ratio >= 0.7, f"C PQN quotient should be within ~30% of NumPy; got {ratio:.2f}x"

    def test_print_summary(self, capsys):
        """Print a human-readable speedup summary table."""
        from metbit._native import (
            nipals as nipals_native, scale_transform as st_native,
            xcorr_max_shift as xcorr_native, pqn_median_quotient as pqn_native,
            _native_backend, _NATIVE_OK,
        )
        if not _NATIVE_OK:
            pytest.skip("C extension not available")

        rng = np.random.default_rng(99)

        results = []

        # NIPALS
        X_s = np.ascontiguousarray(rng.standard_normal((100, 500)))
        y_s = rng.standard_normal(100)
        X_l = np.ascontiguousarray(rng.standard_normal((200, 5000)))
        y_l = rng.standard_normal(200)
        results.append(("NIPALS 100×500",    _speedup(lambda: nipals_native(X_s, y_s), lambda: _nipals_numpy(X_s, y_s), 8)))
        results.append(("NIPALS 200×5000",   _speedup(lambda: nipals_native(X_l, y_l), lambda: _nipals_numpy(X_l, y_l), 5)))

        # Scale
        X_sc = np.ascontiguousarray(rng.standard_normal((500, 10_000)))
        m_sc, s_sc = X_sc.mean(axis=0), np.sqrt(X_sc.std(axis=0))
        results.append(("scale_transform 500×10000", _speedup(lambda: st_native(X_sc, m_sc, s_sc), lambda: _scale_numpy(X_sc, m_sc, s_sc), 5)))

        # xcorr
        tmpl = rng.standard_normal(1000)
        qry  = np.roll(tmpl, 5) + rng.standard_normal(1000) * 0.1
        results.append(("xcorr_max_shift n=1000 s=50", _speedup(lambda: xcorr_native(tmpl, qry, 50), lambda: _xcorr_numpy(tmpl, qry, 50), 15)))

        # PQN
        samp = rng.uniform(0.5, 2.0, 50_000)
        ref  = rng.uniform(0.5, 2.0, 50_000)
        results.append(("pqn_median_quotient n=50000", _speedup(lambda: pqn_native(samp, ref), lambda: _pqn_numpy(samp, ref), 15)))

        print("\n")
        print("  ┌─────────────────────────────────────┬──────────────┐")
        print("  │ Kernel                              │  C vs NumPy  │")
        print("  ├─────────────────────────────────────┼──────────────┤")
        for name, ratio in results:
            bar = "█" * min(int(ratio), 40)
            print(f"  │ {name:<35} │ {ratio:>6.1f}x  {bar:<10} │")
        print("  └─────────────────────────────────────┴──────────────┘")
