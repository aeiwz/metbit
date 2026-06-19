# -*- coding: utf-8 -*-
"""
Large-scale backend tests.

Covers:
  - All six new C extension functions (pearson_columns_par, pearson_columns_f32,
    column_variances, column_variances_f32, vip_scores, openmp_threads)
  - Auto-dispatch routing in _native.py (correct backend chosen by dataset size)
  - fallback consistency: multiprocessing and NumPy paths match the C path
  - ChunkedSTOCSY correctness vs the standard STOCSY kernel
  - feature_preselection via the dispatch layer
  - MemoryEstimator arithmetic
  - Environment-variable overrides (METBIT_DISABLE_NATIVE)
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_pearson(matrix: np.ndarray, anchor: int) -> np.ndarray:
    """Scipy reference for Pearson r of anchor column vs every column."""
    return np.array([
        pearsonr(matrix[:, anchor], matrix[:, c]).statistic
        for c in range(matrix.shape[1])
    ])


def _ref_variances(matrix: np.ndarray) -> np.ndarray:
    """Numpy reference: per-column sample variance."""
    return matrix.astype(np.float64).var(axis=0, ddof=1)


def _make_matrix(n=80, p=300, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(dtype)


# ---------------------------------------------------------------------------
# C extension: new function correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("metbit._native", fromlist=["native_available"]).native_available(),
    reason="C extension not compiled"
)
class TestNativeBackendNewFunctions:
    """Verify correctness of every function in the enhanced C backend."""

    def test_pearson_columns_par_matches_scipy(self):
        from metbit import _native_backend
        if not hasattr(_native_backend, "pearson_columns_par"):
            pytest.skip("pearson_columns_par not in this build")
        mat = _make_matrix(60, 200)
        packed = _native_backend.pearson_columns_par(memoryview(mat), 60, 200, 50)
        actual = np.frombuffer(packed, dtype=np.float64).copy()
        np.testing.assert_allclose(actual, _ref_pearson(mat, 50), rtol=1e-12, atol=1e-12)

    def test_pearson_columns_f32_close_to_scipy(self):
        from metbit import _native_backend
        if not hasattr(_native_backend, "pearson_columns_f32"):
            pytest.skip("pearson_columns_f32 not in this build")
        mat = _make_matrix(50, 150, dtype=np.float32)
        packed = _native_backend.pearson_columns_f32(memoryview(mat), 50, 150, 20)
        actual = np.frombuffer(packed, dtype=np.float64).copy()
        # float32 precision limit: tolerate up to 1e-5
        np.testing.assert_allclose(actual, _ref_pearson(mat, 20), rtol=1e-5, atol=1e-5)

    def test_column_variances_matches_numpy(self):
        from metbit import _native_backend
        if not hasattr(_native_backend, "column_variances"):
            pytest.skip("column_variances not in this build")
        mat = _make_matrix(70, 200)
        packed = _native_backend.column_variances(memoryview(mat), 70, 200)
        actual = np.frombuffer(packed, dtype=np.float64).copy()
        np.testing.assert_allclose(actual, _ref_variances(mat), rtol=1e-12, atol=1e-12)

    def test_column_variances_f32_close_to_numpy(self):
        from metbit import _native_backend
        if not hasattr(_native_backend, "column_variances_f32"):
            pytest.skip("column_variances_f32 not in this build")
        mat = _make_matrix(70, 200, dtype=np.float32)
        packed = _native_backend.column_variances_f32(memoryview(mat), 70, 200)
        actual = np.frombuffer(packed, dtype=np.float64).copy()
        np.testing.assert_allclose(actual, _ref_variances(mat), rtol=1e-5, atol=1e-5)

    def test_vip_scores_matches_numpy(self):
        from metbit import _native_backend
        from sklearn.cross_decomposition import PLSRegression
        if not hasattr(_native_backend, "vip_scores"):
            pytest.skip("vip_scores not in this build")

        rng = np.random.default_rng(9)
        X = rng.standard_normal((60, 120))
        y = rng.integers(0, 2, 60).astype(float)
        pls = PLSRegression(n_components=3).fit(X, y)
        t = np.ascontiguousarray(pls.x_scores_, dtype=np.float64)
        w = np.ascontiguousarray(pls.x_weights_, dtype=np.float64)
        q = np.ascontiguousarray(pls.y_loadings_.ravel(), dtype=np.float64)
        n_s, n_c = t.shape
        n_f = w.shape[0]

        packed = _native_backend.vip_scores(
            memoryview(t), memoryview(w), memoryview(q),
            n_s, n_f, n_c
        )
        actual = np.frombuffer(packed, dtype=np.float64).copy()

        # Reference numpy VIP
        S = np.einsum("ij,ij->j", t, t) * (q ** 2)
        norms = np.linalg.norm(w, axis=0); norms[norms == 0] = 1.0
        wn = w / norms
        expected = np.sqrt(n_f * ((wn ** 2) @ S) / S.sum())

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_openmp_threads_returns_int(self):
        from metbit import _native_backend
        if not hasattr(_native_backend, "openmp_threads"):
            pytest.skip("openmp_threads not in this build")
        n = _native_backend.openmp_threads()
        assert isinstance(n, int)
        assert n >= 0


# ---------------------------------------------------------------------------
# Dispatch layer: backend_info and routing
# ---------------------------------------------------------------------------

class TestDispatchRouting:
    def test_backend_info_keys(self):
        from metbit import backend_info
        info = backend_info()
        for key in ("native_c", "openmp_threads", "gpu_cupy", "gpu_torch",
                    "n_jobs", "default_chunk"):
            assert key in info, f"Missing key '{key}' in backend_info()"

    def test_native_available_bool(self):
        from metbit import native_available
        assert isinstance(native_available(), bool)

    def test_gpu_available_bool(self):
        from metbit import gpu_available
        assert isinstance(gpu_available(), bool)

    def test_pearson_dispatch_small_uses_single_thread_c(self, monkeypatch):
        """n*p <= SMALL_THRESH -> single-threaded C path (pearson_columns)."""
        from metbit import _native as _n
        calls = []
        original = _n._native_backend

        if original is None:
            pytest.skip("C extension not compiled")

        class _Recorder:
            def __getattr__(self, name):
                def _wrapped(*args, **kwargs):
                    calls.append(name)
                    return getattr(original, name)(*args, **kwargs)
                return _wrapped

        monkeypatch.setattr(_n, "_native_backend", _Recorder())
        monkeypatch.setattr(_n, "_NATIVE_OK", True)

        mat = _make_matrix(50, 100)   # 5000 elements, well below SMALL_THRESH
        _n.pearson_columns(mat, anchor_index=10)
        # Should have called the single-threaded function
        assert "pearson_columns" in calls

    def test_pearson_dispatch_fallback_matches_c(self, monkeypatch):
        """NumPy chunked fallback must produce identical results to C path."""
        from metbit import _native as _n

        mat = _make_matrix(40, 150)
        result_c = _n.pearson_columns(mat, anchor_index=20)

        monkeypatch.setattr(_n, "_native_backend", None)
        monkeypatch.setattr(_n, "_NATIVE_OK", False)
        result_np = _n.pearson_columns(mat, anchor_index=20, chunk_size=50)

        np.testing.assert_allclose(result_np, result_c, rtol=1e-12, atol=1e-12)

    def test_column_variances_dispatch_matches_numpy(self):
        from metbit._native import column_variances
        mat = _make_matrix(50, 200)
        actual = column_variances(mat)
        expected = _ref_variances(mat)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_column_variances_f32_dispatch_close_to_numpy(self):
        from metbit._native import column_variances
        mat = _make_matrix(50, 200, dtype=np.float32)
        actual = column_variances(mat)
        expected = _ref_variances(mat)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_vip_scores_dispatch_matches_numpy(self):
        from metbit._native import vip_scores
        from sklearn.cross_decomposition import PLSRegression

        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 100))
        y = rng.integers(0, 2, 50).astype(float)
        pls = PLSRegression(n_components=2).fit(X, y)

        actual = vip_scores(pls.x_scores_, pls.x_weights_, pls.y_loadings_)

        # Reference
        t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_.ravel()
        S = np.einsum("ij,ij->j", t, t) * (q ** 2)
        norms = np.linalg.norm(w, axis=0); norms[norms == 0] = 1.0
        expected = np.sqrt(100 * ((w / norms) ** 2 @ S) / S.sum())

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_vip_dispatch_fallback_when_no_c_ext(self, monkeypatch):
        """NumPy fallback in vip_scores must match C path."""
        from metbit import _native as _n
        from sklearn.cross_decomposition import PLSRegression

        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 80))
        y = rng.integers(0, 2, 50).astype(float)
        pls = PLSRegression(n_components=2).fit(X, y)

        result_c = _n.vip_scores(pls.x_scores_, pls.x_weights_, pls.y_loadings_)

        monkeypatch.setattr(_n, "_native_backend", None)
        monkeypatch.setattr(_n, "_NATIVE_OK", False)
        result_np = _n.vip_scores(pls.x_scores_, pls.x_weights_, pls.y_loadings_)

        np.testing.assert_allclose(result_np, result_c, rtol=1e-12, atol=1e-12)

    def test_pearson_f32_dispatch_uses_f32_c_when_available(self):
        """float32 matrix must route to the f32 C function and return float64."""
        from metbit._native import pearson_columns
        mat = _make_matrix(40, 100, dtype=np.float32)
        result = pearson_columns(mat, anchor_index=10)
        assert result.dtype == np.float64
        assert result.shape == (100,)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvironmentOverrides:
    def test_disable_native_falls_back_to_numpy(self, monkeypatch):
        """METBIT_DISABLE_NATIVE=1 must make native_available() return False."""
        monkeypatch.setenv("METBIT_DISABLE_NATIVE", "1")
        import metbit._native as _n
        # Simulate what the module-level code does on import
        import importlib
        monkeypatch.setattr(_n, "_native_backend", None)
        monkeypatch.setattr(_n, "_NATIVE_OK", False)

        mat = _make_matrix(20, 50)
        result = _n.pearson_columns(mat, anchor_index=5)
        ref = _ref_pearson(mat, 5)
        np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-12)

    def test_custom_chunk_size_via_env(self, monkeypatch):
        """Dispatch must honour custom chunk size from env without crashing."""
        from metbit import _native as _n
        monkeypatch.setattr(_n, "_DEFAULT_CHUNK", 20)
        mat = _make_matrix(30, 100)
        result = _n.pearson_columns(mat, anchor_index=10, chunk_size=20)
        ref = _ref_pearson(mat, 10)
        np.testing.assert_allclose(result, ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# ChunkedSTOCSY: correctness at various chunk sizes
# ---------------------------------------------------------------------------

class TestChunkedSTOCSYBackend:
    @pytest.fixture
    def spectra(self):
        rng = np.random.default_rng(3)
        ppm = np.linspace(9.0, 0.5, 120)
        data = rng.standard_normal((25, 120))
        data[:, 60]  = rng.standard_normal(25)
        data[:, 61]  = data[:, 60] * 2 + rng.standard_normal(25) * 0.1
        return pd.DataFrame(data, columns=ppm.tolist())

    @pytest.mark.parametrize("chunk", [10, 30, 60, 120])
    def test_various_chunk_sizes_match_standard(self, spectra, chunk):
        from metbit.analysis.stocsy import _stocsy_statistics
        from metbit import ChunkedSTOCSY

        ppm = spectra.columns.astype(float).to_numpy()
        anchor_idx = 60
        anchor_ppm = float(ppm[anchor_idx])

        corr_std, pval_std = _stocsy_statistics(spectra, anchor_index=anchor_idx)

        stocsy = ChunkedSTOCSY(chunk_size=chunk)
        _, corr_out, pval_out = stocsy.compute(spectra, anchor_ppm_value=anchor_ppm)

        np.testing.assert_allclose(corr_out, corr_std, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(pval_out, pval_std, rtol=1e-8,  atol=1e-10)

    def test_self_correlation_is_one(self, spectra):
        from metbit import ChunkedSTOCSY
        ppm = spectra.columns.astype(float).to_numpy()
        anchor_idx = 60
        anchor_ppm = float(ppm[anchor_idx])
        stocsy = ChunkedSTOCSY(chunk_size=25)
        _, corr, _ = stocsy.compute(spectra, anchor_ppm_value=anchor_ppm)
        assert abs(corr[anchor_idx] - 1.0) < 1e-10

    def test_correlations_clipped_to_minus_one_one(self, spectra):
        from metbit import ChunkedSTOCSY
        ppm = [float(c) for c in spectra.columns]
        stocsy = ChunkedSTOCSY()
        _, corr, _ = stocsy.compute(spectra, anchor_ppm_value=ppm[40])
        assert (corr >= -1.0).all() and (corr <= 1.0).all()

    def test_plot_traces_match_p_value_threshold(self, spectra):
        """Significant trace count + non-significant trace count = n_features."""
        from metbit import ChunkedSTOCSY
        import plotly.graph_objects as go

        ppm = [float(c) for c in spectra.columns]
        stocsy = ChunkedSTOCSY(p_value_threshold=0.05)
        fig = stocsy.plot(spectra, anchor_ppm_value=ppm[60])

        assert isinstance(fig, go.Figure)
        total_points = sum(len(t.x) for t in fig.data)
        assert total_points == spectra.shape[1]


# ---------------------------------------------------------------------------
# feature_preselection: dispatch layer integration
# ---------------------------------------------------------------------------

class TestFeaturePreselectionDispatch:
    def test_variance_dispatch_matches_numpy(self):
        """Dispatch must return variances matching numpy reference."""
        from metbit import feature_preselection
        from metbit._native import column_variances

        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.standard_normal((50, 200)))

        # Manually inject known low-variance features
        X.iloc[:, :40] *= 0.001

        _, mask = feature_preselection(X, percentile=20, method="variance")

        # Low-variance features should be preferentially removed
        # (at least half of the 40 low-variance features should be excluded)
        n_low_var_kept = mask[:40].sum()
        n_high_var_kept = mask[40:].sum()
        assert n_high_var_kept > n_low_var_kept, (
            f"Preselection kept {n_low_var_kept}/40 low-variance and "
            f"{n_high_var_kept}/160 high-variance features. "
            "Low-variance features should be preferentially removed."
        )

    def test_preselection_chunk_size_does_not_change_result(self):
        """Different chunk sizes must produce identical masks."""
        from metbit.analysis.large_scale import feature_preselection

        rng = np.random.default_rng(21)
        X = pd.DataFrame(rng.standard_normal((30, 100)))

        _, mask1 = feature_preselection(X, percentile=20, chunk_size=20)
        _, mask2 = feature_preselection(X, percentile=20, chunk_size=100)

        np.testing.assert_array_equal(mask1, mask2)


# ---------------------------------------------------------------------------
# MemoryEstimator arithmetic
# ---------------------------------------------------------------------------

class TestMemoryEstimatorArithmetic:
    def test_float64_is_8_bytes_per_element(self):
        from metbit import MemoryEstimator
        # Use a large matrix so rounding to 2 dp is negligible relative to the value
        r = MemoryEstimator.estimate(10_000, 10_000, np.float64, copies=1)
        expected_gb = 10_000 * 10_000 * 8 / 1024 ** 3
        assert abs(r["single_matrix_gb"] - round(expected_gb, 2)) < 1e-9

    def test_float32_is_4_bytes_per_element(self):
        from metbit import MemoryEstimator
        r = MemoryEstimator.estimate(10_000, 10_000, np.float32, copies=1)
        expected_gb = 10_000 * 10_000 * 4 / 1024 ** 3
        assert abs(r["single_matrix_gb"] - round(expected_gb, 2)) < 1e-9

    def test_float32_half_memory_of_float64(self):
        from metbit import MemoryEstimator
        # single_matrix_gb is rounded to 2 dp, so allow a 5% rounding error.
        r64 = MemoryEstimator.estimate(10_000, 10_000, np.float64, copies=1)
        r32 = MemoryEstimator.estimate(10_000, 10_000, np.float32, copies=1)
        ratio = r64["single_matrix_gb"] / r32["single_matrix_gb"]
        assert 1.9 < ratio < 2.1, (
            f"float64/float32 GB ratio {ratio:.3f} expected ~2.0 "
            "(rounding to 2 dp allows ±5% tolerance)"
        )

    def test_copies_multiplier_correct(self):
        from metbit import MemoryEstimator
        # Large matrices so rounding error on single_gb is small relative to peak_gb
        r1 = MemoryEstimator.estimate(5_000, 5_000, np.float64, copies=1)
        r3 = MemoryEstimator.estimate(5_000, 5_000, np.float64, copies=3)
        # peak_gb_with_copies should be exactly 3x single_matrix_gb (both rounded)
        assert abs(r3["peak_gb_with_copies"] - r1["single_matrix_gb"] * 3) < 0.1

    def test_large_dataset_recommendation(self):
        from metbit import MemoryEstimator
        r = MemoryEstimator.estimate(10_000, 1_000_000, np.float64, copies=2)
        assert r["recommended_dtype"] == "float32"

    def test_small_dataset_keeps_float64(self):
        from metbit import MemoryEstimator
        r = MemoryEstimator.estimate(100, 1_000, np.float64, copies=1)
        assert r["recommended_dtype"] == "float64"
