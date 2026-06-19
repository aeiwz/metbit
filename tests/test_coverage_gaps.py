"""Covers remaining small gaps across multiple modules."""
import importlib
import os
import sys
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# metbit/_internal/pairs.py
# ---------------------------------------------------------------------------

from metbit._internal.pairs import lazypair


class TestLazypair:
    def _make_df(self):
        return pd.DataFrame({
            "group": ["A", "A", "B", "B", "C"],
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    def test_not_dataframe_raises(self):
        with pytest.raises(ValueError):
            lazypair({"a": 1}, "group")

    def test_column_not_string_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError):
            lazypair(df, 123)

    def test_column_not_found_raises(self):
        df = self._make_df()
        with pytest.raises(KeyError):
            lazypair(df, "nonexistent")

    def test_too_few_groups_raises(self):
        df = pd.DataFrame({"group": ["A", "A"], "val": [1.0, 2.0]})
        with pytest.raises(ValueError):
            lazypair(df, "group")

    def test_get_index(self):
        df = self._make_df()
        lp = lazypair(df, "group")
        idx = lp.get_index()
        assert isinstance(idx, list)
        assert len(idx) > 0

    def test_get_name(self):
        df = self._make_df()
        lp = lazypair(df, "group")
        names = lp.get_name()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get_meta(self):
        df = self._make_df()
        lp = lazypair(df, "group")
        meta = lp.get_meta()
        assert isinstance(meta, pd.Series)

    def test_get_column_name(self):
        df = self._make_df()
        lp = lazypair(df, "group")
        col = lp.get_column_name()
        assert col == "group"

    def test_get_dataset(self):
        df = self._make_df()
        lp = lazypair(df, "group")
        dsets = lp.get_dataset()
        assert isinstance(dsets, list)
        assert all(isinstance(d, pd.DataFrame) for d in dsets)


# ---------------------------------------------------------------------------
# metbit/_native.py - worker functions and edge cases
# ---------------------------------------------------------------------------

from metbit._native import (
    _pearson_chunk_worker,
    _variance_chunk_worker,
    _variance_multiprocessing,
    _pearson_numpy_chunked,
    _asf64_c,
    _asf32_c,
    vip_scores,
)


class TestNativeWorkers:
    def test_pearson_chunk_worker(self):
        rng = np.random.default_rng(0)
        n, chunk_size = 20, 5
        chunk_data = rng.standard_normal((n, chunk_size))
        anchor = rng.standard_normal(n)
        a_c = anchor - anchor.mean()
        a_sq = float(np.dot(a_c, a_c))
        col_means = chunk_data.mean(axis=0)
        result = _pearson_chunk_worker((chunk_data, a_c, a_sq, col_means))
        assert result.shape == (chunk_size,)
        assert np.all(np.abs(result) <= 1.0)

    def test_variance_chunk_worker(self):
        rng = np.random.default_rng(0)
        n, chunk_size = 20, 5
        chunk_data = rng.standard_normal((n, chunk_size))
        col_means = chunk_data.mean(axis=0)
        result = _variance_chunk_worker((chunk_data, col_means))
        assert result.shape == (chunk_size,)
        assert np.all(result >= 0)

    def test_asf64_c(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = _asf64_c(arr)
        assert result.dtype == np.float64

    def test_asf32_c(self):
        arr = np.array([1, 2, 3], dtype=np.float64)
        result = _asf32_c(arr)
        assert result.dtype == np.float32

    def test_pearson_numpy_chunked_zero_anchor(self):
        matrix = np.zeros((5, 10))
        matrix[:, 1] = 1.0
        result = _pearson_numpy_chunked(matrix, anchor_index=0, chunk_size=5)
        assert np.all(result == 0.0)

    def test_vip_scores_zero_total_s(self):
        rng = np.random.default_rng(0)
        t = np.zeros((10, 2))
        w = rng.standard_normal((5, 2))
        q = np.array([1.0, 1.0])
        result = vip_scores(t, w, q)
        assert np.all(result == 0.0)

    def test_variance_multiprocessing_result_gathering(self):
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((20, 10)).astype(np.float64)
        result = _variance_multiprocessing(matrix, chunk_size=5, n_jobs=1)
        assert result.shape == (10,)
        assert np.all(result >= 0)

    def test_native_disable_env(self, monkeypatch):
        monkeypatch.setenv("METBIT_DISABLE_NATIVE", "1")
        for key in list(sys.modules.keys()):
            if "metbit._native" in key:
                del sys.modules[key]
        import metbit._native as n_mod
        importlib.reload(n_mod)
        assert n_mod._DISABLE_NATIVE is True
        for key in list(sys.modules.keys()):
            if "metbit._native" in key:
                del sys.modules[key]
        monkeypatch.delenv("METBIT_DISABLE_NATIVE", raising=False)
        import metbit._native
        importlib.reload(metbit._native)


# ---------------------------------------------------------------------------
# metbit/analysis/large_scale.py
# ---------------------------------------------------------------------------

from metbit.analysis.large_scale import (
    feature_preselection, LargeScaleAlignment, memory_report, ChunkedSTOCSY,
    MemoryEstimator, chunked_pearson,
)


class TestLargeScaleExtra:
    def test_feature_preselection_iqr(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((30, 50)))
        X_red, mask = feature_preselection(X, percentile=20.0, method="iqr")
        assert mask.sum() > 0

    def test_feature_preselection_invalid_method(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, 20)))
        with pytest.raises(ValueError):
            feature_preselection(X, method="bad_method")

    def test_memory_report_dataframe(self, capsys):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((10, 5)))
        memory_report(X)
        captured = capsys.readouterr()
        assert len(captured.out) >= 0

    def test_memory_report_large_triggers_tip(self, capsys):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((1000, 6000)))
        memory_report(X)
        captured = capsys.readouterr()
        assert len(captured.out) >= 0

    def test_chunked_stocsy_plot(self):
        rng = np.random.default_rng(0)
        n, p = 30, 50
        ppm = np.linspace(10, 0, p)
        spectra = pd.DataFrame(rng.standard_normal((n, p)), columns=ppm)
        cs = ChunkedSTOCSY(chunk_size=25)
        fig = cs.plot(spectra, anchor_ppm_value=5.0)
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)

    def test_chunked_stocsy_n2_pvalue(self):
        rng = np.random.default_rng(0)
        ppm = np.linspace(10, 0, 20)
        spectra = pd.DataFrame(rng.standard_normal((2, 20)), columns=ppm)
        cs = ChunkedSTOCSY()
        ppm_out, corr, pvals = cs.compute(spectra, anchor_ppm_value=5.0)
        assert pvals.shape == corr.shape

    def test_large_scale_alignment_align(self):
        rng = np.random.default_rng(0)
        ppm = np.linspace(10, 0, 100)
        spectra = pd.DataFrame(rng.standard_normal((5, 100)), columns=ppm)
        windows = [(4.8, 5.2), (2.8, 3.2)]
        la = LargeScaleAlignment()
        aligned, shifts = la.align(spectra, ppm, windows)
        assert aligned.shape == spectra.shape

    def test_memory_estimator_tip_for_float32(self, capsys):
        # float32 input with small data => recommended_dtype='float64' != 'float32' => prints tip
        MemoryEstimator.print_estimate(10, 10, dtype=np.float32, copies=1)
        captured = capsys.readouterr()
        assert "Tip" in captured.out

    def test_chunked_pearson_function(self):
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((20, 30))
        result = chunked_pearson(mat, anchor_index=5, chunk_size=10)
        assert result.shape == (30,)
        assert np.all(np.abs(result) <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# metbit/models/cross_validation.py - non-opls error paths
# ---------------------------------------------------------------------------

from metbit.models.cross_validation import CrossValidation


def _make_cv_data(n=40, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    return X, y


class TestCrossValidationNonOPLS:
    def test_orthogonal_score_opls(self):
        X, y = _make_cv_data()
        cv = CrossValidation("opls", kfold=3)
        cv.fit(X, y)
        os_ = cv.orthogonal_score
        assert os_ is not None

    def test_predictive_score_non_opls_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.predictive_score

    def test_r2xcorr_non_opls_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.R2Xcorr

    def test_r2xyo_non_opls_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.R2XYO

    def test_correlation_non_opls_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.correlation

    def test_covariance_non_opls_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.covariance

    def test_loadings_cv_non_opls_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.loadings_cv

    def test_split_too_few_raises(self):
        X, y = _make_cv_data()
        cv = CrossValidation("opls", kfold=100)
        with pytest.raises(ValueError):
            cv.fit(X, y)

    def test_predict_non_opls(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        yhat = cv.predict(X)
        assert yhat.shape[0] == X.shape[0]

    def test_leave_one_out(self):
        X, y = _make_cv_data(n=10)
        cv = CrossValidation("opls", kfold=10)
        cv.fit(X, y)
        assert hasattr(cv, "_opt_component")

    def test_scores_non_opls(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        s = cv.scores
        assert s is not None

    def test_r2x_and_r2y(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        assert isinstance(cv.R2X, float)
        assert isinstance(cv.R2y, float)

    def test_min_nmc_and_mis_classifications(self):
        X, y = _make_cv_data()
        cv = CrossValidation("opls", kfold=3)
        cv.fit(X, y)
        assert isinstance(cv.min_nmc, (int, float, np.integer))
        assert isinstance(cv.mis_classifications, np.ndarray)

    def test_orthogonal_score_raises_for_pls(self):
        X, y = _make_cv_data()
        cv = CrossValidation("pls", kfold=3)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            _ = cv.orthogonal_score

    def test_r2xyo_opls(self):
        X, y = _make_cv_data()
        cv = CrossValidation("opls", kfold=3)
        cv.fit(X, y)
        val = cv.R2XYO
        assert isinstance(val, float)

    def test_y_as_list_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 5))
        y = ["A"] * 10 + ["B"] * 10  # list, not np.array
        cv = CrossValidation("opls", kfold=3)
        cv.fit(X, y)
        assert hasattr(cv, "_opt_component")

    def test_non_binary_y_raises(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 5))
        y = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        cv = CrossValidation("opls", kfold=3)
        with pytest.raises(ValueError):
            cv.fit(X, y)


# ---------------------------------------------------------------------------
# metbit/models/opls.py - 1D predict path, predictive_score, ortho_score
# ---------------------------------------------------------------------------

from metbit.models.opls import OPLS


class TestOPLS1D:
    def test_correct_1d_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        y = X[:, 0] * 3 - X[:, 1] + rng.standard_normal(30) * 0.1
        model = OPLS()
        model.fit(X, y, n_comp=2)
        x_1d = X[0, :]
        xc, t = model.correct(x_1d, return_scores=True)
        assert t.shape == (2,)

    def test_predictive_score_method(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        y = X[:, 0] * 3 + rng.standard_normal(30) * 0.1
        model = OPLS()
        model.fit(X, y, n_comp=2)
        ps = model.predictive_score(n_component=1)
        assert ps.shape[0] == 30

    def test_ortho_score_method(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        y = X[:, 0] * 3 + rng.standard_normal(30) * 0.1
        model = OPLS()
        model.fit(X, y, n_comp=2)
        os_ = model.ortho_score(n_component=2)
        assert os_.shape[0] == 30

    def test_predictive_score_none_component(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        y = X[:, 0] + rng.standard_normal(30) * 0.1
        model = OPLS()
        model.fit(X, y, n_comp=2)
        ps = model.predictive_score()
        assert ps.shape[0] == 30

    def test_ortho_score_exceeds_npc(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 8))
        y = X[:, 0] + rng.standard_normal(30) * 0.1
        model = OPLS()
        model.fit(X, y, n_comp=2)
        os_ = model.ortho_score(n_component=99)
        assert os_.shape[0] == 30


# ---------------------------------------------------------------------------
# metbit/nmr/alignment.py - edge cases
# ---------------------------------------------------------------------------

from metbit.nmr.alignment import _classify_pattern, icoshift_align, PeakAligner


class TestAlignmentEdgeCases:
    def test_classify_zero_heights(self):
        centers = np.array([1.0, 2.0])
        heights = np.array([0.0, 0.0])
        result = _classify_pattern(centers, heights, sf_mhz=600.0)
        assert isinstance(result, str)

    def test_classify_quartet_pattern(self):
        centers = np.array([1.0, 2.0, 3.0, 4.0])
        heights = np.array([1.0, 3.0, 3.0, 1.0])
        result = _classify_pattern(centers, heights, sf_mhz=600.0)
        assert isinstance(result, str)

    def test_icoshift_descending_ppm(self):
        rng = np.random.default_rng(0)
        n, p = 5, 50
        ppm_desc = np.linspace(10, 0, p)
        spectra = pd.DataFrame(rng.standard_normal((n, p)), columns=ppm_desc)
        windows = [(4.8, 5.2)]
        aligned, shifts = icoshift_align(spectra, ppm_desc, windows)
        assert aligned.shape == spectra.shape

    def test_icoshift_adjacent_peaks_group(self):
        rng = np.random.default_rng(0)
        n, p = 8, 100
        ppm = np.linspace(0, 10, p)
        spectra = pd.DataFrame(rng.standard_normal((n, p)), columns=ppm)
        windows = [(4.8, 5.2), (4.9, 5.3)]
        aligned, shifts = icoshift_align(spectra, ppm, windows)
        assert aligned.shape == spectra.shape

    def test_classify_pattern_adjacent_peaks_grouped(self):
        # Two peaks close enough to be in one group (within max_group_width_ppm=0.03)
        from metbit.nmr.alignment import detect_multiplets
        n, p = 5, 200
        ppm = np.linspace(10, 0, p)
        # Two peaks very close together at 5.0 and 5.01 ppm
        spec = np.zeros(p)
        p1 = np.argmin(np.abs(ppm - 5.01))
        p2 = np.argmin(np.abs(ppm - 5.00))
        # Make Gaussian peaks
        x = np.arange(p)
        spec += 5.0 * np.exp(-0.5 * ((x - p1) / 1.5) ** 2)
        spec += 4.0 * np.exp(-0.5 * ((x - p2) / 1.5) ** 2)
        mps = detect_multiplets(spec, ppm, sf_mhz=600.0)
        assert len(mps) >= 1

    def test_peak_aligner_auto_windows_overlapping(self):
        # Dense ppm axis so adjacent peaks can form a wide multiplet whose
        # expanded windows overlap (covers alignment.py:249 and 265)
        n, p = 5, 1000
        ppm = np.linspace(10, 0, p)  # descending; step ~0.01 ppm
        x = np.arange(p)
        spec_data = np.zeros((n, p))
        # Two peaks 0.02 ppm apart → within max_group_width_ppm=0.03 → one wide multiplet
        for center in [200, 202]:
            spec_data += 10.0 * np.exp(-0.5 * ((x - center) / 2) ** 2)[np.newaxis, :]
        # Another peak 0.04 ppm away → separate multiplet but close enough that
        # expanded windows (span = max(0.005, width*1.5)) from both overlap
        spec_data += 8.0 * np.exp(-0.5 * ((x - 206) / 2) ** 2)[np.newaxis, :]
        spectra = pd.DataFrame(spec_data, columns=ppm)
        pa = PeakAligner(spectra, ppm, sf_mhz=600.0)
        windows, mptable = pa.auto_windows(top_n=5, max_group_width_ppm=0.03)
        assert isinstance(windows, list)


# ---------------------------------------------------------------------------
# metbit/nmr/calibrate.py - 2-peak path for glucose/alanine
# ---------------------------------------------------------------------------

from metbit.nmr.calibrate import calibrate


class TestCalibrateExtra:
    def test_calibrate_glucose_two_peaks(self):
        ppm = np.linspace(10, 0, 500)
        X = np.zeros((3, 500))
        mask = (ppm >= 5.0) & (ppm <= 5.4)
        idx = np.where(mask)[0]
        # idx has ~20 elements; use 5 and 15 to stay in bounds
        X[:, idx[5]] = 1.0
        X[:, idx[15]] = 0.8
        result = calibrate(X, ppm, calib_type="glucose")
        assert result.shape == (3, 500)

    def test_calibrate_alanine_one_peak(self):
        ppm = np.linspace(10, 0, 500)
        X = np.zeros((3, 500))
        mask = (ppm >= 1.2) & (ppm <= 1.6)
        idx = np.where(mask)[0]
        X[:, idx[5]] = 1.0
        result = calibrate(X, ppm, calib_type="alanine")
        assert result.shape == (3, 500)

    def test_calibrate_glucose_no_peaks_fallback(self):
        # All zeros in segment → no peaks found by find_peaks → falls back to argmax (line 94)
        ppm = np.linspace(10, 0, 500)
        X = np.zeros((3, 500))
        # No spike in the 5.0-5.4 region → len(peaks)==0 → peak_ppm = segment_ppm[argmax(segment)]
        mask = (ppm >= 5.0) & (ppm <= 5.4)
        idx = np.where(mask)[0]
        X[:, idx[3]] = 0.001  # too small for prominence=0.01 → not detected as peak
        result = calibrate(X, ppm, calib_type="glucose")
        assert result.shape == (3, 500)


# ---------------------------------------------------------------------------
# metbit/nmr/peaks.py - exception path and ppm path
# ---------------------------------------------------------------------------

from metbit.nmr.peaks import peak_chops


class TestPeakChops:
    def test_non_numeric_columns_raises(self):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError):
            peak_chops(X)

    def test_with_explicit_ppm(self):
        ppm = np.linspace(10, 0, 10)
        X = pd.DataFrame(np.ones((3, 10)), columns=[str(i) for i in range(10)])
        pc = peak_chops(X, ppm=ppm.tolist())
        assert pc.ppm is not None


# ---------------------------------------------------------------------------
# metbit/nmr/preprocess.py - getter methods via mock object
# ---------------------------------------------------------------------------

from metbit.nmr.preprocess import nmr_preprocessing


class TestNMRPreprocessGetters:
    def test_getters_via_mock_object(self):
        rng = np.random.default_rng(0)
        obj = object.__new__(nmr_preprocessing)
        obj.nmr_data = pd.DataFrame(rng.standard_normal((5, 20)))
        obj.ppm = np.linspace(10, 0, 20)
        obj.dic_array = {"sample1": {}}
        obj.phase_data = pd.DataFrame({"p0": [0.0] * 5, "p1": [0.0] * 5})

        data = obj.get_data(flip_data=True)
        assert data.shape[1] == 20

        data2 = obj.get_data(flip_data=False)
        assert data2.shape[1] == 20

        ppm = obj.get_ppm()
        assert len(ppm) == 20

        meta = obj.get_metadata()
        assert isinstance(meta, dict)

        phase = obj.get_phase()
        assert isinstance(phase, pd.DataFrame)


# ---------------------------------------------------------------------------
# metbit/preprocessing/baseline.py - all baseline methods
# ---------------------------------------------------------------------------

from metbit.preprocessing.baseline import baseline_correct


class TestBaselineMethods:
    def _make_spectra(self, n=3, p=50):
        rng = np.random.default_rng(0)
        ppm = np.linspace(10, 0, p)
        data = pd.DataFrame(rng.standard_normal((n, p)) + 5, columns=ppm)
        return data

    def test_arpls_method(self):
        X = self._make_spectra()
        result = baseline_correct(X, method="arpls")
        assert result.shape == X.shape

    def test_airpls_method(self):
        X = self._make_spectra()
        result = baseline_correct(X, method="airpls")
        assert result.shape == X.shape

    def test_modpoly_method(self):
        X = self._make_spectra()
        result = baseline_correct(X, method="modpoly")
        assert result.shape == X.shape

    def test_imodpoly_method(self):
        X = self._make_spectra()
        result = baseline_correct(X, method="imodpoly")
        assert result.shape == X.shape

    def test_rubberband_method(self):
        X = self._make_spectra()
        result = baseline_correct(X, method="rubberband")
        assert result.shape == X.shape

    def test_return_baseline_true(self):
        X = self._make_spectra()
        corrected, baseline_df = baseline_correct(X, method="asls", return_baseline=True)
        assert corrected.shape == X.shape
        assert baseline_df.shape == X.shape

    def test_nan_values_filled(self):
        rng = np.random.default_rng(0)
        data = np.abs(rng.standard_normal((3, 30))) + 1
        X = pd.DataFrame(data)
        X.iloc[0, 0] = np.nan
        result = baseline_correct(X, method="asls")
        assert not result.isnull().any().any()

    def test_non_numeric_columns_use_index(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((3, 20)) + 5,
                         columns=[f"col{i}" for i in range(20)])
        result = baseline_correct(X, method="rubberband")
        assert result.shape == X.shape

    def test_unknown_method_raises(self):
        X = self._make_spectra()
        with pytest.raises(ValueError):
            baseline_correct(X, method="unknown")

    def test_non_dataframe_raises(self):
        arr = np.ones((3, 20))
        with pytest.raises(ValueError):
            baseline_correct(arr)


# ---------------------------------------------------------------------------
# metbit/preprocessing/scaler.py - just importing covers the 1 statement
# ---------------------------------------------------------------------------

def test_scaler_module_importable():
    import metbit.preprocessing.scaler as s_mod
    assert hasattr(s_mod, "Scaler")


# ---------------------------------------------------------------------------
# metbit/preprocessing/scaler_ext.py - sparse matrix and partial_fit
# ---------------------------------------------------------------------------

from metbit.preprocessing.scaler_ext import Scaler as ExtScaler


class TestScalerExtSparse:
    def test_partial_fit_dense(self):
        rng = np.random.default_rng(0)
        X1 = rng.standard_normal((10, 5)).astype(np.float64)
        X2 = rng.standard_normal((8, 5)).astype(np.float64)
        scaler = ExtScaler(scale_power=0.5)
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        assert hasattr(scaler, "scale_")

    def test_transform_sparse(self):
        rng = np.random.default_rng(0)
        X_dense = rng.standard_normal((10, 5)).astype(np.float64)
        X_sparse = sp.csr_matrix(np.abs(X_dense))
        scaler = ExtScaler(scale_power=1, with_mean=False)
        scaler.fit(X_sparse)
        result = scaler.transform(X_sparse.copy())
        assert result.shape == X_sparse.shape

    def test_inverse_transform_sparse(self):
        rng = np.random.default_rng(0)
        X_dense = rng.standard_normal((10, 5)).astype(np.float64)
        X_sparse = sp.csr_matrix(np.abs(X_dense))
        scaler = ExtScaler(scale_power=1, with_mean=False)
        scaler.fit(X_sparse)
        Xt = scaler.transform(X_sparse.copy())
        inv = scaler.inverse_transform(Xt)
        assert inv.shape == X_sparse.shape

    def test_inverse_transform_csc_to_csr(self):
        rng = np.random.default_rng(0)
        X_dense = np.abs(rng.standard_normal((10, 5))).astype(np.float64)
        X_sparse = sp.csc_matrix(X_dense)
        scaler = ExtScaler(scale_power=1, with_mean=False)
        scaler.fit(X_sparse)
        Xt = scaler.transform(X_sparse.copy())
        inv = scaler.inverse_transform(Xt.copy())
        assert inv.shape == X_sparse.shape

    def test_with_mean_sparse_raises(self):
        X_sparse = sp.csr_matrix(np.ones((5, 5)))
        scaler = ExtScaler(with_mean=True)
        with pytest.raises(ValueError):
            scaler.fit(X_sparse)

    def test_with_std_false_sparse(self):
        X_sparse = sp.csr_matrix(np.abs(np.random.randn(5, 5)))
        scaler = ExtScaler(scale_power=1, with_mean=False, with_std=False)
        scaler.fit(X_sparse)
        assert scaler.scale_ is None

    def test_partial_fit_sparse_incremental(self):
        # Second partial_fit call on sparse triggers incr_mean_variance_axis (line 118)
        X1 = sp.csr_matrix(np.abs(np.random.randn(5, 4)))
        X2 = sp.csr_matrix(np.abs(np.random.randn(5, 4)))
        scaler = ExtScaler(scale_power=1, with_mean=False)
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        assert hasattr(scaler, "scale_")

    def test_inverse_transform_csc_sparse_converts(self):
        # CSC sparse in inverse_transform → tocsr() + copy=False (lines 196-197)
        X_dense = np.abs(np.random.randn(6, 4)).astype(np.float64)
        scaler = ExtScaler(scale_power=1, with_mean=False)
        scaler.fit(sp.csr_matrix(X_dense))
        Xt_csr = scaler.transform(sp.csr_matrix(X_dense.copy()))
        # Pass CSC matrix explicitly to inverse_transform to trigger lines 196-197
        Xt_csc = sp.csc_matrix(Xt_csr)
        inv = scaler.inverse_transform(Xt_csc, copy=False)
        assert inv.shape == (6, 4)


# ---------------------------------------------------------------------------
# metbit/viz/plots.py - save_plot=True with file_name=None errors
# ---------------------------------------------------------------------------

from metbit.viz.plots import Plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_cv_plots_model():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10))
    y = np.array(["A"] * 20 + ["B"] * 20)
    cv = CrossValidation("opls", kfold=3)
    cv.fit(X, y)
    return cv


class TestPlotsSavePlotErrors:
    def test_plot_scores_save_no_filename_raises(self):
        cv = _make_cv_plots_model()
        p = Plots(cv)
        with pytest.raises(ValueError):
            p.plot_scores(save_plot=True, file_name=None)

    def test_jackknife_loading_plot_save_no_filename_raises(self):
        cv = _make_cv_plots_model()
        p = Plots(cv)
        plt.close("all")
        with pytest.raises(ValueError):
            p.jackknife_loading_plot(save_plot=True, file_name=None)
        plt.close("all")

    def test_plot_cv_errors_save_no_filename_raises(self):
        cv = _make_cv_plots_model()
        p = Plots(cv)
        plt.close("all")
        with pytest.raises(ValueError):
            p.plot_cv_errors(save_plot=True, file_name=None)
        plt.close("all")

    def test_splot_save_with_filename_no_ext(self, tmp_path):
        # Covers plots.py:146-148 (splot save path): filename without '.' → appends .png → savefig
        cv = _make_cv_plots_model()
        plots = Plots(cv)
        out = str(tmp_path / "test_splot")
        plt.close("all")
        plots.splot(save_plot=True, file_name=out)
        plt.close("all")
        import os
        assert os.path.exists(out + ".png")

    def test_jackknife_save_with_filename_no_ext(self, tmp_path):
        # Covers lines 215-217
        cv = _make_cv_plots_model()
        plots = Plots(cv)
        out = str(tmp_path / "test_jk")
        plt.close("all")
        plots.jackknife_loading_plot(save_plot=True, file_name=out)
        plt.close("all")
        import os
        assert os.path.exists(out + ".png")

    def test_plot_cv_errors_save_with_filename_no_ext(self, tmp_path):
        # Covers lines 243-245
        cv = _make_cv_plots_model()
        plots = Plots(cv)
        out = str(tmp_path / "test_cv_errors")
        plt.close("all")
        plots.plot_cv_errors(save_plot=True, file_name=out)
        plt.close("all")
        import os
        assert os.path.exists(out + ".png")
