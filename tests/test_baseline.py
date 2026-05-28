import numpy as np
import pandas as pd
import pytest

from metbit.preprocessing.baseline import (
    _rubberband_baseline,
    _apply_baseline_1d,
    baseline_correct,
    bline,
)


def _make_spectra(n=4, p=50, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, p)
    drift = np.sin(x * 0.5) * 2
    signal = rng.standard_normal((n, p)) * 0.3 + drift
    return pd.DataFrame(signal, columns=x)


class TestRubberbandBaseline:
    def test_returns_same_length(self):
        y = np.array([1.0, 0.5, 0.2, 0.1, 0.3, 0.8, 1.5])
        bl = _rubberband_baseline(y)
        assert len(bl) == len(y)

    def test_baseline_le_signal(self):
        # Rubberband is a lower convex hull - should be <= signal
        y = np.array([3.0, 1.0, 2.0, 0.5, 2.5, 1.0, 3.0])
        bl = _rubberband_baseline(y)
        assert np.all(bl <= y + 1e-10)

    def test_custom_x_axis(self):
        y = np.array([1.0, 2.0, 1.5, 0.5, 2.0])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        bl = _rubberband_baseline(y, x=x)
        assert len(bl) == len(y)


class TestApplyBaseline1d:
    def test_rubberband_returns_array(self):
        y = np.linspace(0, 1, 50) + np.sin(np.linspace(0, np.pi, 50))
        bl = _apply_baseline_1d(y, method="rubberband")
        assert bl.shape == y.shape

    def test_unknown_method_raises(self):
        y = np.ones(20)
        with pytest.raises(ValueError, match="Unknown baseline method"):
            _apply_baseline_1d(y, method="bogus")

    def test_asls_method(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 100)) + np.linspace(0, 1, 100)
        bl = _apply_baseline_1d(y, method="asls")
        assert bl.shape == y.shape

    def test_arpls_method(self):
        y = np.sin(np.linspace(0, 2 * np.pi, 100)) + np.linspace(0, 1, 100)
        bl = _apply_baseline_1d(y, method="arpls")
        assert bl.shape == y.shape


class TestBaselineCorrect:
    def test_returns_dataframe_same_shape(self):
        X = _make_spectra()
        result = baseline_correct(X, method="rubberband")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    def test_preserves_index_and_columns(self):
        X = _make_spectra()
        result = baseline_correct(X, method="rubberband")
        pd.testing.assert_index_equal(result.index, X.index)
        pd.testing.assert_index_equal(result.columns, X.columns)

    def test_non_dataframe_raises(self):
        with pytest.raises(ValueError, match="pandas DataFrame"):
            baseline_correct(np.ones((3, 10)), method="rubberband")

    def test_return_baseline_flag(self):
        X = _make_spectra()
        corrected, baselines = baseline_correct(X, method="rubberband", return_baseline=True)
        assert isinstance(baselines, pd.DataFrame)
        assert baselines.shape == X.shape
        # corrected = X - baseline
        np.testing.assert_allclose(
            corrected.values, (X.values - baselines.values), atol=1e-12
        )

    def test_nan_rows_filled_before_correction(self):
        X = _make_spectra()
        X.iloc[0, 5] = np.nan
        result = baseline_correct(X, method="rubberband")
        assert not result.isnull().values.any()

    def test_explicit_x_axis(self):
        X = _make_spectra()
        x = np.linspace(0, 1, X.shape[1])
        result = baseline_correct(X, method="rubberband", x=x)
        assert result.shape == X.shape

    def test_asls_reduces_drift(self):
        p = 100
        x = np.linspace(0, 10, p)
        drift = np.linspace(0, 5, p)
        signal = np.zeros((3, p)) + drift
        X = pd.DataFrame(signal, columns=x)
        result = baseline_correct(X, method="asls")
        assert result.abs().mean().mean() < drift.mean()


class TestBline:
    def test_bline_is_asls_wrapper(self):
        X = _make_spectra()
        r1 = bline(X)
        r2 = baseline_correct(X, method="asls")
        pd.testing.assert_frame_equal(r1, r2)
