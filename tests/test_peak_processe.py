import numpy as np
import pandas as pd
import pytest

from metbit.nmr.peaks import peak_chops


def _make_df(n_samples=5, ppm_start=0.5, ppm_end=9.5, n_points=100):
    rng = np.random.default_rng(7)
    ppm = np.linspace(ppm_start, ppm_end, n_points)
    data = rng.standard_normal((n_samples, n_points))
    return pd.DataFrame(data, columns=ppm.tolist())


class TestPeakChopsInit:
    def test_infers_ppm_from_numeric_columns(self):
        df = _make_df()
        pc = peak_chops(df)
        assert pc.ppm is not None

    def test_explicit_ppm_accepted(self):
        df = _make_df()
        ppm = np.linspace(0.5, 9.5, 100).tolist()
        pc = peak_chops(df, ppm=ppm)
        assert pc.ppm is not None

    def test_stores_data(self):
        df = _make_df()
        pc = peak_chops(df)
        assert pc.data is df


class TestCutPeak:
    def test_removes_columns_in_range(self):
        df = _make_df()
        original_cols = df.shape[1]
        pc = peak_chops(df.copy())
        result, new_ppm = pc.cut_peak(2.0, 3.0)
        assert result.shape[1] < original_cols

    def test_returns_updated_ppm_list(self):
        df = _make_df()
        pc = peak_chops(df.copy())
        result, new_ppm = pc.cut_peak(2.0, 3.0)
        assert isinstance(new_ppm, list)
        assert len(new_ppm) == result.shape[1]

    def test_cut_range_absent_from_result(self):
        df = _make_df()
        pc = peak_chops(df.copy())
        result, new_ppm = pc.cut_peak(4.0, 5.0)
        remaining = np.array(new_ppm, dtype=float)
        assert not np.any((remaining >= 4.0) & (remaining <= 5.0))
