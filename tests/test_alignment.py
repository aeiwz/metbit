import numpy as np
import pandas as pd
import pytest

from metbit.alignment import (
    _binomial_ratios,
    _classify_pattern,
    detect_multiplets,
    icoshift_align,
    PeakAligner,
    Multiplet,
)


def _synthetic_spectrum(ppm, peaks_at, amplitudes=None, width=0.02):
    """Build a synthetic 1D spectrum with Gaussian peaks."""
    y = np.zeros(len(ppm))
    if amplitudes is None:
        amplitudes = [1.0] * len(peaks_at)
    for center, amp in zip(peaks_at, amplitudes):
        y += amp * np.exp(-0.5 * ((ppm - center) / width) ** 2)
    return y


def _synthetic_spectra(n=10, p=200, seed=0):
    """Build n spectra each with a prominent peak near ppm=2.0 that can shift slightly."""
    rng = np.random.default_rng(seed)
    ppm = np.linspace(0.5, 10.0, p)
    peak_shifts = rng.uniform(-0.05, 0.05, n)
    rows = []
    for shift in peak_shifts:
        rows.append(_synthetic_spectrum(ppm, [2.0 + shift, 5.0 + shift * 0.5],
                                        amplitudes=[50.0, 30.0], width=0.1))
    return pd.DataFrame(rows, columns=ppm)


class TestBinomialRatios:
    def test_singlet_returns_one(self):
        r = _binomial_ratios(1)
        assert r.shape == (1,)
        assert r[0] == pytest.approx(1.0)

    def test_doublet_returns_equal_ratios(self):
        r = _binomial_ratios(2)
        np.testing.assert_allclose(r, [1.0, 1.0])

    def test_triplet_is_normalized(self):
        r = _binomial_ratios(3)
        assert r.max() == pytest.approx(1.0)
        assert r[1] > r[0]

    def test_quartet_peaks_at_both_ends_equal(self):
        r = _binomial_ratios(4)
        assert r[0] == pytest.approx(r[-1])
        assert r.max() == pytest.approx(1.0)


class TestClassifyPattern:
    def test_empty_returns_unknown(self):
        result = _classify_pattern(np.array([]), np.array([]), sf_mhz=600)
        assert result == "unknown"

    def test_single_peak_is_singlet(self):
        result = _classify_pattern(np.array([2.0]), np.array([1.0]), sf_mhz=600)
        assert result == "singlet"

    def test_two_equal_peaks_is_doublet(self):
        centers = np.array([1.95, 2.05])
        heights = np.array([1.0, 1.0])
        result = _classify_pattern(centers, heights, sf_mhz=600)
        assert result == "doublet"

    def test_irregular_n_peaks_is_multiplet(self):
        centers = np.array([1.0, 2.0, 4.0, 7.0, 8.0])
        heights = np.array([1.0, 0.5, 0.8, 0.3, 0.9])
        result = _classify_pattern(centers, heights, sf_mhz=600)
        assert result == "multiplet"

    def test_triplet_pattern(self):
        centers = np.array([1.95, 2.00, 2.05])
        heights = np.array([0.5, 1.0, 0.5])
        result = _classify_pattern(centers, heights, sf_mhz=600)
        assert result == "triplet"


class TestDetectMultiplets:
    def test_no_peaks_returns_empty(self):
        ppm = np.linspace(0, 10, 200)
        spectrum = pd.Series(np.zeros(200))
        result = detect_multiplets(spectrum, ppm, sf_mhz=600)
        assert result == []

    def test_single_peak_returns_one_multiplet(self):
        ppm = np.linspace(0.5, 10.0, 500)
        y = _synthetic_spectrum(ppm, [5.0], amplitudes=[10.0], width=0.05)
        spectrum = pd.Series(y)
        result = detect_multiplets(spectrum, ppm, sf_mhz=600, prominence=0.5, width=2)
        assert len(result) >= 1

    def test_multiplet_dataclass_fields(self):
        ppm = np.linspace(0.5, 10.0, 500)
        y = _synthetic_spectrum(ppm, [3.0], amplitudes=[10.0], width=0.05)
        spectrum = pd.Series(y)
        result = detect_multiplets(spectrum, ppm, sf_mhz=600, prominence=0.5, width=2)
        if result:
            mp = result[0]
            assert hasattr(mp, "start_ppm")
            assert hasattr(mp, "end_ppm")
            assert hasattr(mp, "center_ppm")
            assert hasattr(mp, "n_peaks")
            assert hasattr(mp, "pattern")

    def test_two_distant_peaks_gives_two_multiplets(self):
        ppm = np.linspace(0.5, 10.0, 500)
        y = _synthetic_spectrum(ppm, [2.0, 8.0], amplitudes=[10.0, 10.0], width=0.05)
        spectrum = pd.Series(y)
        result = detect_multiplets(spectrum, ppm, sf_mhz=600, prominence=0.5, width=2,
                                   max_group_width_ppm=0.1)
        assert len(result) == 2

    def test_no_smoothing_still_works(self):
        ppm = np.linspace(0.5, 10.0, 200)
        y = _synthetic_spectrum(ppm, [5.0], amplitudes=[5.0], width=0.1)
        spectrum = pd.Series(y)
        result = detect_multiplets(spectrum, ppm, sf_mhz=600, smooth_window=0, prominence=0.1, width=2)
        assert isinstance(result, list)


class TestIcoshiftAlign:
    def test_returns_dataframe_same_shape(self):
        spectra = _synthetic_spectra()
        ppm = spectra.columns.to_numpy(dtype=float)
        windows = [(1.8, 2.2)]
        aligned, shifts = icoshift_align(spectra, ppm, windows)
        assert isinstance(aligned, pd.DataFrame)
        assert aligned.shape == spectra.shape

    def test_shifts_dict_has_all_samples(self):
        spectra = _synthetic_spectra()
        ppm = spectra.columns.to_numpy(dtype=float)
        windows = [(1.8, 2.2)]
        _, shifts = icoshift_align(spectra, ppm, windows)
        assert len(shifts) == spectra.shape[0]

    def test_mean_reference(self):
        spectra = _synthetic_spectra()
        ppm = spectra.columns.to_numpy(dtype=float)
        windows = [(1.8, 2.2)]
        aligned, _ = icoshift_align(spectra, ppm, windows, reference="mean")
        assert aligned.shape == spectra.shape

    def test_narrow_window_skipped_gracefully(self):
        spectra = _synthetic_spectra()
        ppm = spectra.columns.to_numpy(dtype=float)
        # window spanning only 1-2 ppm points - should be skipped, no crash
        windows = [(2.0, 2.001)]
        aligned, shifts = icoshift_align(spectra, ppm, windows)
        assert aligned.shape == spectra.shape

    def test_descending_ppm_axis(self):
        spectra = _synthetic_spectra()
        ppm_asc = spectra.columns.to_numpy(dtype=float)
        ppm_desc = ppm_asc[::-1]
        # reverse column order of spectra
        spectra_rev = spectra.iloc[:, ::-1].copy()
        spectra_rev.columns = ppm_desc
        windows = [(2.2, 1.8)]  # reversed window for descending axis
        aligned, _ = icoshift_align(spectra_rev, ppm_desc, windows)
        assert aligned.shape == spectra_rev.shape

    def test_no_windows_returns_unchanged(self):
        spectra = _synthetic_spectra()
        ppm = spectra.columns.to_numpy(dtype=float)
        aligned, shifts = icoshift_align(spectra, ppm, [])
        np.testing.assert_array_equal(aligned.values, spectra.values)


class TestPeakAligner:
    def test_auto_windows_returns_list_and_dataframe(self):
        spectra = _synthetic_spectra(n=15)
        ppm = spectra.columns.to_numpy(dtype=float)
        pa = PeakAligner(spectra, ppm, sf_mhz=600)
        windows, df = pa.auto_windows(top_n=5)
        assert isinstance(windows, list)
        assert isinstance(df, pd.DataFrame)

    def test_align_returns_dataframe_and_shifts(self):
        spectra = _synthetic_spectra(n=10)
        ppm = spectra.columns.to_numpy(dtype=float)
        pa = PeakAligner(spectra, ppm, sf_mhz=600)
        windows = [(1.8, 2.2), (4.8, 5.2)]
        aligned, shifts = pa.align(windows)
        assert isinstance(aligned, pd.DataFrame)
        assert aligned.shape == spectra.shape

    def test_align_preserves_index(self):
        spectra = _synthetic_spectra(n=8)
        ppm = spectra.columns.to_numpy(dtype=float)
        pa = PeakAligner(spectra, ppm, sf_mhz=600)
        windows = [(1.8, 2.2)]
        aligned, _ = pa.align(windows)
        pd.testing.assert_index_equal(aligned.index, spectra.index)

    def test_auto_windows_then_align_pipeline(self):
        spectra = _synthetic_spectra(n=12)
        ppm = spectra.columns.to_numpy(dtype=float)
        pa = PeakAligner(spectra, ppm, sf_mhz=600)
        windows, _ = pa.auto_windows(top_n=10)
        if windows:
            aligned, shifts = pa.align(windows)
            assert aligned.shape == spectra.shape
