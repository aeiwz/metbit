import numpy as np
import pandas as pd
import pytest

from metbit.calibrate import calibrate


def _spectrum(ppm, peak_at, amplitude=10.0, width=0.05, n_samples=4):
    """Create a synthetic NMR-like DataFrame with a Gaussian peak at peak_at ppm."""
    rng = np.random.default_rng(0)
    intensities = amplitude * np.exp(-0.5 * ((ppm - peak_at) / width) ** 2)
    noise = rng.normal(0, 0.05, (n_samples, len(ppm)))
    data = intensities[np.newaxis, :] + noise
    return pd.DataFrame(data, columns=ppm)


class TestCalibrateKnownTypes:
    def test_tsp_returns_dataframe_same_shape(self):
        ppm = np.linspace(-0.3, 0.3, 200)
        X = _spectrum(ppm, peak_at=0.05)
        result = calibrate(X, ppm, calib_type="tsp")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    def test_acetate_returns_same_shape(self):
        ppm = np.linspace(1.7, 2.3, 200)
        X = _spectrum(ppm, peak_at=1.95)
        result = calibrate(X, ppm, calib_type="acetate")
        assert result.shape == X.shape

    def test_formate_returns_same_shape(self):
        ppm = np.linspace(7.9, 8.5, 200)
        X = _spectrum(ppm, peak_at=8.45)
        result = calibrate(X, ppm, calib_type="formate")
        assert result.shape == X.shape

    def test_glucose_multi_peak_path(self):
        ppm = np.linspace(4.9, 5.5, 300)
        # Create two peaks so the top-2 branch executes
        intensities = (
            10 * np.exp(-0.5 * ((ppm - 5.1) / 0.03) ** 2)
            + 8 * np.exp(-0.5 * ((ppm - 5.3) / 0.03) ** 2)
        )
        X = pd.DataFrame(intensities[np.newaxis, :], columns=ppm)
        result = calibrate(X, ppm, calib_type="glucose")
        assert result.shape == X.shape

    def test_alanine_returns_same_shape(self):
        ppm = np.linspace(1.1, 1.7, 200)
        X = _spectrum(ppm, peak_at=1.5)
        result = calibrate(X, ppm, calib_type="alanine")
        assert result.shape == X.shape


class TestCalibrateCustom:
    def test_custom_with_range_and_target(self):
        ppm = np.linspace(0.8, 1.3, 200)
        X = _spectrum(ppm, peak_at=0.95)
        result = calibrate(X, ppm, calib_type="custom",
                           custom_range=(0.85, 1.1), custom_target=0.91)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    def test_custom_missing_range_raises(self):
        ppm = np.linspace(0.0, 1.0, 100)
        X = _spectrum(ppm, peak_at=0.5)
        with pytest.raises(ValueError, match="custom_range"):
            calibrate(X, ppm, calib_type="custom", custom_target=0.5)

    def test_custom_missing_target_raises(self):
        ppm = np.linspace(0.0, 1.0, 100)
        X = _spectrum(ppm, peak_at=0.5)
        with pytest.raises(ValueError, match="custom_range"):
            calibrate(X, ppm, calib_type="custom", custom_range=(0.3, 0.7))

    def test_backward_compat_unknown_type_with_range_and_target(self):
        # The elif branch fires when calib_type is unknown but both custom args given
        ppm = np.linspace(0.5, 1.5, 200)
        X = _spectrum(ppm, peak_at=1.0)
        result = calibrate(X, ppm, calib_type="unknown_type",
                           custom_range=(0.7, 1.3), custom_target=1.0)
        assert result.shape == X.shape

    def test_fallback_range_only_uses_mean_as_target(self):
        # The final elif fires when calib_type unknown and only custom_range provided
        ppm = np.linspace(0.5, 1.5, 200)
        X = _spectrum(ppm, peak_at=1.0)
        result = calibrate(X, ppm, calib_type="unknown_type", custom_range=(0.7, 1.3))
        assert result.shape == X.shape


class TestCalibrateErrors:
    def test_unknown_type_without_custom_range_raises(self):
        ppm = np.linspace(0.0, 1.0, 100)
        X = _spectrum(ppm, peak_at=0.5)
        with pytest.raises(ValueError, match="Invalid calibration"):
            calibrate(X, ppm, calib_type="unknown_type")

    def test_ppm_out_of_range_raises(self):
        ppm = np.linspace(5.0, 6.0, 100)
        X = _spectrum(ppm, peak_at=5.5)
        with pytest.raises(ValueError, match="No ppm values"):
            calibrate(X, ppm, calib_type="tsp")  # tsp range is -0.2 to 0.2

    def test_numpy_array_input_accepted(self):
        ppm = np.linspace(-0.3, 0.3, 200)
        X = _spectrum(ppm, peak_at=0.05)
        result = calibrate(X.to_numpy(), ppm, calib_type="tsp")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    def test_preserves_index(self):
        ppm = np.linspace(-0.3, 0.3, 200)
        X = _spectrum(ppm, peak_at=0.05)
        X.index = ["s1", "s2", "s3", "s4"]
        result = calibrate(X, ppm, calib_type="tsp")
        assert list(result.index) == ["s1", "s2", "s3", "s4"]
