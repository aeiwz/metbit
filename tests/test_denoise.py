import numpy as np
import pandas as pd
import pytest

from metbit.nmr.denoise import Denoise


def _spectra_array(n=3, p=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def _spectra_df(n=3, p=50, seed=0):
    return pd.DataFrame(_spectra_array(n, p, seed))


class TestDenoiseDecreaseNoise:
    def test_numpy_input_returns_numpy(self):
        X = _spectra_array()
        result = Denoise.decrease_noise(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == X.shape

    def test_dataframe_input_returns_dataframe(self):
        # window_length=11 needs >= 11 rows when applied column-wise
        X = _spectra_df(n=20, p=5)
        result = Denoise.decrease_noise(X, window_length=11, polyorder=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    def test_filtering_reduces_high_frequency_noise(self):
        rng = np.random.default_rng(42)
        n, p = 5, 200
        signal = np.tile(np.sin(np.linspace(0, 2 * np.pi, p)), (n, 1))
        noise = rng.standard_normal((n, p)) * 2.0
        noisy = signal + noise
        filtered = Denoise.decrease_noise(noisy, window_length=21, polyorder=3)
        # Filtered output should be closer to original signal than noisy input
        assert np.abs(filtered - signal).mean() < np.abs(noisy - signal).mean()

    def test_invalid_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid data type"):
            Denoise.decrease_noise([[1, 2, 3], [4, 5, 6]])

    def test_custom_window_and_polyorder(self):
        X = _spectra_array()
        result = Denoise.decrease_noise(X, window_length=7, polyorder=3)
        assert result.shape == X.shape

    def test_output_dtype_is_float(self):
        X = _spectra_array()
        result = Denoise.decrease_noise(X)
        assert result.dtype.kind == "f"
