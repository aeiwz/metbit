import numpy as np
import pytest
from scipy.stats import pearsonr

from metbit import _native


def _reference_correlations(matrix, anchor_index):
    return np.array(
        [pearsonr(matrix[:, anchor_index], matrix[:, column]).statistic
         for column in range(matrix.shape[1])]
    )


def test_pearson_columns_matches_scipy():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(37, 211))

    actual = _native.pearson_columns(matrix, anchor_index=17)
    expected = _reference_correlations(matrix, anchor_index=17)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_pearson_columns_accepts_float32_and_non_contiguous_input():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(20, 30)).astype(np.float32)[:, ::2]

    actual = _native.pearson_columns(matrix, anchor_index=3)
    expected = _reference_correlations(matrix, anchor_index=3)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_pearson_columns_marks_constant_columns_nan():
    matrix = np.arange(60, dtype=float).reshape(12, 5)
    matrix[:, 2] = 4.0

    correlations = _native.pearson_columns(matrix, anchor_index=0)

    assert np.isnan(correlations[2])
    assert correlations[0] == pytest.approx(1.0)


def test_pearson_columns_fallback_matches_native(monkeypatch):
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(25, 80))
    native_result = _native.pearson_columns(matrix, anchor_index=9)

    monkeypatch.setattr(_native, "_native_backend", None)
    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    fallback_result = _native.pearson_columns(matrix, anchor_index=9)

    np.testing.assert_allclose(
        fallback_result, native_result, rtol=1e-12, atol=1e-12
    )


def test_pearson_columns_multiprocessing_fallback(monkeypatch):
    rng = np.random.default_rng(92)
    matrix = rng.normal(size=(12, 9))

    monkeypatch.setattr(_native, "_native_backend", None)
    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_SMALL_THRESH", 1)

    actual = _native.pearson_columns(
        matrix, anchor_index=4, chunk_size=3, n_jobs=2
    )
    expected = _reference_correlations(matrix, anchor_index=4)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_column_variances_fallback_paths(monkeypatch):
    rng = np.random.default_rng(93)
    matrix = rng.normal(size=(12, 9))
    expected = matrix.var(axis=0, ddof=1)

    monkeypatch.setattr(_native, "_native_backend", None)
    monkeypatch.setattr(_native, "_NATIVE_OK", False)

    actual_numpy = _native.column_variances(matrix, chunk_size=4, n_jobs=1)
    np.testing.assert_allclose(actual_numpy, expected, rtol=1e-12, atol=1e-12)

    monkeypatch.setattr(_native, "_SMALL_THRESH", 1)
    actual_mp = _native.column_variances(matrix, chunk_size=4, n_jobs=2)
    np.testing.assert_allclose(actual_mp, expected, rtol=1e-12, atol=1e-12)


def test_vip_scores_zero_signal_returns_zeros(monkeypatch):
    monkeypatch.setattr(_native, "_native_backend", None)
    monkeypatch.setattr(_native, "_NATIVE_OK", False)

    actual = _native.vip_scores(
        np.zeros((6, 2)),
        np.ones((5, 2)),
        np.array([0.0, 0.0]),
    )

    np.testing.assert_array_equal(actual, np.zeros(5))


@pytest.mark.parametrize(
    ("matrix", "anchor_index", "error"),
    [
        (np.ones(5), 0, ValueError),
        (np.ones((1, 5)), 0, ValueError),
        (np.ones((5, 0)), 0, ValueError),
        (np.ones((5, 3)), -1, IndexError),
        (np.ones((5, 3)), 3, IndexError),
    ],
)
def test_pearson_columns_rejects_invalid_inputs(matrix, anchor_index, error):
    with pytest.raises(error):
        _native.pearson_columns(matrix, anchor_index)
