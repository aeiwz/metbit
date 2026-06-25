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


# ---------------------------------------------------------------------------
# New kernels – fallback (numpy) paths
# ---------------------------------------------------------------------------

def test_nipals_fallback_matches_native(monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 20))
    y = rng.standard_normal(30)

    native_w, native_u, native_c, native_t = _native.nipals(X, y)

    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    fb_w, fb_u, fb_c, fb_t = _native.nipals(X, y)

    np.testing.assert_allclose(fb_w, native_w, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fb_t, native_t, rtol=1e-8, atol=1e-10)
    assert abs(fb_c - native_c) < 1e-8


def test_scale_transform_fallback_matches_native(monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((25, 40))
    mean = X.mean(axis=0)
    s = X.std(axis=0)

    native_out = _native.scale_transform(X, mean, s)

    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    fb_out = _native.scale_transform(X, mean, s)
    np.testing.assert_allclose(fb_out, native_out, rtol=1e-12, atol=1e-14)


def test_scale_transform_fallback_1d(monkeypatch):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 10))
    mean = X.mean(axis=0)
    s = np.sqrt(X.std(axis=0))

    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    row = X[0]
    out = _native.scale_transform(row, mean, s)
    assert out.shape == (10,)


def test_xcorr_max_shift_fallback_matches_native(monkeypatch):
    rng = np.random.default_rng(9)
    template = rng.standard_normal(100)
    query = np.roll(template, 3) + rng.standard_normal(100) * 0.1

    native_shift, native_corr = _native.xcorr_max_shift(template, query, max_shift=10)

    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    fb_shift, fb_corr = _native.xcorr_max_shift(template, query, max_shift=10)
    assert fb_shift == native_shift
    assert abs(fb_corr - native_corr) < 1e-10


def test_pqn_median_quotient_fallback_matches_native(monkeypatch):
    rng = np.random.default_rng(5)
    sample = rng.uniform(0.5, 2.0, 50)
    reference = rng.uniform(0.5, 2.0, 50)

    native_q = _native.pqn_median_quotient(sample, reference)

    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    fb_q = _native.pqn_median_quotient(sample, reference)
    assert abs(fb_q - native_q) < 1e-12


def test_pqn_median_quotient_zero_reference(monkeypatch):
    """Returns 1.0 when all reference values are zero."""
    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    sample = np.ones(10)
    reference = np.zeros(10)
    assert _native.pqn_median_quotient(sample, reference) == 1.0


def test_nipals_fallback_degenerate_cases(monkeypatch):
    """Exercise break conditions: zero u, zero w, zero t."""
    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    # Zero y → u = 0 → utu = 0 → break on first iteration
    X = np.eye(5)
    y_zero = np.zeros(5)
    _native.nipals(X, y_zero, tol=1e-10, max_iter=100)

    # Tight tolerance → converges quickly (exercises normal break path)
    rng = np.random.default_rng(0)
    X2 = rng.standard_normal((20, 10))
    y2 = rng.standard_normal(20)
    _native.nipals(X2, y2, tol=1.0, max_iter=3)


def test_xcorr_max_shift_fallback_empty_window(monkeypatch):
    """Shift larger than signal length → corr loop skips all shifts → returns 0."""
    monkeypatch.setattr(_native, "_NATIVE_OK", False)
    monkeypatch.setattr(_native, "_native_backend", None)

    template = np.array([1.0, 2.0, 3.0])
    query   = np.array([1.0, 2.0, 3.0])
    # max_shift = 10 far exceeds length 3; some shifts produce empty overlap (continue)
    shift, _ = _native.xcorr_max_shift(template, query, max_shift=10)
    assert isinstance(shift, int)


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
