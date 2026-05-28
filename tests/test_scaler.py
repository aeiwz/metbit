import copy

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from metbit.scaler import Scaler, _handle_zeros_in_scale


def test_scaler_pareto_mean_centering():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = Scaler(scale_power=0.5)
    scaler.fit(X)

    transformed = scaler.transform(X)

    np.testing.assert_allclose(transformed, np.array([[-1.0, -1.0], [1.0, 1.0]]))


def test_scaler_unit_variance_standardises():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = Scaler(scale_power=1)
    scaler.fit(X)

    transformed = scaler.transform(X)

    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=0, ddof=0), [1.0, 1.0], atol=1e-12)


def test_scaler_transform_requires_fit():
    X = np.array([[1.0, 2.0]])
    scaler = Scaler()

    with pytest.raises(NotFittedError):
        scaler.transform(X)


def test_scaler_inverse_round_trip():
    X = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 7.0]])
    scaler = Scaler(scale_power=0.5)
    scaler.fit(X)

    transformed = scaler.transform(X)
    restored = scaler.inverse_transform(transformed)

    np.testing.assert_allclose(restored, X)


def test_scaler_partial_fit_accumulates_samples():
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[5.0, 6.0]])
    scaler = Scaler(scale_power=1)
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    assert int(np.asarray(scaler.n_samples_seen_).flat[0]) == 3
    np.testing.assert_allclose(scaler.mean_, [3.0, 4.0], atol=1e-12)


def test_scaler_fit_resets_state():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = Scaler(scale_power=1)
    scaler.fit(X)
    n1 = int(np.asarray(scaler.n_samples_seen_).flat[0])
    scaler.fit(X)
    n2 = int(np.asarray(scaler.n_samples_seen_).flat[0])

    assert n1 == n2 == 2


def test_scaler_no_mean_centering():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = Scaler(with_mean=False, with_std=True, scale_power=1)
    scaler.fit(X)
    transformed = scaler.transform(X.copy())

    assert transformed.shape == X.shape


def test_scaler_no_std_scaling():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = Scaler(with_mean=True, with_std=False)
    scaler.fit(X)
    transformed = scaler.transform(X.copy())

    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-12)


def test_scaler_deepcopy():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = Scaler(scale_power=0.5)
    scaler.fit(X)
    scaler_copy = copy.deepcopy(scaler)

    np.testing.assert_allclose(scaler_copy.mean_, scaler.mean_)
    np.testing.assert_allclose(scaler_copy.scale_, scaler.scale_)


class TestHandleZerosInScale:
    def test_scalar_zero_becomes_one(self):
        assert _handle_zeros_in_scale(0.0) == 1.0

    def test_scalar_nonzero_unchanged(self):
        assert _handle_zeros_in_scale(2.5) == 2.5

    def test_array_zeros_replaced(self):
        scale = np.array([0.0, 1.0, 2.0])
        result = _handle_zeros_in_scale(scale)
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 2.0

    def test_array_no_copy_modifies_in_place(self):
        scale = np.array([0.0, 1.0])
        result = _handle_zeros_in_scale(scale, copy=False)
        assert result[0] == 1.0
