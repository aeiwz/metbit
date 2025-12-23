import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from metbit.scaler import Scaler


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
