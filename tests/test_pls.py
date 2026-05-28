import numpy as np
import pytest

from metbit.models.pls import PLS


def _make_data(n=20, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 2 + rng.standard_normal(n) * 0.1
    return X, y


class TestPLSFit:
    def test_fit_stores_scores_and_loadings(self):
        X, y = _make_data()
        pls = PLS()
        pls.fit(X, y, n_comp=3)

        assert pls.scores_x.shape == (20, 3)
        assert pls.loadings_x.shape == (5, 3)
        assert pls.weights_y.shape == (3,)

    def test_fit_default_ncomp_uses_min_n_p(self):
        X, y = _make_data(n=8, p=5)
        pls = PLS()
        pls.fit(X, y)

        assert pls.coef.shape[0] == min(8, 5)

    def test_fit_ncomp_none_n_greater_than_p(self):
        X, y = _make_data(n=30, p=4)
        pls = PLS()
        pls.fit(X, y)

        assert pls.coef.shape[0] == 4

    def test_fit_single_component(self):
        X, y = _make_data()
        pls = PLS()
        pls.fit(X, y, n_comp=1)

        assert pls.coef.shape[0] == 1


class TestPLSPredict:
    def test_predict_returns_vector_of_length_n(self):
        X, y = _make_data()
        pls = PLS()
        pls.fit(X, y, n_comp=3)

        yhat = pls.predict(X)
        assert yhat.shape == (20,)

    def test_predict_with_explicit_n_component(self):
        X, y = _make_data()
        pls = PLS()
        pls.fit(X, y, n_comp=4)

        yhat_full = pls.predict(X)
        yhat_1 = pls.predict(X, n_component=1)
        assert yhat_full.shape == yhat_1.shape

    def test_predict_ncomp_larger_than_fit_clips_to_max(self):
        X, y = _make_data()
        pls = PLS()
        pls.fit(X, y, n_comp=3)

        yhat_clipped = pls.predict(X, n_component=999)
        yhat_default = pls.predict(X)
        np.testing.assert_array_equal(yhat_clipped, yhat_default)

    def test_predict_returns_float_array(self):
        X, y = _make_data(n=50, p=10)
        pls = PLS()
        pls.fit(X, y, n_comp=3)

        yhat = pls.predict(X)
        assert yhat.dtype.kind == "f"
        assert yhat.shape == (50,)
