import numpy as np
import pytest

from metbit.opls import OPLS


def _make_data(n=30, p=8, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 3 - X[:, 1] + rng.standard_normal(n) * 0.1
    return X.copy(), y.copy()


class TestOPLSFit:
    def test_fit_populates_internal_matrices(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=2)

        assert model._Tortho is not None
        assert model._Portho is not None
        assert model._T is not None
        assert model.coef is not None

    def test_fit_stores_npc(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=3)

        assert model.npc == 3

    def test_fit_ncomp_none_uses_min_n_p(self):
        n, p = 10, 5
        X, y = _make_data(n=n, p=p)
        model = OPLS()
        model.fit(X, y)

        assert model.npc == min(n, p)

    def test_fit_ncomp_larger_than_data_clips(self):
        n, p = 15, 6
        X, y = _make_data(n=n, p=p)
        model = OPLS()
        model.fit(X, y, n_comp=999)

        assert model.npc == min(n, p)


class TestOPLSPredict:
    def test_predict_shape(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=2)

        yhat = model.predict(X)
        assert yhat.shape == (len(X),)

    def test_predict_correlation_with_true_y(self):
        X, y = _make_data(n=60, p=8)
        model = OPLS()
        model.fit(X, y, n_comp=3)

        yhat = model.predict(X)
        corr = np.corrcoef(y, yhat)[0, 1]
        assert corr > 0.9

    def test_predict_returns_scores_when_requested(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=2)

        result = model.predict(X, return_scores=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_n_component_clips_to_npc(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=3)

        yhat_capped = model.predict(X, n_component=999)
        yhat_default = model.predict(X)
        np.testing.assert_array_equal(yhat_capped, yhat_default)


class TestOPLSCorrect:
    def test_correct_returns_same_shape(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=2)

        Xc = model.correct(X)
        assert Xc.shape == X.shape

    def test_correct_with_scores_returns_tuple(self):
        X, y = _make_data()
        model = OPLS()
        model.fit(X, y, n_comp=2)

        result = model.correct(X, return_scores=True)
        assert isinstance(result, tuple)
        Xc, t = result
        assert Xc.shape == X.shape
        assert t.shape[0] == X.shape[0]
