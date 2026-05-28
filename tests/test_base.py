import numpy as np
import pytest

from metbit.models.base import nipals


def _simple_xy(n=20, p=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] + rng.standard_normal(n) * 0.05
    return X.copy(), y.copy()


class TestNIPALS:
    def test_returns_four_arrays(self):
        X, y = _simple_xy()
        result = nipals(X, y)

        assert len(result) == 4

    def test_weight_vector_is_unit_norm(self):
        X, y = _simple_xy()
        w, u, c, t = nipals(X, y)

        np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-10)

    def test_shapes_consistent_with_input(self):
        n, p = 15, 6
        X, y = _simple_xy(n=n, p=p)
        w, u, c, t = nipals(X, y)

        assert w.shape == (p,)
        assert u.shape == (n,)
        assert t.shape == (n,)
        assert np.isscalar(c) or c.shape == ()

    def test_scores_align_with_weights(self):
        X, y = _simple_xy()
        w, u, c, t = nipals(X, y)

        t_reconstructed = X @ w
        np.testing.assert_allclose(t, t_reconstructed, atol=1e-8)

    def test_custom_tolerance_converges(self):
        X, y = _simple_xy()
        w_tight, *_ = nipals(X, y, tol=1e-14)
        w_loose, *_ = nipals(X, y, tol=1e-3)

        np.testing.assert_allclose(np.abs(w_tight), np.abs(w_loose), atol=1e-2)
