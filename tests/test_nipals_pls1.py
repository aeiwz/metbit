"""PLS1 NIPALS in one pass: the same answer as iterating to convergence, without idle iterations."""
import numpy as np

from metbit import _native


def _loop_nipals(x, y, tol=1e-10, max_iter=1000):
    """The iterative NIPALS-PLS1 this replaces, kept as the reference."""
    u, w, t, c = y.copy(), np.zeros(x.shape[1]), np.zeros(x.shape[0]), 0.0
    for _ in range(max_iter):
        w = x.T @ u / (u @ u)
        w /= np.linalg.norm(w)
        t = x @ w
        c = t @ y / (t @ t)
        u_new = y / c
        d = np.linalg.norm(u_new - u) / np.linalg.norm(u_new)
        u = u_new
        if d <= tol:
            break
    return w, u, c, t


def test_one_pass_equals_the_converged_loop():
    rng = np.random.default_rng(0)
    for n, p in ((30, 400), (120, 3000)):  # both above the C path's size limit
        x, y = rng.normal(size=(n, p)), rng.normal(size=n)
        for got, want in zip(_native.nipals(x, y), _loop_nipals(x, y)):
            assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_degenerate_inputs_return_like_the_old_loop():
    x = np.ones((4, 20_000))  # every sample identical, so X'y = 0 for a centred y (n*p > 50k: the NumPy path)
    y = np.array([1.0, -1.0, 1.0, -1.0])
    w, u, c, t = _native.nipals(x, y)
    assert c == 0.0 and not t.any() and np.array_equal(u, y) and not w.any()
    w, u, c, t = _native.nipals(np.ones((4, 20_000)), np.zeros(4))
    assert c == 0.0 and not w.any() and not t.any()


def test_used_up_residual_does_not_spin():
    """X'y at rounding level (as after deep OPLS deflation) finishes at once instead of 1000 idle passes."""
    import time

    rng = np.random.default_rng(1)
    y = np.array([1.0, -1.0] * 60)
    x = rng.normal(size=(120, 3000))
    x -= np.outer(y, y @ x) / (y @ y)  # remove all y-direction: X'y is now ~1e-13
    start = time.perf_counter()
    _native.nipals(x, y)
    assert time.perf_counter() - start < 0.05


def test_cross_validation_stops_at_max_components():
    from metbit.models.cross_validation import CrossValidation

    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 500))
    y = np.array([0, 1] * 30)
    x[y == 1, :5] += 1.0
    cv = CrossValidation(kfold=3)
    cv.fit(x, y)
    assert cv.max_components == 15 and len(cv._q2) == 15
    small = CrossValidation(kfold=3, max_components=4)
    small.fit(x, y)
    assert len(small._q2) == 4 and np.allclose(small._q2, cv._q2[:4])
