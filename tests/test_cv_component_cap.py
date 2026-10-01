import numpy as np

from metbit.models.cross_validation import CrossValidation


def test_small_folds_never_fit_more_components_than_they_can_hold():
    # 20 samples in 3 folds: 12 training samples, so 11 components or more used to blow up
    rng = np.random.default_rng(0)
    for _ in range(20):
        X = rng.random((20, 500))
        y = np.array(["A"] * 10 + ["B"] * 10)[rng.permutation(20)]  # random labels: no real difference
        cv = CrossValidation(kfold=3, estimator="opls", scaler="pareto")
        cv.fit(X, y)
        assert np.all(np.abs(cv._q2) < 5), cv._q2
        assert -5 < cv.q2 <= 1
