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


def test_given_splits_replace_the_stratified_blocks():
    from sklearn.model_selection import StratifiedKFold

    from metbit import opls_da

    rng = np.random.default_rng(2)
    X = rng.random((24, 100))
    y = np.array(["A"] * 12 + ["B"] * 12)
    X[12:, :5] += 1.0
    default = CrossValidation(kfold=3, estimator="opls", scaler="pareto")
    default.fit(X, y)
    same = CrossValidation(kfold=3, estimator="opls", scaler="pareto", splits=list(default._split(default._reset_y(y))))
    same.fit(X, y)
    assert np.allclose(default._q2, same._q2)  # handing over metbit's own folds changes nothing
    folds = list(StratifiedKFold(4, shuffle=True, random_state=0).split(X, y))
    model = opls_da(X=X, y=y, kfold=3, cv_splits=folds)
    model.fit()
    assert model.pipeline.named_steps["opls"].splits is not None and -1 < model.q2 <= 1
