import numpy as np
import pytest

from metbit.models.cross_validation import CrossValidation


def _make_xy(n=40, p=8, seed=0):
    """Balanced binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X[y == 1] += 1.5   # add separability
    return X, y


class TestCrossValidationInit:
    def test_default_estimator_is_opls(self):
        cv = CrossValidation()
        assert cv.estimator_id == "opls"

    def test_pls_estimator_selected(self):
        cv = CrossValidation(estimator="pls")
        assert cv.estimator_id == "pls"

    def test_kfold_stored(self):
        cv = CrossValidation(kfold=5)
        assert cv.kfold == 5

    def test_unknown_estimator_silently_ignores(self):
        # estimator stored as None but estimator_id set
        cv = CrossValidation(estimator="unknown")
        assert cv.estimator_id == "unknown"


class TestCrossValidationFitPLS:
    def setup_method(self):
        self.X, self.y = _make_xy()
        self.cv = CrossValidation(estimator="pls", kfold=4, scaler="pareto")

    def test_fit_runs_without_error(self):
        self.cv.fit(self.X, self.y)

    def test_q2_after_fit(self):
        self.cv.fit(self.X, self.y)
        q2 = self.cv.q2
        assert isinstance(q2, (float, int, np.floating))
        assert -1.0 <= float(q2) <= 1.0

    def test_mis_classifications_length(self):
        self.cv.fit(self.X, self.y)
        mc = self.cv.mis_classifications
        assert len(mc) > 0

    def test_optimal_component_positive(self):
        self.cv.fit(self.X, self.y)
        assert self.cv._opt_component >= 0

    def test_predict_returns_array(self):
        self.cv.fit(self.X, self.y)
        yhat = self.cv.predict(self.X)
        assert yhat.shape[0] == self.X.shape[0]


class TestCrossValidationFitOPLS:
    def setup_method(self):
        self.X, self.y = _make_xy()
        self.cv = CrossValidation(estimator="opls", kfold=4, scaler="pareto")

    def test_fit_runs_without_error(self):
        self.cv.fit(self.X, self.y)

    def test_predictive_score_shape(self):
        self.cv.fit(self.X, self.y)
        tp = self.cv.predictive_score
        assert tp.shape[0] == self.X.shape[0]

    def test_orthogonal_score_shape(self):
        self.cv.fit(self.X, self.y)
        to = self.cv.orthogonal_score
        assert to.shape[0] == self.X.shape[0]

    def test_predict_returns_array(self):
        self.cv.fit(self.X, self.y)
        yhat = self.cv.predict(self.X)
        assert yhat.shape[0] == self.X.shape[0]


class TestResetOptimalComponent:
    def test_reset_to_valid_component(self):
        X, y = _make_xy()
        cv = CrossValidation(estimator="pls", kfold=4)
        cv.fit(X, y)
        cv.reset_optimal_num_component(1)
        assert cv._opt_component == 0
        assert cv.optimal_component_num == 1

    def test_reset_to_zero_raises(self):
        X, y = _make_xy()
        cv = CrossValidation(estimator="pls", kfold=4)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            cv.reset_optimal_num_component(0)

    def test_reset_exceeding_max_raises(self):
        X, y = _make_xy()
        cv = CrossValidation(estimator="pls", kfold=4)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            cv.reset_optimal_num_component(cv._npc0 + 10)

    def test_reset_non_int_raises(self):
        X, y = _make_xy()
        cv = CrossValidation(estimator="pls", kfold=4)
        cv.fit(X, y)
        with pytest.raises(ValueError):
            cv.reset_optimal_num_component(1.5)
