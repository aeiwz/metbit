"""Tests for TrainTestSplit and CrossValidator."""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def Xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 10)),
                     columns=[f"f{i}" for i in range(10)])
    y = pd.Series(["A"] * 30 + ["B"] * 30)
    return X, y


@pytest.fixture
def clf():
    return RandomForestClassifier(n_estimators=5, random_state=0)


# ── TrainTestSplit ────────────────────────────────────────────────────────────

class TestTrainTestSplit:
    def test_split_shapes(self, Xy):
        from metbit.validation.splitter import TrainTestSplit
        X, y = Xy
        tts = TrainTestSplit(X, y, test_size=0.2)
        Xtr, Xte, ytr, yte = tts.split()
        assert len(Xtr) == 48 and len(Xte) == 12

    def test_no_stratify(self, Xy):
        from metbit.validation.splitter import TrainTestSplit
        X, y = Xy
        tts = TrainTestSplit(X, y, test_size=0.25, stratify=False)
        Xtr, Xte, ytr, yte = tts.split()
        assert len(Xtr) + len(Xte) == 60

    def test_get_summary(self, Xy):
        from metbit.validation.splitter import TrainTestSplit
        X, y = Xy
        tts = TrainTestSplit(X, y)
        tts.split()
        summary = tts.get_summary()
        assert isinstance(summary, pd.DataFrame)
        assert set(summary.columns) >= {"class", "train_n", "test_n"}

    def test_get_summary_auto_splits(self, Xy):
        from metbit.validation.splitter import TrainTestSplit
        X, y = Xy
        tts = TrainTestSplit(X, y)
        # get_summary calls split() if not done yet
        summary = tts.get_summary()
        assert len(summary) == 2  # two classes

    def test_plot_split(self, Xy):
        from metbit.validation.splitter import TrainTestSplit
        X, y = Xy
        tts = TrainTestSplit(X, y)
        tts.split()
        fig = tts.plot_split()
        assert isinstance(fig, go.Figure)

    def test_plot_split_custom_title(self, Xy):
        from metbit.validation.splitter import TrainTestSplit
        X, y = Xy
        tts = TrainTestSplit(X, y)
        tts.split()
        fig = tts.plot_split(title="My Split")
        assert isinstance(fig, go.Figure)

    def test_ndarray_input(self):
        from metbit.validation.splitter import TrainTestSplit
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 5))
        y = np.array([0] * 20 + [1] * 20)
        tts = TrainTestSplit(X, y)
        Xtr, Xte, ytr, yte = tts.split()
        assert len(Xtr) + len(Xte) == 40


# ── CrossValidator ────────────────────────────────────────────────────────────

class TestCrossValidator:
    @pytest.mark.parametrize("strategy", [
        "kfold", "stratified_kfold", "shuffle_split", "repeated_kfold",
    ])
    def test_fit_all_common_strategies(self, Xy, clf, strategy):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, cv_strategy=strategy, n_splits=3)
        cv.fit()
        assert cv.scores_ is not None
        assert len(cv.scores_) >= 3

    def test_loo(self, clf):
        from metbit.validation.splitter import CrossValidator
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 5)))
        y = pd.Series(["A"] * 10 + ["B"] * 10)
        cv = CrossValidator(clf, X, y, cv_strategy="loo")
        cv.fit()
        assert len(cv.scores_) == 20

    def test_leave_p_out(self, clf):
        from metbit.validation.splitter import CrossValidator
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((12, 5)))
        y = pd.Series(["A"] * 6 + ["B"] * 6)
        cv = CrossValidator(clf, X, y, cv_strategy="leave_p_out", p=2)
        cv.fit()
        assert cv.scores_ is not None

    def test_time_series(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, cv_strategy="time_series", n_splits=3)
        cv.fit()
        assert len(cv.scores_) == 3

    def test_group_kfold(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        groups = np.repeat(np.arange(10), 6)
        cv = CrossValidator(clf, X, y, cv_strategy="group_kfold",
                            n_splits=5, groups=groups)
        cv.fit()
        assert cv.scores_ is not None

    def test_get_scores(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=3)
        cv.fit()
        df = cv.get_scores()
        assert isinstance(df, pd.DataFrame)
        assert "fold" in df.columns and "score" in df.columns
        assert len(df) == 3

    def test_get_summary(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=4)
        cv.fit()
        summary = cv.get_summary()
        assert isinstance(summary, pd.DataFrame)
        assert set(summary.columns) >= {"mean", "std", "min", "max", "strategy"}

    def test_plot_scores(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=3)
        cv.fit()
        fig = cv.plot_scores()
        assert isinstance(fig, go.Figure)

    def test_plot_score_distribution(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=5)
        cv.fit()
        fig = cv.plot_score_distribution()
        assert isinstance(fig, go.Figure)

    def test_plot_learning_curve(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=3)
        fig = cv.plot_learning_curve(train_sizes=np.array([0.3, 0.6, 1.0]))
        assert isinstance(fig, go.Figure)

    def test_compare_strategies(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=3)
        fig = cv.compare_strategies(
            strategies=["kfold", "stratified_kfold", "shuffle_split"]
        )
        assert isinstance(fig, go.Figure)

    def test_bad_strategy_raises(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        with pytest.raises(ValueError):
            CrossValidator(clf, X, y, cv_strategy="bad_strategy")

    def test_plot_before_fit_raises(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=3)
        with pytest.raises(RuntimeError):
            cv.plot_scores()

    def test_ndarray_input(self, clf):
        from metbit.validation.splitter import CrossValidator
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8))
        y = np.array(["A"] * 20 + ["B"] * 20)
        cv = CrossValidator(clf, X, y, n_splits=3)
        cv.fit()
        assert cv.scores_ is not None

    def test_custom_scoring(self, Xy, clf):
        from metbit.validation.splitter import CrossValidator
        X, y = Xy
        cv = CrossValidator(clf, X, y, n_splits=3, scoring="accuracy")
        cv.fit()
        assert all(0 <= s <= 1 for s in cv.scores_)


# ── available_cv_strategies ───────────────────────────────────────────────────

def test_available_cv_strategies():
    from metbit.validation.splitter import available_cv_strategies
    df = available_cv_strategies()
    assert isinstance(df, pd.DataFrame)
    assert "key" in df.columns and "name" in df.columns
    assert len(df) == 8
    assert "stratified_kfold" in df["key"].values
