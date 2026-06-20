"""Extra tests for metbit/analysis/opls_da.py - covers remaining gaps."""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from metbit.analysis.opls_da import opls_da


def _make_opls_data(n=40, p=15, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(["A"] * (n // 2) + ["B"] * (n // 2))
    return X, y


@pytest.fixture(scope="module")
def fitted_opls():
    X, y = _make_opls_data()
    model = opls_da(X=X, y=y, n_components=1, scaling_method="pareto",
                    kfold=3, estimator="opls", random_state=42)
    model.fit()
    model.vip_scores()
    return model


@pytest.fixture(scope="module")
def fitted_pls():
    X, y = _make_opls_data(seed=1)
    model = opls_da(X=X, y=y, n_components=1, scaling_method="pareto",
                    kfold=3, estimator="pls", random_state=42)
    model.fit()
    return model


class TestOPLSDAValidation:
    def test_none_X_raises(self):
        _, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=None, y=y)

    def test_none_y_raises(self):
        X, _ = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=None)

    def test_invalid_n_components_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, n_components=0)

    def test_non_int_n_components_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, n_components="one")

    def test_X_bad_type_raises(self):
        _, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=[[1, 2], [3, 4]], y=y.head(2))

    def test_y_bad_type_raises(self):
        X, _ = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y={"a": 1})

    def test_X_y_shape_mismatch_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y.head(5))

    def test_scaling_method_bad_type_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, scaling_method=123)

    def test_kfold_bad_type_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, kfold="three")

    def test_estimator_bad_type_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, estimator=42)

    def test_random_state_bad_type_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, random_state="abc")

    def test_features_name_bad_type_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, features_name="not_a_list")

    def test_features_name_wrong_length_raises(self):
        X, y = _make_opls_data()
        with pytest.raises(ValueError):
            opls_da(X=X, y=y, features_name=["a", "b"])


class TestOPLSDAFitPaths:
    def test_auto_ncomp_false_path(self):
        X, y = _make_opls_data()
        model = opls_da(X=X, y=y, n_components=1, scaling_method="pareto",
                        kfold=3, estimator="opls", random_state=42,
                        auto_ncomp=False)
        model.fit()
        assert hasattr(model, "cv_model")

    def test_estimator_pls_path(self, fitted_pls):
        assert hasattr(fitted_pls, "cv_model")

    def test_get_oplsda_model(self, fitted_opls):
        m = fitted_opls.get_oplsda_model()
        assert m is not None

    def test_get_cv_model(self, fitted_opls):
        m = fitted_opls.get_cv_model()
        assert m is not None


class TestOPLSDAVipPlot:
    def test_vip_plot_with_transform(self, fitted_opls):
        fig = fitted_opls.vip_plot(vip_transform=True)
        assert isinstance(fig, go.Figure)

    def test_vip_plot_without_transform(self, fitted_opls):
        fig = fitted_opls.vip_plot(vip_transform=False)
        assert isinstance(fig, go.Figure)

    def test_vip_plot_with_filter(self, fitted_opls):
        fig = fitted_opls.vip_plot(filter_=True, threshold=1.0)
        assert isinstance(fig, go.Figure)


class TestOPLSDAScoresPlot:
    def test_plot_oplsda_scores_default(self, fitted_opls):
        fig = fitted_opls.plot_oplsda_scores()
        assert isinstance(fig, go.Figure)

    def test_plot_oplsda_scores_individual_ellipse_false(self, fitted_opls):
        fig = fitted_opls.plot_oplsda_scores(individual_ellipse=False)
        assert isinstance(fig, go.Figure)

    def test_plot_oplsda_scores_color_mismatch_raises(self, fitted_opls):
        with pytest.raises(ValueError):
            fitted_opls.plot_oplsda_scores(color_=pd.Series(["X", "Y"]))

    def test_plot_oplsda_scores_symbol_mismatch_raises(self, fitted_opls):
        with pytest.raises(ValueError):
            fitted_opls.plot_oplsda_scores(symbol_=pd.Series(["X", "Y"]))

    def test_plot_oplsda_scores_symbol_dict_bad_raises(self, fitted_opls):
        with pytest.raises(ValueError):
            fitted_opls.plot_oplsda_scores(symbol_dict="not_dict")


class TestOPLSDAPermutationPlot:
    def test_permutation_test_records_scores(self, fitted_opls, monkeypatch):
        def fake_permutation_test_score(*args, **kwargs):
            return 0.75, np.array([0.2, 0.3]), 0.05

        monkeypatch.setattr(
            "metbit.analysis.opls_da.permutation_test_score",
            fake_permutation_test_score,
        )

        fitted_opls.permutation_test(n_permutations=2, cv=2, n_jobs=1, verbose=0)

        assert fitted_opls.acc_score == pytest.approx(0.75)
        np.testing.assert_array_equal(
            fitted_opls.permutation_scores, np.array([0.2, 0.3])
        )
        assert fitted_opls.p_value == pytest.approx(0.05)

    def test_plot_hist_after_mock_permutation(self, fitted_opls):
        fitted_opls.permutation_scores = np.random.uniform(0.4, 0.8, 100)
        fitted_opls.acc_score = 0.9
        fitted_opls.p_value = 0.01
        fitted_opls.n_permutations = 100
        fig = fitted_opls.plot_hist()
        assert isinstance(fig, go.Figure)

    def test_plot_hist_before_permutation_raises(self):
        X, y = _make_opls_data()
        model = opls_da(X=X, y=y, n_components=1, kfold=3, random_state=42)
        model.fit()
        model.permutation_scores = None
        with pytest.raises(ValueError):
            model.plot_hist()

    def test_get_permutation_scores(self, fitted_opls):
        # Covers opls_da.py:418-419 get_permutation_scores() getter
        fitted_opls.permutation_scores = np.array([0.5, 0.6, 0.7])
        scores = fitted_opls.get_permutation_scores()
        assert scores is not None


class TestOPLSDAScoresPlotValidColor:
    def _fresh_model(self):
        X, y = _make_opls_data()
        model = opls_da(X=X, y=y, n_components=1, scaling_method="pareto",
                        kfold=3, estimator="opls", random_state=42)
        model.fit()
        return model

    def test_plot_oplsda_scores_valid_color(self):
        # Covers lines 668-669 (color_ = color_) and 688 (df_opls_scores['Group'] = color_)
        model = self._fresh_model()
        n = len(model.y)
        color_ = pd.Series(["red"] * (n // 2) + ["blue"] * (n - n // 2))
        fig = model.plot_oplsda_scores(color_=color_)
        assert isinstance(fig, go.Figure)

    def test_plot_oplsda_scores_valid_symbol(self):
        # Covers line 692 (df_opls_scores['symbol'] = symbol_)
        model = self._fresh_model()
        n = len(model.y)
        symbol_ = pd.Series(["circle"] * (n // 2) + ["square"] * (n - n // 2))
        fig = model.plot_oplsda_scores(symbol_=symbol_)
        assert isinstance(fig, go.Figure)

    def test_plot_oplsda_scores_color_dict(self):
        # Covers line 699 (color_dict_2 = color_dict)
        model = self._fresh_model()
        color_dict = {"A": "red", "B": "blue"}
        fig = model.plot_oplsda_scores(color_dict=color_dict)
        assert isinstance(fig, go.Figure)
