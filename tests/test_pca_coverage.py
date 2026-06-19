"""Tests to achieve 100% coverage of metbit/analysis/pca.py."""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from metbit.analysis.pca import pca


def _make_data(n=40, p=15, n_groups=2, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    label = pd.Series([f"G{i % n_groups}" for i in range(n)])
    return X, label


@pytest.fixture(scope="module")
def fitted_2d():
    X, label = _make_data()
    model = pca(X=X, label=label, n_components=2, scaling_method="pareto",
                random_state=42, test_size=0.3)
    model.fit()
    return model


@pytest.fixture(scope="module")
def fitted_3d():
    X, label = _make_data(n_groups=3)
    model = pca(X=X, label=label, n_components=3, scaling_method="pareto",
                random_state=42, test_size=0.3)
    model.fit()
    return model


class TestPCAInit:
    def test_features_name_bad_type(self):
        X, label = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label=label, features_name="not_a_list")

    def test_features_name_wrong_length(self):
        X, label = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label=label, features_name=["a", "b"])

    def test_label_none_fills_default(self):
        X, _ = _make_data()
        model = pca(X=X)
        assert len(model.label) == X.shape[0]

    def test_X_bad_type(self):
        with pytest.raises(ValueError):
            pca(X=[[1, 2], [3, 4]], label=["A", "B"])

    def test_n_components_bad_type(self):
        X, label = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label=label, n_components="two")

    def test_scaling_method_bad_type(self):
        X, label = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label=label, scaling_method=3)

    def test_random_state_bad_type(self):
        X, label = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label=label, random_state="abc")

    def test_label_bad_type(self):
        X, _ = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label={"a": 1})

    def test_label_length_mismatch(self):
        X, _ = _make_data()
        with pytest.raises(ValueError):
            pca(X=X, label=pd.Series(["A", "B"]))

    def test_ndarray_X_sets_arange_features(self):
        rng = np.random.default_rng(0)
        X_arr = rng.standard_normal((20, 5))
        label = np.array(["A"] * 10 + ["B"] * 10)
        model = pca(X=X_arr, label=label)
        assert len(model.features_name) == 5

    def test_missing_values_dataframe_raises(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        with pytest.raises(ValueError):
            pca(X=X, label=["A", "A", "B"])

    def test_missing_values_ndarray_raises(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        with pytest.raises(ValueError):
            pca(X=X, label=["A", "B", "C"])


class TestPCAFit:
    def test_fit_with_ndarray_X_and_list_label(self):
        rng = np.random.default_rng(0)
        X_arr = rng.standard_normal((30, 8))
        label_list = ["A"] * 15 + ["B"] * 15
        features = [f"feat_{i}" for i in range(8)]
        model = pca(X=X_arr, label=label_list, features_name=features, n_components=2)
        result = model.fit()
        assert result is not None

    def test_fit_with_list_features(self):
        X, label = _make_data()
        feat_list = [f"x{i}" for i in range(X.shape[1])]
        model = pca(X=X, label=label, features_name=feat_list, n_components=2)
        model.fit()
        scores = model.get_scores()
        assert scores.shape[0] == X.shape[0]


class TestPCAGetters:
    def test_get_explained_variance(self, fitted_2d):
        ev = fitted_2d.get_explained_variance()
        assert "Explained variance" in ev.columns

    def test_get_scores(self, fitted_2d):
        s = fitted_2d.get_scores()
        assert "PC1" in s.columns

    def test_get_loadings(self, fitted_2d):
        lo = fitted_2d.get_loadings()
        assert "PC1" in lo.columns

    def test_get_q2_test(self, fitted_2d):
        q2 = fitted_2d.get_q2_test()
        assert isinstance(q2, float)


class TestPCAPlotVariance:
    def test_plot_observe_variance(self, fitted_2d):
        fig = fitted_2d.plot_observe_variance()
        assert isinstance(fig, go.Figure)

    def test_plot_cumulative_observed(self, fitted_2d):
        fig = fitted_2d.plot_cumulative_observed()
        assert isinstance(fig, go.Figure)


class TestPCAPlotScores:
    def test_plot_pca_scores_default(self, fitted_2d):
        fig = fitted_2d.plot_pca_scores()
        assert isinstance(fig, go.Figure)

    def test_plot_pca_scores_with_color_dict(self, fitted_2d):
        fig = fitted_2d.plot_pca_scores(color_dict={"G0": "red", "G1": "blue"})
        assert isinstance(fig, go.Figure)

    def test_plot_pca_scores_individual_ellipse_false(self, fitted_2d):
        fig = fitted_2d.plot_pca_scores(individual_ellipse=False)
        assert isinstance(fig, go.Figure)

    def test_plot_pca_scores_pc_not_list_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(pc="PC1")

    def test_plot_pca_scores_pc_wrong_length_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(pc=["PC1"])

    def test_plot_pca_scores_invalid_pc_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(pc=["PC1", "PC99"])

    def test_plot_pca_scores_pc1_invalid_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(pc=["PC99", "PC1"])

    def test_plot_pca_scores_color_mismatch_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(color_=pd.Series(["A", "B"]))

    def test_plot_pca_scores_symbol_mismatch_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(symbol_=pd.Series(["X", "Y"]))

    def test_plot_pca_scores_symbol_dict_bad_type_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_scores(symbol_dict="not_a_dict")


class TestPCAPlotLoadings:
    def test_plot_loading_(self, fitted_2d):
        fig = fitted_2d.plot_loading_()
        assert isinstance(fig, go.Figure)


class TestPCAPlotTrajectory:
    def test_plot_pca_trajectory_basic(self, fitted_2d):
        n = len(fitted_2d.label)
        time_ = pd.Series(["T1" if i < n // 2 else "T2" for i in range(n)])
        time_order = {"T1": 0, "T2": 1}
        fig = fitted_2d.plot_pca_trajectory(time_=time_, time_order=time_order)
        assert isinstance(fig, go.Figure)

    def test_plot_pca_trajectory_with_color_dict(self, fitted_2d):
        n = len(fitted_2d.label)
        time_ = pd.Series(["T1" if i < n // 2 else "T2" for i in range(n)])
        time_order = {"T1": 0, "T2": 1}
        color_dict = {"G0": "#636EFA", "G1": "#EF553B"}
        fig = fitted_2d.plot_pca_trajectory(time_=time_, time_order=time_order,
                                             color_dict=color_dict,
                                             stat_=["median", "std"])
        assert isinstance(fig, go.Figure)

    def test_plot_pca_trajectory_bad_time_order_raises(self, fitted_2d):
        n = len(fitted_2d.label)
        time_ = pd.Series(["T1"] * n)
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_trajectory(time_=time_, time_order="wrong")

    def test_plot_pca_trajectory_time_mismatch_raises(self, fitted_2d):
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_trajectory(time_=None, time_order={"T1": 0})

    def test_plot_pca_trajectory_bad_stat_raises(self, fitted_2d):
        n = len(fitted_2d.label)
        time_ = pd.Series(["T1"] * n)
        with pytest.raises(ValueError):
            fitted_2d.plot_pca_trajectory(time_=time_, time_order={"T1": 0},
                                          stat_=["bad", "sem"])


class TestPCAPlot3D:
    def test_plot_3d_pca_basic(self, fitted_3d):
        fig = fitted_3d.plot_3d_pca()
        assert isinstance(fig, go.Figure)

    def test_plot_3d_pca_with_color_dict(self, fitted_3d):
        color_dict = {"G0": "red", "G1": "blue", "G2": "green"}
        fig = fitted_3d.plot_3d_pca(color_dict=color_dict)
        assert isinstance(fig, go.Figure)

    def test_plot_3d_pca_pc_not_list_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(pc="PC1")

    def test_plot_3d_pca_wrong_length_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(pc=["PC1", "PC2"])

    def test_plot_3d_pca_invalid_pc3_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(pc=["PC1", "PC2", "PC99"])

    def test_plot_3d_pca_invalid_pc1_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(pc=["PC99", "PC1", "PC2"])

    def test_plot_3d_pca_invalid_pc2_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(pc=["PC1", "PC99", "PC2"])

    def test_plot_3d_pca_color_mismatch_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(color_=pd.Series(["A", "B"]))

    def test_plot_3d_pca_symbol_mismatch_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(symbol_=pd.Series(["X", "Y"]))

    def test_plot_3d_pca_symbol_dict_bad_raises(self, fitted_3d):
        with pytest.raises(ValueError):
            fitted_3d.plot_3d_pca(symbol_dict="not_a_dict")

    def test_plot_3d_pca_valid_color(self, fitted_3d):
        # Covers pca.py:753 (color_ = color_) and 761 (df_scores_['Group'] = color_)
        n = len(fitted_3d.label)
        color_ = pd.Series(["red"] * (n // 3) + ["blue"] * (n // 3) + ["green"] * (n - 2 * (n // 3)))
        fig = fitted_3d.plot_3d_pca(color_=color_)
        assert isinstance(fig, go.Figure)

    def test_plot_3d_pca_valid_symbol(self, fitted_3d):
        # Covers pca.py:767-768 duplicate symbol_ length check
        n = len(fitted_3d.label)
        symbol_ = pd.Series(["circle"] * (n // 3) + ["square"] * (n // 3) + ["diamond"] * (n - 2 * (n // 3)))
        fig = fitted_3d.plot_3d_pca(symbol_=symbol_)
        assert isinstance(fig, go.Figure)


class TestPCAScoresValidColor:
    def test_plot_pca_scores_valid_color(self, fitted_2d):
        # Covers pca.py:401 (color_ = color_) and 409 (df_scores_['Group'] = color_)
        n = len(fitted_2d.label)
        color_ = pd.Series(["red"] * (n // 2) + ["blue"] * (n - n // 2))
        fig = fitted_2d.plot_pca_scores(color_=color_)
        assert isinstance(fig, go.Figure)

    def test_plot_pca_scores_valid_symbol(self, fitted_2d):
        # Covers pca.py:415-416 duplicate symbol_ check
        n = len(fitted_2d.label)
        symbol_ = pd.Series(["circle"] * (n // 2) + ["square"] * (n - n // 2))
        fig = fitted_2d.plot_pca_scores(symbol_=symbol_)
        assert isinstance(fig, go.Figure)
