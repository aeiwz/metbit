import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_diabetes

from metbit.vip import vip_scores


def _fitted_pls(n_components=3):
    data = load_diabetes()
    model = PLSRegression(n_components=n_components)
    model.fit(data.data, data.target)
    return model, list(data.feature_names)


class TestVIPScores:
    def test_get_scores_returns_dataframe(self):
        model, _ = _fitted_pls()
        vip = vip_scores(model)

        result = vip.get_scores()

        assert isinstance(result, pd.DataFrame)
        assert "VIP" in result.columns
        assert "Features" in result.columns

    def test_scores_length_matches_features(self):
        model, names = _fitted_pls()
        vip = vip_scores(model, features_name=names)

        result = vip.get_scores()

        assert len(result) == len(names)

    def test_features_column_contains_provided_names(self):
        model, names = _fitted_pls()
        vip = vip_scores(model, features_name=names)

        result = vip.get_scores()

        assert list(result["Features"]) == names

    def test_vip_scores_are_nonnegative(self):
        model, _ = _fitted_pls()
        vip = vip_scores(model)

        result = vip.get_scores()

        assert (result["VIP"] >= 0).all()

    def test_no_features_name_uses_index(self):
        model, _ = _fitted_pls()
        vip = vip_scores(model)

        result = vip.get_scores()

        assert list(result["Features"]) == list(result.index)

    def test_model_stored_as_attribute(self):
        model, _ = _fitted_pls()
        vip = vip_scores(model)

        assert vip.model is model

    def test_vip_plot_returns_plotly_figure(self):
        import plotly.graph_objects as go

        model, _ = _fitted_pls()
        vip = vip_scores(model)

        fig = vip.vip_plot(threshold=1.0)

        assert isinstance(fig, go.Figure)

    def test_vip_plot_custom_threshold(self):
        import plotly.graph_objects as go

        model, names = _fitted_pls()
        vip = vip_scores(model, features_name=names)

        fig = vip.vip_plot(threshold=2.0)

        assert isinstance(fig, go.Figure)
