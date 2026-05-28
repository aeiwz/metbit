import matplotlib
matplotlib.use("Agg")  # headless backend - must be set before pyplot import

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from metbit.plotting import Plots


def _pls_model():
    """Minimal mock of a fitted PLS cross-validation model."""
    m = MagicMock()
    m.estimator_id = "pls"
    rng = np.random.default_rng(0)
    m.scores = rng.standard_normal((20, 2))
    m.y = np.array([0] * 10 + [1] * 10)
    m.groups = ["Control", "Treatment"]
    return m


def _opls_model():
    """Minimal mock of a fitted OPLS cross-validation model."""
    m = MagicMock()
    m.estimator_id = "opls"
    rng = np.random.default_rng(1)
    m.predictive_score = rng.standard_normal(20)
    m.orthogonal_score = rng.standard_normal(20)
    m.y = np.array([0] * 10 + [1] * 10)
    m.groups = ["Control", "Treatment"]
    m.covariance = rng.standard_normal(10)
    m.correlation = rng.uniform(-1, 1, 10)
    return m


class TestPlotsScores:
    @patch("matplotlib.pyplot.show")
    def test_pls_score_plot_runs_without_error(self, _show):
        Plots(_pls_model()).plot_scores()
        _show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_opls_score_plot_runs_without_error(self, _show):
        Plots(_opls_model()).plot_scores()
        _show.assert_called_once()

    def test_save_plot_requires_file_name(self):
        with pytest.raises(ValueError, match="file_name"):
            Plots(_pls_model()).plot_scores(save_plot=True)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_save_plot_adds_png_extension(self, mock_save, _show, tmp_path):
        out = str(tmp_path / "scores")
        Plots(_pls_model()).plot_scores(save_plot=True, file_name=out)
        saved_path = mock_save.call_args[0][0]
        assert saved_path.endswith(".png")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_save_plot_preserves_explicit_extension(self, mock_save, _show, tmp_path):
        out = str(tmp_path / "scores.svg")
        Plots(_pls_model()).plot_scores(save_plot=True, file_name=out)
        saved_path = mock_save.call_args[0][0]
        assert saved_path.endswith(".svg")


class TestSPlot:
    @patch("matplotlib.pyplot.show")
    def test_splot_runs_for_opls(self, _show):
        Plots(_opls_model()).splot()
        _show.assert_called_once()

    def test_splot_raises_for_non_opls(self):
        with pytest.raises(ValueError, match="OPLS"):
            Plots(_pls_model()).splot()

    def test_splot_save_requires_file_name(self):
        with pytest.raises(ValueError, match="file_name"):
            Plots(_opls_model()).splot(save_plot=True)


class TestJackknifeLoadingPlot:
    def _jk_model(self):
        m = _pls_model()
        rng = np.random.default_rng(2)
        m.loadings_cv = rng.standard_normal((5, 10))
        m.kfold = 5
        return m

    @patch("matplotlib.pyplot.show")
    def test_returns_mean_and_intervals(self, _show):
        mean, intervals = Plots(self._jk_model()).jackknife_loading_plot()
        assert mean.shape == (10,)
        assert intervals.shape == (10,)

    @patch("matplotlib.pyplot.show")
    def test_intervals_are_nonnegative(self, _show):
        _, intervals = Plots(self._jk_model()).jackknife_loading_plot()
        assert np.all(intervals >= 0)

    def test_save_requires_file_name(self):
        with pytest.raises(ValueError, match="file_name"):
            Plots(self._jk_model()).jackknife_loading_plot(save_plot=True)

    @patch("matplotlib.pyplot.show")
    def test_custom_alpha_accepted(self, _show):
        mean, _ = Plots(self._jk_model()).jackknife_loading_plot(alpha=0.01)
        assert mean.shape == (10,)


class TestPlotCVErrors:
    @patch("matplotlib.pyplot.show")
    def test_runs_without_error(self, _show):
        m = _pls_model()
        m.mis_classifications = [8, 5, 3, 2]
        Plots(m).plot_cv_errors()
        _show.assert_called_once()

    def test_save_requires_file_name(self):
        m = _pls_model()
        m.mis_classifications = [4, 2, 1]
        with pytest.raises(ValueError, match="file_name"):
            Plots(m).plot_cv_errors(save_plot=True)
