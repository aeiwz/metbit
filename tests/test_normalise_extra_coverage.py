"""Extra tests for metbit/stats/normalise.py - covers Normality_distribution and remaining Normalise methods."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from metbit.stats.normalise import Normality_distribution, Normalise


def _make_medium_data(n=60, p=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, p)),
                        columns=[f"f{i}" for i in range(p)])


class TestNormalityDistribution:
    def test_init_prints_info(self, capsys):
        data = _make_medium_data()
        nd = Normality_distribution(data)
        captured = capsys.readouterr()
        assert "features" in captured.out

    def test_init_with_large_data_covers_division(self, capsys):
        rng = np.random.default_rng(1)
        data = pd.DataFrame(rng.standard_normal((200, 200)),
                            columns=[f"f{i}" for i in range(200)])
        nd = Normality_distribution(data)
        captured = capsys.readouterr()
        assert "samples" in captured.out

    def test_plot_distribution(self):
        data = _make_medium_data()
        nd = Normality_distribution(data)
        with patch("matplotlib.pyplot.show"):
            result = nd.plot_distribution("f0")
        assert result is not None

    def test_pca_distributions(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.standard_normal((40, 10)),
                            columns=[f"f{i}" for i in range(10)])
        nd = Normality_distribution(data)
        with patch("matplotlib.pyplot.show"):
            result = nd.pca_distributions()
        assert result is not None


class TestNormaliseMediumData:
    def test_init_covers_kb_division(self, capsys):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.standard_normal((200, 100)),
                            columns=[f"f{i}" for i in range(100)])
        normaliser = Normalise(data, compute_missing=False)
        captured = capsys.readouterr()
        assert "features" in captured.out

    def test_pqn_normalise_with_plot(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(np.abs(rng.standard_normal((20, 10))) + 0.1,
                            columns=[f"f{i}" for i in range(10)])
        normaliser = Normalise(data, compute_missing=False)
        with patch("matplotlib.pyplot.show"):
            result = normaliser.pqn_normalise(plot=True)
        assert result.shape == data.shape

    def test_normalize_to_100(self):
        data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        normaliser = Normalise(data, compute_missing=False)
        result = normaliser.normalize_to_100()
        assert result.shape == data.shape

    def test_clipping_normalisation(self):
        data = pd.DataFrame({"a": [0.1, 5.0, 10.0], "b": [2.0, 4.0, 8.0]})
        normaliser = Normalise(data, compute_missing=False)
        result = normaliser.clipping_normalisation(lower=1.0, upper=7.0)
        assert result.min().min() >= 1.0
        assert result.max().max() <= 7.0

    def test_standard_deviation_normalisation(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.standard_normal((20, 5)),
                            columns=[f"f{i}" for i in range(5)])
        normaliser = Normalise(data, compute_missing=False)
        result = normaliser.standard_deviation_normalisation()
        assert result.shape == data.shape
