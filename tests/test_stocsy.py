import numpy as np
import pandas as pd
import pytest

from metbit.analysis.stocsy import STOCSY, _stocsy_statistics


def _make_spectra(n=20, p=50, seed=0):
    rng = np.random.default_rng(seed)
    ppm = np.linspace(0.5, 5.0, p)
    data = rng.standard_normal((n, p))
    # Add a correlated signal at two positions for later testing
    data[:, 10] = rng.standard_normal(n)
    data[:, 20] = data[:, 10] * 2 + rng.standard_normal(n) * 0.1
    return pd.DataFrame(data, columns=ppm.tolist())


class TestSTOCSY:
    def test_statistics_match_scipy(self):
        from scipy.stats import pearsonr

        spectra = _make_spectra(n=32, p=75)
        correlations, p_values = _stocsy_statistics(spectra, anchor_index=10)
        expected = [
            pearsonr(spectra.iloc[:, 10], spectra.iloc[:, column])
            for column in range(spectra.shape[1])
        ]

        np.testing.assert_allclose(
            correlations,
            [result.statistic for result in expected],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            p_values,
            [result.pvalue for result in expected],
            rtol=1e-10,
            atol=1e-14,
        )

    def test_statistics_with_two_samples_use_unit_p_values(self):
        spectra = _make_spectra(n=2, p=25)

        correlations, p_values = _stocsy_statistics(spectra, anchor_index=1)

        assert correlations.shape == (25,)
        np.testing.assert_allclose(p_values, np.ones(25))

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        anchor = ppm_values[10]
        fig = STOCSY(spectra, anchor_ppm_value=anchor)
        assert isinstance(fig, go.Figure)

    def test_figure_has_two_traces(self):
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        fig = STOCSY(spectra, anchor_ppm_value=ppm_values[10])
        assert len(fig.data) == 2

    def test_custom_p_value_threshold(self):
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        fig = STOCSY(spectra, anchor_ppm_value=ppm_values[10], p_value_threshold=0.01)
        assert len(fig.data) == 2

    def test_anchor_at_first_column(self):
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        fig = STOCSY(spectra, anchor_ppm_value=ppm_values[0])
        assert fig is not None

    def test_anchor_at_last_column(self):
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        fig = STOCSY(spectra, anchor_ppm_value=ppm_values[-1])
        assert fig is not None

    def test_title_contains_anchor_ppm(self):
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        anchor = ppm_values[10]
        fig = STOCSY(spectra, anchor_ppm_value=anchor)
        assert "STOCSY" in fig.layout.title.text
