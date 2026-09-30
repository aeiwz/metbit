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
        fig = STOCSY(spectra, anchor_ppm_value=ppm_values[10], plot="markers")
        assert len(fig.data) == 2

    def test_custom_p_value_threshold(self):
        spectra = _make_spectra()
        ppm_values = [float(c) for c in spectra.columns]
        fig = STOCSY(spectra, anchor_ppm_value=ppm_values[10], p_value_threshold=0.01, plot="markers")
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


class TestSTOCSYGradientLine:
    """The default plot: the median spectrum as a line coloured by r^2 (Plotly lines are single-coloured,
    so segments are grouped into colour levels, one trace per level)."""

    def _fig(self, **kw):
        # a broad peak (columns 15-35) driven by one factor: r^2 falls off from its centre
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 60)) * 0.3 + 5
        factor = rng.normal(size=40)
        for j in range(15, 35):
            X[:, j] += factor * 3 * np.exp(-(((j - 25) / 4) ** 2))
        spectra = pd.DataFrame(X, columns=np.linspace(5.0, 0.5, 60).tolist())
        kw.setdefault("p_value_threshold", 0.05)
        return spectra, STOCSY(spectra, anchor_ppm_value=float(spectra.columns[25]), **kw)

    def test_segments_carry_the_colour_of_their_r2(self):
        spectra, fig = self._fig()
        lines = [t for t in fig.data if t.mode == "lines"]
        assert lines and fig.layout.showlegend is False  # the colour bar explains the colours
        segments = sum(sum(v is None for v in t.x) for t in lines)
        assert segments == spectra.shape[1] - 1  # every neighbouring pair is drawn exactly once
        colours = {t.line.color for t in lines if not t.name.startswith("Not significant")}
        assert len(colours) > 1  # a gradient, not one colour

    def test_hover_layer_holds_the_colorbar_and_statistics(self):
        spectra, fig = self._fig()
        hover = [t for t in fig.data if t.mode == "markers"]
        assert len(hover) == 1
        h = hover[0]
        assert h.marker.showscale is True and (h.marker.cmin, h.marker.cmax) == (0, 1)
        assert len(h.x) == spectra.shape[1] and h.marker.opacity == 0
        assert h.customdata.shape == (spectra.shape[1], 2)  # r and p per point
        np.testing.assert_allclose(h.marker.color[20], h.customdata[20][0] ** 2)
        np.testing.assert_allclose(h.y, spectra.median().to_numpy())

    def test_non_significant_segments_are_grey(self):
        _, fig = self._fig(p_value_threshold=1e-12)  # nothing but the anchor itself is significant
        grey = [t for t in fig.data if (t.name or "").startswith("Not significant")]
        assert grey and grey[0].line.color == "#b0b0b0"
        assert "grey: p ≥ 1e-12" in fig.data[-1].marker.colorbar.title.text

    def test_colorscale_is_configurable_and_markers_keep_the_classic_plot(self):
        _, fig = self._fig(colorscale="Viridis")
        assert fig.data[-1].marker.colorscale[0][1].lower() == "#440154"
        _, classic = self._fig(plot="markers")
        assert len(classic.data) == 2 and {t.mode for t in classic.data} == {"markers"}
        with pytest.raises(ValueError, match="plot"):
            self._fig(plot="bars")

    def test_axis_titles_describe_what_is_drawn(self):
        _, fig = self._fig()
        assert "median" in fig.layout.yaxis.title.text.lower()
        assert fig.layout.xaxis.autorange == "reversed"
