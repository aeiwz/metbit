"""Tests to achieve 100% coverage of metbit/stats/univariate.py."""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from metbit.stats.univariate import UnivarStats


def _make_df(groups=2, n_per_group=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        for _ in range(n_per_group):
            rows.append({"group": f"G{g}", "value": rng.normal(g * 2, 1)})
    return pd.DataFrame(rows)


class TestUnivarStatsPlot:
    def test_basic_t_test_plot(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value", stats_options=["t-test"])
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_nonparametric_plot(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["nonparametric"],
                         correct_p="bonferroni")
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_effect_size_plot(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test", "effect-size"])
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_anova_plot_three_groups(self):
        df = _make_df(groups=3)
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["anova", "effect-size"])
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_annotate_style(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"], annotate_style="symbol",
                         show_non_significant=True)
        fig = us.plot(show_description=True)
        assert isinstance(fig, go.Figure)

    def test_violin_plot_type(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         plot_type="violin")
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_custom_colors(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         custom_colors={"G0": "red", "G1": "blue"})
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_custom_colors_missing_raises(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         custom_colors={"G0": "red"})  # G1 missing
        with pytest.raises(ValueError):
            us.plot()

    def test_empty_df_raises(self):
        df = pd.DataFrame({"group": [], "value": []})
        us = UnivarStats(df, x_col="group", y_col="value")
        with pytest.raises(ValueError):
            us.plot()

    def test_invalid_plot_type_raises(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value", plot_type="bar")
        with pytest.raises(ValueError):
            us.plot()

    def test_invalid_annotate_style_raises(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         annotate_style="stars")
        with pytest.raises(ValueError):
            us.plot()

    def test_invalid_stats_option_raises(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["unknown"])
        with pytest.raises(ValueError):
            us.plot()

    def test_show_non_significant_false(self):
        df = _make_df(seed=99)
        us = UnivarStats(df, x_col="group", y_col="value",
                         show_non_significant=False)
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_group_with_small_sample_warns(self):
        df = pd.DataFrame({
            "group": ["G0", "G1", "G1", "G1"],
            "value": [1.0, 2.0, 3.0, 4.0]
        })
        us = UnivarStats(df, x_col="group", y_col="value", stats_options=["t-test"])
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_get_stats_table(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value")
        us.plot()
        tbl = us.get_stats_table()
        assert "Comparison" in tbl.columns
        assert "Raw P-Value" in tbl.columns

    def test_compute_effsize_cohen(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        d = UnivarStats.compute_effsize(a, b)
        assert isinstance(d, float)

    def test_compute_effsize_unsupported_raises(self):
        with pytest.raises(ValueError):
            UnivarStats.compute_effsize([1, 2], [3, 4], eftype="unknown")

    def test_fdr_correction(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         correct_p="fdr_bh")
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_show_axis_lines_false(self):
        df = _make_df()
        us = UnivarStats(df, x_col="group", y_col="value",
                         show_axis_lines=False)
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_effect_size_with_small_group(self):
        # Group G0 has only 1 sample → skips t-test and appends np.nan for effect-size (line 146)
        df = pd.DataFrame({
            "group": ["G0", "G1", "G1", "G1", "G1"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test", "effect-size"])
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_annotate_non_significant_skipped(self):
        # Non-significant data with symbol style and show_non_significant=False -> line 203 continue
        rng = np.random.default_rng(42)
        rows = []
        for g in range(2):
            for _ in range(15):
                rows.append({"group": f"G{g}", "value": rng.normal(0, 1)})
        df = pd.DataFrame(rows)
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"],
                         annotate_style="symbol",
                         show_non_significant=False,
                         p_value_threshold=0.001)
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_triple_star(self):
        # Groups far apart → p < 0.001 → *** (line 214)
        rows = []
        for _ in range(30):
            rows.append({"group": "G0", "value": 100.0})
            rows.append({"group": "G1", "value": 0.0})
        df = pd.DataFrame(rows)
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"],
                         annotate_style="symbol")
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_double_star(self):
        # Groups moderately apart → p in [0.001, 0.01) → ** (line 215-216)
        rng = np.random.default_rng(7)
        rows = []
        for _ in range(10):
            rows.append({"group": "G0", "value": rng.normal(0, 1)})
            rows.append({"group": "G1", "value": rng.normal(5, 1)})
        df = pd.DataFrame(rows)
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"],
                         annotate_style="symbol",
                         p_value_threshold=0.05)
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_ns(self):
        # Identical groups → p=1.0 → ns branch (line 220)
        df = pd.DataFrame({
            "group": ["G0"] * 5 + ["G1"] * 5,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0] * 2
        })
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"],
                         annotate_style="symbol",
                         show_non_significant=True,
                         p_value_threshold=0.05)
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_single_star(self):
        # Deterministic data: p ≈ 0.037 → * branch (lines 217-218)
        # G0 mean=2, G1 mean=4.5, std≈1.58, n=5 → t≈2.5, p≈0.037
        df = pd.DataFrame({
            "group": ["G0"] * 5 + ["G1"] * 5,
            "value": [0.0, 1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 4.5, 5.5, 6.5]
        })
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"],
                         annotate_style="symbol",
                         show_non_significant=True,
                         p_value_threshold=0.05,
                         correct_p="bonferroni")
        fig = us.plot()
        assert isinstance(fig, go.Figure)

    def test_symbol_double_star_deterministic(self):
        # Deterministic data: p ≈ 0.004 → ** branch (lines 215-216)
        # G0 mean=2, G1 mean=6, std≈1.58, n=5 → t≈4, p≈0.004
        df = pd.DataFrame({
            "group": ["G0"] * 5 + ["G1"] * 5,
            "value": [0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        })
        us = UnivarStats(df, x_col="group", y_col="value",
                         stats_options=["t-test"],
                         annotate_style="symbol",
                         show_non_significant=True,
                         p_value_threshold=0.05,
                         correct_p="bonferroni")
        fig = us.plot()
        assert isinstance(fig, go.Figure)
