# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from typing import List, Optional, Dict
import warnings


class UnivarStats:
    """
    Perform univariate statistical analysis and visualization using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the measurement and group columns.
    x_col : str
        Column name for the grouping variable.
    y_col : str
        Column name for the measurement variable.
    group_order : list of str, optional
        Custom group plotting order.
    custom_colors : dict of str -> str, optional
        Mapping from group name to color.
    stats_options : list of str, optional
        Supported: ["t-test", "anova", "nonparametric", "effect-size"].
    p_value_threshold : float, default=0.05
        Significance threshold.
    annotate_style : {'value', 'symbol'}, default='value'
        Annotation style: numeric or stars.
    y_offset_factor : float, default=0.35
        Vertical spacing factor for annotations.
    show_non_significant : bool, default=True
        Whether to display 'ns'.
    correct_p : str or None, default='bonferroni'
        Method for multiple testing correction. Supported:
            - 'bonferroni', 'holm', 'hochberg', 'hommel'
            - 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
            - None or 'none' = no correction
    title_ : str, optional
        Plot title.
    y_label : str, optional
        Y-axis label.
    x_label : str, optional
        X-axis label.
    fig_height : int, default=800
        Figure height.
    fig_width : int, default=600
        Figure width.
    plot_type : {'box', 'violin'}, default='box'
        Plot type.
    show_axis_lines : bool, default=True
        Whether to show axis lines.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_order: Optional[List[str]] = None,
        custom_colors: Optional[Dict[str, str]] = None,
        stats_options: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
        annotate_style: str = "value",
        y_offset_factor: float = 0.35,
        show_non_significant: bool = True,
        correct_p: Optional[str] = "bonferroni",
        title_: Optional[str] = None,
        y_label: Optional[str] = None,
        x_label: Optional[str] = None,
        fig_height: int = 800,
        fig_width: int = 600,
        plot_type: str = "box",
        show_axis_lines: bool = True,
    ):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.group_order = group_order
        self.custom_colors = custom_colors
        self.stats_options = stats_options or ["t-test"]
        self.p_value_threshold = p_value_threshold
        self.annotate_style = annotate_style
        self.y_offset_factor = y_offset_factor
        self.show_non_significant = show_non_significant
        self.correct_p = correct_p
        self.title_ = title_ or y_col
        self.y_label = y_label or y_col
        self.x_label = x_label or x_col
        self.fig_height = fig_height
        self.fig_width = fig_width
        self.plot_type = plot_type
        self.show_axis_lines = show_axis_lines

    @staticmethod
    def compute_effsize(a, b, eftype: str = "cohen") -> float:
        if eftype == "cohen":
            pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
            return (np.mean(a) - np.mean(b)) / pooled_std
        raise ValueError("Unsupported effect size type.")

    def plot(self, show_description: bool = True) -> go.Figure:
        warnings.filterwarnings("ignore")
        df = self.df
        if df.empty:
            raise ValueError("The DataFrame is empty.")

        grouped = df.groupby(self.x_col)[self.y_col]
        group_order = self.group_order or list(grouped.groups.keys())

        if self.custom_colors:
            missing = set(group_order) - set(self.custom_colors)
            if missing:
                raise ValueError(f"Missing colors for groups: {missing}")

        comparisons = list(combinations(group_order, 2))
        y_range = df[self.y_col].max() - df[self.y_col].min()
        y_offset = self.y_offset_factor * y_range
        max_y = df[self.y_col].max()

        raw_p_values = []
        effect_sizes = []

        if "anova" in self.stats_options and len(group_order) > 2:
            f_stat, anova_p = f_oneway(*(grouped.get_group(g).values for g in group_order))
            raw_p_values = [anova_p] * len(comparisons)
            corrected_p_values = raw_p_values
            if "effect-size" in self.stats_options:
                print("Effect sizes are skipped when using only ANOVA.")
        else:
            for g1, g2 in comparisons:
                group1 = grouped.get_group(g1).dropna().values
                group2 = grouped.get_group(g2).dropna().values
                print(f"Comparing {g1} vs {g2}: {len(group1)} vs {len(group2)} samples")

                if len(group1) < 2 or len(group2) < 2:
                    warnings.warn(f"Skipping {g1} vs {g2}: one of the groups has <2 samples.")
                    raw_p_values.append(np.nan)
                    if "effect-size" in self.stats_options:
                        effect_sizes.append(np.nan)
                    continue

                if "t-test" in self.stats_options:
                    _, p_val = ttest_ind(group1, group2, equal_var=False)
                elif "nonparametric" in self.stats_options:
                    _, p_val = mannwhitneyu(group1, group2, alternative="two-sided")
                else:
                    raise ValueError("Invalid stats_options.")

                raw_p_values.append(p_val)

                if "effect-size" in self.stats_options:
                    d = self.compute_effsize(group1, group2)
                    effect_sizes.append(d)

            # Correct p-values safely
            raw_array = np.array(raw_p_values, dtype=np.float64)
            corrected_array = np.full_like(raw_array, np.nan)

            valid_idx = [i for i, p in enumerate(raw_array) if not np.isnan(p)]
            if valid_idx:
                _, corrected_vals, _, _ = multipletests(raw_array[valid_idx], method=self.correct_p)
                corrected_array[valid_idx] = corrected_vals

            corrected_p_values = corrected_array.tolist()

        # Store results
        self._results = {
            "comparisons": comparisons,
            "raw_p_values": raw_p_values,
            "corrected_p_values": corrected_p_values,
            "effect_sizes": effect_sizes if effect_sizes else [None] * len(comparisons),
        }

        # Plot
        if self.plot_type == "box":
            fig = px.box(
                df, x=self.x_col, y=self.y_col, color=self.x_col,
                points="all", category_orders={self.x_col: group_order},
                color_discrete_map=self.custom_colors,
            )
        elif self.plot_type == "violin":
            fig = px.violin(
                df, x=self.x_col, y=self.y_col, color=self.x_col,
                box=True, points="all", category_orders={self.x_col: group_order},
                color_discrete_map=self.custom_colors,
            )
        else:
            raise ValueError("Invalid plot_type. Use 'box' or 'violin'.")

        annotations = []
        lines = []
        for i, ((g1, g2), p_val) in enumerate(zip(comparisons, corrected_p_values)):
            if np.isnan(p_val):
                continue
            if p_val > self.p_value_threshold and not self.show_non_significant:
                continue

            x1 = group_order.index(g1)
            x2 = group_order.index(g2)
            x_center = (x1 + x2) / 2
            y_pos = max_y + 0.15 + (i + 1) * y_offset

            if self.annotate_style == "value":
                p_text = f"p={p_val:.4f}"
            elif self.annotate_style == "symbol":
                if p_val < 0.001:
                    p_text = "***"
                elif p_val < 0.01:
                    p_text = "**"
                elif p_val < 0.05:
                    p_text = "*"
                else:
                    p_text = "ns"
            else:
                raise ValueError("Invalid annotate_style.")

            if "effect-size" in self.stats_options and len(effect_sizes) > i:
                if not np.isnan(effect_sizes[i]):
                    p_text += f", d={effect_sizes[i]:.2f}"

            annotations.append(dict(
                x=x_center,
                y=y_pos + y_offset * 0.75,
                text=p_text,
                showarrow=False,
                xref="x", yref="y",
                font=dict(size=12),
            ))

            lines.append(go.Scatter(
                x=[g1, g1, g2, g2],
                y=[y_pos, y_pos + y_offset * 0.5, y_pos + y_offset * 0.5, y_pos],
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="skip"
            ))

        for line in lines:
            fig.add_trace(line)

        axis_config = dict(showline=self.show_axis_lines, linewidth=2, linecolor="black")
        fig.update_layout(
            annotations=annotations,
            title=dict(text=f"<b>{self.title_}</b>", x=0.5),
            yaxis_title=self.y_label,
            xaxis_title=self.x_label,
            legend_title=self.x_col,
            width=self.fig_width,
            height=self.fig_height,
            showlegend=False,
            yaxis=dict(tickformat=".2e", **axis_config),
            xaxis=axis_config,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(b=140)
        )

        if show_description and self.annotate_style == "symbol":
            legend_text = (
                f"<b>stat:</b> {' '.join(self.stats_options)}<br>"
                f"<b>corrected:</b> {self.correct_p}<br>"
                "* p < 0.05, ** p < 0.01, *** p < 0.001"
            )
            if self.show_non_significant:
                legend_text += ", ns = not significant"
            fig.add_annotation(
                text=legend_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.18,
                showarrow=False,
                font=dict(size=12),
                align="left"
            )

        fig.update_yaxes(range=[None, max_y + y_offset * (len(comparisons) + 2)])
        return fig

    def get_stats_table(self) -> pd.DataFrame:
        """Return a DataFrame of statistical results."""
        res = self._results
        return pd.DataFrame({
            "Comparison": [f"{a} vs {b}" for a, b in res["comparisons"]],
            "Raw P-Value": res["raw_p_values"],
            "Corrected P-Value": res["corrected_p_values"],
            "Effect Size (Cohen's d)": res["effect_sizes"]
        })
