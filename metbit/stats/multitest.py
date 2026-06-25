from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "aeiwz"
__copyright__ = "Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p_to_stars(p: float, threshold: float = 0.05) -> str:
    """Convert a p-value to a significance star string.

    Parameters
    ----------
    p : float
        P-value to convert.
    threshold : float, default=0.05
        Significance threshold.

    Returns
    -------
    str
        Star annotation string: "***", "**", "*", or "ns".
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < threshold:
        return "*"
    return "ns"


def _apply_correction(
    p_values: np.ndarray, method: Optional[str]
) -> np.ndarray:
    """Apply multiple testing correction to an array of p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values (may contain NaN).
    method : str or None
        Correction method passed to ``statsmodels.stats.multitest.multipletests``.
        Pass ``None`` to skip correction.

    Returns
    -------
    np.ndarray
        Corrected p-values of the same length as input.
    """
    if method is None:
        return p_values.copy()

    corrected = np.full_like(p_values, np.nan, dtype=float)
    valid = ~np.isnan(p_values)
    if valid.any():
        _, corrected_vals, _, _ = multipletests(p_values[valid], method=method)
        corrected[valid] = corrected_vals
    return corrected


# ---------------------------------------------------------------------------
# CLASS 1: VolcanoPlot
# ---------------------------------------------------------------------------

class VolcanoPlot:
    """Volcano plot for two-group differential analysis.

    Computes per-feature log2 fold change and p-values (Welch's t-test),
    optionally applies multiple testing correction, classifies each feature
    as Up / Down / NS, and renders an interactive Plotly volcano plot.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy DataFrame containing group labels and numeric feature columns.
    group_col : str
        Column name containing group labels. Must have exactly two unique values.
    value_cols : list of str, optional
        Subset of numeric columns to analyse. If ``None``, all numeric columns
        (excluding ``group_col``) are used.
    group_a : str, optional
        Label of the reference group ("control"). If ``None``, the
        lexicographically first unique value in ``group_col`` is used.
    group_b : str, optional
        Label of the comparison group ("treatment"). If ``None``, the
        lexicographically second unique value in ``group_col`` is used.
    p_value_threshold : float, default=0.05
        Significance threshold applied to the (corrected) p-value.
    fc_threshold : float, default=1.0
        |log2FC| threshold for calling a feature "changed".
    correct_p : str or None, default="fdr_bh"
        Multiple testing correction method passed to
        ``statsmodels.stats.multitest.multipletests``.
        Common values: ``"fdr_bh"``, ``"bonferroni"``. Pass ``None`` to
        skip correction.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from metbit.stats.multitest import VolcanoPlot
    >>> np.random.seed(42)
    >>> n = 30
    >>> bins = [f"bin_{i:.2f}" for i in np.linspace(0.5, 10.0, 50)]
    >>> ctrl = pd.DataFrame(np.random.normal(5, 1, (n, 50)), columns=bins)
    >>> treat = pd.DataFrame(np.random.normal(6, 1, (n, 50)), columns=bins)
    >>> ctrl["group"] = "Control"
    >>> treat["group"] = "Treatment"
    >>> df = pd.concat([ctrl, treat], ignore_index=True)
    >>> vp = VolcanoPlot(df, group_col="group", correct_p="fdr_bh")
    >>> fig = vp.plot(title="NMR Metabolomics Volcano Plot")
    >>> table = vp.get_table()
    >>> print(table.head())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_cols: Optional[List[str]] = None,
        group_a: Optional[str] = None,
        group_b: Optional[str] = None,
        p_value_threshold: float = 0.05,
        fc_threshold: float = 1.0,
        correct_p: Optional[str] = "fdr_bh",
    ) -> None:
        if group_col not in df.columns:
            raise ValueError(f"group_col '{group_col}' not found in DataFrame.")

        unique_groups = sorted(df[group_col].dropna().unique())
        if len(unique_groups) != 2:
            raise ValueError(
                f"VolcanoPlot requires exactly 2 unique group values; "
                f"found {len(unique_groups)}: {unique_groups}."
            )

        self.df = df.copy()
        self.group_col = group_col
        self.group_a = group_a if group_a is not None else unique_groups[0]
        self.group_b = group_b if group_b is not None else unique_groups[1]
        self.p_value_threshold = p_value_threshold
        self.fc_threshold = fc_threshold
        self.correct_p = correct_p

        if value_cols is not None:
            self.value_cols = list(value_cols)
        else:
            self.value_cols = [
                c for c in df.select_dtypes(include="number").columns
                if c != group_col
            ]

        self._result_df: Optional[pd.DataFrame] = None
        self._compute()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        """Compute log2FC, p-values, correction, and feature labels."""
        mask_a = self.df[self.group_col] == self.group_a
        mask_b = self.df[self.group_col] == self.group_b
        data_a = self.df.loc[mask_a, self.value_cols]
        data_b = self.df.loc[mask_b, self.value_cols]

        rows = []
        for feat in self.value_cols:
            a_vals = data_a[feat].dropna().values
            b_vals = data_b[feat].dropna().values

            mean_a = np.mean(a_vals) if len(a_vals) else np.nan
            mean_b = np.mean(b_vals) if len(b_vals) else np.nan

            # log2FC: log2(mean_b / mean_a)
            if mean_a > 0 and mean_b > 0:
                log2fc = np.log2(mean_b / mean_a)
            elif mean_a == 0 and mean_b == 0:
                log2fc = 0.0
            else:
                log2fc = np.nan

            # Welch t-test
            if len(a_vals) >= 2 and len(b_vals) >= 2:
                _, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
            else:
                p_val = np.nan

            rows.append({"feature": feat, "log2FC": log2fc, "p_value": p_val})

        result = pd.DataFrame(rows)

        # Multiple testing correction
        p_raw = result["p_value"].values.astype(float)
        p_adj = _apply_correction(p_raw, self.correct_p)
        result["p_adj"] = p_adj

        # Use corrected p if correction applied, else raw
        p_for_threshold = p_adj if self.correct_p is not None else p_raw
        result["neg_log10_p"] = -np.log10(np.where(p_for_threshold > 0, p_for_threshold, np.nan))

        # Classify features
        sig = p_for_threshold < self.p_value_threshold
        up = sig & (result["log2FC"] > self.fc_threshold)
        down = sig & (result["log2FC"] < -self.fc_threshold)
        result["label"] = "NS"
        result.loc[up, "label"] = "Up"
        result.loc[down, "label"] = "Down"

        self._result_df = result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_table(self) -> pd.DataFrame:
        """Return the per-feature statistical results.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``feature``, ``log2FC``, ``p_value``,
            ``p_adj``, ``neg_log10_p``, ``label``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import VolcanoPlot
        >>> np.random.seed(0)
        >>> bins = [f"bin_{i:.2f}" for i in np.linspace(0.5, 10.0, 20)]
        >>> ctrl = pd.DataFrame(np.random.normal(5, 1, (20, 20)), columns=bins)
        >>> treat = pd.DataFrame(np.random.normal(6, 1, (20, 20)), columns=bins)
        >>> ctrl["group"] = "Control"
        >>> treat["group"] = "Treatment"
        >>> df = pd.concat([ctrl, treat], ignore_index=True)
        >>> vp = VolcanoPlot(df, group_col="group")
        >>> tbl = vp.get_table()
        >>> print(tbl.columns.tolist())
        ['feature', 'log2FC', 'p_value', 'p_adj', 'neg_log10_p', 'label']
        """
        return self._result_df.copy()

    def plot(
        self,
        title: Optional[str] = None,
        fig_width: int = 900,
        fig_height: int = 700,
        font_size: int = 14,
        label_top_n: int = 10,
    ) -> go.Figure:
        """Render the volcano plot.

        Parameters
        ----------
        title : str, optional
            Plot title. Defaults to a generated title including group names.
        fig_width : int, default=900
            Figure width in pixels.
        fig_height : int, default=700
            Figure height in pixels.
        font_size : int, default=14
            Base font size for axis labels and tick marks.
        label_top_n : int, default=10
            Number of top significant features to label by name.

        Returns
        -------
        go.Figure
            Interactive Plotly volcano plot.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import VolcanoPlot
        >>> np.random.seed(1)
        >>> bins = [f"bin_{i:.2f}" for i in np.linspace(0.5, 10.0, 40)]
        >>> ctrl = pd.DataFrame(np.random.normal(5, 1, (25, 40)), columns=bins)
        >>> treat = pd.DataFrame(np.random.normal(5.8, 1, (25, 40)), columns=bins)
        >>> ctrl["group"] = "Control"
        >>> treat["group"] = "Treatment"
        >>> df = pd.concat([ctrl, treat], ignore_index=True)
        >>> vp = VolcanoPlot(df, group_col="group")
        >>> fig = vp.plot(title="NMR Differential Analysis")
        >>> fig.show()  # doctest: +SKIP
        """
        result = self._result_df.copy()
        color_map = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#aec7e8"}
        plot_title = title or (
            f"Volcano Plot: {self.group_b} vs {self.group_a}"
        )

        fig = go.Figure()
        for label_cat in ["NS", "Up", "Down"]:
            sub = result[result["label"] == label_cat]
            fig.add_trace(
                go.Scatter(
                    x=sub["log2FC"],
                    y=sub["neg_log10_p"],
                    mode="markers",
                    name=label_cat,
                    marker=dict(
                        color=color_map[label_cat],
                        size=7,
                        opacity=0.75,
                        line=dict(width=0.5, color="white"),
                    ),
                    text=sub["feature"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "log2FC: %{x:.3f}<br>"
                        "-log10(p): %{y:.3f}<extra></extra>"
                    ),
                )
            )

        # Label top_n most significant changed features
        changed = result[result["label"].isin(["Up", "Down"])].copy()
        changed = changed.sort_values("neg_log10_p", ascending=False).head(label_top_n)
        for _, row in changed.iterrows():
            if pd.isna(row["neg_log10_p"]):
                continue
            fig.add_annotation(
                x=row["log2FC"],
                y=row["neg_log10_p"],
                text=str(row["feature"]),
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1,
                ax=20,
                ay=-20,
                font=dict(size=10),
            )

        # Threshold lines
        y_max_val = result["neg_log10_p"].replace([np.inf, -np.inf], np.nan).dropna().max()
        y_line = -np.log10(self.p_value_threshold)
        fig.add_hline(
            y=y_line,
            line_dash="dash",
            line_color="grey",
            opacity=0.6,
            annotation_text=f"p={self.p_value_threshold}",
            annotation_position="top right",
        )
        fig.add_vline(
            x=self.fc_threshold,
            line_dash="dash",
            line_color="grey",
            opacity=0.6,
        )
        fig.add_vline(
            x=-self.fc_threshold,
            line_dash="dash",
            line_color="grey",
            opacity=0.6,
        )

        p_label = (
            f"-log10(p_adj [{self.correct_p}])"
            if self.correct_p
            else "-log10(p_value)"
        )
        fig.update_layout(
            title=dict(text=f"<b>{plot_title}</b>", x=0.5),
            xaxis_title="log2 Fold Change",
            yaxis_title=p_label,
            width=fig_width,
            height=fig_height,
            font=dict(size=font_size),
            legend_title="Regulation",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showline=True, linewidth=1.5, linecolor="black", zeroline=False),
            yaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
        )
        return fig


# ---------------------------------------------------------------------------
# CLASS 2: ANOVAStats
# ---------------------------------------------------------------------------

class ANOVAStats:
    """One-way ANOVA with Tukey HSD post-hoc for multi-group comparisons.

    Fits a one-way ANOVA across all groups in ``x_col`` for the numeric
    response ``y_col``, then runs pairwise Tukey HSD comparisons. Results
    can be visualised as annotated box or violin plots.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy DataFrame containing group labels and the response variable.
    x_col : str
        Column name for the grouping variable (categorical).
    y_col : str
        Column name for the numeric response variable.
    group_order : list of str, optional
        Display order of groups. If ``None``, groups are sorted alphabetically.
    p_value_threshold : float, default=0.05
        Significance threshold for bracket annotations.
    correct_p : str or None, default="fdr_bh"
        Multiple testing correction applied to Tukey HSD p-values.
        Note: Tukey HSD already controls FWER; this parameter allows
        additional FDR correction if desired.
    fig_height : int, default=600
        Figure height in pixels.
    fig_width : int, default=800
        Figure width in pixels.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from metbit.stats.multitest import ANOVAStats
    >>> np.random.seed(42)
    >>> n = 20
    >>> bins = "bin_3.50"
    >>> groups = (
    ...     ["Control"] * n + ["Low_Dose"] * n + ["High_Dose"] * n
    ... )
    >>> values = np.concatenate([
    ...     np.random.normal(5.0, 0.8, n),
    ...     np.random.normal(5.8, 0.8, n),
    ...     np.random.normal(7.2, 0.8, n),
    ... ])
    >>> df = pd.DataFrame({"group": groups, "intensity": values})
    >>> an = ANOVAStats(df, x_col="group", y_col="intensity")
    >>> an.fit()
    ANOVAStats(x_col='group', y_col='intensity')
    >>> fig = an.plot(title="NMR Bin 3.50 ppm")
    >>> print(an.get_anova_table())
    >>> print(an.get_posthoc_table())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_order: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
        correct_p: Optional[str] = "fdr_bh",
        fig_height: int = 600,
        fig_width: int = 800,
    ) -> None:
        self.df = df.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.group_order = group_order or sorted(df[x_col].dropna().unique().tolist())
        self.p_value_threshold = p_value_threshold
        self.correct_p = correct_p
        self.fig_height = fig_height
        self.fig_width = fig_width

        self._anova_f: Optional[float] = None
        self._anova_p: Optional[float] = None
        self._posthoc_df: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> "ANOVAStats":
        """Run one-way ANOVA and Tukey HSD post-hoc test.

        Returns
        -------
        ANOVAStats
            Returns ``self`` to allow method chaining.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import ANOVAStats
        >>> np.random.seed(0)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        ...     "val": np.concatenate([
        ...         np.random.normal(4, 1, 15),
        ...         np.random.normal(6, 1, 15),
        ...         np.random.normal(5, 1, 15),
        ...     ])
        ... })
        >>> an = ANOVAStats(df, x_col="group", y_col="val").fit()
        >>> print(an.get_anova_table())
        """
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        group_data = [
            self.df.loc[self.df[self.x_col] == g, self.y_col].dropna().values
            for g in self.group_order
        ]
        f_stat, p_val = stats.f_oneway(*group_data)
        self._anova_f = float(f_stat)
        self._anova_p = float(p_val)

        # Tukey HSD on the full (stacked) data
        tukey = pairwise_tukeyhsd(
            endog=self.df[self.y_col].dropna().values,
            groups=self.df.loc[self.df[self.y_col].notna(), self.x_col].values,
            alpha=self.p_value_threshold,
        )
        summary_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
        summary_df.columns = ["group1", "group2", "meandiff", "p_adj", "lower", "upper", "reject"]
        summary_df["meandiff"] = summary_df["meandiff"].astype(float)
        summary_df["p_adj"] = summary_df["p_adj"].astype(float)
        summary_df["reject"] = summary_df["reject"].astype(bool)

        # Optional additional correction on top of Tukey
        if self.correct_p is not None:
            p_vals_raw = summary_df["p_adj"].values.astype(float)
            summary_df["p_adj"] = _apply_correction(p_vals_raw, self.correct_p)

        self._posthoc_df = summary_df[["group1", "group2", "meandiff", "p_adj", "reject"]].copy()
        self._fitted = True
        return self

    def get_anova_table(self) -> pd.DataFrame:
        """Return overall ANOVA F-statistic and p-value.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns: ``F_statistic``, ``p_value``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import ANOVAStats
        >>> np.random.seed(7)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        ...     "val": np.concatenate([
        ...         np.random.normal(3, 1, 20),
        ...         np.random.normal(5, 1, 20),
        ...         np.random.normal(4, 1, 20),
        ...     ])
        ... })
        >>> an = ANOVAStats(df, x_col="group", y_col="val").fit()
        >>> print(an.get_anova_table())
        """
        self._check_fitted()
        return pd.DataFrame(
            {"F_statistic": [self._anova_f], "p_value": [self._anova_p]}
        )

    def get_posthoc_table(self) -> pd.DataFrame:
        """Return pairwise Tukey HSD results.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``group1``, ``group2``, ``meandiff``,
            ``p_adj``, ``reject``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import ANOVAStats
        >>> np.random.seed(3)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        ...     "val": np.concatenate([
        ...         np.random.normal(2, 1, 15),
        ...         np.random.normal(5, 1, 15),
        ...         np.random.normal(3, 1, 15),
        ...     ])
        ... })
        >>> an = ANOVAStats(df, x_col="group", y_col="val").fit()
        >>> print(an.get_posthoc_table())
        """
        self._check_fitted()
        return self._posthoc_df.copy()

    def plot(
        self,
        plot_type: str = "box",
        font_size: int = 14,
        title: Optional[str] = None,
        custom_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """Render an annotated box or violin plot with Tukey significance brackets.

        Parameters
        ----------
        plot_type : str, default="box"
            Either ``"box"`` or ``"violin"``.
        font_size : int, default=14
            Base font size.
        title : str, optional
            Plot title. Defaults to ``y_col``.
        custom_colors : dict of str -> str, optional
            Mapping from group name to hex color string.

        Returns
        -------
        go.Figure
            Annotated Plotly figure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import ANOVAStats
        >>> np.random.seed(5)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        ...     "intensity": np.concatenate([
        ...         np.random.normal(4, 0.8, 20),
        ...         np.random.normal(6, 0.8, 20),
        ...         np.random.normal(5, 0.8, 20),
        ...     ])
        ... })
        >>> fig = ANOVAStats(df, x_col="group", y_col="intensity").fit().plot()
        >>> fig.show()  # doctest: +SKIP
        """
        self._check_fitted()
        plot_title = title or self.y_col
        color_map = custom_colors

        if plot_type == "box":
            fig = px.box(
                self.df,
                x=self.x_col,
                y=self.y_col,
                color=self.x_col,
                points="all",
                category_orders={self.x_col: self.group_order},
                color_discrete_map=color_map,
            )
        elif plot_type == "violin":
            fig = px.violin(
                self.df,
                x=self.x_col,
                y=self.y_col,
                color=self.x_col,
                box=True,
                points="all",
                category_orders={self.x_col: self.group_order},
                color_discrete_map=color_map,
            )
        else:
            raise ValueError("plot_type must be 'box' or 'violin'.")

        fig = self._add_significance_brackets(fig)

        fig.update_layout(
            title=dict(text=f"<b>{plot_title}</b>", x=0.5),
            xaxis_title=self.x_col,
            yaxis_title=self.y_col,
            font=dict(size=font_size),
            width=self.fig_width,
            height=self.fig_height,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
            yaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
        )
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing results or plots.")

    def _add_significance_brackets(self, fig: go.Figure) -> go.Figure:
        """Add Tukey HSD significance brackets to an existing figure.

        Parameters
        ----------
        fig : go.Figure
            Base Plotly figure to annotate.

        Returns
        -------
        go.Figure
            Annotated figure.
        """
        y_vals = self.df[self.y_col].dropna()
        y_max = float(y_vals.max())
        y_range = float(y_vals.max() - y_vals.min())
        step = y_range * 0.12

        annotations = []
        pairs = self._posthoc_df[
            self._posthoc_df["p_adj"] < self.p_value_threshold
        ].copy()

        for idx, (_, row) in enumerate(pairs.iterrows()):
            g1, g2 = str(row["group1"]), str(row["group2"])
            if g1 not in self.group_order or g2 not in self.group_order:
                continue
            x1 = self.group_order.index(g1)
            x2 = self.group_order.index(g2)
            y_pos = y_max + step * (idx + 1)

            star = _p_to_stars(float(row["p_adj"]), self.p_value_threshold)

            # Bracket line
            fig.add_trace(
                go.Scatter(
                    x=[g1, g1, g2, g2],
                    y=[y_pos, y_pos + step * 0.4, y_pos + step * 0.4, y_pos],
                    mode="lines",
                    line=dict(color="black", width=1.2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            annotations.append(
                dict(
                    x=(x1 + x2) / 2,
                    y=y_pos + step * 0.55,
                    xref="x",
                    yref="y",
                    text=star,
                    showarrow=False,
                    font=dict(size=13),
                )
            )

        fig.update_layout(annotations=annotations)
        if pairs.shape[0] > 0:
            n_brackets = pairs.shape[0]
            fig.update_yaxes(range=[None, y_max + step * (n_brackets + 2)])
        return fig

    def __repr__(self) -> str:
        return f"ANOVAStats(x_col={self.x_col!r}, y_col={self.y_col!r})"


# ---------------------------------------------------------------------------
# CLASS 3: KruskalStats
# ---------------------------------------------------------------------------

class KruskalStats:
    """Kruskal-Wallis test with Dunn post-hoc for non-parametric multi-group comparisons.

    A non-parametric alternative to :class:`ANOVAStats`. Uses
    ``scipy.stats.kruskal`` for the overall test and implements Dunn's test
    manually (rank-sum z-scores) for pairwise comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy DataFrame containing group labels and the response variable.
    x_col : str
        Column name for the grouping variable (categorical).
    y_col : str
        Column name for the numeric response variable.
    group_order : list of str, optional
        Display order of groups. If ``None``, groups are sorted alphabetically.
    p_value_threshold : float, default=0.05
        Significance threshold for bracket annotations.
    correct_p : str or None, default="fdr_bh"
        Multiple testing correction applied to Dunn post-hoc p-values.
        Common values: ``"fdr_bh"``, ``"bonferroni"``. Pass ``None`` to skip.
    fig_height : int, default=600
        Figure height in pixels.
    fig_width : int, default=800
        Figure width in pixels.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from metbit.stats.multitest import KruskalStats
    >>> np.random.seed(42)
    >>> n = 20
    >>> groups = ["Control"] * n + ["Low_Dose"] * n + ["High_Dose"] * n
    >>> values = np.concatenate([
    ...     np.random.exponential(2, n),
    ...     np.random.exponential(4, n),
    ...     np.random.exponential(7, n),
    ... ])
    >>> df = pd.DataFrame({"group": groups, "intensity": values})
    >>> kr = KruskalStats(df, x_col="group", y_col="intensity")
    >>> kr.fit()
    KruskalStats(x_col='group', y_col='intensity')
    >>> fig = kr.plot(title="NMR Bin Kruskal-Wallis")
    >>> print(kr.get_kruskal_table())
    >>> print(kr.get_posthoc_table())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_order: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
        correct_p: Optional[str] = "fdr_bh",
        fig_height: int = 600,
        fig_width: int = 800,
    ) -> None:
        self.df = df.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.group_order = group_order or sorted(df[x_col].dropna().unique().tolist())
        self.p_value_threshold = p_value_threshold
        self.correct_p = correct_p
        self.fig_height = fig_height
        self.fig_width = fig_width

        self._kruskal_h: Optional[float] = None
        self._kruskal_p: Optional[float] = None
        self._posthoc_df: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> "KruskalStats":
        """Run Kruskal-Wallis test and Dunn post-hoc pairwise comparisons.

        Dunn's test ranks all observations jointly, then computes a z-score
        for each pair ``(i, j)``:

        .. math::

            z = \\frac{\\bar{R}_i - \\bar{R}_j}
                      {\\sqrt{\\frac{N(N+1)}{12}
                      \\left(\\frac{1}{n_i} + \\frac{1}{n_j}\\right)}}

        The two-sided p-value follows from the standard normal distribution.
        Multiple testing correction is applied if ``correct_p`` is set.

        Returns
        -------
        KruskalStats
            Returns ``self`` to allow method chaining.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import KruskalStats
        >>> np.random.seed(0)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        ...     "val": np.concatenate([
        ...         np.random.exponential(2, 15),
        ...         np.random.exponential(5, 15),
        ...         np.random.exponential(3, 15),
        ...     ])
        ... })
        >>> kr = KruskalStats(df, x_col="group", y_col="val").fit()
        >>> print(kr.get_kruskal_table())
        """
        clean = self.df[[self.x_col, self.y_col]].dropna()
        group_data = [
            clean.loc[clean[self.x_col] == g, self.y_col].values
            for g in self.group_order
        ]

        h_stat, p_val = stats.kruskal(*group_data)
        self._kruskal_h = float(h_stat)
        self._kruskal_p = float(p_val)

        # Dunn's test: rank all observations jointly
        all_values = clean[self.y_col].values
        N = len(all_values)
        ranks = stats.rankdata(all_values)

        # Build rank array per group using the same row order as clean
        group_ranks: Dict[str, np.ndarray] = {}
        for g in self.group_order:
            mask = clean[self.x_col].values == g
            group_ranks[g] = ranks[mask]

        pairs = list(combinations(self.group_order, 2))
        rows = []
        raw_p_list = []
        for g1, g2 in pairs:
            r1 = group_ranks[g1]
            r2 = group_ranks[g2]
            n1, n2 = len(r1), len(r2)
            mean_r1 = np.mean(r1)
            mean_r2 = np.mean(r2)
            se = np.sqrt(N * (N + 1) / 12.0 * (1.0 / n1 + 1.0 / n2))
            if se == 0:
                z = 0.0
            else:
                z = (mean_r1 - mean_r2) / se
            p_raw = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            raw_p_list.append(p_raw)
            rows.append({"group1": g1, "group2": g2, "z_score": z, "p_value": p_raw})

        posthoc = pd.DataFrame(rows)

        # Correction
        raw_arr = np.array(raw_p_list, dtype=float)
        adj_arr = _apply_correction(raw_arr, self.correct_p)
        posthoc["p_adj"] = adj_arr
        posthoc["reject"] = posthoc["p_adj"] < self.p_value_threshold

        self._posthoc_df = posthoc[["group1", "group2", "z_score", "p_value", "p_adj", "reject"]].copy()
        self._fitted = True
        return self

    def get_kruskal_table(self) -> pd.DataFrame:
        """Return overall Kruskal-Wallis H-statistic and p-value.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns: ``H_statistic``, ``p_value``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import KruskalStats
        >>> np.random.seed(9)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        ...     "val": np.concatenate([
        ...         np.random.exponential(1, 20),
        ...         np.random.exponential(3, 20),
        ...         np.random.exponential(2, 20),
        ...     ])
        ... })
        >>> kr = KruskalStats(df, x_col="group", y_col="val").fit()
        >>> print(kr.get_kruskal_table())
        """
        self._check_fitted()
        return pd.DataFrame(
            {"H_statistic": [self._kruskal_h], "p_value": [self._kruskal_p]}
        )

    def get_posthoc_table(self) -> pd.DataFrame:
        """Return pairwise Dunn post-hoc results.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``group1``, ``group2``, ``z_score``,
            ``p_value``, ``p_adj``, ``reject``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import KruskalStats
        >>> np.random.seed(11)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
        ...     "val": np.concatenate([
        ...         np.random.exponential(1, 15),
        ...         np.random.exponential(4, 15),
        ...         np.random.exponential(2, 15),
        ...     ])
        ... })
        >>> kr = KruskalStats(df, x_col="group", y_col="val").fit()
        >>> print(kr.get_posthoc_table())
        """
        self._check_fitted()
        return self._posthoc_df.copy()

    def plot(
        self,
        plot_type: str = "box",
        font_size: int = 14,
        title: Optional[str] = None,
        custom_colors: Optional[Dict[str, str]] = None,
    ) -> go.Figure:
        """Render an annotated box or violin plot with Dunn significance brackets.

        Parameters
        ----------
        plot_type : str, default="box"
            Either ``"box"`` or ``"violin"``.
        font_size : int, default=14
            Base font size.
        title : str, optional
            Plot title. Defaults to ``y_col``.
        custom_colors : dict of str -> str, optional
            Mapping from group name to hex color string.

        Returns
        -------
        go.Figure
            Annotated Plotly figure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from metbit.stats.multitest import KruskalStats
        >>> np.random.seed(6)
        >>> df = pd.DataFrame({
        ...     "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        ...     "intensity": np.concatenate([
        ...         np.random.exponential(2, 20),
        ...         np.random.exponential(6, 20),
        ...         np.random.exponential(4, 20),
        ...     ])
        ... })
        >>> fig = KruskalStats(df, x_col="group", y_col="intensity").fit().plot()
        >>> fig.show()  # doctest: +SKIP
        """
        self._check_fitted()
        plot_title = title or self.y_col
        color_map = custom_colors

        if plot_type == "box":
            fig = px.box(
                self.df,
                x=self.x_col,
                y=self.y_col,
                color=self.x_col,
                points="all",
                category_orders={self.x_col: self.group_order},
                color_discrete_map=color_map,
            )
        elif plot_type == "violin":
            fig = px.violin(
                self.df,
                x=self.x_col,
                y=self.y_col,
                color=self.x_col,
                box=True,
                points="all",
                category_orders={self.x_col: self.group_order},
                color_discrete_map=color_map,
            )
        else:
            raise ValueError("plot_type must be 'box' or 'violin'.")

        fig = self._add_significance_brackets(fig)

        fig.update_layout(
            title=dict(text=f"<b>{plot_title}</b>", x=0.5),
            xaxis_title=self.x_col,
            yaxis_title=self.y_col,
            font=dict(size=font_size),
            width=self.fig_width,
            height=self.fig_height,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
            yaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
        )
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing results or plots.")

    def _add_significance_brackets(self, fig: go.Figure) -> go.Figure:
        """Add Dunn post-hoc significance brackets to an existing figure.

        Parameters
        ----------
        fig : go.Figure
            Base Plotly figure to annotate.

        Returns
        -------
        go.Figure
            Annotated figure.
        """
        y_vals = self.df[self.y_col].dropna()
        y_max = float(y_vals.max())
        y_range = float(y_vals.max() - y_vals.min())
        step = y_range * 0.12

        annotations = []
        pairs = self._posthoc_df[
            self._posthoc_df["p_adj"] < self.p_value_threshold
        ].copy()

        for idx, (_, row) in enumerate(pairs.iterrows()):
            g1, g2 = str(row["group1"]), str(row["group2"])
            if g1 not in self.group_order or g2 not in self.group_order:
                continue
            x1 = self.group_order.index(g1)
            x2 = self.group_order.index(g2)
            y_pos = y_max + step * (idx + 1)

            star = _p_to_stars(float(row["p_adj"]), self.p_value_threshold)

            fig.add_trace(
                go.Scatter(
                    x=[g1, g1, g2, g2],
                    y=[y_pos, y_pos + step * 0.4, y_pos + step * 0.4, y_pos],
                    mode="lines",
                    line=dict(color="black", width=1.2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            annotations.append(
                dict(
                    x=(x1 + x2) / 2,
                    y=y_pos + step * 0.55,
                    xref="x",
                    yref="y",
                    text=star,
                    showarrow=False,
                    font=dict(size=13),
                )
            )

        fig.update_layout(annotations=annotations)
        if pairs.shape[0] > 0:
            n_brackets = pairs.shape[0]
            fig.update_yaxes(range=[None, y_max + step * (n_brackets + 2)])
        return fig

    def __repr__(self) -> str:
        return f"KruskalStats(x_col={self.x_col!r}, y_col={self.y_col!r})"
