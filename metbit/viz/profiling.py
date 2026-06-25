from __future__ import annotations

# -*- coding: utf-8 -*-

__author__ = "aeiwz"
__copyright__ = "Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

"""
Metabolite profiling visualisations: FoldChangePlot, GroupComparison,
and MetaboliteDashboard (interactive Dash app).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CORRECTION_METHODS = ("fdr_bh", "bonferroni", "holm", "fdr_by", None)

_UP_COLOR = "#2563eb"    # blue
_DOWN_COLOR = "#e11d48"  # red
_NS_COLOR = "#9ca3af"    # grey


def _apply_correction(p_values: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Apply multiple-testing correction to an array of p-values.

    Parameters:
        p_values: Raw p-values (may contain NaN).
        method: Correction method accepted by
            ``statsmodels.stats.multitest.multipletests``.
            Pass ``None`` to skip correction.

    Returns:
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


def _label_feature(
    log2fc: float,
    p_adj: float,
    fc_threshold: float,
    p_threshold: float,
) -> str:
    """Classify a feature as 'Up', 'Down', or 'NS'.

    Parameters:
        log2fc: Log2 fold change value.
        p_adj: Adjusted p-value.
        fc_threshold: Absolute log2FC threshold for significance.
        p_threshold: Adjusted p-value threshold.

    Returns:
        One of ``'Up'``, ``'Down'``, or ``'NS'``.
    """
    if np.isnan(log2fc) or np.isnan(p_adj):
        return "NS"
    if p_adj < p_threshold and log2fc >= fc_threshold:
        return "Up"
    if p_adj < p_threshold and log2fc <= -fc_threshold:
        return "Down"
    return "NS"


# ---------------------------------------------------------------------------
# CLASS 1: FoldChangePlot
# ---------------------------------------------------------------------------

class FoldChangePlot:
    """Horizontal bar chart of fold changes with significance colouring.

    For each numeric feature the class computes a log2 fold change
    (group_b / group_a) and a Welch's t-test p-value. Multiple-testing
    correction is applied optionally. Features are then coloured as
    Up (blue), Down (red), or Not-Significant (grey).

    Parameters:
        df: Tidy DataFrame containing group labels and numeric feature columns.
        group_col: Column name containing group labels. Must have at least two
            unique values. When more than two unique values are present,
            ``group_a`` and ``group_b`` must be specified explicitly.
        group_a: Reference group label (denominator of the fold change).
            If ``None`` the first unique value (sorted) is used.
        group_b: Comparison group label (numerator). If ``None`` the second
            unique value (sorted) is used.
        value_cols: Subset of numeric columns to analyse. If ``None`` all
            numeric columns except ``group_col`` are used.
        log2: If ``True`` (default) the fold change is expressed in log2
            scale. If ``False`` the raw ratio is returned instead, and
            ``fc_threshold`` is interpreted as a raw ratio.
        p_value_threshold: Significance threshold applied to (optionally
            corrected) p-values. Default ``0.05``.
        fc_threshold: Fold-change magnitude threshold for colouring. When
            ``log2=True`` this is an absolute log2FC threshold (default
            ``1.0``). When ``log2=False`` it is an absolute raw-ratio
            threshold.
        correct_p: Multiple-testing correction method forwarded to
            ``statsmodels.stats.multitest.multipletests``. Pass ``None``
            to skip correction. Default ``"fdr_bh"``.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.profiling import FoldChangePlot
        >>> X = pd.DataFrame(np.random.rand(40, 10), columns=[f"f{i}" for i in range(10)])
        >>> X["group"] = ["A"]*20 + ["B"]*20
        >>> fc = FoldChangePlot(X, group_col="group")
        >>> fig = fc.plot()
        >>> tbl = fc.get_table()
        >>> assert "log2FC" in tbl.columns
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_col: str,
        group_a: Optional[str] = None,
        group_b: Optional[str] = None,
        value_cols: Optional[List[str]] = None,
        log2: bool = True,
        p_value_threshold: float = 0.05,
        fc_threshold: float = 1.0,
        correct_p: Optional[str] = "fdr_bh",
    ) -> None:
        self._df = df.copy()
        self._group_col = group_col
        self._log2 = log2
        self._p_threshold = p_value_threshold
        self._fc_threshold = fc_threshold
        self._correct_p = correct_p

        groups = sorted(self._df[group_col].dropna().unique().tolist())
        if len(groups) < 2:
            raise ValueError(
                f"group_col '{group_col}' must contain at least two unique values."
            )
        self._group_a = group_a if group_a is not None else groups[0]
        self._group_b = group_b if group_b is not None else groups[1]

        if value_cols is None:
            self._value_cols = [
                c for c in df.columns
                if c != group_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            self._value_cols = list(value_cols)

        self._result: Optional[pd.DataFrame] = None
        self._compute()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        """Compute per-feature log2 FC and p-values (Welch t-test)."""
        mask_a = self._df[self._group_col] == self._group_a
        mask_b = self._df[self._group_col] == self._group_b

        records = []
        for feat in self._value_cols:
            vals_a = self._df.loc[mask_a, feat].dropna().values.astype(float)
            vals_b = self._df.loc[mask_b, feat].dropna().values.astype(float)

            if len(vals_a) < 2 or len(vals_b) < 2:
                records.append(
                    {"feature": feat, "log2FC": np.nan, "p_value": np.nan}
                )
                continue

            mean_a = vals_a.mean()
            mean_b = vals_b.mean()

            # Avoid division by zero
            if mean_a == 0:
                fc_val = np.nan
            else:
                ratio = mean_b / mean_a
                fc_val = np.log2(ratio) if self._log2 else ratio

            _, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            records.append(
                {"feature": feat, "log2FC": fc_val, "p_value": p_val}
            )

        res = pd.DataFrame(records)
        p_arr = res["p_value"].values.astype(float)
        res["p_adj"] = _apply_correction(p_arr, self._correct_p)
        res["label"] = res.apply(
            lambda r: _label_feature(
                r["log2FC"], r["p_adj"], self._fc_threshold, self._p_threshold
            ),
            axis=1,
        )
        self._result = res

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_table(self) -> pd.DataFrame:
        """Return the per-feature statistics table.

        Returns:
            DataFrame with columns ``feature``, ``log2FC``, ``p_value``,
            ``p_adj``, and ``label``.

        Examples:
            >>> import pandas as pd, numpy as np
            >>> from metbit.viz.profiling import FoldChangePlot
            >>> X = pd.DataFrame(np.random.rand(20, 5), columns=list("ABCDE"))
            >>> X["g"] = ["X"]*10 + ["Y"]*10
            >>> tbl = FoldChangePlot(X, "g").get_table()
            >>> list(tbl.columns)
            ['feature', 'log2FC', 'p_value', 'p_adj', 'label']
        """
        return self._result.copy()

    def plot(
        self,
        top_n: int = 30,
        sort_by: str = "fc",
        fig_height: int = 700,
        fig_width: int = 900,
        font_size: int = 13,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render a horizontal bar chart of fold changes.

        Bars are colour-coded: Up (blue), Down (red), NS (grey). Vertical
        dashed lines mark ±fc_threshold.

        Parameters:
            top_n: Maximum number of features to display (by absolute FC).
                Default ``30``.
            sort_by: Sorting criterion. One of:
                ``"fc"`` - sort by absolute log2FC (descending),
                ``"p_value"`` - sort by adjusted p-value (ascending),
                ``"feature"`` - sort alphabetically.
                Default ``"fc"``.
            fig_height: Figure height in pixels. Default ``700``.
            fig_width: Figure width in pixels. Default ``900``.
            font_size: Global font size. Default ``13``.
            title: Plot title. If ``None`` a default title is generated.

        Returns:
            Plotly ``go.Figure`` object.

        Examples:
            >>> import pandas as pd, numpy as np
            >>> from metbit.viz.profiling import FoldChangePlot
            >>> X = pd.DataFrame(np.random.rand(40, 6), columns=list("ABCDEF"))
            >>> X["g"] = ["P"]*20 + ["Q"]*20
            >>> fig = FoldChangePlot(X, "g").plot(top_n=6)
            >>> assert fig is not None
        """
        res = self._result.dropna(subset=["log2FC"]).copy()

        # Select top_n by absolute FC
        res["abs_fc"] = res["log2FC"].abs()
        res = res.nlargest(top_n, "abs_fc")

        # Sort
        if sort_by == "p_value":
            res = res.sort_values("p_adj", ascending=False)
        elif sort_by == "feature":
            res = res.sort_values("feature", ascending=True)
        else:  # default: fc
            res = res.sort_values("log2FC", ascending=True)

        color_map = {"Up": _UP_COLOR, "Down": _DOWN_COLOR, "NS": _NS_COLOR}
        bar_colors = res["label"].map(color_map).tolist()

        fc_col = "log2FC" if self._log2 else "FC"

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=res["log2FC"],
                y=res["feature"],
                orientation="h",
                marker_color=bar_colors,
                text=res["p_adj"].apply(lambda v: f"p_adj={v:.3g}" if not np.isnan(v) else ""),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + ("log2FC" if self._log2 else "FC")
                    + ": %{x:.3f}<br>%{text}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Threshold lines
        for sign, label in [(self._fc_threshold, f"+{self._fc_threshold}"), (-self._fc_threshold, f"-{self._fc_threshold}")]:
            fig.add_vline(
                x=sign,
                line_dash="dash",
                line_color="#374151",
                line_width=1.2,
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=font_size - 1,
            )

        # Dummy traces for legend
        for lbl, col in color_map.items():
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    orientation="h",
                    marker_color=col,
                    name=lbl,
                    showlegend=True,
                )
            )

        axis_label = "log2 Fold Change" if self._log2 else "Fold Change"
        default_title = (
            f"Fold Change: {self._group_b} vs {self._group_a}"
        )

        fig.update_layout(
            title=dict(text=title or default_title, font_size=font_size + 2),
            xaxis_title=axis_label,
            yaxis_title="Feature",
            height=fig_height,
            width=fig_width,
            font_size=font_size,
            legend=dict(title="Regulation", orientation="v"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#374151"),
            yaxis=dict(gridcolor="#e5e7eb"),
            bargap=0.3,
        )
        return fig


# ---------------------------------------------------------------------------
# CLASS 2: GroupComparison
# ---------------------------------------------------------------------------

class GroupComparison:
    """Side-by-side box/violin plots for a set of selected features.

    Renders a subplot grid where each panel shows the distribution of one
    feature across all groups using a box plot or violin plot. Groups are
    coloured consistently across all panels.

    Parameters:
        df: Tidy DataFrame containing group labels and numeric feature columns.
        group_col: Column name containing group labels.
        feature_cols: Numeric columns to visualise. If ``None`` all numeric
            columns (excluding ``group_col``) are used.
        group_order: Explicit ordering of groups on the x-axis. If ``None``
            groups are sorted alphabetically.
        color_dict: Mapping of group label to hex/named colour. If ``None``
            Plotly's default qualitative palette is used.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> from metbit.viz.profiling import GroupComparison
        >>> X = pd.DataFrame(np.random.rand(30, 4), columns=list("ABCD"))
        >>> X["group"] = ["X"]*10 + ["Y"]*10 + ["Z"]*10
        >>> gc = GroupComparison(X, group_col="group")
        >>> fig = gc.plot(features=["A", "B"], n_cols=2)
        >>> assert fig is not None
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_col: str,
        feature_cols: Optional[List[str]] = None,
        group_order: Optional[List[str]] = None,
        color_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        self._df = df.copy()
        self._group_col = group_col

        if feature_cols is None:
            self._feature_cols = [
                c for c in df.columns
                if c != group_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            self._feature_cols = list(feature_cols)

        all_groups = sorted(self._df[group_col].dropna().unique().tolist())
        self._group_order = group_order if group_order is not None else all_groups

        if color_dict is None:
            palette = px.colors.qualitative.Plotly
            self._color_dict: Dict[str, str] = {
                g: palette[i % len(palette)]
                for i, g in enumerate(self._group_order)
            }
        else:
            self._color_dict = color_dict

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        features: Optional[List[str]] = None,
        n_cols: int = 4,
        plot_type: str = "box",
        show_points: bool = True,
        fig_height: int = 300,
        fig_width: int = 300,
        font_size: int = 12,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create a subplot grid with one box/violin per feature.

        Parameters:
            features: Features to plot. If ``None`` ``self.feature_cols``
                (up to 20) are used.
            n_cols: Number of subplot columns. Default ``4``.
            plot_type: ``"box"`` or ``"violin"``. Default ``"box"``.
            show_points: Overlay individual data points. Default ``True``.
            fig_height: Height of each individual subplot panel in pixels.
                Default ``300``.
            fig_width: Width of each individual subplot panel in pixels.
                Default ``300``.
            font_size: Global font size. Default ``12``.
            title: Overall figure title. If ``None`` no title is set.

        Returns:
            Plotly ``go.Figure`` object.

        Examples:
            >>> import pandas as pd, numpy as np
            >>> from metbit.viz.profiling import GroupComparison
            >>> X = pd.DataFrame(np.random.rand(20, 3), columns=list("ABC"))
            >>> X["g"] = ["M"]*10 + ["N"]*10
            >>> fig = GroupComparison(X, "g").plot(features=["A", "B"], n_cols=2)
            >>> assert fig is not None
        """
        if features is None:
            features = self._feature_cols[:20]

        n_feats = len(features)
        if n_feats == 0:
            raise ValueError("No features to plot.")

        n_cols = min(n_cols, n_feats)
        n_rows = int(np.ceil(n_feats / n_cols))

        total_height = fig_height * n_rows
        total_width = fig_width * n_cols

        subplot_titles = list(features)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        shown_in_legend: set = set()

        for idx, feat in enumerate(features):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            for grp in self._group_order:
                mask = self._df[self._group_col] == grp
                vals = self._df.loc[mask, feat].dropna().values

                color = self._color_dict.get(grp, "#9ca3af")
                show_legend = grp not in shown_in_legend

                if plot_type == "violin":
                    trace = go.Violin(
                        y=vals,
                        name=grp,
                        legendgroup=grp,
                        showlegend=show_legend,
                        line_color=color,
                        fillcolor=color,
                        opacity=0.65,
                        box_visible=True,
                        meanline_visible=True,
                        points="all" if show_points else False,
                        jitter=0.3,
                        marker=dict(size=4, opacity=0.6),
                        hovertemplate=f"<b>{grp}</b><br>{feat}: %{{y:.3f}}<extra></extra>",
                    )
                else:  # box
                    trace = go.Box(
                        y=vals,
                        name=grp,
                        legendgroup=grp,
                        showlegend=show_legend,
                        marker_color=color,
                        line_color=color,
                        fillcolor=color,
                        opacity=0.65,
                        boxpoints="all" if show_points else False,
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(size=4, opacity=0.6),
                        hovertemplate=f"<b>{grp}</b><br>{feat}: %{{y:.3f}}<extra></extra>",
                    )

                fig.add_trace(trace, row=row, col=col)
                shown_in_legend.add(grp)

        fig.update_layout(
            title=dict(text=title or "", font_size=font_size + 2) if title else {},
            height=total_height,
            width=total_width,
            font_size=font_size,
            legend=dict(title="Group"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            boxmode="group",
            violinmode="group",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#e5e7eb")
        return fig


# ---------------------------------------------------------------------------
# CLASS 3: MetaboliteDashboard  (Dash app)
# ---------------------------------------------------------------------------

class MetaboliteDashboard:
    """Interactive Dash dashboard combining a volcano plot and group comparison.

    The left panel shows an interactive volcano plot powered by
    :class:`metbit.stats.multitest.VolcanoPlot`. Clicking a point selects
    that feature and updates the right panel, which shows a box plot for the
    selected feature across all groups. Dropdowns allow live adjustment of
    the FC threshold, p-value threshold, and multiple-testing correction
    method.

    Parameters:
        df: Tidy DataFrame containing group labels and numeric feature columns.
        group_col: Column name containing group labels.
        value_cols: Numeric columns to analyse. If ``None`` all numeric columns
            (excluding ``group_col``) are used.
        group_order: Explicit ordering of groups. If ``None`` groups are sorted
            alphabetically.
        color_dict: Mapping of group label to colour. If ``None`` Plotly's
            default qualitative palette is used.
        p_value_threshold: Initial p-value significance threshold. Default
            ``0.05``.
        fc_threshold: Initial log2FC threshold. Default ``1.0``.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> from metbit.viz.profiling import MetaboliteDashboard
        >>> X = pd.DataFrame(np.random.rand(40, 8), columns=[f"m{i}" for i in range(8)])
        >>> X["group"] = ["A"]*20 + ["B"]*20
        >>> dashboard = MetaboliteDashboard(X, group_col="group")
        >>> app = dashboard.run_ui()
        >>> app  # doctest: +ELLIPSIS
        <...Dash...>
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_cols: Optional[List[str]] = None,
        group_order: Optional[List[str]] = None,
        color_dict: Optional[Dict[str, str]] = None,
        p_value_threshold: float = 0.05,
        fc_threshold: float = 1.0,
    ) -> None:
        try:
            import dash
            from dash import dcc, html, Input, Output
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MetaboliteDashboard requires dash. "
                "Install it with: pip install dash"
            ) from exc

        try:
            from metbit.stats.multitest import VolcanoPlot
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MetaboliteDashboard requires metbit.stats.multitest.VolcanoPlot."
            ) from exc

        self._df = df.copy()
        self._group_col = group_col
        self._p_threshold = p_value_threshold
        self._fc_threshold = fc_threshold

        if value_cols is None:
            self._value_cols = [
                c for c in df.columns
                if c != group_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            self._value_cols = list(value_cols)

        all_groups = sorted(self._df[group_col].dropna().unique().tolist())
        self._group_order = group_order if group_order is not None else all_groups

        if color_dict is None:
            palette = px.colors.qualitative.Plotly
            self._color_dict: Dict[str, str] = {
                g: palette[i % len(palette)]
                for i, g in enumerate(self._group_order)
            }
        else:
            self._color_dict = color_dict

        self._VolcanoPlot = VolcanoPlot
        self._dash = dash
        self._dcc = dcc
        self._html = html
        self._Input = Input
        self._Output = Output

        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _build_app(self):
        """Build and return the configured Dash application."""
        dash = self._dash
        dcc = self._dcc
        html = self._html
        Input = self._Input
        Output = self._Output

        app = dash.Dash(__name__)
        app.title = "Metabolite Dashboard"

        # --- layout -------------------------------------------------------
        app.layout = html.Div(
            style={
                "fontFamily": "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
                "background": "#f5f7fb",
                "minHeight": "100vh",
                "padding": "16px",
            },
            children=[
                # Header
                html.Div(
                    "Metabolite Dashboard",
                    style={
                        "fontSize": "22px",
                        "fontWeight": "700",
                        "color": "#1b2530",
                        "marginBottom": "14px",
                    },
                ),
                # Controls bar
                html.Div(
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "16px",
                        "alignItems": "center",
                        "background": "#ffffff",
                        "borderRadius": "12px",
                        "padding": "12px 16px",
                        "marginBottom": "14px",
                        "boxShadow": "0 2px 12px rgba(20,22,35,.06)",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "FC threshold",
                                    style={"fontWeight": "600", "fontSize": "12px", "color": "#6d7890"},
                                ),
                                dcc.Dropdown(
                                    id="dd-fc",
                                    options=[
                                        {"label": str(v), "value": v}
                                        for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                                    ],
                                    value=self._fc_threshold,
                                    clearable=False,
                                    style={"width": "110px", "fontSize": "13px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "p threshold",
                                    style={"fontWeight": "600", "fontSize": "12px", "color": "#6d7890"},
                                ),
                                dcc.Dropdown(
                                    id="dd-pval",
                                    options=[
                                        {"label": str(v), "value": v}
                                        for v in [0.001, 0.01, 0.05, 0.1]
                                    ],
                                    value=self._p_threshold,
                                    clearable=False,
                                    style={"width": "110px", "fontSize": "13px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Correction method",
                                    style={"fontWeight": "600", "fontSize": "12px", "color": "#6d7890"},
                                ),
                                dcc.Dropdown(
                                    id="dd-correction",
                                    options=[
                                        {"label": "FDR (BH)", "value": "fdr_bh"},
                                        {"label": "Bonferroni", "value": "bonferroni"},
                                        {"label": "Holm", "value": "holm"},
                                        {"label": "None", "value": "__none__"},
                                    ],
                                    value="fdr_bh",
                                    clearable=False,
                                    style={"width": "160px", "fontSize": "13px"},
                                ),
                            ]
                        ),
                        html.Div(
                            id="selected-feature-pill",
                            style={
                                "marginLeft": "auto",
                                "background": "#eff6ff",
                                "border": "1px solid #bfdbfe",
                                "borderRadius": "999px",
                                "padding": "6px 14px",
                                "fontSize": "13px",
                                "fontWeight": "600",
                                "color": "#2563eb",
                            },
                            children="Click a point to select a feature",
                        ),
                    ],
                ),
                # Main panels
                html.Div(
                    style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                    children=[
                        # Left panel - volcano
                        html.Div(
                            style={
                                "flex": "0 0 40%",
                                "minWidth": "320px",
                                "background": "#ffffff",
                                "borderRadius": "12px",
                                "padding": "12px",
                                "boxShadow": "0 2px 12px rgba(20,22,35,.06)",
                            },
                            children=[
                                html.Div(
                                    "Volcano Plot",
                                    style={"fontWeight": "700", "fontSize": "14px", "marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="volcano-graph",
                                    config={"displayModeBar": True},
                                    style={"height": "520px"},
                                ),
                            ],
                        ),
                        # Right panel - box plot
                        html.Div(
                            style={
                                "flex": "1 1 55%",
                                "minWidth": "320px",
                                "background": "#ffffff",
                                "borderRadius": "12px",
                                "padding": "12px",
                                "boxShadow": "0 2px 12px rgba(20,22,35,.06)",
                            },
                            children=[
                                html.Div(
                                    "Group Comparison",
                                    style={"fontWeight": "700", "fontSize": "14px", "marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="box-graph",
                                    config={"displayModeBar": True},
                                    style={"height": "520px"},
                                ),
                            ],
                        ),
                    ],
                ),
                # Hidden store for selected feature
                dcc.Store(id="selected-feature-store", data=None),
            ],
        )

        # --- callbacks ----------------------------------------------------

        @app.callback(
            Output("volcano-graph", "figure"),
            Input("dd-fc", "value"),
            Input("dd-pval", "value"),
            Input("dd-correction", "value"),
        )
        def update_volcano(fc_val, p_val, correction):  # pragma: no cover
            method = None if correction == "__none__" else correction
            vp = self._VolcanoPlot(
                df=self._df,
                group_col=self._group_col,
                value_cols=self._value_cols,
                p_value_threshold=float(p_val),
                fc_threshold=float(fc_val),
                correct_p=method,
            )
            fig = vp.plot(fig_height=500, fig_width=None)
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            return fig

        @app.callback(
            Output("selected-feature-store", "data"),
            Output("selected-feature-pill", "children"),
            Input("volcano-graph", "clickData"),
        )
        def store_selected_feature(click_data):  # pragma: no cover
            if click_data is None:
                return None, "Click a point to select a feature"
            point = click_data.get("points", [{}])[0]
            feature = point.get("text") or point.get("customdata") or point.get("hovertext")
            if feature is None:
                # Try to extract from hovertemplate data
                feature = str(point.get("y") or point.get("x") or "")
            feature = str(feature).strip()
            if not feature:
                return None, "Click a point to select a feature"
            return feature, f"Selected: {feature}"

        @app.callback(
            Output("box-graph", "figure"),
            Input("selected-feature-store", "data"),
        )
        def update_box(feature):  # pragma: no cover
            if feature is None or feature not in self._df.columns:
                # Default: show first feature
                feature = self._value_cols[0] if self._value_cols else None

            if feature is None:
                return go.Figure()

            gc = GroupComparison(
                df=self._df,
                group_col=self._group_col,
                feature_cols=[feature],
                group_order=self._group_order,
                color_dict=self._color_dict,
            )
            fig = gc.plot(
                features=[feature],
                n_cols=1,
                plot_type="box",
                show_points=True,
                fig_height=480,
                fig_width=600,
                title=f"Distribution: {feature}",
            )
            fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
            return fig

        return app

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_ui(self):
        """Return the configured Dash app object without starting the server.

        Returns:
            Configured ``dash.Dash`` instance.

        Examples:
            >>> import pandas as pd, numpy as np
            >>> from metbit.viz.profiling import MetaboliteDashboard
            >>> X = pd.DataFrame(np.random.rand(20, 4), columns=list("ABCD"))
            >>> X["g"] = ["X"]*10 + ["Y"]*10
            >>> app = MetaboliteDashboard(X, group_col="g").run_ui()
        """
        return self._app

    def run(self, debug: bool = True, port: int = 8052) -> None:
        """Launch the Dash server.

        Parameters:
            debug: Enable Dash debug mode. Default ``True``.
            port: TCP port the server listens on. Default ``8052``.

        Examples:
            >>> import pandas as pd, numpy as np
            >>> from metbit.viz.profiling import MetaboliteDashboard
            >>> X = pd.DataFrame(np.random.rand(20, 4), columns=list("ABCD"))
            >>> X["g"] = ["X"]*10 + ["Y"]*10
            >>> dashboard = MetaboliteDashboard(X, group_col="g")
            >>> dashboard.run(debug=False, port=8052)  # doctest: +SKIP
        """
        self._app.run(debug=debug, port=port)  # pragma: no cover
