from __future__ import annotations

# -*- coding: utf-8 -*-

__author__ = "aeiwz"
__copyright__ = "Copyright 2024, Theerayut"
__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Union, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_color_sequence(
    label: Optional[pd.Series],
    color_dict: Optional[dict],
) -> tuple[list, dict]:
    """Return (unique_groups, group_to_color) from label + optional color_dict.

    Args:
        label: Sample group labels.
        color_dict: Optional mapping of group -> hex/named colour.

    Returns:
        Tuple of (unique_groups list, colour_map dict).
    """
    if label is None:
        return [], {}

    groups = list(pd.Series(label).unique())
    default_colors = px.colors.qualitative.Plotly
    color_map: dict = {}
    for i, g in enumerate(groups):
        if color_dict and g in color_dict:
            color_map[g] = color_dict[g]
        else:
            color_map[g] = default_colors[i % len(default_colors)]
    return groups, color_map


# ---------------------------------------------------------------------------
# CLASS 1: Biplot
# ---------------------------------------------------------------------------

class Biplot:
    """Combined scores + loadings biplot for PCA, LDA, or OPLS-DA models.

    Overlays sample score scatter (coloured by group label) with loading
    arrows so that both scores and loadings can be interpreted together
    on a single Plotly figure.

    Args:
        scores: DataFrame of shape (n_samples, >=2) containing score values
            (e.g. columns ``["PC1", "PC2"]``).
        loadings: DataFrame of shape (n_features, >=2) containing loading
            values aligned to the same components as ``scores``.
        label: Group label for each sample used to colour the score scatter.
            May be a Series or list of length n_samples.  If ``None`` all
            points share one colour.
        color_dict: Optional mapping ``{group_name: colour_string}`` to
            override default Plotly colour cycle.
        features_name: Feature labels for loading annotations.  If ``None``
            ``loadings.index`` is used.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.interpretation import Biplot
        >>> scores = pd.DataFrame(np.random.rand(40, 2), columns=["PC1", "PC2"])
        >>> loadings = pd.DataFrame(np.random.rand(20, 2), columns=["PC1", "PC2"])
        >>> label = pd.Series(["A"] * 20 + ["B"] * 20)
        >>> bp = Biplot(scores, loadings, label=label)
        >>> fig = bp.plot()
        >>> fig.show()  # doctest: +SKIP
    """

    def __init__(
        self,
        scores: pd.DataFrame,
        loadings: pd.DataFrame,
        label: Union[pd.Series, list, None] = None,
        color_dict: Optional[dict] = None,
        features_name: Optional[list] = None,
    ) -> None:
        if not isinstance(scores, pd.DataFrame):
            raise TypeError("scores must be a pandas DataFrame.")
        if not isinstance(loadings, pd.DataFrame):
            raise TypeError("loadings must be a pandas DataFrame.")
        if scores.shape[1] < 2:
            raise ValueError("scores must have at least 2 columns.")
        if loadings.shape[1] < 2:
            raise ValueError("loadings must have at least 2 columns.")

        self.scores = scores.reset_index(drop=True)
        self.loadings = loadings.reset_index(drop=True)
        self.label = pd.Series(label).reset_index(drop=True) if label is not None else None
        self.color_dict = color_dict or {}
        self.features_name = (
            list(features_name)
            if features_name is not None
            else list(loadings.index)
        )

    # ------------------------------------------------------------------
    def plot(
        self,
        pc: Optional[list[str]] = None,
        scale_loadings: float = 0.7,
        top_n_loadings: int = 20,
        marker_size: int = 10,
        fig_height: int = 700,
        fig_width: int = 900,
        font_size: int = 14,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render the biplot.

        Args:
            pc: Two component column names to plot, e.g. ``["PC1", "PC2"]``.
                Defaults to the first two columns of ``scores``.
            scale_loadings: Fraction of the score range used to scale loading
                arrows so they fit within the score cloud.  Values in (0, 1].
            top_n_loadings: Number of loadings (highest L2 magnitude) to
                annotate with text labels.
            marker_size: Diameter of score scatter markers in pixels.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size in points.
            title: Optional figure title.  If ``None`` a default title is
                constructed from the component names.

        Returns:
            A Plotly ``go.Figure`` object with score scatter traces and loading
            arrow annotations.

        Raises:
            ValueError: If a requested component name is not found in the
                scores or loadings DataFrames.
        """
        # --- resolve component names ---
        if pc is None:
            pc = list(self.scores.columns[:2])
        if len(pc) != 2:
            raise ValueError("pc must contain exactly two component names.")
        for name in pc:
            if name not in self.scores.columns:
                raise ValueError(f"Component '{name}' not found in scores columns.")
            if name not in self.loadings.columns:
                raise ValueError(f"Component '{name}' not found in loadings columns.")

        x_col, y_col = pc[0], pc[1]

        s_x = self.scores[x_col].values
        s_y = self.scores[y_col].values
        l_x = self.loadings[x_col].values
        l_y = self.loadings[y_col].values

        # --- scale loadings to fit the score range ---
        score_range_x = np.ptp(s_x) if np.ptp(s_x) > 0 else 1.0
        score_range_y = np.ptp(s_y) if np.ptp(s_y) > 0 else 1.0
        loading_range_x = np.ptp(l_x) if np.ptp(l_x) > 0 else 1.0
        loading_range_y = np.ptp(l_y) if np.ptp(l_y) > 0 else 1.0
        sx = (score_range_x / loading_range_x) * scale_loadings
        sy = (score_range_y / loading_range_y) * scale_loadings
        scale = min(sx, sy)

        scaled_lx = l_x * scale
        scaled_ly = l_y * scale

        # --- select top_n loadings by magnitude ---
        magnitudes = np.sqrt(l_x ** 2 + l_y ** 2)
        top_idx = np.argsort(magnitudes)[::-1][:top_n_loadings]

        fig = go.Figure()

        # --- score traces ---
        if self.label is not None:
            groups, color_map = _build_color_sequence(self.label, self.color_dict)
            for grp in groups:
                mask = self.label == grp
                fig.add_trace(
                    go.Scatter(
                        x=s_x[mask],
                        y=s_y[mask],
                        mode="markers",
                        name=str(grp),
                        marker=dict(
                            size=marker_size,
                            color=color_map[grp],
                            opacity=0.85,
                        ),
                        xaxis="x",
                        yaxis="y",
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=s_x,
                    y=s_y,
                    mode="markers",
                    name="Samples",
                    marker=dict(size=marker_size, opacity=0.85),
                    xaxis="x",
                    yaxis="y",
                )
            )

        # --- loading arrows using annotations ---
        annotations = []
        for i in range(len(scaled_lx)):
            in_top = i in top_idx
            arrow_color = "rgba(100,100,100,0.55)" if not in_top else "rgba(60,60,60,0.85)"
            annotations.append(
                dict(
                    x=scaled_lx[i],
                    y=scaled_ly[i],
                    ax=0,
                    ay=0,
                    xref="x2",
                    yref="y2",
                    axref="x2",
                    ayref="y2",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=1.5 if not in_top else 2.0,
                    arrowcolor=arrow_color,
                )
            )
            if in_top:
                feature_label = (
                    self.features_name[i]
                    if i < len(self.features_name)
                    else str(i)
                )
                annotations.append(
                    dict(
                        x=scaled_lx[i] * 1.05,
                        y=scaled_ly[i] * 1.05,
                        xref="x2",
                        yref="y2",
                        text=str(feature_label),
                        showarrow=False,
                        font=dict(size=font_size - 3, color="rgba(40,40,40,0.9)"),
                    )
                )

        # --- invisible loading scatter to drive secondary axes ---
        fig.add_trace(
            go.Scatter(
                x=scaled_lx,
                y=scaled_ly,
                mode="markers",
                marker=dict(size=1, opacity=0),
                showlegend=False,
                name="loadings_range",
                xaxis="x2",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title=title or f"Biplot ({x_col} vs {y_col})",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            legend=dict(title="Group"),
            annotations=annotations,
            xaxis=dict(
                title=x_col,
                zeroline=True,
                zerolinecolor="lightgrey",
                domain=[0, 1],
            ),
            yaxis=dict(
                title=y_col,
                zeroline=True,
                zerolinecolor="lightgrey",
            ),
            xaxis2=dict(
                title=f"{x_col} (loadings)",
                overlaying="x",
                side="top",
                showgrid=False,
            ),
            yaxis2=dict(
                title=f"{y_col} (loadings)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


# ---------------------------------------------------------------------------
# CLASS 2: CoefficientPlot
# ---------------------------------------------------------------------------

class CoefficientPlot:
    """Horizontal bar chart of model coefficients or loadings.

    Useful for interpreting OPLS-DA coefficients, PLS regression coefficients,
    or any signed per-feature statistic.

    Args:
        coef: Coefficient / loading values as a Series (with feature names as
            index) or a 1-D NumPy array.
        features_name: Feature labels.  If ``None``, uses ``coef.index`` when
            ``coef`` is a Series, or integer indices otherwise.
        ci_lower: Optional lower bound of confidence intervals aligned to
            ``coef``.
        ci_upper: Optional upper bound of confidence intervals aligned to
            ``coef``.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.interpretation import CoefficientPlot
        >>> rng = np.random.default_rng(0)
        >>> coef = pd.Series(rng.normal(size=50), name="coefficient")
        >>> cp = CoefficientPlot(coef)
        >>> fig = cp.plot(top_n=20)
        >>> fig.show()  # doctest: +SKIP
    """

    def __init__(
        self,
        coef: Union[pd.Series, np.ndarray],
        features_name: Optional[list] = None,
        ci_lower: Union[pd.Series, np.ndarray, None] = None,
        ci_upper: Union[pd.Series, np.ndarray, None] = None,
    ) -> None:
        if isinstance(coef, pd.Series):
            self._coef = coef.values.astype(float)
            default_names = list(coef.index)
        else:
            self._coef = np.asarray(coef, dtype=float).ravel()
            default_names = [str(i) for i in range(len(self._coef))]

        self.features_name = list(features_name) if features_name is not None else default_names

        if len(self.features_name) != len(self._coef):
            raise ValueError(
                f"features_name length ({len(self.features_name)}) does not match "
                f"coef length ({len(self._coef)})."
            )

        self._ci_lower = (
            np.asarray(ci_lower, dtype=float).ravel() if ci_lower is not None else None
        )
        self._ci_upper = (
            np.asarray(ci_upper, dtype=float).ravel() if ci_upper is not None else None
        )

    # ------------------------------------------------------------------
    def plot(
        self,
        top_n: int = 30,
        sort_by: str = "magnitude",
        color_positive: str = "#2563eb",
        color_negative: str = "#ef4444",
        fig_height: int = 700,
        fig_width: int = 800,
        font_size: int = 13,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Render the coefficient bar chart.

        Args:
            top_n: Maximum number of features to display, selected by highest
                absolute coefficient value before sorting.
            sort_by: Ordering of bars.  ``"magnitude"`` sorts by absolute
                value (largest at top); ``"value"`` sorts by signed value
                (most positive at top).
            color_positive: Fill colour for positive bars.
            color_negative: Fill colour for negative bars.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size in points.
            title: Optional figure title.

        Returns:
            A Plotly ``go.Figure`` with a horizontal bar chart.

        Raises:
            ValueError: If ``sort_by`` is not ``"magnitude"`` or ``"value"``.
        """
        if sort_by not in ("magnitude", "value"):
            raise ValueError("sort_by must be 'magnitude' or 'value'.")

        coef = self._coef
        names = np.array(self.features_name)

        # Select top_n by magnitude
        abs_vals = np.abs(coef)
        top_idx = np.argsort(abs_vals)[::-1][:top_n]
        coef_sel = coef[top_idx]
        names_sel = names[top_idx]
        ci_lower_sel = self._ci_lower[top_idx] if self._ci_lower is not None else None
        ci_upper_sel = self._ci_upper[top_idx] if self._ci_upper is not None else None

        # Sort for display
        if sort_by == "magnitude":
            order = np.argsort(np.abs(coef_sel))
        else:
            order = np.argsort(coef_sel)

        coef_plot = coef_sel[order]
        names_plot = names_sel[order]

        # Build error bar arrays
        error_x = None
        if ci_lower_sel is not None and ci_upper_sel is not None:
            ci_lo = ci_lower_sel[order]
            ci_hi = ci_upper_sel[order]
            error_minus = np.abs(coef_plot - ci_lo)
            error_plus = np.abs(ci_hi - coef_plot)
            error_x = dict(
                type="data",
                symmetric=False,
                array=error_plus.tolist(),
                arrayminus=error_minus.tolist(),
                color="rgba(0,0,0,0.5)",
                thickness=1.5,
                width=4,
            )

        bar_colors = [
            color_positive if v >= 0 else color_negative for v in coef_plot
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=coef_plot.tolist(),
                y=[str(n) for n in names_plot],
                orientation="h",
                marker=dict(color=bar_colors),
                error_x=error_x,
                showlegend=False,
            )
        )

        # Vertical line at x=0
        fig.add_vline(x=0, line=dict(color="black", width=1.5, dash="solid"))

        fig.update_layout(
            title=title or "Coefficient Plot",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis=dict(title="Coefficient", zeroline=False),
            yaxis=dict(title="Feature", automargin=True),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


# ---------------------------------------------------------------------------
# CLASS 3: FeatureImportancePlot
# ---------------------------------------------------------------------------

class FeatureImportancePlot:
    """Unified feature importance visualisation for multiple model types.

    Supports VIP scores, random-forest ``feature_importances_``, SHAP-style
    values, and loading magnitudes.  When ``importance`` is a DataFrame with
    multiple columns, grouped bars allow side-by-side comparison of metrics.

    Args:
        importance: Feature importances as:

            - ``pd.Series`` - one score per feature (index = feature names).
            - ``pd.DataFrame`` - multiple importance metrics as columns
              (index = feature names, columns = metric names).
            - ``np.ndarray`` - 1-D array treated as a Series with integer
              index.

        features_name: Override for feature labels.  If ``None``, uses
            ``importance.index`` (Series / DataFrame) or integer indices
            (ndarray).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.interpretation import FeatureImportancePlot
        >>> rng = np.random.default_rng(1)
        >>> vip = pd.Series(rng.exponential(size=60), name="VIP")
        >>> fip = FeatureImportancePlot(vip)
        >>> fig = fip.plot(top_n=20, threshold=1.0)
        >>> fig.show()  # doctest: +SKIP
        >>> fig_cum = fip.plot_cumulative()
        >>> fig_cum.show()  # doctest: +SKIP
    """

    def __init__(
        self,
        importance: Union[pd.Series, pd.DataFrame, np.ndarray],
        features_name: Optional[list] = None,
    ) -> None:
        if isinstance(importance, pd.DataFrame):
            self._data = importance.copy()
            default_names = list(importance.index)
            self._is_frame = True
        elif isinstance(importance, pd.Series):
            self._data = importance.copy()
            default_names = list(importance.index)
            self._is_frame = False
        else:
            arr = np.asarray(importance, dtype=float).ravel()
            self._data = pd.Series(arr)
            default_names = [str(i) for i in range(len(arr))]
            self._is_frame = False

        self.features_name = list(features_name) if features_name is not None else default_names

        n_features = (
            self._data.shape[0] if not self._is_frame else self._data.shape[0]
        )
        if len(self.features_name) != n_features:
            raise ValueError(
                f"features_name length ({len(self.features_name)}) does not match "
                f"importance length ({n_features})."
            )

        # Apply feature names as index
        self._data.index = self.features_name

    # ------------------------------------------------------------------
    def plot(
        self,
        top_n: int = 30,
        fig_height: int = 700,
        fig_width: int = 900,
        font_size: int = 13,
        title: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> go.Figure:
        """Horizontal bar chart of top-n features by importance.

        Args:
            top_n: Number of top features to display.
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size in points.
            title: Optional figure title.
            threshold: If provided, a vertical dashed line is drawn at this
                value (e.g. ``threshold=1.0`` for the VIP >= 1 rule).

        Returns:
            A Plotly ``go.Figure`` with horizontal bar(s).
        """
        fig = go.Figure()

        if self._is_frame:
            df: pd.DataFrame = self._data

            # Select top_n by mean importance across metrics
            mean_imp = df.mean(axis=1)
            top_idx = mean_imp.nlargest(top_n).index
            df_sel = df.loc[top_idx]

            # Sort by mean for display
            order = mean_imp.loc[top_idx].sort_values(ascending=True).index
            df_plot = df_sel.loc[order]

            colors = px.colors.qualitative.Plotly
            for i, col in enumerate(df_plot.columns):
                fig.add_trace(
                    go.Bar(
                        x=df_plot[col].values.tolist(),
                        y=[str(n) for n in df_plot.index],
                        orientation="h",
                        name=str(col),
                        marker=dict(color=colors[i % len(colors)], opacity=0.85),
                    )
                )
            fig.update_layout(barmode="group")
        else:
            series: pd.Series = self._data
            top_series = series.nlargest(top_n).sort_values(ascending=True)

            bar_colors = px.colors.qualitative.Plotly
            fig.add_trace(
                go.Bar(
                    x=top_series.values.tolist(),
                    y=[str(n) for n in top_series.index],
                    orientation="h",
                    marker=dict(color=bar_colors[0], opacity=0.85),
                    showlegend=False,
                )
            )

        if threshold is not None:
            fig.add_vline(
                x=threshold,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text=f"threshold={threshold}",
                annotation_position="top right",
                annotation_font=dict(size=font_size - 2, color="red"),
            )

        fig.update_layout(
            title=title or "Feature Importance",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis=dict(title="Importance", zeroline=False),
            yaxis=dict(title="Feature", automargin=True),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    # ------------------------------------------------------------------
    def plot_cumulative(
        self,
        fig_height: int = 400,
        fig_width: int = 700,
        font_size: int = 13,
    ) -> go.Figure:
        """Cumulative importance curve sorted in descending order.

        The "elbow" point where cumulative importance first reaches 90% of
        the total is highlighted with a vertical dashed line.

        Args:
            fig_height: Figure height in pixels.
            fig_width: Figure width in pixels.
            font_size: Global font size in points.

        Returns:
            A Plotly ``go.Figure`` with a line chart of cumulative importance
            and an elbow marker.
        """
        if self._is_frame:
            # Use mean across metrics for the cumulative curve
            series = self._data.mean(axis=1)
        else:
            series = self._data.copy()

        sorted_vals = series.sort_values(ascending=False).values.astype(float)
        total = sorted_vals.sum()
        if total == 0:
            cumulative = np.zeros(len(sorted_vals))
        else:
            cumulative = np.cumsum(sorted_vals) / total

        n_features = np.arange(1, len(cumulative) + 1)

        # Find elbow at 90% cumulative importance
        elbow_idx = int(np.searchsorted(cumulative, 0.90))
        elbow_idx = min(elbow_idx, len(cumulative) - 1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=n_features.tolist(),
                y=cumulative.tolist(),
                mode="lines+markers",
                name="Cumulative importance",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=4),
            )
        )

        # 90% horizontal reference
        fig.add_hline(
            y=0.90,
            line=dict(color="grey", dash="dot", width=1),
            annotation_text="90%",
            annotation_position="right",
            annotation_font=dict(size=font_size - 2, color="grey"),
        )

        # Elbow vertical line
        fig.add_vline(
            x=int(elbow_idx + 1),
            line=dict(color="red", dash="dash", width=1.5),
            annotation_text=f"elbow (n={elbow_idx + 1})",
            annotation_position="top right",
            annotation_font=dict(size=font_size - 2, color="red"),
        )

        fig.update_layout(
            title="Cumulative Feature Importance",
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis=dict(title="Number of features"),
            yaxis=dict(title="Cumulative importance", range=[0, 1.05]),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig
