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
from typing import List, Optional, Union


class SpectraPlot:
    """Visualisation of NMR spectra from a DataFrame.

    Columns are ppm values (or numeric column names); rows are samples.
    Supports overlay, mean-with-SD-envelope, stacked, and single-spectrum plots.

    Parameters:
        spectra (pd.DataFrame): DataFrame with rows=samples and columns=ppm values
            (float column names) or any numeric column names interpretable as ppm.
        ppm (list or np.ndarray, optional): Explicit ppm axis. If None, inferred
            from ``spectra.columns``.
        label (pd.Series or list, optional): Group labels per sample used for
            colour-coding. Length must match ``len(spectra)``.
        color_dict (dict, optional): Mapping ``{group: color}`` for each unique
            label value. Auto-generated from Plotly's qualitative palette when
            None.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from metbit.viz.spectra import SpectraPlot
        >>> ppm = np.linspace(0.5, 9.5, 500)
        >>> spectra = pd.DataFrame(np.random.rand(20, 500), columns=ppm)
        >>> label = pd.Series(["A"]*10 + ["B"]*10)
        >>> sp = SpectraPlot(spectra, label=label)
        >>> fig = sp.overlay()
        >>> fig.show()
    """

    def __init__(
        self,
        spectra: pd.DataFrame,
        ppm: Optional[Union[list, np.ndarray]] = None,
        label: Optional[Union[pd.Series, list]] = None,
        color_dict: Optional[dict] = None,
    ) -> None:
        self._spectra = spectra.copy()

        if ppm is not None:
            self._ppm = np.asarray(ppm, dtype=float)
        else:
            self._ppm = np.asarray(spectra.columns, dtype=float)

        if label is not None:
            self._label = pd.Series(label, index=spectra.index).astype(str)
        else:
            self._label = None

        self._groups: List[str] = (
            list(self._label.unique()) if self._label is not None else []
        )

        if color_dict is not None:
            self._color_dict = color_dict
        else:
            palette = px.colors.qualitative.Plotly
            self._color_dict = {
                g: palette[i % len(palette)]
                for i, g in enumerate(self._groups)
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_color(self, idx: int) -> str:
        """Return the colour string for sample at position ``idx``."""
        if self._label is None:
            return px.colors.qualitative.Plotly[0]
        grp = self._label.iloc[idx]
        return self._color_dict.get(grp, "#636EFA")

    def _sample_label(self, idx: int) -> str:
        """Return the group label string for sample at position ``idx``."""
        if self._label is None:
            return "Sample"
        return str(self._label.iloc[idx])

    @staticmethod
    def _hex_to_rgba(color: str, alpha: float) -> str:
        """Convert a hex or named colour to an rgba string for Plotly."""
        import re

        hex_match = re.fullmatch(r"#([0-9A-Fa-f]{6})", color.strip())
        if hex_match:
            r, g, b = (
                int(hex_match.group(1)[0:2], 16),
                int(hex_match.group(1)[2:4], 16),
                int(hex_match.group(1)[4:6], 16),
            )
            return f"rgba({r},{g},{b},{alpha})"
        # If already rgba/rgb just return as-is (best effort)
        return color

    def _apply_common_layout(
        self,
        fig: go.Figure,
        title: Optional[str],
        xaxis_title: str,
        yaxis_title: str,
        fig_height: int,
        fig_width: int,
        font_size: int,
        xaxis_reversed: bool,
    ) -> go.Figure:
        """Apply shared layout settings to *fig* and return it."""
        fig.update_layout(
            title=title,
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                autorange="reversed" if xaxis_reversed else True,
                showgrid=True,
                gridcolor="lightgrey",
                zeroline=False,
            ),
            yaxis=dict(showgrid=True, gridcolor="lightgrey", zeroline=False),
        )
        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def overlay(
        self,
        alpha: float = 0.5,
        linewidth: float = 1.0,
        fig_height: int = 500,
        fig_width: int = 1400,
        font_size: int = 14,
        title: Optional[str] = None,
        xaxis_title: str = "δ¹H (ppm)",
        yaxis_title: str = "Intensity",
        xaxis_reversed: bool = True,
    ) -> go.Figure:
        """Overlay all spectra on a single plot, coloured by label.

        Parameters:
            alpha (float): Line opacity. Default is 0.5.
            linewidth (float): Line width in pixels. Default is 1.0.
            fig_height (int): Figure height in pixels. Default is 500.
            fig_width (int): Figure width in pixels. Default is 1400.
            font_size (int): Base font size. Default is 14.
            title (str, optional): Plot title. Default is None.
            xaxis_title (str): X-axis label. Default is ``"delta1H (ppm)"``.
            yaxis_title (str): Y-axis label. Default is ``"Intensity"``.
            xaxis_reversed (bool): Reverse the x-axis (NMR convention).
                Default is True.

        Returns:
            go.Figure: Plotly figure with all spectra overlaid.

        Examples:
            >>> fig = sp.overlay(alpha=0.4, linewidth=0.8)
            >>> fig.show()
        """
        fig = go.Figure()
        added_groups: set = set()

        for i in range(len(self._spectra)):
            row = self._spectra.iloc[i].values
            grp = self._sample_label(i)
            color_rgba = self._hex_to_rgba(self._sample_color(i), alpha)
            show_legend = (self._label is not None) and (grp not in added_groups)
            if show_legend:
                added_groups.add(grp)

            fig.add_trace(
                go.Scatter(
                    x=self._ppm,
                    y=row,
                    mode="lines",
                    name=grp,
                    legendgroup=grp,
                    showlegend=bool(show_legend),
                    line=dict(color=color_rgba, width=linewidth),
                    opacity=alpha,
                )
            )

        self._apply_common_layout(
            fig,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            fig_height=fig_height,
            fig_width=fig_width,
            font_size=font_size,
            xaxis_reversed=xaxis_reversed,
        )
        return fig

    def mean_sd(
        self,
        fig_height: int = 500,
        fig_width: int = 1400,
        font_size: int = 14,
        title: Optional[str] = None,
        show_individual: bool = False,
        xaxis_title: str = "δ¹H (ppm)",
        yaxis_title: str = "Intensity",
        xaxis_reversed: bool = True,
    ) -> go.Figure:
        """Plot mean spectrum with shaded SD envelope per group.

        For each group (or for all samples when no label is set), a solid mean
        line and a shaded standard-deviation band are drawn using
        ``go.Scatter`` with ``fill="tonexty"``.

        Parameters:
            fig_height (int): Figure height in pixels. Default is 500.
            fig_width (int): Figure width in pixels. Default is 1400.
            font_size (int): Base font size. Default is 14.
            title (str, optional): Plot title. Default is None.
            show_individual (bool): When True, individual spectra are drawn as
                faint background lines. Default is False.
            xaxis_title (str): X-axis label. Default is ``"delta1H (ppm)"``.
            yaxis_title (str): Y-axis label. Default is ``"Intensity"``.
            xaxis_reversed (bool): Reverse the x-axis (NMR convention).
                Default is True.

        Returns:
            go.Figure: Plotly figure with mean and SD envelope traces.

        Examples:
            >>> fig = sp.mean_sd(show_individual=True)
            >>> fig.show()
        """
        fig = go.Figure()

        if self._label is None:
            groups_iter = [("All samples", self._spectra)]
            color_map = {"All samples": px.colors.qualitative.Plotly[0]}
        else:
            groups_iter = [
                (g, self._spectra.loc[self._label == g]) for g in self._groups
            ]
            color_map = self._color_dict

        for grp, subset in groups_iter:
            color = color_map.get(grp, "#636EFA")
            color_rgba_fill = self._hex_to_rgba(color, 0.2)
            color_line = self._hex_to_rgba(color, 1.0)

            arr = subset.values.astype(float)
            mean_vals = arr.mean(axis=0)
            sd_vals = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_vals)

            # Optionally show individual spectra
            if show_individual:
                for j in range(arr.shape[0]):
                    fig.add_trace(
                        go.Scatter(
                            x=self._ppm,
                            y=arr[j],
                            mode="lines",
                            name=grp,
                            legendgroup=grp,
                            showlegend=False,
                            line=dict(color=self._hex_to_rgba(color, 0.15), width=0.5),
                        )
                    )

            # Lower SD boundary (invisible, used as fill base)
            fig.add_trace(
                go.Scatter(
                    x=self._ppm,
                    y=mean_vals - sd_vals,
                    mode="lines",
                    name=grp,
                    legendgroup=grp,
                    showlegend=False,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                )
            )

            # Upper SD boundary filled to lower
            fig.add_trace(
                go.Scatter(
                    x=self._ppm,
                    y=mean_vals + sd_vals,
                    mode="lines",
                    name=grp,
                    legendgroup=grp,
                    showlegend=False,
                    fill="tonexty",
                    fillcolor=color_rgba_fill,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                )
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=self._ppm,
                    y=mean_vals,
                    mode="lines",
                    name=grp,
                    legendgroup=grp,
                    showlegend=True,
                    line=dict(color=color_line, width=2),
                )
            )

        self._apply_common_layout(
            fig,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            fig_height=fig_height,
            fig_width=fig_width,
            font_size=font_size,
            xaxis_reversed=xaxis_reversed,
        )
        return fig

    def stacked(
        self,
        offset_factor: float = 0.3,
        fig_height: int = 800,
        fig_width: int = 1400,
        font_size: int = 12,
        title: Optional[str] = None,
        xaxis_title: str = "δ¹H (ppm)",
        xaxis_reversed: bool = True,
    ) -> go.Figure:
        """Plot spectra stacked vertically with a uniform offset.

        Each spectrum is shifted upward by ``offset_factor * max(|spectra|)``
        relative to the previous one so they do not overlap. Tick labels on the
        y-axis show sample IDs.

        Parameters:
            offset_factor (float): Fraction of the global intensity range used
                as the vertical step between spectra. Default is 0.3.
            fig_height (int): Figure height in pixels. Default is 800.
            fig_width (int): Figure width in pixels. Default is 1400.
            font_size (int): Base font size. Default is 12.
            title (str, optional): Plot title. Default is None.
            xaxis_title (str): X-axis label. Default is ``"delta1H (ppm)"``.
            xaxis_reversed (bool): Reverse the x-axis (NMR convention).
                Default is True.

        Returns:
            go.Figure: Plotly figure with stacked spectrum traces.

        Examples:
            >>> fig = sp.stacked(offset_factor=0.5)
            >>> fig.show()
        """
        fig = go.Figure()

        global_max = np.abs(self._spectra.values).max()
        step = offset_factor * global_max
        added_groups: set = set()

        tickvals = []
        ticktext = []

        for i in range(len(self._spectra)):
            row = self._spectra.iloc[i].values.astype(float)
            offset = i * step
            y_shifted = row + offset

            grp = self._sample_label(i)
            color = self._sample_color(i)
            show_legend = (self._label is not None) and (grp not in added_groups)
            if show_legend:
                added_groups.add(grp)

            sample_id = (
                self._spectra.index[i]
                if self._spectra.index is not None
                else str(i)
            )
            tickvals.append(offset)
            ticktext.append(str(sample_id))

            fig.add_trace(
                go.Scatter(
                    x=self._ppm,
                    y=y_shifted,
                    mode="lines",
                    name=grp,
                    legendgroup=grp,
                    showlegend=bool(show_legend),
                    line=dict(color=color, width=1),
                )
            )

        fig.update_layout(
            title=title,
            height=fig_height,
            width=fig_width,
            font=dict(size=font_size),
            xaxis_title=xaxis_title,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                autorange="reversed" if xaxis_reversed else True,
                showgrid=True,
                gridcolor="lightgrey",
                zeroline=False,
            ),
            yaxis=dict(
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=False,
                zeroline=False,
            ),
        )
        return fig

    def single(
        self,
        sample_id: Optional[Union[int, str]] = None,
        annotate_peaks: Optional[List[float]] = None,
        fig_height: int = 400,
        fig_width: int = 1400,
        font_size: int = 14,
        title: Optional[str] = None,
        xaxis_title: str = "δ¹H (ppm)",
        yaxis_title: str = "Intensity",
        xaxis_reversed: bool = True,
    ) -> go.Figure:
        """Plot a single NMR spectrum.

        When ``sample_id`` is None the median spectrum (element-wise median
        across all samples) is plotted. Vertical dashed lines with ppm
        annotations can be added via ``annotate_peaks``.

        Parameters:
            sample_id (int or str, optional): Index label or integer position
                of the sample to plot. When None the median spectrum is used.
                Default is None.
            annotate_peaks (list of float, optional): ppm positions at which
                vertical dashed annotation lines are drawn. Default is None.
            fig_height (int): Figure height in pixels. Default is 400.
            fig_width (int): Figure width in pixels. Default is 1400.
            font_size (int): Base font size. Default is 14.
            title (str, optional): Plot title. Default is None.
            xaxis_title (str): X-axis label. Default is ``"delta1H (ppm)"``.
            yaxis_title (str): Y-axis label. Default is ``"Intensity"``.
            xaxis_reversed (bool): Reverse the x-axis (NMR convention).
                Default is True.

        Returns:
            go.Figure: Plotly figure containing the single spectrum trace.

        Examples:
            >>> fig = sp.single(sample_id=0, annotate_peaks=[1.33, 3.05])
            >>> fig.show()
        """
        if sample_id is None:
            y = np.median(self._spectra.values.astype(float), axis=0)
            name = "Median spectrum"
            color = px.colors.qualitative.Plotly[0]
        else:
            if isinstance(sample_id, int) and sample_id not in self._spectra.index:
                row = self._spectra.iloc[sample_id]
            else:
                row = self._spectra.loc[sample_id]
            y = row.values.astype(float)
            name = str(sample_id)
            # Determine position index for colour
            try:
                pos = list(self._spectra.index).index(
                    sample_id if not isinstance(sample_id, int)
                    else self._spectra.index[sample_id]
                )
            except (ValueError, IndexError):
                pos = 0
            color = self._sample_color(pos)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self._ppm,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
            )
        )

        if annotate_peaks:
            y_max = float(np.max(np.abs(y)))
            for peak_ppm in annotate_peaks:
                fig.add_vline(
                    x=peak_ppm,
                    line=dict(color="red", dash="dash", width=1),
                    annotation_text=f"{peak_ppm:.3f}",
                    annotation_position="top",
                    annotation=dict(font_size=font_size - 2, textangle=-90),
                )

        self._apply_common_layout(
            fig,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            fig_height=fig_height,
            fig_width=fig_width,
            font_size=font_size,
            xaxis_reversed=xaxis_reversed,
        )
        return fig
