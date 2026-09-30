# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import stdtr

from .._native import pearson_columns


def _stocsy_statistics(spectra: pd.DataFrame, anchor_index: int):
    """Return vectorized Pearson correlations and two-sided p-values."""
    values = spectra.to_numpy(dtype=np.float64, copy=False)
    correlations = pearson_columns(values, anchor_index)
    degrees_of_freedom = values.shape[0] - 2

    if values.shape[0] == 2:
        p_values = np.ones(values.shape[1], dtype=np.float64)
        p_values[np.isnan(correlations)] = np.nan
        return correlations, p_values

    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = np.maximum(1.0 - correlations ** 2, 0.0)
        t_statistics = correlations * np.sqrt(
            degrees_of_freedom / denominator
        )
    p_values = 2.0 * stdtr(degrees_of_freedom, -np.abs(t_statistics))
    return correlations, p_values


GREY = "#b0b0b0"
PLOT_TYPES = ("line", "markers")


def _segments(x: np.ndarray, y: np.ndarray, starts: np.ndarray):
    """x / y for line segments i -> i+1 (for each i in `starts`), separated by gaps (None)."""
    xs = np.empty(len(starts) * 3, dtype=object)
    ys = np.empty(len(starts) * 3, dtype=object)
    xs[0::3], xs[1::3], xs[2::3] = x[starts], x[starts + 1], None
    ys[0::3], ys[1::3], ys[2::3] = y[starts], y[starts + 1], None
    return xs, ys


def stocsy_figure(ppm, y, correlations, p_values, anchor_ppm_value, p_value_threshold=0.0001,
                  plot="line", colorscale="Jet", n_levels=64, title="STOCSY"):
    """
    Draw STOCSY results.

    ``plot="line"`` (default) draws ``y`` (the median spectrum) as a line whose colour follows r^2
    (the squared correlation with the anchor), with a colour bar; segments with no significant end
    (p >= ``p_value_threshold`` at both ends) are grey. Plotly gives one line trace a single colour,
    so segments are grouped into ``n_levels`` colour levels, one trace per level. An invisible marker
    layer carries the colour bar and hover values (delta, r^2, r, p).

    ``plot="markers"`` keeps the classic plot: grey and red points for non-significant and significant.
    """
    if plot not in PLOT_TYPES:
        raise ValueError(f"plot must be one of {PLOT_TYPES}, got {plot!r}")
    from plotly.colors import get_colorscale, sample_colorscale

    x = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.asarray(correlations, dtype=float)
    p = np.asarray(p_values, dtype=float)
    r2 = np.nan_to_num(r ** 2)
    significant = p < p_value_threshold
    anchor = np.round(anchor_ppm_value, decimals=4)
    fig = go.Figure()

    if plot == "markers":
        for mask, colour, name in ((~significant, "gray", "Non-significant"),
                                   (significant, "red", f"Significant (<i>p</i> < {p_value_threshold})")):
            fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", marker=dict(size=3, color=colour), name=name))
        y_title = f"Median intensity (red: <i>p</i> < {p_value_threshold})"
    else:
        starts = np.arange(len(x) - 1)
        coloured = significant[:-1] | significant[1:]  # a segment touching a significant point
        level = np.minimum(((r2[:-1] + r2[1:]) / 2 * n_levels).astype(int), n_levels - 1)
        palette = sample_colorscale(get_colorscale(colorscale), (np.arange(n_levels) + 0.5) / n_levels, colortype="rgb")
        grey = starts[~coloured]
        if len(grey):
            gx, gy = _segments(x, y, grey)
            fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color=GREY, width=1.2),
                                     name=f"Not significant (<i>p</i> ≥ {p_value_threshold})", hoverinfo="skip",
                                     showlegend=False))  # the colour bar title explains grey
        for k in np.unique(level[coloured]):
            lx, ly = _segments(x, y, starts[coloured & (level == k)])
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=palette[k], width=1.6),
                                     showlegend=False, hoverinfo="skip", name=f"r² ≈ {(k + 0.5) / n_levels:.2f}"))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name="r²", showlegend=False,
            marker=dict(size=6, opacity=0, color=r2, colorscale=colorscale, cmin=0, cmax=1, showscale=True,
                        colorbar=dict(title=dict(text=f"r²<br><sup>grey: p ≥ {p_value_threshold:g}</sup>"), thickness=14)),
            customdata=np.column_stack([r, p]),
            hovertemplate="δ %{x:.4f}<br>r² %{marker.color:.3f}<br>r %{customdata[0]:.3f}"
                          "<br>p %{customdata[1]:.2g}<extra></extra>",
        ))
        y_title = "Median intensity (colour: r²)"
        fig.add_vline(x=float(anchor_ppm_value), line_dash="dot", line_color="#888888")

    fig.update_layout(
        title={'text': f'<b>{title}: δ {anchor}</b>', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='<b>δ<sup>1</sup>H</b>',
        yaxis_title=y_title,
        showlegend=plot == "markers",  # the line plot explains itself with the colour bar
    )
    fig.update_xaxes(autorange="reversed")
    return fig


def STOCSY(spectra: pd.DataFrame, anchor_ppm_value, p_value_threshold=0.0001, plot="line", colorscale="Jet"):

    """
    Performs a STOCSY (Statistic Total Correlation Spectroscopy) analysis on NMR spectra data.

    This function calculates the Pearson correlation between a specified anchor signal
    (identified by its PPM value) and all other signals in the NMR spectra, and draws the
    median spectrum coloured by the squared correlation (r^2) with the anchor.

    Parameters:
    -----------
    spectra : pd.DataFrame
        A DataFrame containing the NMR spectra data, where each column represents a
        chemical shift in ppm and each row represents a sample.

    anchor_ppm_value : float
        The PPM value of the anchor signal used for correlation analysis.

    p_value_threshold : float, optional
        The threshold for determining significance. Segments with a p-value at or above
        this threshold are drawn grey. Default is 0.0001.

    plot : {"line", "markers"}, optional
        "line" (default): the median spectrum as a line with an r^2 colour gradient and colour bar.
        "markers": the classic plot with grey (non-significant) and red (significant) points.

    colorscale : str, optional
        Any Plotly colour scale name for the r^2 gradient, e.g. "Jet" (default), "Turbo", "Viridis".

    Returns:
    --------
    fig : go.Figure
        A Plotly figure object of the correlation results.

    Example:
    ---------
    >>> fig = STOCSY(spectra=spectra, anchor_ppm_value=1.29275, p_value_threshold=0.0000001)
    >>> fig.show()
    """
    ppm = spectra.columns.astype(float).to_numpy()
    anchor_index = int(np.argmin(np.abs(ppm - anchor_ppm_value)))
    correlations, p_values = _stocsy_statistics(spectra, anchor_index)
    return stocsy_figure(ppm, spectra.median().to_numpy(), correlations, p_values, anchor_ppm_value,
                         p_value_threshold, plot=plot, colorscale=colorscale)
