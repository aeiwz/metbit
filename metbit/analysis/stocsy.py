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


def STOCSY(spectra: pd.DataFrame, anchor_ppm_value, p_value_threshold=0.0001):

    """
    Performs a STOCSY (Statistic Total Correlation Spectroscopy) analysis on NMR spectra data.

    This function calculates the Pearson correlation between a specified anchor signal 
    (identified by its PPM value) and all other signals in the NMR spectra. It identifies 
    significant correlations based on the specified p-value threshold and visualizes 
    the results in a scatter plot.

    Parameters:
    -----------
    spectra : pd.DataFrame
        A DataFrame containing the NMR spectra data, where each column represents a 
        chemical shift in ppm and each row represents a sample.

    anchor_ppm_value : float
        The PPM value of the anchor signal used for correlation analysis.

    p_value_threshold : float, optional
        The threshold for determining significance. Correlations with a p-value less than 
        this threshold will be marked as significant. Default is 0.0001.

    Returns:
    --------
    fig : go.Figure
        A Plotly figure object containing the scatter plot of the correlation results.

    Example:
    ---------
    >>> fig = STOCSY(spectra=spectra, anchor_ppm_value=1.29275, p_value_threshold=0.0000001)
    >>> fig.show()
    """
    # Step 1: Load NMR spectra data

    ppm = spectra.columns.astype(float).to_list()  # Convert column names to floats (ppm values)

    # Step 2: NMR spectra data (X is already a DataFrame)
    X = spectra

    # Step 3: Find the index of the anchor ppm in the list of ppm values
    anchor_index = np.argmin(np.abs(np.array(ppm) - anchor_ppm_value))

    # Step 4: Calculate correlations and p-values in one matrix pass.
    correlations, p_values = _stocsy_statistics(X, anchor_index)

    # Step 5: Calculate r^2 (squared correlation) for each point
    r_squared = correlations ** 2

    # Step 6: Prepare plotly scatter plot
    fig = go.Figure()

    # Scatter plot of non-significant points
    non_significant_mask = p_values >= p_value_threshold
    fig.add_trace(go.Scatter(
        x=np.array(ppm)[non_significant_mask],
        y=X.median()[non_significant_mask],
        mode='markers',
        marker=dict(
            size=3,
            color='gray',
        ),
        name='Non-significant'
    ))

    # Scatter plot of significant points (marked in red)
    significant_mask = p_values < p_value_threshold
    fig.add_trace(go.Scatter(
        x=np.array(ppm)[significant_mask],
        y=X.median()[significant_mask],
        mode='markers',
        marker=dict(
            size=3,
            color='red',  # Red color for significant points
        ),
        name=f'Significant (<i>p</i> < {p_value_threshold})'
    ))

    # Add labels and title
    fig.update_layout(
        title={'text':f'<b>STOCSY: δ {np.round(anchor_ppm_value, decimals=4)}</b>',
                'y':0.9,
                'x':0.5,
                'xanchor':'center',
                'yanchor':'top'},
        xaxis_title='<b>δ<sup>1</sup>H</b>',
        yaxis_title=f'Correlation (r<sup>2</sup>) δ = {np.round(anchor_ppm_value, decimals=4)}',
        showlegend=True
    )

    #invert x-axis
    fig.update_xaxes(autorange="reversed")
    # Display the interactive plot
    return fig

# Example usage
#plot_nmr_correlation(spectra=spectra, anchor_ppm_value=1.29275, p_value_threshold=0.0000001)
