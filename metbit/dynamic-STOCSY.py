# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor

__author__ = 'aeiwz'
author_email = 'theerayut_aeiw_123@hotmail.com'


def calculate_correlation(anchor_data, target_data):
    """Helper function to calculate correlation and p-value for parallel execution."""
    return pearsonr(anchor_data, target_data)


def STOCSY(spectra: pd.DataFrame, anchor_ppm_value, p_value_threshold=0.0001):
    """
    Performs a STOCSY (Statistic Total Correlation Spectroscopy) analysis on NMR spectra data.

    Parameters:
    -----------
    spectra : pd.DataFrame
        A DataFrame containing the NMR spectra data, where each column represents a 
        chemical shift in ppm and each row represents a sample.

    anchor_ppm_value : float
        The PPM value of the anchor signal used for correlation analysis.

    p_value_threshold : float, optional
        The threshold for determining significance. Default is 0.0001.

    Returns:
    --------
    fig : go.Figure
        A Plotly figure object containing the scatter plot of the correlation results.

    Example:
    ---------
    >>> fig = STOCSY(spectra=spectra, anchor_ppm_value=1.29275, p_value_threshold=0.0000001)
    >>> fig.show()
    """
    # Step 1: Convert column names to floats (ppm values)
    ppm = spectra.columns.astype(float).to_list()
    
    # Step 2: Find the index of the anchor ppm
    anchor_index = np.argmin(np.abs(np.array(ppm) - anchor_ppm_value))
    anchor_data = spectra.iloc[:, anchor_index]

    # Step 3: Calculate Pearson correlations in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda col: calculate_correlation(anchor_data, spectra[col]), spectra.columns))

    # Extract correlation coefficients and p-values
    correlations, p_values = zip(*results)
    correlations = np.array(correlations)
    p_values = np.array(p_values)
    
    # Step 4: Calculate r^2 (squared correlation) for each point
    r_squared = correlations ** 2

    # Step 5: Prepare Plotly scatter plot
    fig = go.Figure()

    # Scatter plot of non-significant points
    non_significant_mask = p_values >= p_value_threshold
    fig.add_trace(go.Scatter(
        x=np.array(ppm)[non_significant_mask],
        y=r_squared[non_significant_mask],
        mode='markers',
        marker=dict(size=3, color='gray'),
        name='Non-significant'
    ))

    # Scatter plot of significant points (marked in red)
    significant_mask = p_values < p_value_threshold
    fig.add_trace(go.Scatter(
        x=np.array(ppm)[significant_mask],
        y=r_squared[significant_mask],
        mode='markers',
        marker=dict(size=3, color='red'),
        name=f'Significant (<i>p</i> < {p_value_threshold})'
    ))

    # Add labels and title
    fig.update_layout(
        title={'text':f'<b>STOCSY: δ {np.round(anchor_ppm_value, decimals=4)}</b>',
               'y':0.9, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
        xaxis_title='<b>δ<sup>1</sup>H</b>',
        yaxis_title=f'Correlation (r<sup>2</sup>) δ = {np.round(anchor_ppm_value, decimals=4)}',
        showlegend=True
    )

    # Invert x-axis
    fig.update_xaxes(autorange="reversed")
    
    # Return the interactive plot
    return fig