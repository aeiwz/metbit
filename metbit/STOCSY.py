# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr



def STOCSY(spectra: pd.DataFrame, anchor_ppm_value, p_value_threshold=0.0001):

    """
    Performs a STOCSY (Soft Toward Correlation Spectroscopy) analysis on NMR spectra data.

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

    # Step 4: Calculate Pearson correlation and p-values for the anchor point against all others
    correlations = []
    p_values = []

    for col in X.columns:
        # Calculate correlation between the anchor signal and each other signal
        corr, p_val = pearsonr(X.iloc[:, anchor_index], X[col])
        correlations.append(corr)
        p_values.append(p_val)

    correlations = np.array(correlations)
    p_values = np.array(p_values)

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
        title={'text':f'<b>STOCSY: δ {anchor_ppm_value}</b>',
                'y':0.9,
                'x':0.5,
                'xanchor':'center',
                'yanchor':'top'},
        xaxis_title='Chemical Shift (ppm)',
        yaxis_title=f'Correlation (r<sup>2</sup>) δ = {anchor_ppm_value}',
        showlegend=True
    )

    #invert x-axis
    fig.update_xaxes(autorange="reversed")
    # Display the interactive plot
    return fig

# Example usage
#plot_nmr_correlation(spectra=spectra, anchor_ppm_value=1.29275, p_value_threshold=0.0000001)