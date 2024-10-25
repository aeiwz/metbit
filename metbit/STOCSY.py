import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr



def plot_nmr_correlation(spectra: pd.DataFrame, anchor_ppm_value, p_value_threshold=0.0001):
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
        name=f'Significant (p < {p_value_threshold})'
    ))

    # Add labels and title
    fig.update_layout(
        title=f'STOCSY: Correlation Profile with Anchor ppm {anchor_ppm_value}',
        xaxis_title='Chemical Shift (ppm)',
        yaxis_title=r'Correlation (r^2)',
        showlegend=True
    )

    # Display the interactive plot
    fig.show()

# Example usage
#plot_nmr_correlation(spectra=spectra, anchor_ppm_value=1.29275, p_value_threshold=0.0000001)