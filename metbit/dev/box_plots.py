# -*- coding: utf-8 -*-

__author__ = 'aeiwz'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Development"


'''def create_boxplot_with_pvalues(
    df, 
    x_col, 
    y_col, 
    group_order=None, 
    custom_colors=None, 
    p_value_threshold=0.05, 
    annotate_style="value", 
    figure_size=(800, 600),
    y_offset=5,
    show_non_significant=True,
    title_=None,
    y_label=None,
    x_label=None,
    fig_height=800,
    fig_width=600
    ):
    """
    Creates a box plot with tiered p-value annotations.

    Parameters:
    - df (DataFrame): The input data.
    - x_col (str): Column name for categorical grouping.
    - y_col (str): Column name for numerical values.
    - group_order (list, optional): Order of groups for visualization.
    - custom_colors (dict, optional): Custom colors for each group.
    - p_value_threshold (float, optional): Threshold for significance annotations.
    - annotate_style (str, optional): 'value' for p-value text, 'symbol' for stars ('*').
    - figure_size (tuple, optional): Width and height of the figure.
    - y_offset (int, optional): Vertical spacing for p-value annotations.

    Returns:
    - fig (Figure): The Plotly figure object.
    """
    if title_ is None:
        title_ = y_col
    else:
        title_ = title_

    if y_label is None:
        y_label = y_col
    else: 
        y_label = y_label

    if x_label is None:
        x_label = x_col
    else:
        x_label = x_label

    # Group data and calculate p-values
    grouped = df.groupby(x_col)[y_col]
    if group_order is None:
        group_order = list(grouped.groups.keys())
    comparisons = list(combinations(group_order, 2))
    p_values = []
    for g1, g2 in comparisons:
        t_stat, p_val = ttest_ind(grouped.get_group(g1), grouped.get_group(g2))
        p_values.append((g1, g2, p_val))

    # Create box plot
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col, 
        color=x_col, 
        points="all", 
        category_orders={x_col: group_order}, 
        color_discrete_map=custom_colors
    )

    # Add p-value annotations
    max_y = df[y_col].max()  # Maximum y-value for positioning annotations
    annotations = []
    lines = []

    for i, (g1, g2, p_val) in enumerate(p_values):
        y_pos = max_y + (i + 1) * y_offset  # Increment vertical position for each comparison
        if annotate_style == "value":
            p_text = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        elif annotate_style == "symbol":
            if p_val < p_value_threshold:
                p_text = "*" if p_val < 0.05 else ""
                p_text += "*" if p_val < 0.01 else ""
                p_text += "*" if p_val < 0.001 else ""
            else:
                if show_non_significant is not True:
                    continue  # Skip insignificant comparisons
                else:
                    p_text = "ns"
        else:
            raise ValueError("Invalid annotate_style. Use 'value' or 'symbol'.")

        # Add horizontal line for the comparison
        lines.append(
            go.Scatter(
                x=[g1, g1, g2, g2],  # Two vertical bars and one horizontal
                y=[y_pos, y_pos + 1, y_pos + 1, y_pos],  # Create the bracket shape
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="none",
            )
        )
        
        # Add text annotation for the p-value
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=(group_order.index(g1) + group_order.index(g2)) / 2,
                y=y_pos + 2.5,
                text=p_text,
                showarrow=False,
                font=dict(size=12),
            )
        )

    


    # Add lines to the figure
    for line in lines:
        fig.add_trace(line)

    # Update layout with annotations and figure size
    fig.update_layout(
        annotations=annotations,
        title=f'<b>{title_}</b>',
        yaxis_title=y_label,
        xaxis_title=x_label,
        legend_title=x_col,
        width=fig_width,
        height=fig_height,
        showlegend=False
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    #update title position to center
    fig.update_layout(
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    

    return fig
'''

'''
import numpy as np

def boxplot_stats(
    df, 
    x_col, 
    y_col, 
    group_order=None, 
    custom_colors=None, 
    p_value_threshold=0.05, 
    annotate_style="value", 
    figure_size=(800, 600),
    y_offset_factor=0.05,  # Use a factor to scale offset based on y-axis range
    show_non_significant=True,
    title_=None,
    y_label=None,
    x_label=None,
    fig_height=800,
    fig_width=600
    ):
    """
    Creates a box plot with tiered p-value annotations with improved y-axis positioning.

    Parameters:
    - df (DataFrame): The input data.
    - x_col (str): Column name for categorical grouping.
    - y_col (str): Column name for numerical values.
    - group_order (list, optional): Order of groups for visualization.
    - custom_colors (dict, optional): Custom colors for each group.
    - p_value_threshold (float, optional): Threshold for significance annotations.
    - annotate_style (str, optional): 'value' for p-value text, 'symbol' for stars ('*').
    - figure_size (tuple, optional): Width and height of the figure.
    - y_offset_factor (float, optional): Scaling factor for vertical spacing based on y-axis range.

    Returns:
    - fig (Figure): The Plotly figure object.
    """
    # Basic setup
    if title_ is None:
        title_ = y_col

    y_label = y_label or y_col
    x_label = x_label or x_col

    # Calculate dynamic y-offset based on data range
    y_range = df[y_col].max() - df[y_col].min()
    y_offset = y_offset_factor * y_range

    # Group data and calculate p-values
    grouped = df.groupby(x_col)[y_col]
    if group_order is None:
        group_order = list(grouped.groups.keys())
    comparisons = list(combinations(group_order, 2))
    p_values = []
    for g1, g2 in comparisons:
        t_stat, p_val = ttest_ind(grouped.get_group(g1), grouped.get_group(g2))
        p_values.append((g1, g2, p_val))

    # Create box plot
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col, 
        color=x_col, 
        points="all", 
        category_orders={x_col: group_order}, 
        color_discrete_map=custom_colors
    )

    # Add p-value annotations with dynamic y-offset
    max_y = df[y_col].max()
    annotations = []
    lines = []

    for i, (g1, g2, p_val) in enumerate(p_values):
        y_pos = max_y + (i + 1) * y_offset  # Dynamically adjusted based on data
        if annotate_style == "value":
            p_text = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        elif annotate_style == "symbol":
            if p_val < p_value_threshold:
                p_text = "*" if p_val < 0.05 else ""
                p_text += "*" if p_val < 0.01 else ""
                p_text += "*" if p_val < 0.001 else ""
            else:
                if show_non_significant:
                    p_text = "ns"
                else:
                    continue
        else:
            raise ValueError("Invalid annotate_style. Use 'value' or 'symbol'.")

        # Add horizontal line for the comparison
        lines.append(
            go.Scatter(
                x=[g1, g1, g2, g2], 
                y=[y_pos, y_pos + y_offset * 0.5, y_pos + y_offset * 0.5, y_pos], 
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="none",
            )
        )
        
        # Add text annotation for the p-value
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=(group_order.index(g1) + group_order.index(g2)) / 2,
                y=y_pos + y_offset * 0.75,  # Position text above line
                text=p_text,
                showarrow=False,
                font=dict(size=12),
            )
        )

    # Add lines to the figure
    for line in lines:
        fig.add_trace(line)

    # Update layout with annotations and figure size
    fig.update_layout(
        annotations=annotations,
        title=f'<b>{title_}</b>',
        yaxis_title=y_label,
        xaxis_title=x_label,
        legend_title=x_col,
        width=fig_width,
        height=fig_height,
        showlegend=False
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    #update title position to center
    fig.update_layout(
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    

    return fig'''

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from pingouin import compute_effsize
from scipy.stats import ttest_ind, f_oneway
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations


def boxplot_stats(
    df, 
    x_col, 
    y_col, 
    group_order=None, 
    custom_colors=None, 
    stats_options=None,  # User-selectable statistical methods
    p_value_threshold=0.05, 
    annotate_style="value", 
    figure_size=(800, 600),
    y_offset_factor=0.05,  
    show_non_significant=True,
    correct_p="bonferroni",  # Option for multiple testing correction
    title_=None,
    y_label=None,
    x_label=None,
    fig_height=800,
    fig_width=600
):
    """

    Enhanced box plot function with customizable statistical analysis and annotation.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data for the plot.
    x_col : str
        The name of the column representing the categorical variable (e.g., treatment groups).
    y_col : str
        The name of the column representing the numerical variable (e.g., scores).
    group_order : list, optional
        Custom order of groups for the x-axis. Defaults to the natural group order in the data.
    custom_colors : dict, optional
        A dictionary mapping group names to specific colors (e.g., {"A": "red", "B": "blue"}).
    stats_options : list of str, optional
        Statistical tests and calculations to perform. Options:
        - "t-test": Perform pairwise Student's t-tests between groups.
        - "nonparametric": Use Mann-Whitney U test for pairwise comparisons.
        - "anova": Perform a one-way ANOVA (requires more than two groups).
        - "effect-size": Calculate Cohen's d for pairwise comparisons (not supported for ANOVA).
        Defaults to ["t-test"].
    p_value_threshold : float, optional
        Threshold for considering p-values as significant. Default is 0.05.
    annotate_style : str, optional
        Style for annotations. Options:
        - "value": Show exact p-values (e.g., "p=0.0123").
        - "symbol": Use significance symbols (e.g., "***", "**", "*", or "ns" for not significant).
        Default is "value".
    figure_size : tuple, optional
        Tuple specifying the width and height of the plot (in pixels). Default is (800, 600).
    y_offset_factor : float, optional
        Proportion of the y-axis range to use for spacing annotations. Default is 0.05.
    show_non_significant : bool, optional
        Whether to display annotations for non-significant comparisons. Default is True.
    correct_p : str, optional
        Method for correcting p-values for multiple comparisons. Options include:
        - "bonferroni"
        - "holm"
        - "fdr_bh" (Benjamini-Hochberg)
        - None (no correction)
        Default is "bonferroni".
    title_ : str, optional
        Title of the plot. Defaults to the name of the y_col column.
    y_label : str, optional
        Label for the y-axis. Defaults to the name of the y_col column.
    x_label : str, optional
        Label for the x-axis. Defaults to the name of the x_col column.
    fig_height : int, optional
        Height of the figure in pixels. Default is 800.
    fig_width : int, optional
        Width of the figure in pixels. Default is 600.

    Returns:
    --------
    plotly.graph_objects.Figure
        A Plotly Figure object containing the enhanced box plot with statistical annotations.

    Examples:
    ---------
    Example 1: Basic box plot with t-tests and Bonferroni correction:
        data = {
            "treatment": ["A"] * 10 + ["B"] * 10,
            "score": [0.5, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.5,
                      0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8, 0.5, 0.4],
        }
        df = pd.DataFrame(data)
        fig = boxplot_stats(
            df, 
            x_col="treatment", 
            y_col="score", 
            stats_options=["t-test"], 
            correct_p="bonferroni", 
            p_value_threshold=0.05
        )
        fig.show()

    Example 2: Advanced plot with custom colors, ANOVA, and effect sizes:
        data = {
            "treatment": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "score": np.random.rand(30),
        }
        df = pd.DataFrame(data)
        custom_colors = {"A": "red", "B": "blue", "C": "green"}
        fig = boxplot_stats(
            df, 
            x_col="treatment", 
            y_col="score", 
            stats_options=["anova", "effect-size"], 
            custom_colors=custom_colors
        )
        fig.show()
    
    """
    # Set default title and labels
    if title_ is None:
        title_ = y_col

    y_label = y_label or y_col
    x_label = x_label or x_col

    # Calculate y-offset for annotations
    y_range = df[y_col].max() - df[y_col].min()
    y_offset = y_offset_factor * y_range

    # Prepare group data and combinations
    grouped = df.groupby(x_col)[y_col]
    if group_order is None:
        group_order = list(grouped.groups.keys())
    comparisons = list(combinations(group_order, 2))

    # Default to Student's t-test if no methods are selected
    if stats_options is None:
        stats_options = ["t-test"]

    # Initialize storage for results
    p_values = []
    effect_sizes = []
    annotations = []
    lines = []

    # Perform statistical analysis
    if "anova" in stats_options and len(group_order) > 2:
        # Perform one-way ANOVA
        group_data = [grouped.get_group(group) for group in group_order]
        f_stat, anova_p_val = f_oneway(*group_data)
        p_values = [anova_p_val] * len(comparisons)  # Same p-value for all comparisons in ANOVA

    else:
        # Pairwise comparisons
        for g1, g2 in comparisons:
            group1 = grouped.get_group(g1)
            group2 = grouped.get_group(g2)

            # Perform Student's t-test or nonparametric test
            if "t-test" in stats_options:
                t_stat, p_val = ttest_ind(group1, group2)
            elif "nonparametric" in stats_options:
                from scipy.stats import mannwhitneyu
                _, p_val = mannwhitneyu(group1, group2, alternative="two-sided")
            else:
                raise ValueError("Invalid stats_options. Use 't-test', 'nonparametric', or 'anova'.")

            p_values.append(p_val)

            # Compute effect size if applicable
            if "effect-size" in stats_options:
                effect_size = compute_effsize(group1, group2, eftype="cohen")
                effect_sizes.append(effect_size)

    # Apply multiple testing correction
    if correct_p and "anova" not in stats_options:  # No correction for a single ANOVA test
        _, corrected_p_values, _, _ = multipletests(p_values, method=correct_p)
        p_values = corrected_p_values

    # Create the box plot
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col, 
        color=x_col, 
        points="all", 
        category_orders={x_col: group_order}, 
        color_discrete_map=custom_colors
    )

    # Add annotations for p-values, effect sizes, and confidence intervals
    max_y = df[y_col].max()
    for i, ((g1, g2), p_val) in enumerate(zip(comparisons, p_values)):
        y_pos = max_y + (i + 1) * y_offset  # Adjust y-position dynamically
        annotation_text = ""

        if p_val < p_value_threshold or show_non_significant:
            # Format p-value annotation
            if annotate_style == "value":
                p_text = f"p={p_val:.4f}" if p_val >= 0.0001 else "p<0.0001"
            elif annotate_style == "symbol":
                if p_val > p_value_threshold:
                    p_text = "ns" if show_non_significant else ""
                elif p_val < 0.001:
                    p_text = "***"
                elif p_val < 0.01:
                    p_text = "**"
                elif p_val < 0.05:
                    p_text = "*"
                elif p_val > p_value_threshold:
                    p_text = "ns" if show_non_significant else ""
                else:
                    p_text = "ns" if show_non_significant else ""
            else:
                raise ValueError("Invalid annotate_style. Use 'value' or 'symbol'.")
            annotation_text += p_text



            # Add effect size annotation if applicable
            if "effect-size" in stats_options and "anova" not in stats_options:
                annotation_text += f", d={effect_sizes[i]:.2f}"

            # Draw horizontal line for the comparison
            lines.append(
                go.Scatter(
                    x=[g1, g1, g2, g2], 
                    y=[y_pos, y_pos + y_offset * 0.5, y_pos + y_offset * 0.5, y_pos], 
                    mode="lines",
                    line=dict(color="black", width=1),
                    hoverinfo="none",
                )
            )

            # Add text annotation
            annotations.append(
                dict(
                    xref="x",
                    yref="y",
                    x=(group_order.index(g1) + group_order.index(g2)) / 2,
                    y=y_pos + y_offset * 0.90,
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=12),
                )
            )

    # Add lines and annotations to the plot
    for line in lines:
        fig.add_trace(line)

    # Update figure layout
    fig.update_layout(
        annotations=annotations,
        title=f'<b>{title_}</b>',
        yaxis_title=y_label,
        xaxis_title=x_label,
        legend_title=x_col,
        width=fig_width,
        height=fig_height,
        showlegend=False,
    )
    fig.update_layout(yaxis=dict(tickformat=".2e"))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        title={
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    return fig


# Example Usage
if __name__ == '__main__':
    data = {
        "treatment": ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10,
        "score": [0.5, 0.7, 0.6, 0.5, 0.6, 0.8, 0.5, 0.6, 0.7, 0.8, 0.5, 0.7, 0.6, 0.5, 0.6, 0.8, 0.5, 0.6, 0.7, 0.8,
                  0.40, 0.45, 0.50, 0.48, 0.41, 0.42, 0.47, 0.44, 0.45, 0.43, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4],
    }

    '''    import pandas as pd
    import numpy as np

    # Define parameters
    n_samples_per_group = 100
    n_groups = 10
    total_samples = n_samples_per_group * n_groups

    # Generate random data
    treatments = [f"Group_{i+1}" for i in range(n_groups)]
    treatment_column = np.repeat(treatments, n_samples_per_group)
    score_column = np.random.normal(loc=50, scale=10, size=total_samples)  # Random scores with mean=50, std=10

    # Create DataFrame
    df = pd.DataFrame({
        'treatment': treatment_column,
        'score': score_column
    })
    '''

    df = pd.DataFrame(data)
    custom_colors = {"A": "red", "B": "blue", "C": "green", "D": "purple"}

    fig = boxplot_stats(
        df, 
        x_col="treatment", 
        y_col="score", 
        stats_options=["t-test"], 
        correct_p="bonferroni", 
        p_value_threshold=0.05, 
        annotate_style="symbol", 
        show_non_significant=True,
        y_label="Score",
        fig_height=1000,
        fig_width=800
    )
    fig.show()
