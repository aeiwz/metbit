import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations

def create_boxplot_with_pvalues(
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



if __name__ == '__main__':
    # Example usage
    data = {
        "treatment": ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10,
        "score": [5, 7, 6, 5, 6, 8, 5, 6, 7, 8, 5, 7, 6, 5, 6, 8, 5, 6, 7, 8,
                40, 45, 50, 48, 41, 42, 47, 44, 45, 43, 5, 6, 5, 4, 5, 6, 4, 5, 6, 4],
    }

    df = pd.DataFrame(data)
    custom_colors = {"A": "red", "B": "blue", "C": "green", "D": "purple"}
    fig = create_boxplot_with_pvalues(
        df, 
        x_col="treatment", 
        y_col="score", 
        group_order=["A", "B", "C", "D"], 
        custom_colors=custom_colors, 
        p_value_threshold=0.05, 
        annotate_style="symbol", 
        fig_height=800,
        fig_width=600,
        y_offset=7,
        show_non_significant=True
    )
    fig.show()