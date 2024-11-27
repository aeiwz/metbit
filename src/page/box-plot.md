# Enhanced Box Plot with Statistical Analysis

`boxplot_stats` is a Python function for creating enhanced box plots with integrated statistical analysis and visual annotations. It utilizes libraries such as `pandas`, `plotly`, and `scipy` to provide an interactive and customizable visualization experience.

## Features

- **Customizable Box Plots**: Generate box plots with flexible options for color, group order, and layout.
- **Statistical Analysis**: Perform pairwise **t-tests**, **Mann-Whitney U tests**, or **one-way ANOVA**.
- **Effect Size Calculation**: Compute Cohen's d for pairwise comparisons.
- **Multiple Testing Correction**: Apply Bonferroni, Holm, or FDR-BH corrections for p-values.
- **Dynamic Annotations**: Display p-values or significance symbols (`***`, `**`, `*`, `ns`) directly on the plot.
- **Interactive Visualizations**: Create interactive plots using Plotly.

## Installation

Ensure the following Python packages are installed:

- `pandas`
- `numpy`
- `scipy`
- `plotly`
- `pingouin`
- `statsmodels`

Install them using pip if necessary:

```bash
pip install pandas numpy scipy plotly pingouin statsmodels
```
Usage

**Function Parameters**

| Parameter | Description |	Default Value |
|-----------|-------------|---------------|
| df |	Input DataFrame containing the data. |	- | 
| x_col	| Column name for the categorical variable. |	- |
| y_col	| Column name for the numerical variable. |	- |
| group_order |	Custom order of groups for the x-axis. |	None |
| custom_colors |	Dictionary mapping group names to colors. |	None |
| stats_options	| Statistical methods: ["t-test"], ["nonparametric"], ["anova"], or a combination. |	["t-test"] |
| p_value_threshold	| Threshold for statistical significance. |	default = 0.05 |
| annotate_style |	Style for annotations: "value" (exact p-values) or "symbol" (e.g., ***, **, *, ns). |	"value" |
| y_offset_factor |	Proportion of y-axis range for annotation spacing. |	0.05 |
| show_non_significant |	Display annotations for non-significant results. |	True |
| correct_p |	Method for correcting p-values: "bonferroni", "holm", "fdr_bh", or None. |	"bonferroni" |
| title_ |	Title of the plot. |	None |
| y_label |	Label for the y-axis. |	None |
| x_label |	Label for the x-axis. |	None |
| fig_height |	Height of the figure in pixels. |	800 |
| fig_width |	Width of the figure in pixels. |	600 |

Example

import pandas as pd

# Sample data
data = {
    "treatment": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
    "score": [0.5, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.5,
              0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8, 0.5, 0.4,
              0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4],
}
df = pd.DataFrame(data)

# Generate the plot
fig = boxplot_stats(
    df,
    x_col="treatment",
    y_col="score",
    stats_options=["t-test", "effect-size"],
    custom_colors={"A": "red", "B": "blue", "C": "green"},
    correct_p="bonferroni",
    annotate_style="symbol"
)

fig.show()

Output

The function produces an interactive Plotly box plot with statistical annotations.

Contributing

Contributions, bug fixes, and feature requests are welcome. Please fork the repository and submit a pull request.

License

This project is licensed under the MIT License.

