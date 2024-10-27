# **STOCSY Dash Application**

This is a Dash-based web application for visualizing NMR spectra and performing STOCSY (**S**tatistical **TO**tal **C**orrelation **S**pectroscop**Y**) analysis. The application allows users to select peaks from NMR spectra, set a *p-value* threshold, and perform STOCSY analysis, with caching for improved performance.

Features

•	Interactive NMR Plotting: Visualize and explore NMR spectra by clicking on peaks to select them.
•	STOCSY Analysis: Perform **STOCSY** analysis on selected peaks with adjustable *p-value* threshold.
•	Caching: Uses caching to avoid redundant **STOCSY** computations for repeated requests with the same peak and p-value threshold.

Requirements

The application requires the following libraries:

	•	Dash and dash-bootstrap-components
	•	Plotly
	•	Pandas

Install the necessary libraries using:

```bash
pip install dash dash-bootstrap-components plotly pandas
```
Example Usage

To use the **STOCSY Dash app** with your own data, ensure your .csv file has the same format as shown above. Update the data loading section in app.py if needed.

```python
import pandas as pd
from metbit import STOCSY_app

df = pd.read_csv("path_to_your_file.csv")
spectra = df.iloc[:,1:]
ppm = spectra.columns.astype(float).to_list()

stocsy_app = STOCSY_app(spectra, ppm)
app = stocsy_app.run_ui()
app.run_server(debug=True, port=8051)
```

Run the Application:

```bash
python app.py
```

The application will run locally on http://localhost:8051. Open this URL in your web browser.

The STOCSY_app class initializes the NMR data, sets up the Dash layout, and manages callbacks for the application. Key functionalities include:

	1.	Data Initialization:
	•	Loads NMR spectra data (spectra) and corresponding PPM values (ppm).
	2.	Interactive Plotting:
	•	The plot_NMR_spec inner class generates interactive plots of the NMR spectra.
	3.	STOCSY Analysis with Caching:
	•	The update_stocsy_plot method performs STOCSY analysis using cached results for previously computed peaks and thresholds.

Application Callbacks

	•	update_peaks: Updates peak selection based on user clicks on the NMR plot.
	•	update_stocsy_plot: Runs STOCSY analysis and updates the STOCSY plot. Cached results are used if available.

