# -*- coding: utf-8 -*-

__auther__ ='aeiwz'
author_email='theerayut_aeiw_123@hotmail.com'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Develop"

import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

class pickie_peak:

    """
    A class to display and interact with NMR spectral data, enabling users to pick peaks from an interactive plot.

    Parameters:
    - spectra (pd.DataFrame): A DataFrame containing NMR spectral data with rows representing different spectra and columns representing intensity values at each chemical shift (ppm).
    - ppm (list): A list of ppm values corresponding to the columns in the `spectra` DataFrame.

    Methods:
    - run_ui: Launches a Dash web application with a user interface to visualize NMR spectra, select peaks interactively, and export peak positions.

    Example:
    ```python
        # Load your NMR spectra data
        df = pd.read_csv("https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv")
        spectra = df.iloc[:,1:]
        ppm = spectra.columns.astype(float).to_list()
        label = df['Group']

        # Create instance of the class with spectra and ppm data
        picker_ui = pickie_peak(spectra, ppm, label)


        # Get the app instance
        app = picker_ui.run_ui()

        # Run the app
        app.run_server(debug=True, port=8051)
    ```

    Usage:
    1. **Plot Visualization**: Displays the spectra using an interactive Plotly graph.
    2. **Peak Selection**: Allows users to click on the plot to select peaks, storing their positions.
    3. **Export Functionality**: Offers options to export selected peak positions as a CSV file.
    """

    def __init__(self, spectra: pd.DataFrame, ppm: list, label: list):
        self.spectra = spectra
        self.ppm = ppm
        self.label = label

    def run_ui(self):


        """
        Launches the Dash application for visualizing and selecting NMR spectral peaks.

        Components:
        - Interactive Plot: Displays NMR spectra with options to click on peaks.
        - Peak Selection: Allows users to store selected peak positions.
        - Export Data: Provides an option to download selected peak positions as a CSV file.

        Returns:
        - Dash app: An interactive web application with plot, data selection, and export features.
        """


        import dash
        from dash.dependencies import Input, Output, State
        import plotly.graph_objects as go
        import pandas as pd
        import dash_bootstrap_components as dbc
        from dash import dcc, html

        from lingress import plot_NMR_spec
        
        if self.label is None:
            if self.spectra.shape[0] >= 20:
                spectra_plot = self.spectra.sample(20)
                plotter = plot_NMR_spec(spectra_plot, self.ppm, self.label).single_spectra()
            else:
                plotter = plot_NMR_spec(self.spectra, self.ppm, self.label).single_spectra()
        else:
            plotter = plot_NMR_spec(self.spectra, self.ppm, self.label).median_spectra_group()


        # Create the Dash app with Bootstrap stylesheet
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Layout for the app
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='nmr-plot',
                        figure=plotter
                    ),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='peak-data', children='Click on the plot to select peaks.'),
                    dcc.Store(id='stored-peaks', data=[]),
                    dbc.Button("Export X Positions", id="export-button", color="primary", className="mt-2"),
                    dbc.Button("Clear Data", id="clear-button", color="danger", className="mt-2 ml-2"),
                    dcc.Download(id="download-dataframe-csv")
                ], width=12)
            ])
        ], fluid=True)

        @app.callback(
            [Output('stored-peaks', 'data'), Output('peak-data', 'children')],
            [Input('nmr-plot', 'clickData'), Input('clear-button', 'n_clicks')],
            [State('stored-peaks', 'data')]
        )
        def update_peaks(clickData, clear_clicks, stored_peaks):

            """
            Callback to update stored peaks upon user click or clear action.

            Returns:
            - Updated peaks data and display text with selected peak positions.
            """

            ctx = dash.callback_context

            if not ctx.triggered:
                return stored_peaks, 'Click on the plot to select peaks.'
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'clear-button':
                return [], 'Click on the plot to select peaks.'

            if clickData:
                click_point = clickData['points'][0]
                x_peak = click_point['x']
                y_peak = click_point['y']
                stored_peaks.append({'x': x_peak, 'y': y_peak})
                peak_text = f'Picked peaks (X): {[p["x"] for p in stored_peaks]}'
                return stored_peaks, peak_text

            return stored_peaks, 'Click on the plot to select peaks.'

        @app.callback(
            Output("download-dataframe-csv", "data"),
            [Input("export-button", "n_clicks")],
            [State("stored-peaks", "data")],
            prevent_initial_call=True
        )
        def export_x_positions(n_clicks, stored_peaks):
            if not stored_peaks:
                return dash.no_update

            x_positions = [p['x'] for p in stored_peaks]
            #create unique list of x positions
            x_positions = list(set(x_positions))
            df = pd.DataFrame(x_positions, columns=["X Positions"])
            csv_string = df.to_csv(index=False)

            return dict(content=csv_string, filename="x_positions.csv")

        return app


if __name__ == '__main__':
    # Load your NMR spectra data
    df = pd.read_csv("https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv")
    spectra = df.iloc[:,1:]
    ppm = spectra.columns.astype(float).to_list()
    label = df['Group']

    # Create instance of the class with spectra and ppm data
    picker_ui = pickie_peak(spectra, ppm, label)


    # Get the app instance
    app = picker_ui.run_ui()

    # Run the app
    app.run_server(debug=True, port=8051)