# -*- coding: utf-8 -*-

__author__ = 'aeiwz'

import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

# Importing the custom STOCSY function
from STOCSY import STOCSY  # Adjust as necessary based on your file

class STOCSY_app:

    """
    A Dash application for visualizing NMR spectra and performing STOCSY analysis.

    Parameters:
    -----------
    spectra : pd.DataFrame
        DataFrame containing the NMR spectra data.
    ppm : list
        List of PPM (parts per million) values corresponding to the spectra.

    Methods:
    --------
    run_ui() -> dash.Dash:
        Sets up the Dash UI layout and initializes the application callbacks.
    """

    def __init__(self, spectra: pd.DataFrame, ppm: list):

        """
        Initializes the STOCSY_app with NMR spectra and corresponding PPM values.

        Parameters:
        -----------
        spectra : pd.DataFrame
            A DataFrame containing the NMR spectra data, where each row represents a sample.
        ppm : list
            A list of PPM values corresponding to the spectral data columns.
        """
        self.spectra = spectra
        self.ppm = ppm

    def run_ui(self):

        """
        Sets up the Dash application layout and callbacks.

        Returns:
        --------
        app : dash.Dash
            The initialized Dash application instance.
        """

        class plot_NMR_spec:

            """
            A class to handle the plotting of NMR spectra.

            Parameters:
            -----------
            spectra : pd.DataFrame
                DataFrame containing the NMR spectra data.
            ppm : list
                List of PPM values corresponding to the spectra.
            """
            def __init__(self, spectra, ppm):

                """
                Initializes the plot_NMR_spec with the spectra and PPM values.

                Parameters:
                -----------
                spectra : pd.DataFrame
                    DataFrame containing the NMR spectra data.
                ppm : list
                    List of PPM values corresponding to the spectra.
                """
                self.spectra = spectra
                self.ppm = ppm

            def single_spectra(self, color_map=None, title='<b>Spectra of <sup>1</sup>H NMR data</b>',
                                title_font_size=28, legend_name='<b>Sample</b>', legend_font_size=16,
                                axis_font_size=20, line_width=1.5):

                """
                Generates a plot of the NMR spectra.

                Parameters:
                -----------
                color_map : str, optional
                    Color mapping for the plot (default is None).
                title : str, optional
                    Title of the plot (default is '<b>Spectra of <sup>1</sup>H NMR data</b>').
                title_font_size : int, optional
                    Font size for the title (default is 28).
                legend_name : str, optional
                    Title for the legend (default is '<b>Sample</b>').
                legend_font_size : int, optional
                    Font size for the legend (default is 16).
                axis_font_size : int, optional
                    Font size for the axis labels (default is 20).
                line_width : float, optional
                    Width of the plot lines (default is 1.5).

                Returns:
                --------
                fig : go.Figure
                    A Plotly figure object containing the plotted NMR spectra.
                """
                from plotly import express as px

                df_spectra = pd.DataFrame(self.spectra)
                df_spectra.columns = self.ppm

                fig = go.Figure()
                for i in df_spectra.index:
                    fig.add_trace(go.Scatter(x=self.ppm, y=df_spectra.loc[i, :], mode='lines', name=i,
                                            line=dict(width=line_width)))
                fig.update_layout(
                    title={'text': title, 'xanchor': 'center', 'yanchor': 'top'}, title_x=0.5,
                    xaxis_title="<b>δ<sup>1</sup>H</b>", yaxis_title="<b>Intensity</b>",
                    title_font_size=title_font_size, legend=dict(title=legend_name, font=dict(size=legend_font_size)),
                    xaxis_autorange="reversed", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickformat=".2e")
                )
                return fig

        plotter = plot_NMR_spec(self.spectra, self.ppm)
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([dcc.Graph(id='nmr-plot', figure=plotter.single_spectra())], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='peak-data', children='Click on the plot to select peaks.'),
                    dcc.Store(id='stored-peaks', data=[]),
                    html.Label("P-value Threshold:", className="mr-2"),  # Add this line for the label
                    dbc.Input(id='pvalue-threshold', type='number', value=0.001, step=0.000000000000000000000000000000000000000000000001, placeholder="P-value threshold"),
                    dbc.Button("Clear Data", id="clear-button", color="danger", className="mt-2 ml-2"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id='stocsy-plot')], width=12)
            ]),
        ], fluid=True)

        @app.callback(
            [Output('stored-peaks', 'data'), Output('peak-data', 'children')],
            [Input('nmr-plot', 'clickData'), Input('clear-button', 'n_clicks')],
            [State('stored-peaks', 'data')]
        )
        def update_peaks(clickData, clear_clicks, stored_peaks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return stored_peaks, 'Click on the plot to select peaks.'
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'clear-button':
                return [], 'Click on the plot to select peaks.'
            if clickData:
                x_peak = clickData['points'][0]['x']
                stored_peaks = [{'x': x_peak}]
                peak_text = f'Selected peak (X): {x_peak}'
                return stored_peaks, peak_text
            return stored_peaks, 'Click on the plot to select peaks.'

        @app.callback(
            Output("stocsy-plot", "figure"),
            [Input("stored-peaks", "data"), Input("pvalue-threshold", "value")]
        )
        def update_stocsy_plot(stored_peaks, pvalue_threshold):
            # If no peak is selected, do not update
            if not stored_peaks:
                print("No peak selected.")
                return dash.no_update
            
            # Extract the x position of the selected peak
            try:
                x_peak = float(stored_peaks[0]['x'])
            except (KeyError, TypeError, ValueError) as e:
                print("Error extracting peak value:", e)
                return dash.no_update
            
            # Ensure p-value threshold is not None; if None, use a default value (e.g., 0.05)
            if pvalue_threshold is None:
                pvalue_threshold = 0.05  # Default p-value if user input is cleared
                print("p-value threshold was None, setting to default:", pvalue_threshold)
            else:
                try:
                    pvalue_threshold = float(pvalue_threshold)
                except ValueError as e:
                    print("Error converting p-value threshold to float:", e)
                    return dash.no_update

            # Print the values for debugging
            print(f"Running STOCSY with x_peak={x_peak}, p-value threshold={pvalue_threshold}")

            # Run the STOCSY function and catch any issues
            try:
                fig = STOCSY(spectra=spectra, anchor_ppm_value=x_peak, p_value_threshold=pvalue_threshold)
                if not isinstance(fig, go.Figure):
                    raise ValueError("STOCSY did not return a Plotly figure.")
            except Exception as e:
                print("Error in STOCSY:", e)
                return dash.no_update
            
            return fig
        '''
        @app.callback(
            Output("download-dataframe-csv", "data"),
            [Input("export-button", "n_clicks")],
            [State("stored-peaks", "data")],
            prevent_initial_call=True
        )
        '''
        def export_x_positions(n_clicks, stored_peaks):
            if not stored_peaks:
                return dash.no_update
            x_positions = [p['x'] for p in stored_peaks]
            df = pd.DataFrame(x_positions, columns=["X Positions"])
            csv_string = df.to_csv(index=False)
            return dict(content=csv_string, filename="x_positions.csv")

        # Inside STOCSY_app class, replace `app.run_server(debug=True)` with:
        return app  # At the end of the `run_ui` method



if __name__ == '__main__':

    # Step 1: Load your NMR spectra data
    df = pd.read_csv("https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv")
    spectra = df.iloc[:,1:]  # Assuming first column is metadata and not part of the spectra
    ppm = spectra.columns.astype(float).to_list()  # Convert column names to floats (ppm values)
    # Outside the class, in the main block
    # Create instance of the class with spectra and ppm data


    stocsy_app = STOCSY_app(spectra, ppm)
    
    # Get the app instance
    app = stocsy_app.run_ui()
    
    # Run the app
    app.run_server(debug=True, port=8051)