# -*- coding: utf-8 -*-


__auther__ ='aeiwz'
author_email='theerayut_aeiw_123@hotmail.com'

import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html
from functools import lru_cache

# Importing the custom STOCSY function
from .STOCSY import STOCSY  # Adjust as necessary based on your file

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
        Sets up the Dash UI layout, initializes the application callbacks, and returns the app instance.

    
    Example:
    -------
    # Load your NMR spectra data
    df = pd.read_csv("https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv")
    spectra = df.iloc[:,1:]
    ppm = spectra.columns.astype(float).to_list()

    # Create instance of the class with spectra and ppm data
    stocsy_app = STOCSY_app(spectra, ppm)

    # Get the app instance
    app = stocsy_app.run_ui()

    # Run the app
    app.run_server(debug=True, port=8051)
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
        self.stocsy_cache = {}

    @lru_cache(maxsize=32)
    def cached_stocsy(self, x_peak, pvalue_threshold):
        """
        Cache the STOCSY analysis results to avoid redundant computation.
        """
        return STOCSY(spectra=self.spectra, anchor_ppm_value=x_peak, p_value_threshold=pvalue_threshold)

    def run_ui(self):
        class plot_NMR_spec:
            def __init__(self, spectra, ppm):
                self.spectra = spectra
                self.ppm = ppm

            def single_spectra(self, color_map=None, title='<b>Spectra of <sup>1</sup>H NMR data</b>',
                               title_font_size=28, legend_name='<b>Sample</b>', legend_font_size=16,
                               axis_font_size=20, line_width=1.5):
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

        # Define the layout for the Dash app
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([dcc.Graph(id='nmr-plot', figure=plotter.single_spectra())], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='peak-data', children='Click on the plot to select peaks.'),
                    dcc.Store(id='stored-peaks', data=[]),
                    html.Label("P-value Threshold:", className="mr-2"),
                    dbc.Input(id='pvalue-threshold', type='number', value=0.001, placeholder="P-value threshold"),
                    dbc.Button("Run STOCSY", id="run-stocsy-button", color="primary", className="mt-2"),
                    dbc.Button("Clear Data", id="clear-button", color="danger", className="mt-2 ml-2"),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-stocsy",
                        type="circle",
                        children=[dcc.Graph(id='stocsy-plot')]
                    )
                ], width=12)
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
            [Input("run-stocsy-button", "n_clicks")],
            [State("stored-peaks", "data"), State("pvalue-threshold", "value")]
        )
        def update_stocsy_plot(n_clicks, stored_peaks, pvalue_threshold):
            if not stored_peaks or n_clicks is None:
                return dash.no_update

            try:
                x_peak = float(stored_peaks[0]['x'])
            except (KeyError, TypeError, ValueError):
                return dash.no_update

            if pvalue_threshold is None:
                pvalue_threshold = 0.05
            else:
                try:
                    pvalue_threshold = float(pvalue_threshold)
                except ValueError:
                    return dash.no_update

            cache_key = (x_peak, pvalue_threshold)
            if cache_key in self.stocsy_cache:
                print(f"Using cached STOCSY result for δ={x_peak}, p-value threshold={pvalue_threshold}")
                return self.stocsy_cache[cache_key]

            print(f"Running STOCSY with δ={x_peak}, p-value threshold={pvalue_threshold}")
            try:
                fig = self.cached_stocsy(x_peak, pvalue_threshold)
                if not isinstance(fig, go.Figure):
                    raise ValueError("STOCSY did not return a Plotly figure.")
                self.stocsy_cache[cache_key] = fig
                return fig
            except Exception as e:
                print("Error in STOCSY:", e)
                return dash.no_update

        return app


if __name__ == '__main__':
    # Load your NMR spectra data
    df = pd.read_csv("https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv")
    spectra = df.iloc[:,1:]
    ppm = spectra.columns.astype(float).to_list()

    # Create instance of the class with spectra and ppm data
    stocsy_app = STOCSY_app(spectra, ppm)

    # Get the app instance
    app = stocsy_app.run_ui()

    # Run the app
    app.run_server(debug=True, port=8051)