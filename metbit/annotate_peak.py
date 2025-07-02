import dash
from dash import dcc, html, Input, Output, State
import plotly.io as pio
import pandas as pd
import numpy as np
import io
import json
import base64

class annotate_peak:

    '''
    A Dash application for annotating NMR spectra with interactive features.
    This application allows users to visualize NMR spectra, add annotations,
    style lines, and export annotations in HTML and JSON formats.
    Parameters:
    - spectra: DataFrame containing NMR spectra data.
    - ppm: List of ppm values corresponding to the spectra.
    - label: Series or DataFrame containing labels for the spectra.
    Usage:
    ```python
    import pandas as pd
    df = pd.read_csv('path_to_your_data.csv')
    spectra = df.iloc[:, 1:]  # Assuming first column is not part of spectra
    ppm = spectra.columns.astype(float).to_list()
    label = df['Group']  # Assuming 'Group' is the label column
    annotator = annotate_peak(spectra, ppm, label)
    annotator.run(debug=True, port=8050)
    ```

    '''
    def __init__(self, spectra, ppm, label):
        self.spectra = spectra
        self.ppm = ppm
        self.label = label
        self.app = dash.Dash(__name__)
        self.app.title = "🧪 NMR Annotator"
        self._configure_layout()
        self._configure_callbacks()

    def _configure_layout(self):
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
        <head>
            {%metas%}
            <title>🧪 NMR Annotator</title>
            {%favicon%}
            {%css%}
            <style>
                body { font-family: 'Segoe UI', sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px; }
                h2, h4 { color: #2c3e50; }
                .card { background-color: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
                .control-group { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
                input[type="text"], input[type="number"], .Select-control { min-width: 180px; }
                button, .Select { cursor: pointer; }
                button { background-color: #3498db; color: white; border: none; padding: 8px 14px; border-radius: 6px; transition: background-color 0.2s; }
                button:hover { background-color: #2980b9; }
                .label-list { list-style: none; padding-left: 0; }
                .label-list li { display: flex; justify-content: space-between; background-color: #ecf0f1; margin-bottom: 5px; padding: 6px 10px; border-radius: 6px; }
                .label-list button { background-color: #e74c3c; padding: 4px 8px; }
                .label-list button:hover { background-color: #c0392b; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
        </html>
        '''
        self.app.layout = html.Div([
            html.H2("🧪 NMR Annotator"),
            html.Div([
                html.Div([
                    html.Label("Select Display Mode:"),
                    dcc.RadioItems(
                        id='mode-selector',
                        options=[
                            {'label': 'Single Spectra', 'value': 'single'},
                            {'label': 'Median Spectra', 'value': 'median'}
                        ],
                        value='single',
                        inline=True
                    ),
                ], className="card"),

                html.Div([
                    dcc.Graph(id='nmr-plot', config={'editable': True})
                ], className="card"),

                html.Div([
                    html.H4("📍 Add Annotation"),
                    html.Div(className="control-group", children=[
                        dcc.Input(id='label-input', type='text', placeholder='Enter peak label'),
                        dcc.Input(id='label-angle', type='number', placeholder='Angle (°)', min=-180, max=180, step=5),
                        html.Button("Add Label", id='add-btn'),
                    ]),
                    html.Div(className="control-group", children=[
                        html.Button("Download HTML", id='download-html-btn'),
                        html.Button("Download JSON", id='download-json-btn'),
                        dcc.Upload(id='upload-json', children=html.Button("Upload JSON")),
                        dcc.Download(id='html-download'),
                        dcc.Download(id='json-download'),
                    ])
                ], className="card"),

                html.Div([
                    html.H4("🎨 Style Line"),
                    html.Div(className="control-group", children=[
                        dcc.Dropdown(id='line-selector', placeholder='Select line to style'),
                        dcc.Input(id='line-color', type='text', placeholder='e.g. red or #00FF00'),
                        dcc.Dropdown(
                            id='line-style',
                            options=[
                                {'label': 'Solid', 'value': 'solid'},
                                {'label': 'Dash', 'value': 'dash'},
                                {'label': 'Dot', 'value': 'dot'},
                                {'label': 'DashDot', 'value': 'dashdot'}
                            ],
                            placeholder='Select line style'
                        ),
                        dcc.Input(id='line-width', type='number', placeholder='Line width (e.g. 2)', min=0.1, step=0.1),
                        html.Button("Apply Style", id='apply-style'),
                    ])
                ], className="card"),

                html.Div([
                    html.H4("📌 Current Labels"),
                    html.Ul(id='label-list', className="label-list"),
                ], className="card")
            ]),

            dcc.Store(id='annotations-store', data=[]),
            dcc.Store(id='style-store', data={}),
        ])

    def _generate_figure(self, mode, annotations, style_data):
        from lingress import plot_NMR_spec
        plotter = plot_NMR_spec(self.spectra, self.ppm, self.label)
        if mode == 'median':
            groups = self.label.unique()
            color_map = {g: style_data.get(str(g), {}).get('color', 'blue') for g in groups}
            line_styles = {g: style_data.get(str(g), {}).get('dash', 'solid') for g in groups}
            line_widths = {g: style_data.get(str(g), {}).get('width', 2) for g in groups}
            fig = plotter.median_spectra_group(color_map=color_map, line_width=2)
        else:
            samples = self.spectra.index
            color_map = {i: style_data.get(str(i), {}).get('color', 'blue') for i in samples}
            line_styles = {i: style_data.get(str(i), {}).get('dash', 'solid') for i in samples}
            line_widths = {i: style_data.get(str(i), {}).get('width', 2) for i in samples}
            fig = plotter.single_spectra(color_map=color_map, line_width=2)

        for trace in fig.data:
            key = trace.name
            if key in line_styles:
                trace.line['dash'] = line_styles[key]
            if key in line_widths:
                trace.line['width'] = line_widths[key]

        fig.update_layout(annotations=annotations)
        return fig

    def _configure_callbacks(self):
        app = self.app

        @app.callback(
            Output('line-selector', 'options'),
            Input('mode-selector', 'value')
        )
        def update_line_options(mode):
            if mode == 'median':
                return [{'label': str(g), 'value': str(g)} for g in self.label.unique()]
            else:
                return [{'label': f"Sample {i}", 'value': str(i)} for i in self.spectra.index]

        @app.callback(
            Output('nmr-plot', 'figure'),
            Output('annotations-store', 'data'),
            Output('style-store', 'data'),
            Input('add-btn', 'n_clicks'),
            Input({'type': 'delete-btn', 'index': dash.ALL}, 'n_clicks'),
            Input('apply-style', 'n_clicks'),
            Input('upload-json', 'contents'),
            Input('mode-selector', 'value'),
            State('nmr-plot', 'clickData'),
            State('label-input', 'value'),
            State('label-angle', 'value'),
            State('annotations-store', 'data'),
            State('line-color', 'value'),
            State('line-style', 'value'),
            State('line-width', 'value'),
            State('style-store', 'data'),
            State('line-selector', 'value'),
            prevent_initial_call=True
        )
        def update_all(add_btn, delete_btns, apply_btn, upload_content, mode,
                       clickData, label_text, label_angle, annotations,
                       color, dash_style, line_width, style_data, selected_line):
            ctx = dash.callback_context
            trig = ctx.triggered[0]['prop_id'].split('.')[0]
            annotations = annotations or []
            style_data = style_data or {}

            if trig == 'add-btn' and clickData and label_text:
                point = clickData['points'][0]
                annotations.append({
                    'x': point['x'], 'y': point['y'],
                    'xref': 'x', 'yref': 'y',
                    'text': label_text,
                    'showarrow': True,
                    'ax': 0, 'ay': -40,
                    'textangle': label_angle if label_angle is not None else 0
                })
            elif 'delete-btn' in trig:
                index = int(json.loads(trig)['index'])
                if 0 <= index < len(annotations):
                    annotations.pop(index)
            elif trig == 'upload-json':
                try:
                    _, content_string = upload_content.split(',')
                    decoded = base64.b64decode(content_string).decode()
                    annotations = json.loads(decoded)
                except Exception:
                    pass
            elif trig == 'apply-style' and selected_line:
                if selected_line not in style_data:
                    style_data[selected_line] = {}
                if color:
                    style_data[selected_line]['color'] = color
                if dash_style:
                    style_data[selected_line]['dash'] = dash_style
                if line_width:
                    style_data[selected_line]['width'] = line_width

            fig = self._generate_figure(mode, annotations, style_data)
            return fig, annotations, style_data

        @app.callback(
            Output('label-list', 'children'),
            Input('annotations-store', 'data')
        )
        def render_labels(annotations):
            return [
                html.Li([
                    f"{ann['text']} @ x={ann['x']:.2f}",
                    html.Button("❌", id={'type': 'delete-btn', 'index': i}, n_clicks=0)
                ]) for i, ann in enumerate(annotations)
            ]

        @app.callback(
            Output('html-download', 'data'),
            Input('download-html-btn', 'n_clicks'),
            State('annotations-store', 'data'),
            State('style-store', 'data'),
            State('mode-selector', 'value'),
            prevent_initial_call=True
        )
        def export_html(_, annotations, style_data, mode):
            fig = self._generate_figure(mode, annotations, style_data)
            buffer = io.StringIO()
            pio.write_html(fig, buffer, include_plotlyjs='cdn')
            return dcc.send_string(buffer.getvalue(), filename="nmr_plot.html")

        @app.callback(
            Output('json-download', 'data'),
            Input('download-json-btn', 'n_clicks'),
            State('annotations-store', 'data'),
            prevent_initial_call=True
        )
        def export_json(_, annotations):
            return dcc.send_string(json.dumps(annotations, indent=2), filename="annotations.json")

    def run(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    import nmrglue as ng
    import pandas as pd

    df = pd.read_csv('https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv')
    spectra = df.iloc[:, 1:]
    ppm = spectra.columns.astype(float).to_list()
    label = df['Group']

    annotator = annotate_peak(spectra, ppm, label)
    annotator.run(debug=True, port=8051)