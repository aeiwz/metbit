import dash
from dash import dcc, html, Input, Output, State
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import json
import base64


class annotate_peak:
    """
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
    annotator = annotate_peak(label, spectra, ppm, label)
    annotator.run(debug=True, port=8050)
    ```
    """

    def __init__(self, meta, spectra, ppm=None, label=None, colour_dict: dict=None):
        # ---------- data
        self.spectra = spectra.copy()
        self.spectra.columns = self.spectra.columns.astype(float)
        self.ppm = self.spectra.columns.to_numpy(dtype=float)
        self.colour_dict = colour_dict

        # label/meta
        if label is None and isinstance(meta, pd.Series):
            self.label = meta.copy()
            self.meta = meta.to_frame(name=meta.name or "Label")
        else:
            self.label = label.copy() if isinstance(label, pd.Series) else pd.Series(index=self.spectra.index, dtype="object")
            self.meta = meta.copy() if isinstance(meta, (pd.Series, pd.DataFrame)) else pd.DataFrame(index=self.spectra.index)

        common = self.spectra.index.intersection(self.label.index).intersection(self.meta.index)
        self.spectra = self.spectra.loc[common]
        self.label = self.label.loc[common]
        self.meta = self.meta.loc[common] if isinstance(self.meta, pd.DataFrame) else self.meta.to_frame().loc[common]

        self.spectra.index = self.spectra.index.astype(str)
        self.label.index = self.spectra.index
        self.meta.index = self.spectra.index

        # ---------- dash
        self.app = dash.Dash(__name__)
        self.app.title = "NMR Annotator"
        self._configure_layout()
        self._configure_callbacks()

    # ===================== LAYOUT + CSS =====================
    def _configure_layout(self):
        self.app.index_string = """
        <!DOCTYPE html>
        <html>
        <head>
            {%metas%}
            <title>NMR Annotator</title>
            {%favicon%}
            {%css%}
        <style>
        :root{
            --bg:#f5f7fb; --ink:#1b2530; --muted:#6d7890; --ring:#e6ecff;
            --brand:#2563eb; --brand-600:#1d4ed8; --danger:#e11d48;
            --card:#ffffff; --radius:14px; --shadow:0 6px 28px rgba(20,22,35,.06);
        }
        *{box-sizing:border-box}
        html,body{height:100%}
        body{
            margin:0;background:var(--bg);color:var(--ink);
            font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;
            overflow:hidden;
        }
        .app-shell{display:flex;flex-direction:column;width:100vw;height:100vh}

        /* Toolbar */
        .toolbar{
            height:64px;min-height:64px;display:flex;align-items:center;gap:16px;
            padding:10px 16px;background:var(--card);
            border-bottom:1px solid var(--ring);box-shadow:var(--shadow)
        }
        .toolbar .spacer{flex:1}
        .segmented{display:inline-flex;border:1px solid var(--ring);border-radius:999px;overflow:hidden;background:#f7f9ff}
        .segmented .opt{padding:6px 12px;font-weight:600;color:var(--muted);cursor:pointer;user-select:none;border:none;background:transparent}
        .segmented .opt.active{background:var(--brand);color:#fff}
        .ppm-pill{font-weight:700;color:#a31313;background:#ffe8ea;border:1px solid #ffd3d7;border-radius:999px;padding:6px 10px}

        /* Plot area */
        .plot-wrap{height:calc(100vh - 64px);width:100vw;padding:10px 16px 16px;overflow:auto}
        .plot-card{background:var(--card);border-radius:12px;box-shadow:var(--shadow);padding:8px}
        #graph-wrap{height:78vh}
        @media (max-width:992px){#graph-wrap{height:70vh}}
        @media (max-width:576px){#graph-wrap{height:62vh}}
        #graph-wrap .js-plotly-plot,#graph-wrap .dash-graph{width:100%!important;height:100%!important;cursor:crosshair}

        /* Panels */
        details.panel{margin:12px 0;border:1px solid var(--ring);border-radius:12px;background:#fff}
        details.panel summary{list-style:none;cursor:pointer;padding:12px 16px;border-radius:12px;display:flex;align-items:center;gap:10px;font-weight:700;color:var(--ink)}
        details.panel[open] summary{border-bottom:1px solid var(--ring);border-radius:12px 12px 0 0}
        .panel .content{padding:14px 16px}
        .row{display:flex;flex-wrap:wrap;gap:12px;align-items:center}

        /* Buttons */
        .btn{background:var(--brand);color:#fff;border:none;border-radius:12px;padding:10px 16px;cursor:pointer;transition:background .2s;font-size:15px;font-weight:600}
        .btn:hover{background:var(--brand-600)}
        .btn-ghost{background:#eaf0ff;color:var(--brand);border:1px solid var(--ring)}
        .btn-pill{border-radius:999px}
        .btn-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px}
        .btn-eq{flex:1 1 180px;justify-content:center}
        .dash-upload>button.btn{width:100%}
        @media (max-width:640px){.btn-row{gap:8px}.btn-eq{flex:1 1 140px}}

        /* Inputs (text/number) */
        .input, input[type="text"], input[type="number"]{
            appearance:none;-webkit-appearance:none;height:44px;padding:0 14px;
            border-radius:12px;border:1px solid var(--ring);background:#fff;color:var(--ink);
            outline:none;transition:box-shadow .15s,border-color .15s,background .15s;
            font-size:15px;box-shadow:0 1px 1px rgba(20,22,35,.04)
        }
        .input:hover, input[type="text"]:hover, input[type="number"]:hover{background:#f9fbff}
        .input:focus, input[type="text"]:focus, input[type="number"]:focus{
            border-color:#c7d5ff; box-shadow:0 0 0 3px rgba(37,99,235,.12); background:#fff;
        }
        .input::placeholder, input::placeholder{color:#9aa5b5;font-size:14px}
        input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}

        /* React-Select (Dash dcc.Dropdown) — single clean shell (no double halo) */
        .Select{border:0!important;background:transparent!important;box-shadow:none!important}
        .Select-control{
            border:1px solid var(--ring)!important;border-radius:12px!important;background:#fff!important;
            min-height:44px!important;box-shadow:0 1px 2px rgba(20,22,35,.05)!important;
            transition:border-color .15s,box-shadow .15s,background .15s;font-size:15px!important
        }
        .Select-control:hover{border-color:#c7d5ff!important;background:#f9fbff!important}
        .is-focused .Select-control{border-color:#2563eb!important;box-shadow:0 0 0 3px rgba(37,99,235,.12)!important;background:#fff!important}
        .Select-placeholder,.Select--single>.Select-control .Select-value{
            color:#4b5563!important;font-size:15px!important;line-height:42px!important;padding-left:12px!important
        }
        .Select-menu-outer{
            border-radius:12px!important;border:1px solid #e6ecff!important;
            box-shadow:0 6px 16px rgba(20,22,35,.08)!important;margin-top:4px!important
        }
        .Select-option{padding:10px 14px!important;font-size:15px!important;color:#374151!important;cursor:pointer;transition:background .12s}
        .Select-option.is-focused{background:#f3f6ff!important;color:var(--brand)!important}
        .Select-option.is-selected{background:var(--brand)!important;color:#fff!important}
        .Select-arrow{border-top-color:#9aa5b5!important}
        .is-open .Select-arrow{border-top-color:var(--brand)!important}

        /* Wider dropdowns for line controls */
        #line-selector .Select-control,
        #line-style .Select-control{
            min-width:220px!important; width:240px!important;
        }

        /* Labels list */
        .labels-list{list-style:none;padding:0;margin:0}
        .labels-list li{display:flex;gap:8px;align-items:center;background:#f4f6fb;border:1px solid #edf1ff;padding:6px 10px;border-radius:10px;margin:6px 0}
        .del{background:var(--danger);color:#fff;border:none;border-radius:8px;padding:4px 8px;cursor:pointer}
        .meta-note{color:var(--muted);font-size:13px}
        </style>
        </head>
        <body>
            {%app_entry%}
            <footer>{%config%}{%scripts%}{%renderer%}</footer>
        </body>
        </html>
        """

        self.app.layout = html.Div(className="app-shell", children=[
            # Toolbar
            html.Div(className="toolbar", children=[
                html.Div(className="segmented", children=[
                    html.Button("Single Spectra", id="opt-single",  className="opt active"),
                    html.Button("Median Spectra", id="opt-median", className="opt"),
                ]),
                # invisible control that callbacks use
                dcc.RadioItems(
                    id='mode-selector',
                    options=[{'label':'Single Spectra','value':'single'},
                            {'label':'Median Spectra','value':'median'}],
                    value='single',
                    style={'display':'none'}
                ),

                html.Span("Font Size"),
                html.Div(
                    dcc.Slider(
                        id="font-size", min=6, max=48, step=1, value=16,
                        tooltip={"placement":"bottom", "always_visible":False},
                        marks=None, updatemode="drag",
                    ),
                    style={"width":"220px"}  # wrapper carries width (v3.1.1 sliders don't accept style)
                ),
                html.Span(id="font-size-val", style={"fontWeight":"700"}),
                html.Div(className="spacer"),
                html.Span(id='current-ppm', className="ppm-pill", children="Current peak: –")
            ]),

            # Plot + panels
            html.Div(className="plot-wrap", children=[
                html.Div(className="plot-card", children=[
                    html.Div(id="graph-wrap", children=[
                        dcc.Graph(
                            id='nmr-plot',
                            config={'editable': True, 'responsive': True, 'displaylogo': False},
                            style={'height':'100%','width':'100%'}
                        )
                    ])
                ]),

                html.Details(className="panel", open=True, children=[
                    html.Summary("📍 Add Annotation"),
                    html.Div(className="content", children=[
                        html.Div(className="row", children=[
                            dcc.Input(id='label-input', type='text', placeholder='Enter peak label'),
                            dcc.Input(id='label-angle', type='number', placeholder='Angle (°)', min=-180, max=180, step=5),
                            html.Button("Add Label", id='add-btn', className="btn"),
                        ]),
                        html.Div(style={"height":"10px"}),
                        html.Div(className="row", children=[
                            html.Button("Download HTML", id='download-html-btn', className="btn btn-ghost"),
                            html.Button("Download JSON", id='download-json-btn', className="btn btn-ghost"),
                            dcc.Upload(id='upload-json', children=html.Button("Upload JSON", className="btn btn-ghost")),
                            html.Button("Intensity Table (CSV)", id='download-intensity-btn', className="btn btn-ghost"),
                            dcc.Download(id='html-download'),
                            dcc.Download(id='json-download'),
                            dcc.Download(id='intensity-download'),
                        ]),
                        html.Div(className="meta-note", children="Click a peak (exact x), type a label, then ‘Add Label’.")
                    ])
                ]),

                html.Details(className="panel", open=False, children=[
                    html.Summary("🎨 Style Line"),
                    html.Div(className="content", children=[
                        html.Div(className="row", children=[
                            dcc.Dropdown(id='line-selector', placeholder='Select line to style', style={"minWidth":"200px"}),
                            dcc.Input(id='line-color', type='text', placeholder='e.g. red or #00FF00'),
                            dcc.Dropdown(
                                id='line-style',
                                options=[{'label':'Solid','value':'solid'},
                                         {'label':'Dash','value':'dash'},
                                         {'label':'Dot','value':'dot'},
                                         {'label':'DashDot','value':'dashdot'}],
                                placeholder='Line style'
                            ),
                            dcc.Input(id='line-width', type='number', placeholder='Width (e.g. 2)', min=0.1, step=0.1),
                            html.Button("Apply Style", id='apply-style', className="btn"),
                        ]),
                        html.Div(className="meta-note", children="Custom width overrides auto thickness.")
                    ])
                ]),

                html.Details(className="panel", open=False, children=[
                    html.Summary("📌 Current Labels"),
                    html.Div(className="content", children=[ html.Ul(id='label-list', className="labels-list") ])
                ]),

                # Stores
                dcc.Store(id='annotations-store', data=[]),
                dcc.Store(id='style-store', data={}),
            ])
        ])

    # ===================== PLOTTING =====================
    def _figure(self, mode, annotations, style_data, font_size):
        fig = go.Figure()

        style_data = style_data or {}
        colour_map = dict(getattr(self, "colour_dict", {}) or {})  # optional group/sample -> color

        # priority: UI override (style_data) > colour_dict > Plotly default
        def pick_color(trace_key: str, group_key: str | None = None):
            st = style_data.get(trace_key, {})
            if st.get("color"):
                return st["color"]
            if trace_key in colour_map:
                return colour_map[trace_key]
            if group_key and group_key in colour_map:
                return colour_map[group_key]
            return None  # let Plotly choose

        if mode == "median":
            grp = self.label.astype(str)
            for g, idx in grp.groupby(grp).groups.items():
                Y = self.spectra.loc[idx].to_numpy(dtype=float)
                y_med = np.median(Y, axis=0)
                key = str(g)
                st = style_data.get(key, {})
                color = pick_color(trace_key=key, group_key=key)
                fig.add_trace(go.Scatter(
                    x=self.ppm, y=y_med, name=key, mode="lines",
                    line=dict(
                        dash=st.get("dash", "solid"),
                        width=st.get("width", 1.6),
                        color=color
                    )
                ))
        else:
            # single spectra: allow per-sample override; fall back to its group's color
            for s, row in self.spectra.iterrows():
                key = str(s)
                try:
                    group_key = str(self.label.loc[s])
                except Exception:
                    group_key = None
                st = style_data.get(key, {})
                color = pick_color(trace_key=key, group_key=group_key)
                fig.add_trace(go.Scatter(
                    x=self.ppm, y=row.to_numpy(dtype=float), name=key, mode="lines",
                    line=dict(
                        dash=st.get("dash", "solid"),
                        width=st.get("width", 1.2),
                        color=color
                    )
                ))

        # apply uniform annotation font
        anns = []
        for a in annotations or []:
            a2 = a.copy()
            a2.setdefault('font', {})
            a2['font']['size'] = font_size
            anns.append(a2)

        fig.update_xaxes(
            autorange="reversed",
            title=dict(
                text="<b>δ <sup>1</sup>H</b>",
                font=dict(size=int(font_size*1.5))
            ),
            tickfont=dict(size=max(font_size-2, 8)),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False
        )

        fig.update_yaxes(
            title=dict(
                text="<b>Intensity</b>",
                font=dict(size=int(font_size*1.5))
            ),
            tickfont=dict(size=max(font_size-2, 8)),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False
        )

        fig.update_layout(
            uirevision="constant",
            template="plotly_white",
            annotations=anns,
            autosize=True,
            margin=dict(l=50, r=12, t=70, b=40),
            title=dict(
                text=("<b>Median Spectra of <sup>1</sup>H NMR data</b>"
                    if mode=="median" else "<b>Spectra of <sup>1</sup>H NMR data</b>"),
                x=0.5, y=0.97, font=dict(size=font_size*1.5)
            ),
            legend=dict(
                title=dict(text="Groups", font=dict(size=font_size)),
                orientation="v",
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                font=dict(size=max(font_size-2, 9)),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(200,200,200,0.5)",
                borderwidth=1
            ),
            hovermode="closest",
            clickmode="event+select",

            # === Axis lines ===
            xaxis=dict(
                showline=True,          # enable axis line
                linecolor="black",      # axis line color
                linewidth=2,            # axis line width
                mirror=False,           # show line only on bottom
                title=dict(
                    text="<b>δ <sup>1</sup>H</b>",
                    font=dict(size=int(font_size*1.5), color="black")
                ),
                tickfont=dict(size=max(font_size-2, 9), color="black")
            ),
            yaxis=dict(
                showline=True,
                linecolor="black",
                linewidth=2,
                mirror=False,           # show line only on left
                title=dict(
                    text="<b>Intensity</b>",
                    font=dict(size=int(font_size*1.5), color="black")
                ),
                tickfont=dict(size=max(font_size-2, 9), color="black"),
                exponentformat="power",   # use 10^n style
                showexponent="all"
            ),
        )
        return fig

    # ===================== CALLBACKS =====================
    def _configure_callbacks(self):
        app = self.app

        # Show slider value
        @app.callback(Output('font-size-val', 'children'), Input('font-size', 'value'))
        def _show_font_size(v): return f"{int(v)} px"

        # Segmented buttons -> radio + button classes (server-side; no client JS)
        @app.callback(
            Output('mode-selector', 'value'),
            Output('opt-single', 'className'),
            Output('opt-median', 'className'),
            Input('opt-single', 'n_clicks'),
            Input('opt-median', 'n_clicks'),
            State('mode-selector', 'value')
        )
        def _choose_mode(n_single, n_median, current):
            ctx = dash.callback_context
            if not ctx.triggered:
                val = current or 'single'
            else:
                which = ctx.triggered[0]['prop_id'].split('.')[0]
                val = 'single' if which == 'opt-single' else 'median'
            return val, ('opt active' if val == 'single' else 'opt'), ('opt active' if val == 'median' else 'opt')

        # Options for the style dropdown
        @app.callback(Output('line-selector', 'options'), Input('mode-selector', 'value'))
        def update_line_options(mode):
            opts = (sorted(self.label.astype(str).unique())
                    if mode == 'median'
                    else list(self.spectra.index.astype(str)))
            return [{'label': o, 'value': o} for o in opts]

        # Main update (also draws on first load)
        @app.callback(
            Output('nmr-plot', 'figure'),
            Output('annotations-store', 'data'),
            Output('style-store', 'data'),
            Input('add-btn', 'n_clicks'),
            Input({'type': 'delete-btn', 'index': dash.ALL}, 'n_clicks'),
            Input('apply-style', 'n_clicks'),
            Input('upload-json', 'contents'),
            Input('mode-selector', 'value'),
            Input('font-size', 'value'),
            State('nmr-plot', 'clickData'),   # click only
            State('label-input', 'value'),
            State('label-angle', 'value'),
            State('annotations-store', 'data'),
            State('line-color', 'value'),
            State('line-style', 'value'),
            State('line-width', 'value'),
            State('style-store', 'data'),
            State('line-selector', 'value')
        )
        def update_all(add_btn, delete_btns, apply_btn, upload_content, mode, font_size,
                       clickData, label_text, label_angle, annotations,
                       color, dash_style, line_width, style_data, selected_line):
            ctx = dash.callback_context
            trig = (ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else "")
            annotations = annotations or []
            style_data = style_data or {}

            if trig == 'add-btn' and clickData and label_text:
                pt = clickData['points'][0]
                annotations.append({
                    'x': float(pt['x']), 'y': float(pt['y']),
                    'xref': 'x', 'yref': 'y',
                    'text': str(label_text),
                    'showarrow': True, 'ax': 0, 'ay': -40,
                    'textangle': (label_angle if label_angle is not None else 0),
                    'font': {'size': font_size}
                })
            elif 'delete-btn' in trig:
                try:
                    idx = int(json.loads(trig)['index'])
                    if 0 <= idx < len(annotations):
                        annotations.pop(idx)
                except Exception:
                    pass
            elif trig == 'upload-json' and upload_content:
                try:
                    _, payload = upload_content.split(',', 1)
                    loaded = json.loads(base64.b64decode(payload).decode('utf-8'))
                    if isinstance(loaded, list):
                        anns = []
                        for a in loaded:
                            if isinstance(a, dict) and 'x' in a and 'y' in a and 'text' in a:
                                a.setdefault('xref', 'x'); a.setdefault('yref', 'y')
                                a.setdefault('showarrow', True); a.setdefault('ax', 0); a.setdefault('ay', -40)
                                a.setdefault('textangle', 0); a.setdefault('font', {'size': font_size})
                                anns.append(a)
                        if anns: annotations = anns
                except Exception:
                    pass
            elif trig == 'apply-style' and selected_line:
                key = str(selected_line)
                style_data.setdefault(key, {})
                if color:      style_data[key]['color'] = color
                if dash_style: style_data[key]['dash']  = dash_style
                if line_width: style_data[key]['width'] = float(line_width)

            fig = self._figure(mode or 'single', annotations, style_data, font_size or 14)
            return fig, annotations, style_data

        @app.callback(Output('label-list', 'children'), Input('annotations-store', 'data'))
        def render_labels(annotations):
            labels = []
            for i, ann in enumerate(annotations or []):
                x_txt = f"{float(ann.get('x', np.nan)):.4f}" if 'x' in ann else "?"
                labels.append(html.Li([
                    html.Button("❌", id={'type': 'delete-btn', 'index': i}, n_clicks=0, className="del"),
                    html.Span(f"{ann.get('text','?')} @ ppm {x_txt}")
                ]))
            return labels

        # Downloads
        @app.callback(
            Output('html-download', 'data'),
            Input('download-html-btn', 'n_clicks'),
            State('annotations-store', 'data'),
            State('style-store', 'data'),
            State('mode-selector', 'value'),
            State('font-size', 'value'),
            prevent_initial_call=True
        )
        def export_html(_, annotations, style_data, mode, font_size):
            fig = self._figure(mode or 'single', annotations or [], style_data or {}, font_size or 14)
            buf = io.StringIO()
            pio.write_html(fig, buf, include_plotlyjs='cdn')
            return dcc.send_string(buf.getvalue(), filename="nmr_plot.html")

        @app.callback(Output('json-download', 'data'),
                      Input('download-json-btn', 'n_clicks'),
                      State('annotations-store', 'data'),
                      prevent_initial_call=True)
        def export_json(_, annotations):
            return dcc.send_string(json.dumps(annotations or [], indent=2), filename="annotations.json")

        @app.callback(Output('intensity-download', 'data'),
                      Input('download-intensity-btn', 'n_clicks'),
                      State('annotations-store', 'data'),
                      prevent_initial_call=True)
        def export_intensity_table(_, annotations):
            if not annotations:
                return dash.no_update
            ppm_axis = self.ppm
            cols = self.spectra.columns.values
            out_cols = {}
            for ann in annotations:
                try:
                    x_ppm = float(ann['x']); lbl = str(ann['text'])
                except Exception:
                    continue
                j = int(np.argmin(np.abs(ppm_axis - x_ppm)))
                col_ppm = float(cols[j])
                out_cols[f"{lbl}_{col_ppm:.3f}"] = self.spectra.iloc[:, j].to_numpy()
            intensity_df = pd.DataFrame(out_cols, index=self.spectra.index)

            meta_df = self.meta.copy()
            if not isinstance(meta_df, pd.DataFrame):
                meta_df = meta_df.to_frame(name=str(meta_df.name or "Meta"))
            meta_df = meta_df.loc[self.spectra.index]

            out = pd.concat([pd.Series(self.spectra.index, name="Sample", index=self.spectra.index),
                             meta_df.reset_index(drop=True),
                             intensity_df.reset_index(drop=True)], axis=1)
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            return dcc.send_string(buf.getvalue(), filename="annotated_intensities.csv")

        # Click-only ppm readout
        @app.callback(Output('current-ppm', 'children'), Input('nmr-plot', 'clickData'))
        def show_ppm(clickData):
            if clickData and "points" in clickData and clickData["points"]:
                try:
                    x_val = float(clickData["points"][0].get("x", None))
                    return f"Current peak: {x_val:.3f} ppm"
                except Exception:
                    return "Current peak: –"
            return "Current peak: –"

    def run(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)


# ===================== DEMO ENTRYPOINT =====================
if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/aeiwz/example_data/main/dataset/Example_NMR_data.csv')
    spectra = df.iloc[:, 1:]
    label = df['Group']
    annotator = annotate_peak(meta=label, spectra=spectra, label=label)
    annotator.run(debug=True, port=8052)