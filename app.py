from dash import Dash, dcc, html, Input, Output, State, callback_context, ALL
import dash
import pandas as pd
import plotly.graph_objs as go
import base64
import io
import re
import numpy as np
import os
import zipfile
import tempfile
import math
from dash import ctx
import plotly.io as pio

app = Dash(__name__)

def format_subscript(text):
    return re.sub(r'(\d+(\.\d+)?)', r'<sub>\1</sub>', text)

app.layout = html.Div([
    html.H1("XRD Refinement Plotter", style={
        "fontSize": "32px", 
        "fontWeight": "bold", 
        "fontFamily": "DejaVu Sans, Arial, sans-serif", 
        "textAlign": "center", 
        "marginBottom": "20px", 
        "color": "black"
    }),
    html.Div([
        # Graph container (card style)
        html.Div([
            dcc.Graph(id='graph', style={
                "backgroundColor": "white",
                "marginTop": "30px",
                "width": "1000px",
                "height": "700px"
            }),
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Upload Files', style={
                        "backgroundColor": "#007BFF", 
                        "color": "white", 
                        "padding": "12px 20px", 
                        "border": "none", 
                        "borderRadius": "5px", 
                        "cursor": "pointer",
                        "fontSize": "16px", 
                        "fontWeight": "bold", 
                        "fontFamily": "DejaVu Sans, Arial, sans-serif",
                        "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
                        "marginRight": "10px"
                    }),
                    multiple=True
                ),
                html.Button("Demo File", id="demo-file", n_clicks=0, style={
                    "backgroundColor": "#28a745",
                    "color": "white",
                    "padding": "10px 20px",
                    "border": "none",
                    "borderRadius": "5px",
                    "cursor": "pointer",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif",
                    "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
                    "marginRight": "10px"
                }),
                html.Button("Save Plot (png)", id="save-plot", n_clicks=0, style={
                    "backgroundColor": "#007BFF", 
                    "color": "white", 
                    "padding": "10px 20px", 
                    "border": "none", 
                    "borderRadius": "5px", 
                    "cursor": "pointer",
                    "fontSize": "16px", 
                    "fontWeight": "bold", 
                    "fontFamily": "DejaVu Sans, Arial, sans-serif",
                    "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
                }),
                dcc.Download(id="download-plot"),
            ], style={"display": "flex", "justifyContent": "flex-start", "marginTop": "20px"}),
            html.Div(id='output-data-upload', style={
                "fontFamily": "DejaVu Sans, Arial, sans-serif",
                "fontSize": "16px",
                "color": "black",
                "marginTop": "10px",
                "textAlign": "center"
            }),
            html.Div(id="save-confirmation", style={
    "marginTop": "10px", 
    "color": "#4CAF50", 
    "fontFamily": "DejaVu Sans, Arial, sans-serif"
}),
        ], style={
            "flex": "none",
            "width": "1000px",
            "backgroundColor": "#F9F9F9",
            "borderRadius": "10px",
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
            "padding": "30px",
            "marginRight": "30px"
        }),
        # Graph adjustments on the right
        html.Div([
            html.H3("Graph adjustments", style={
                "fontFamily": "DejaVu Sans, Arial, sans-serif", 
                "color": "black", 
                "marginBottom": "10px",
                "textDecoration": "underline",
                "fontSize": "20px"
            }),
            html.Div([
                html.Label("X-axis limits:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px", "color": "black"}),
                dcc.Input(id='xmin', type='number', style={
                    "marginRight": "10px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px", "color": "black"
                }),
                dcc.Input(id='xmax', type='number', style={
                    "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px", "color": "black"
                }),
            ], style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Y-axis limits:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px", "color": "black"}),
                dcc.Input(id='ymin', type='number', style={
                    "marginRight": "10px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px", "color": "black"
                }),
                dcc.Input(id='ymax', type='number', style={
                    "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px", "color": "black"
                }),
            ], style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Legend Y-position:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px", "color": "black"}),
                dcc.Input(id='legend-y', type='number', min=0, max=1, step=0.01, style={
                    "width": "80px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "color": "black"
                }),
            ], style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Difference shift:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px", "color": "black"}),
                dcc.Input(id='diff-shift', type='number', value=0, step=0.1, style={
                    "width": "80px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "color": "black"
                }),
            ], style={"marginBottom": "15px"}),
            html.Div(id='hkl-shift-inputs', style={"marginBottom": "15px"}),
            html.Div(id='legend-controls-container', style={"marginBottom": "20px"}),
            html.Div([
                html.Button("Update Plot", id="update-plot", n_clicks=0, style={
                    "backgroundColor": "#007BFF", 
                    "color": "white", 
                    "padding": "10px 20px", 
                    "border": "none", 
                    "borderRadius": "5px", 
                    "cursor": "pointer",
                    "fontSize": "16px", 
                    "fontWeight": "bold", 
                    "fontFamily": "DejaVu Sans, Arial, sans-serif",
                    "marginRight": "10px",
                    "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
                }),
                html.Button("Reset Axes", id="reset-axes", n_clicks=0, style={
                    "backgroundColor": "#FF5733", 
                    "color": "white", 
                    "padding": "10px 20px", 
                    "border": "none",
                    "borderRadius": "5px", 
                    "cursor": "pointer",
                    "fontSize": "16px", 
                    "fontWeight": "bold", 
                    "fontFamily": "DejaVu Sans, Arial, sans-serif",
                    "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
                }),
            ], style={"display": "flex", "flexDirection": "row", "marginTop": "10px"}),
        ], style={
            "flex": "1",
            "backgroundColor": "#F9F9F9",
            "borderRadius": "10px",
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
            "padding": "30px"
        }),
    ], style={"display": "flex", "justifyContent": "center", "alignItems": "flex-start", "width": "100%"}),
    dcc.Store(id='demo-store'),
    dcc.Store(id='legend-settings'),
], style={"padding": "3% 5%", "color": "black"})

@app.callback(
    Output('demo-store', 'data'),
    Input('demo-file', 'n_clicks'),
    prevent_initial_call=True
)
def load_demo_file(n_clicks):
    if n_clicks:
        demo_dir = os.path.join(os.path.dirname(__file__), "demo_folder")
        if not os.path.exists(demo_dir):
            return dash.no_update
        contents = []
        filenames = []
        for file in sorted(os.listdir(demo_dir)):
            file_path = os.path.join(demo_dir, file)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                ext = os.path.splitext(file)[1].lower()
                mime = "text/plain" if ext in [".txt", ".hkl"] or "-hkl" in file else "application/octet-stream"
                contents.append(f"data:{mime};base64,{encoded}")
                filenames.append(file)
        return {"contents": contents, "filenames": filenames}
    return dash.no_update

@app.callback(
    Output('graph', 'figure'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('demo-store', 'data'),
    Input('update-plot', 'n_clicks'),
    State('xmin', 'value'),
    State('xmax', 'value'),
    State('ymin', 'value'),
    State('ymax', 'value'),
    State('legend-y', 'value'),
    Input('diff-shift', 'value'),
    Input({'type': 'hkl-shift', 'index': ALL}, 'value'),
    State('legend-settings', 'data'),  # <-- add this line
)
def update_graph(list_of_contents, list_of_names, demo_data, update_n_clicks,
                 xmin, xmax, ymin, ymax, legend_y, diff_shift, hkl_shifts, legend_settings):
    # Use demo data if present
    if demo_data and demo_data.get("contents") and demo_data.get("filenames"):
        list_of_contents = demo_data["contents"]
        list_of_names = demo_data["filenames"]
    if not list_of_contents or not list_of_names:
        return {}, "Please upload the main .txt file and one or more -hkl files."

    # Normalize file extensions to lower case for robust matching
    lower_names = [name.lower().strip() for name in list_of_names]
    txt_idx = next((i for i, name in enumerate(lower_names) if name.endswith('.txt')), None)
    hkl_idxs = [i for i, name in enumerate(lower_names) if name.endswith('-hkl') or name.endswith('.hkl')]

    hkl_idxs = sorted(hkl_idxs, key=lambda i: list_of_names[i])

    if txt_idx is None or not hkl_idxs:
        return {}, f"Please upload one .txt file and at least one .hkl or -hkl file together.<br>Detected: {lower_names}"

    # Parse main data file
    content_type, content_string = list_of_contents[txt_idx].split(',')
    decoded = base64.b64decode(content_string)
    lines = decoded.decode('utf-8').splitlines()
    filtered_lines = [line for line in lines if re.match(r'^\d', line)]
    data = pd.read_csv(io.StringIO('\n'.join(filtered_lines)), header=None)
    data.columns = ['x', '31.5.xy', 'Ycalc', 'Diff']
    xy_columns = [col for col in data.columns if col.endswith('.xy')]
    xy_column = xy_columns[0]
    # Read again with header=1 for correct values
    decoded_full = base64.b64decode(content_string)
    data_full = pd.read_csv(io.StringIO(decoded_full.decode('utf-8')), header=1)
    data_full.columns = ['x', xy_column, 'Ycalc', 'Diff']
    data = data_full

    # Parse hkl files
    hkl_paths = []
    hkl_labels = []
    peak_positions_list = []
    for idx in hkl_idxs:
        hkl_name = list_of_names[idx]
        hkl_paths.append(hkl_name)
        hkl_labels.append(format_subscript(hkl_name.replace('-hkl', '').replace('.hkl', '')))
        content_type, content_string = list_of_contents[idx].split(',')
        decoded = base64.b64decode(content_string)
        peak_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None, sep=r'\s+')
        peak_positions = peak_data.iloc[:, 0].values
        peak_positions_list.append(peak_positions)

    # Find y-limits and offsets
    max_y_value = max(data['Ycalc'].max(), data['Diff'].max(), data[xy_column].max())
    min_y_value_diff = data['Diff'].min()
    min_y_value_exp = data[xy_column].min()
    scaled_y_max = max_y_value / 0.97
    scaled_y_min = min_y_value_diff * 2.5
    diff_offset = 1.2 * (min_y_value_exp - data['Diff'].max())
    data['Diff_shifted'] = data['Diff'] + diff_offset + (diff_shift or 0)

    # Colors for peaks
    colors = ['red', 'green', 'purple', 'blue', 'orange', 'cyan', 'magenta', 'yellow']
    # Plotly traces
    traces = [
        go.Scatter(x=data['x'], y=data[xy_column], mode='markers', name='data',
                   marker=dict(color='black', symbol='circle-open'), showlegend=True),
        go.Scatter(x=data['x'], y=data['Ycalc'], mode='lines', name='fit',
                   line=dict(color='orange'), showlegend=True),
        go.Scatter(x=data['x'], y=data['Diff_shifted'], mode='lines', name='difference',
                   line=dict(color='lightgrey'), showlegend=True)
    ]

    # Plot hkl peaks as vertical lines (actual lines)
    hkl_offset_step = (scaled_y_max - scaled_y_min) * 0.02
    base_hkl_position = scaled_y_min * .9
    fixed_hkl_height = (scaled_y_max - scaled_y_min) * 0.015

    # Apply legend_settings order and colors
    n = len(hkl_labels)
    order = list(range(n))
    colors = COMPOSITION_COLORS[:n]
    if legend_settings:
        order = legend_settings.get('order', order)
        colors = legend_settings.get('colors', colors)
        if len(colors) < n:
            colors += ['black'] * (n - len(colors))
        order = order[:n]
        colors = colors[:n]
    hkl_labels = [hkl_labels[i] for i in order]
    peak_positions_list = [peak_positions_list[i] for i in order]
    colors = [colors[i] for i in order]

    # Now plot HKL peaks using the user-selected order and color
    num_hkls = len(peak_positions_list)
    for i, (peak_positions, label) in enumerate(zip(peak_positions_list, hkl_labels)):
        hkl_shift = hkl_shifts[i] if hkl_shifts and i < len(hkl_shifts) and hkl_shifts[i] is not None else 0
        hkl_offset = base_hkl_position + (num_hkls - 1 - i) * hkl_offset_step + hkl_shift
        for peak in peak_positions:
            traces.append(
                go.Scatter(
                    x=[peak, peak],
                    y=[hkl_offset, hkl_offset + fixed_hkl_height],
                    mode='lines',
                    line=dict(color=colors[i], width=2),
                    name=None,
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        traces.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    color=colors[i],
                    size=18,
                    line=dict(width=2, color=colors[i])
                ),
                name=label,
                showlegend=True,
                hoverinfo='skip'
            )
        )

    txt_file_name = list_of_names[txt_idx]
    plot_title_raw = f"{os.path.splitext(txt_file_name)[0]} refinement plot"
    plot_title = format_subscript(plot_title_raw)

    fig = go.Figure(traces)
    # Default axis ranges
    x_range = [21, 79]
    y_range = [scaled_y_min, scaled_y_max]

    # If user provided axis limits, use them
    if xmin is not None and xmax is not None:
        x_range = [xmin, xmax]
    if ymin is not None and ymax is not None:
        y_range = [ymin, ymax]

    # Clamp legend_y to [0, 1]
    default_legend_y = 0.98
    legend_y_val = default_legend_y + (legend_y or 0)

    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=28, family="DejaVu Sans, Arial, sans-serif", color="black"),  # <-- set title color
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=r"diffraction angle, 2<i>θ</i>",
            title_font=dict(size=28, color="black", family="DejaVu Sans, Arial, sans-serif"),  # <-- set x-axis title color
            range=x_range,
            tickfont=dict(size=28, color="black", family="DejaVu Sans, Arial, sans-serif"),    # <-- set x-axis tick color
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='inside',
            ticklen=10,
            tickwidth=1.5
        ),
        yaxis=dict(
            title='intensity, a.u.',
            title_font=dict(size=28, color="black", family="DejaVu Sans, Arial, sans-serif"),  # <-- set y-axis title color
            range=y_range,
            tickfont=dict(size=28, color="black", family="DejaVu Sans, Arial, sans-serif"),    # <-- set y-axis tick color
            showticklabels=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        legend=dict(
            font=dict(size=28, color="black", family="DejaVu Sans, Arial, sans-serif"),        # <-- set legend color
            x=0.98,
            y=legend_y_val,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        meta={
            "custom_scaled_y_min": scaled_y_min,
            "custom_scaled_y_max": scaled_y_max,
        }
    )

    hkl_files = [list_of_names[i] for i in hkl_idxs]
    txt_file_name = list_of_names[txt_idx]
    plot_title = f"{os.path.splitext(txt_file_name)[0]} refinement plot"
    return fig, html.Span(
        f"Uploaded: {list_of_names[txt_idx]} and HKL files: {', '.join(hkl_files)}",
        style={
            "color": "blue",
            "fontWeight": "bold",
            "fontSize": "16px"
        }
    )

# Add this callback to sync axis limits to the input fields
@app.callback(
    Output('xmin', 'value'),
    Output('xmax', 'value'),
    Output('ymin', 'value'),
    Output('ymax', 'value'),
    Input('reset-axes', 'n_clicks'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def reset_axis_inputs(n_clicks, fig):
    if not fig or 'layout' not in fig:
        return None, None, None, None
    xaxis = fig['layout'].get('xaxis', {})
    yaxis = fig['layout'].get('yaxis', {})
    meta = fig['layout'].get('meta', {})
    ymin = math.floor(meta.get('custom_scaled_y_min', yaxis.get('range', [None, None])[0]))
    ymax = math.ceil(meta.get('custom_scaled_y_max', yaxis.get('range', [None, None])[1]))
    xmin = math.floor(xaxis.get('range', [None, None])[0]) if xaxis.get('range') else None
    xmax = math.ceil(xaxis.get('range', [None, None])[1]) if xaxis.get('range') else None
    return xmin, xmax, ymin, ymax

@app.callback(
    Output('hkl-shift-inputs', 'children'),
    Input('upload-data', 'filename'),
    Input('demo-store', 'data'),
)
def generate_hkl_shift_inputs(filenames, demo_data):
    # Get HKL file names
    if demo_data and demo_data.get("filenames"):
        filenames = demo_data["filenames"]
    if not filenames:
        return []
    lower_names = [name.lower().strip() for name in filenames]
    hkl_names = [name for name in filenames if name.lower().endswith('-hkl') or name.lower().endswith('.hkl')]
    return [
        html.Div([
            dcc.Markdown(
                f"HKL shift: {format_subscript(hkl_name.replace('-hkl', '').replace('.hkl', ''))}",
                dangerously_allow_html=True,
                style={
                    "fontWeight": "bold",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif",
                    "marginBottom": "5px",
                    "color": "black"
                }
            ),
            dcc.Input(id={'type': 'hkl-shift', 'index': i}, type='number', value=0, step=0.1, style={
                "width": "80px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                "fontFamily": "DejaVu Sans, Arial, sans-serif",
                "color": "black"
            }),
        ], style={"marginBottom": "10px"})
        for i, hkl_name in enumerate(hkl_names)
    ]

@app.callback(
    Output('download-plot', 'data'),
    Output('save-confirmation', 'children'),
    Input('save-plot', 'n_clicks'),
    State('graph', 'figure'),
    State('output-data-upload', 'children'),
    prevent_initial_call=True
)
def save_plot(n_clicks, figure, output_children):
    if n_clicks:
        # Try to extract a filename from the output message, fallback to generic
        filename = "refinement_plot.png"
        if output_children and hasattr(output_children, 'children'):
            # Try to parse the txt file name from the upload message
            import re
            match = re.search(r"Uploaded: ([^ ]+)", output_children.children)
            if match:
                filename = f"{os.path.splitext(match.group(1))[0]}_refinement_plot.png"
        fig = pio.from_json(pio.to_json(figure))
        def write_image_to_bytesio(output_buffer):
            fig.write_image(output_buffer, format="png", scale=4)
        return dcc.send_bytes(write_image_to_bytesio, filename), html.Span(
            f"Plot downloaded as '{filename}'!",
            style={"fontWeight": "bold", "fontSize": "18px", "color": "green", "fontFamily": "DejaVu Sans, Arial, sans-serif"}
        )
    return dash.no_update, ""

@app.callback(
    Output('legend-controls-container', 'children'),
    Input('upload-data', 'filename'),
    Input('demo-store', 'data'),
    State('legend-settings', 'data')
)
def generate_legend_controls(filenames, demo_data, legend_settings):
    if demo_data and demo_data.get("filenames"):
        filenames = demo_data["filenames"]
    if not filenames:
        return []
    hkl_names = [name for name in filenames if name.lower().endswith('-hkl') or name.lower().endswith('.hkl')]
    items = []
    for i, hkl_name in enumerate(hkl_names):
        label = format_subscript(hkl_name.replace('-hkl', '').replace('.hkl', ''))
        items.append({"label": label, "id": f"hkl-{i}"})

    default_colors = COMPOSITION_COLORS[:len(items)]
    color_options = [
        {"label": html.Span([
            html.Div(style={
                "backgroundColor": c, "width": "15px", "height": "15px",
                "display": "inline-block", "marginRight": "8px"
            }), c.capitalize()
        ], style={"display": "flex", "alignItems": "center", "fontFamily": "DejaVu Sans, Arial, sans-serif"}),
         "value": c}
        for c in COMPOSITION_COLORS
    ]

    n = len(items)
    order = list(range(n))
    colors = default_colors
    if legend_settings:
        order = legend_settings.get('order', order)
        colors = legend_settings.get('colors', colors)
        if len(colors) < n:
            colors += ['black'] * (n - len(colors))
    items = [items[i] for i in order]
    colors = [colors[i] for i in order]

    controls = []
    for i, item in enumerate(items):
        controls.append(
            html.Div([
                dcc.Markdown(
                    item["label"],
                    dangerously_allow_html=True,
                    style={
                        "fontWeight": "bold",
                        "fontFamily": "DejaVu Sans, Arial, sans-serif",
                        "marginBottom": "5px",
                        "color": "black"
                    }
                ),
                dcc.Dropdown(
                    id={'type': 'color-dropdown', 'index': i},
                    options=color_options,
                    value=colors[i],
                    clearable=False,
                    style={"width": "120px", "marginRight": "10px", "fontFamily": "DejaVu Sans, Arial, sans-serif"}
                ),
                html.Button("↑", id={'type': 'legend-up', 'index': i}, n_clicks=0, style={"marginRight": "5px"}),
                html.Button("↓", id={'type': 'legend-down', 'index': i}, n_clicks=0),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})
        )
    if not controls:
        return []
    return [
        html.H4("Composition legend controls", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px", "color": "black"}),
        html.Div(controls)
    ]

@app.callback(
    Output('legend-settings', 'data'),
    Input({'type': 'color-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'legend-up', 'index': ALL}, 'n_clicks'),
    Input({'type': 'legend-down', 'index': ALL}, 'n_clicks'),
    State('legend-settings', 'data'),
    State('upload-data', 'filename'),
    State('demo-store', 'data'),
    prevent_initial_call=True
)
def update_legend_settings(colors, ups, downs, current, filenames, demo_data):
    # Only handle HKL files
    if demo_data and demo_data.get("filenames"):
        filenames = demo_data["filenames"]
    if not filenames:
        return dash.no_update
    hkl_names = [name for name in filenames if name.lower().endswith('-hkl') or name.lower().endswith('.hkl')]
    n = len(hkl_names)
    order = list(range(n))
    if current and 'order' in current and len(current['order']) == n:
        order = current['order']

    # Detect which up/down button was pressed
    changed_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if changed_id:
        try:
            changed = eval(changed_id)
            idx = changed['index']
            if changed['type'] == 'legend-up' and idx > 0:
                order[idx-1], order[idx] = order[idx], order[idx-1]
            elif changed['type'] == 'legend-down' and idx < n-1:
                order[idx+1], order[idx] = order[idx], order[idx+1]
        except Exception:
            pass

    # Clamp colors to n
    if len(colors) < n:
        colors += ['black'] * (n - len(colors))
    colors = colors[:n]

    return {'order': order, 'colors': colors}

COMPOSITION_COLORS = ['red', 'green', 'blue', 'magenta', 'orange', 'cyan', 'black', 'purple']

# (Optional) You can add a callback for "Update Plot" to update the graph using the input values if you want interactive axis control.

if __name__ == '__main__':
    app.run_server(debug=True)