"""
Dashboard for EEG mental state classifier.
Provides a web interface for visualizing EEG data and classification results.
"""

import os
import sys
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import mne
import base64
import tempfile
import io
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.preprocessor import EEGPreprocessor
from feature_extraction.brainwave_features import BrainwaveFeatureExtractor
from models.classifier import MentalStateClassifier

# Initialize Dash app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = "EEG Mental State Classifier"

# Define standard EEG channel positions (10-20 system) for topographic plots
CHANNEL_POSITIONS = {
    'Fp1': [-0.25, 0.9], 'Fp2': [0.25, 0.9],
    'F7': [-0.8, 0.6], 'F3': [-0.5, 0.6], 'Fz': [0, 0.6], 'F4': [0.5, 0.6], 'F8': [0.8, 0.6],
    'T3': [-0.8, 0], 'C3': [-0.5, 0], 'Cz': [0, 0], 'C4': [0.5, 0], 'T4': [0.8, 0],
    'T5': [-0.8, -0.6], 'P3': [-0.5, -0.6], 'Pz': [0, -0.6], 'P4': [0.5, -0.6], 'T6': [0.8, -0.6],
    'O1': [-0.25, -0.9], 'O2': [0.25, -0.9]
}

# Define colors for different frequency bands
BAND_COLORS = {
    'delta': '#4C72B0',  # Blue
    'theta': '#55A868',  # Green
    'alpha': '#C44E52',  # Red
    'beta': '#8172B3',   # Purple
    'gamma': '#CCB974'   # Yellow
}

# Helper functions
def parse_edf_content(contents, filename):
    """Parse uploaded EDF file content."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as temp:
            temp_name = temp.name
            temp.write(decoded)
        
        # Load the EEG data using MNE
        raw = mne.io.read_raw_edf(temp_name, preload=True)
        
        # Clean up temporary file
        try:
            os.unlink(temp_name)
        except:
            pass
            
        return raw
    except Exception as e:
        print(f"Error parsing EDF file: {e}")
        return None

def process_eeg_data(raw):
    """Process EEG data and extract features."""
    try:
        # Preprocess the data
        preprocessor = EEGPreprocessor()
        raw_processed = preprocessor.preprocess(raw)
        
        # Extract features
        feature_extractor = BrainwaveFeatureExtractor(sfreq=raw.info['sfreq'])
        features_df = feature_extractor.extract_features(raw_processed)
        
        # Get a short sample of the processed signal for display
        data, times = raw_processed[:, :int(raw_processed.info['sfreq'] * 10)]  # 10 seconds
        
        return {
            'features': features_df,
            'sample_data': data,
            'sample_times': times,
            'channel_names': raw_processed.ch_names,
            'sfreq': raw_processed.info['sfreq']
        }
    except Exception as e:
        print(f"Error processing EEG data: {e}")
        return None

def classify_mental_state(features_df):
    """Classify mental state using pre-trained model."""
    try:
        # Path to the saved model
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "saved_models", "mental_state_classifier.joblib"
        )
        
        # Load model if it exists, or create a default one if it doesn't
        if os.path.exists(model_path):
            try:
                import joblib
                model = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        else:
            # Create a default model (not trained)
            print("Model not found, creating a new instance (not trained)")
            model = MentalStateClassifier(model_type='svm')
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        result = {
            'prediction': int(prediction),
            'state': "Focused (Mental Arithmetic)" if prediction == 1 else "Relaxed (Baseline)",
            'confidence': float(max(probabilities)),
            'probabilities': {
                'relaxed': float(probabilities[0]),
                'focused': float(probabilities[1])
            }
        }
        return result
    except Exception as e:
        print(f"Error classifying mental state: {e}")
        return None

# App layout
app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.H2("EEG Mental State Classifier", className="app-header--title"),
                html.P("Upload EEG data to classify mental states (focused vs. relaxed)"),
            ],
            className="app-header",
        ),
        
        # Main content
        html.Div(
            [
                # Left column - Upload and controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Upload EEG Data"),
                                dcc.Upload(
                                    id="upload-eeg",
                                    children=html.Div(
                                        ["Drag and drop or click to select an EDF file"]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px 0",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="upload-status"),
                                
                                html.Hr(),
                                
                                html.H4("Visualization Controls"),
                                html.Label("Select Channel:"),
                                dcc.Dropdown(id="channel-dropdown", disabled=True),
                                
                                html.Label("Select Time Window (seconds):"),
                                dcc.RangeSlider(
                                    id="time-slider",
                                    min=0,
                                    max=10,
                                    step=0.1,
                                    value=[0, 5],
                                    marks={i: f"{i}s" for i in range(11)},
                                    disabled=True,
                                ),
                                
                                html.Label("Select Frequency Bands:"),
                                dcc.Checklist(
                                    id="band-checklist",
                                    options=[
                                        {'label': 'Delta (0.5-4 Hz)', 'value': 'delta'},
                                        {'label': 'Theta (4-8 Hz)', 'value': 'theta'},
                                        {'label': 'Alpha (8-12 Hz)', 'value': 'alpha'},
                                        {'label': 'Beta (12-30 Hz)', 'value': 'beta'},
                                        {'label': 'Gamma (30-45 Hz)', 'value': 'gamma'},
                                    ],
                                    value=['alpha'],
                                    className="checklist-band",
                                    labelStyle={"display": "block", "margin-bottom": "5px"},
                                ),
                            ],
                            className="control-panel",
                        ),
                        
                        # Classification Results
                        html.Div(
                            [
                                html.H4("Classification Results"),
                                html.Div(id="classification-results", className="results-panel"),
                            ],
                        ),
                        
                        # Data info
                        html.Div(
                            [
                                html.H4("Data Information"),
                                html.Div(id="data-info", className="info-panel"),
                            ],
                        ),
                    ],
                    className="four columns sidebar",
                ),
                
                # Right column - Visualizations
                html.Div(
                    [
                        # Signal Plot
                        html.Div(
                            [
                                html.H4("EEG Signal"),
                                dcc.Graph(id="eeg-signal-plot"),
                            ],
                            className="pretty-container",
                        ),
                        
                        # Spectral Power Plot
                        html.Div(
                            [
                                html.H4("Frequency Spectrum"),
                                dcc.Graph(id="spectral-plot"),
                            ],
                            className="pretty-container",
                        ),
                        
                        # Band Power Topographic Map
                        html.Div(
                            [
                                html.H4("Brain Activity Map"),
                                dcc.Graph(id="topo-plot"),
                            ],
                            className="pretty-container",
                        ),
                        
                        # Feature Importance
                        html.Div(
                            [
                                html.H4("Feature Importance"),
                                dcc.Graph(id="feature-importance-plot"),
                            ],
                            className="pretty-container",
                        ),
                    ],
                    className="eight columns main-content",
                ),
            ],
            className="app-body",
        ),
        
        # Store intermediate data
        dcc.Store(id="processed-data"),
        dcc.Store(id="classification-data"),
    ],
    className="app-container",
)

# Callbacks
@app.callback(
    [
        Output("upload-status", "children"),
        Output("processed-data", "data"),
        Output("channel-dropdown", "options"),
        Output("channel-dropdown", "value"),
        Output("channel-dropdown", "disabled"),
        Output("time-slider", "disabled"),
        Output("band-checklist", "disabled"),
    ],
    [Input("upload-eeg", "contents")],
    [State("upload-eeg", "filename")],
)
def process_upload(contents, filename):
    if contents is None:
        return html.Div("No file uploaded yet."), None, [], None, True, True, True
    
    if not filename.endswith('.edf'):
        return html.Div("Please upload an EDF file.", style={"color": "red"}), None, [], None, True, True, True
    
    # Parse and process the uploaded file
    raw = parse_edf_content(contents, filename)
    if raw is None:
        return html.Div("Error parsing the EDF file.", style={"color": "red"}), None, [], None, True, True, True
    
    # Process the EEG data
    processed_data = process_eeg_data(raw)
    if processed_data is None:
        return html.Div("Error processing the EEG data.", style={"color": "red"}), None, [], None, True, True, True
    
    # Prepare channel dropdown options
    channel_options = [{"label": ch, "value": i} for i, ch in enumerate(processed_data['channel_names'])]
    
    # Prepare data for storage
    store_data = {
        'features': processed_data['features'].to_dict('records'),
        'sample_data': processed_data['sample_data'].tolist(),
        'sample_times': processed_data['sample_times'].tolist(),
        'channel_names': processed_data['channel_names'],
        'sfreq': processed_data['sfreq'],
        'filename': filename
    }
    
    return (
        html.Div(f"File processed: {filename}", style={"color": "green"}),
        store_data,
        channel_options,
        0,  # Default to first channel
        False,
        False,
        False
    )

@app.callback(
    Output("classification-data", "data"),
    [Input("processed-data", "data")],
)
def run_classification(processed_data):
    if processed_data is None:
        raise PreventUpdate
    
    # Convert features back to DataFrame
    features_df = pd.DataFrame(processed_data['features'])
    
    # Classify mental state
    classification_result = classify_mental_state(features_df)
    
    return classification_result

@app.callback(
    Output("classification-results", "children"),
    [Input("classification-data", "data")],
)
def update_classification_results(classification_data):
    if classification_data is None:
        return html.Div("Upload EEG data to see classification results.")
    
    state_class = "focused" if classification_data['prediction'] == 1 else "relaxed"
    confidence = classification_data['confidence'] * 100
    
    return html.Div([
        html.H3(classification_data['state'], className=f"result-state {state_class}"),
        html.Div([
            html.Span("Confidence: "),
            html.Span(f"{confidence:.1f}%", style={"font-weight": "bold"}),
        ]),
        html.Div(className="probability-bar-container", children=[
            html.Div("Relaxed", className="prob-label"),
            html.Div(className="probability-bar", children=[
                html.Div(
                    className="probability-fill relaxed",
                    style={"width": f"{classification_data['probabilities']['relaxed'] * 100}%"}
                )
            ]),
            html.Div(f"{classification_data['probabilities']['relaxed'] * 100:.1f}%", className="prob-value"),
        ]),
        html.Div(className="probability-bar-container", children=[
            html.Div("Focused", className="prob-label"),
            html.Div(className="probability-bar", children=[
                html.Div(
                    className="probability-fill focused",
                    style={"width": f"{classification_data['probabilities']['focused'] * 100}%"}
                )
            ]),
            html.Div(f"{classification_data['probabilities']['focused'] * 100:.1f}%", className="prob-value"),
        ]),
    ])

@app.callback(
    Output("data-info", "children"),
    [Input("processed-data", "data")],
)
def update_data_info(processed_data):
    if processed_data is None:
        return html.Div("Upload EEG data to see information.")
    
    return html.Div([
        html.Div([html.Strong("Filename: "), processed_data['filename']]),
        html.Div([html.Strong("Channels: "), str(len(processed_data['channel_names']))]),
        html.Div([html.Strong("Sampling Rate: "), f"{processed_data['sfreq']} Hz"]),
        html.Div([html.Strong("Duration: "), f"{len(processed_data['sample_times']) / processed_data['sfreq']:.1f} seconds"]),
    ])

@app.callback(
    Output("eeg-signal-plot", "figure"),
    [
        Input("processed-data", "data"),
        Input("channel-dropdown", "value"),
        Input("time-slider", "value"),
    ],
)
def update_eeg_signal_plot(processed_data, channel_idx, time_range):
    if processed_data is None or channel_idx is None:
        return {
            "layout": {
                "title": "Upload EEG data to visualize",
                "xaxis": {"title": "Time (s)"},
                "yaxis": {"title": "Amplitude (µV)"},
            }
        }
    
    # Extract data for the selected channel
    channel_data = processed_data['sample_data'][channel_idx]
    times = processed_data['sample_times']
    channel_name = processed_data['channel_names'][channel_idx]
    
    # Apply time range filter
    start_idx = int(time_range[0] * processed_data['sfreq'])
    end_idx = int(time_range[1] * processed_data['sfreq'])
    start_idx = max(0, start_idx)
    end_idx = min(len(times), end_idx)
    
    filtered_times = times[start_idx:end_idx]
    filtered_data = channel_data[start_idx:end_idx]
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_times,
            y=filtered_data,
            mode="lines",
            name=channel_name,
            line=dict(color="#2C3E50", width=1.5),
        )
    )
    
    fig.update_layout(
        title=f"EEG Signal - Channel {channel_name}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

@app.callback(
    Output("spectral-plot", "figure"),
    [
        Input("processed-data", "data"),
        Input("channel-dropdown", "value"),
        Input("band-checklist", "value"),
    ],
)
def update_spectral_plot(processed_data, channel_idx, selected_bands):
    if processed_data is None or channel_idx is None or not selected_bands:
        return {
            "layout": {
                "title": "Upload EEG data and select frequency bands",
                "xaxis": {"title": "Frequency (Hz)"},
                "yaxis": {"title": "Power Spectral Density"},
            }
        }
    
    # Extract data for the selected channel
    channel_data = processed_data['sample_data'][channel_idx]
    channel_name = processed_data['channel_names'][channel_idx]
    sfreq = processed_data['sfreq']
    
    # Calculate power spectrum
    from scipy import signal
    f, Pxx = signal.welch(channel_data, fs=sfreq, nperseg=int(sfreq * 2))
    
    # Define frequency bands
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Create the plot
    fig = go.Figure()
    
    # Add the full spectrum line
    fig.add_trace(
        go.Scatter(
            x=f,
            y=Pxx,
            mode="lines",
            name="Full Spectrum",
            line=dict(color="#7F7F7F", width=1),
            opacity=0.5,
        )
    )
    
    # Add colored regions for each selected frequency band
    for band in selected_bands:
        band_range = freq_bands[band]
        band_mask = (f >= band_range[0]) & (f <= band_range[1])
        
        fig.add_trace(
            go.Scatter(
                x=f[band_mask],
                y=Pxx[band_mask],
                mode="lines",
                name=f"{band.capitalize()} ({band_range[0]}-{band_range[1]} Hz)",
                line=dict(color=BAND_COLORS[band], width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(int(BAND_COLORS[band].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}",
            )
        )
    
    fig.update_layout(
        title=f"Frequency Spectrum - Channel {channel_name}",
        xaxis_title="Frequency (Hz)",
        xaxis=dict(range=[0, 45]),
        yaxis_title="Power Spectral Density",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig

@app.callback(
    Output("topo-plot", "figure"),
    [
        Input("processed-data", "data"),
        Input("band-checklist", "value"),
    ],
)
def update_topo_plot(processed_data, selected_bands):
    if processed_data is None or not selected_bands:
        return {
            "layout": {
                "title": "Upload EEG data and select frequency bands",
            }
        }
    
    # Get features
    features_df = pd.DataFrame(processed_data['features'])
    
    # Extract band power features
    band_powers = {}
    for band in selected_bands:
        band_cols = [col for col in features_df.columns if f"{band}_power" in col]
        if band_cols:
            powers = features_df[band_cols].values[0]
            # Extract channel names from feature names
            channels = [col.split('_')[0] for col in band_cols]
            band_powers[band] = dict(zip(channels, powers))
    
    if not band_powers:
        return {
            "layout": {
                "title": "No band power features found in data",
            }
        }
    
    # Create a topomap for each selected band
    fig = go.Figure()
    
    # Common channels between features and standard positions
    for band in selected_bands:
        if band not in band_powers:
            continue
            
        # Match feature channels with standard positions
        positions = []
        values = []
        
        for ch, power in band_powers[band].items():
            if ch in CHANNEL_POSITIONS:
                positions.append(CHANNEL_POSITIONS[ch])
                values.append(power)
        
        if not positions:
            continue
        
        # Convert to numpy arrays
        positions = np.array(positions)
        values = np.array(values)
        
        # Create scatter plot for electrode positions
        fig.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode="markers",
                marker=dict(
                    size=15,
                    color=values,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=f"{band.capitalize()} Power",
                        x=1.1,
                        xanchor="left",
                    ),
                    showscale=(band == selected_bands[-1]),  # Only show colorbar for the last band
                ),
                text=[f"{ch}: {val:.2f}" for ch, val in zip(band_powers[band].keys(), values)],
                hoverinfo="text",
                name=band.capitalize(),
                visible=(band == selected_bands[0]),  # Only show first band by default
            )
        )
    
    # Add head outline
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Add nose and ears to make the head recognizable
    # Nose
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0.95, 1.1],
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Left ear
    fig.add_trace(
        go.Scatter(
            x=[-1, -1.1, -1.05, -1],
            y=[0, -0.1, 0.1, 0],
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Right ear
    fig.add_trace(
        go.Scatter(
            x=[1, 1.1, 1.05, 1],
            y=[0, -0.1, 0.1, 0],
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Create buttons for each band
    if len(selected_bands) > 1:
        buttons = []
        for i, band in enumerate(selected_bands):
            visible = [False] * len(fig.data)
            # Make head outline, nose, and ears always visible
            visible[-4:] = [True, True, True, True]
            # Make the current band visible
            visible[i] = True
            
            buttons.append(
                dict(
                    method="update",
                    label=band.capitalize(),
                    args=[{"visible": visible}],
                )
            )
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    buttons=buttons,
                    x=0.5,
                    y=1.15,
                    xanchor="center",
                    yanchor="top",
                )
            ]
        )
    
    fig.update_layout(
        title="Brain Activity Map",
        xaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    return fig

@app.callback(
    Output("feature-importance-plot", "figure"),
    [Input("processed-data", "data"),
     Input("classification-data", "data")],
)
def update_feature_importance(processed_data, classification_data):
    if processed_data is None or classification_data is None:
        return {
            "layout": {
                "title": "Upload EEG data to see feature importance",
                "xaxis": {"title": "Importance"},
                "yaxis": {"title": "Feature"},
            }
        }
    
    # Since we don't have actual feature importance from the model,
    # we'll create a visualization based on the features and their values
    features_df = pd.DataFrame(processed_data['features'])
    
    # Select a subset of interesting features (top 15 by absolute value)
    features = features_df.iloc[0].sort_values(key=abs, ascending=False).head(15)
    
    # Create a color scale based on the values
    colors = ['#FF4136' if val < 0 else '#0074D9' for val in features.values]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=features.index,
            x=features.values,
            orientation='h',
            marker_color=colors,
            text=[f"{val:.3f}" for val in features.values],
            textposition="outside",
        )
    )
    
    fig.update_layout(
        title="Top Features by Value",
        xaxis_title="Feature Value",
        yaxis_title="Feature",
        template="plotly_white",
        margin=dict(l=150, r=40, t=40, b=40),
        height=500,
    )
    
    return fig

# Run the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run_server(debug=debug, host="0.0.0.0", port=port)
