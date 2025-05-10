import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

model_names = ['Random Forest', 'XGBoost', 'LightGBM', 'ANN']
accuracies = [0.95, 0.96, 0.94, 0.93]
precisions = [0.92, 0.94, 0.91, 0.90]
recalls = [0.90, 0.92, 0.89, 0.88]
epochs = list(range(1, 11))
accuracy_epochs = [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]

system_stats = {
    'total_scanned': 1000,
    'fake_detected': 170,
    'detection_rate': 17.0,
    'timeline': pd.DataFrame({
        'time': pd.date_range(start='2025-01-01', periods=10, freq='D'),
        'scanned': np.random.randint(80, 120, 10),
        'fake_detected': np.random.randint(10, 20, 10)
    })
}

live_profiles = [
    {'username': 'user123', 'followerCount': 50, 'bioLength': 10, 'isFake': True},
    {'username': 'real_guy', 'followerCount': 500, 'bioLength': 100, 'isFake': False},
    {'username': 'bot_456', 'followerCount': 5, 'bioLength': 5, 'isFake': True},
    {'username': 'legit_user', 'followerCount': 300, 'bioLength': 80, 'isFake': False},
]

feature_names = [
    'userFollowerCount',
    'userFollowingCount',
    'userBiographyLength',
    'userMediaCount',
    'userHasProfilPic',
    'userIsPrivate',
    'usernameDigitCount',
    'usernameLength'
]

def generate_input_field(name, label):
    return dbc.InputGroup([
        dbc.InputGroupText(label),
        dbc.Input(type="number", id=name, min=0, step=1, value=0)
    ], className="mb-3")

def create_confusion_matrix_figure():
    z = [[196, 3], [7, 33]]
    x = ['Predicted Real', 'Predicted Fake']
    y = ['Actual Real', 'Actual Fake']
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='Viridis'))
    fig.update_layout(title='Confusion Matrix', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig
