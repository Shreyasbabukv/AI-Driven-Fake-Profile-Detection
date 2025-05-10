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

def create_roc_curve_figure():
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(1 - fpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def create_pr_curve_figure():
    recall = np.linspace(0, 1, 100)
    precision = 1 - np.sqrt(recall)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve', line=dict(color='magenta')))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def create_model_comparison_figure():
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies, marker_color='cyan'))
    fig.add_trace(go.Bar(name='Precision', x=model_names, y=precisions, marker_color='magenta'))
    fig.add_trace(go.Bar(name='Recall', x=model_names, y=recalls, marker_color='yellow'))
    fig.update_layout(barmode='group', title='Model Comparison', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def create_risk_score_meter(score):
    colorscale = [
        [0, "green"],
        [0.5, "yellow"],
        [1, "red"]
    ]
    fig = go.Figure(go.Bar(
        x=[score],
        y=["Risk Score"],
        orientation='h',
        marker=dict(
            color=score,
            colorscale=colorscale,
            cmin=0,
            cmax=100,
            colorbar=dict(title="Confidence")
        ),
        width=0.5
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False),
        height=100,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_accuracy_epochs_figure():
    fig = go.Figure(data=go.Scatter(x=epochs, y=accuracy_epochs, mode='lines+markers', line=dict(color='cyan')))
    fig.update_layout(title='Accuracy vs Epochs', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

app.layout = dbc.Container([
    html.H1("Fake Profile Detection Dashboard", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            html.H4("Input Profile Features"),
            dbc.Form(
                [generate_input_field(name, name.replace('user', '').replace('Count', ' Count').replace('Is', ' Is ').replace('Pic', 'Pic').replace('BiographyLength', 'Biography Length').replace('DigitCount', 'Digit Count').replace('Length', ' Length')) for name in feature_names] +
                [dbc.Button("Predict", id="predict-button", color="primary")]
            ),
            html.Div(id="prediction-output", className="mt-3")
        ], md=4),
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label='Model Metrics', children=[
                    dcc.Graph(id='accuracy-epochs', figure=create_accuracy_epochs_figure()),
                    dcc.Graph(id='confusion-matrix', figure=create_confusion_matrix_figure()),
                    dcc.Graph(id='roc-curve', figure=create_roc_curve_figure()),
                    dcc.Graph(id='pr-curve', figure=create_pr_curve_figure()),
                    dcc.Graph(id='model-comparison', figure=create_model_comparison_figure()),
                ]),
                dcc.Tab(label='Risk Score', children=[
                    dcc.Graph(id='risk-score-meter', figure=create_risk_score_meter(0)),
                    html.Div(id='explainability-text', className="mt-3 p-3 bg-secondary text-white rounded"),
                ]),
                dcc.Tab(label='Live Monitoring', children=[
                    html.Div(id='live-monitoring-output', className="mt-3"),
                    dcc.Interval(id='live-monitoring-interval', interval=3000, n_intervals=0)
                ]),
                dcc.Tab(label='System Stats', children=[
                    html.Div([
                        html.P(f"Total Profiles Scanned: {system_stats['total_scanned']}"),
                        html.P(f"Fake Profiles Detected: {system_stats['fake_detected']}"),
                        html.P(f"Detection Rate: {system_stats['detection_rate']}%"),
                        dcc.Graph(
                            figure=go.Figure(
                                data=[
                                    go.Scatter(x=system_stats['timeline']['time'], y=system_stats['timeline']['scanned'], mode='lines', name='Scanned', line=dict(color='cyan')),
                                    go.Scatter(x=system_stats['timeline']['time'], y=system_stats['timeline']['fake_detected'], mode='lines', name='Fake Detected', line=dict(color='magenta'))
                                ],
                                layout=go.Layout(title='Scanning Timeline', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                            )
                        )
                    ], className="p-3")
                ])
            ])
        ], md=8)
    ])
], fluid=True)

@app.callback(
    Output("prediction-output", "children"),
    Output("risk-score-meter", "figure"),
    Output("explainability-text", "children"),
    Input("predict-button", "n_clicks"),
    [State(name, "value") for name in feature_names]
)
def predict(n_clicks, *values):
    if not n_clicks:
        return "", create_risk_score_meter(0), ""
    input_data = dict(zip(feature_names, values))
    try:
        response = requests.post("http://localhost:5000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", None)
            risk_score = int(confidence * 100) if confidence else 0
            explainability = []
            if input_data.get('userBiographyLength', 0) < 20:
                explainability.append("Short biography length")
            if input_data.get('userFollowerCount', 0) < 50:
                explainability.append("Low follower count")
            explanation_text = "Profile flagged due to: " + ", ".join(explainability) if explainability else "Profile looks normal."
            alert = dbc.Alert(
                f"Prediction: {'Fake' if prediction == 1 else 'Real'} (Confidence: {confidence:.2f})",
                color="danger" if prediction == 1 else "success"
            )
            risk_figure = create_risk_score_meter(risk_score)
            return alert, risk_figure, explanation_text
        else:
            return dbc.Alert(f"Error from prediction API: {response.status_code}", color="warning"), create_risk_score_meter(0), ""
    except Exception as e:
        return dbc.Alert(f"Error calling prediction API: {str(e)}", color="danger"), create_risk_score_meter(0), ""

@app.callback(
    Output('live-monitoring-output', 'children'),
    Input('live-monitoring-interval', 'n_intervals')
)
def update_live_monitoring(n):
    profile = live_profiles[n % len(live_profiles)]
    status = "Fake" if profile['isFake'] else "Real"
    color = "danger" if profile['isFake'] else "success"
    return dbc.Alert(f"Scanned profile: {profile['username']} - Followers: {profile['followerCount']}, Bio Length: {profile['bioLength']} - Status: {status}", color=color)

if __name__ == "__main__":
    app.run(debug=True)
</create_file>
