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
        dbc.InputGroupText(label, style={'backgroundColor': '#2c2c2c', 'color': '#e0e0e0'}),
        dbc.Input(type="number", id=name, min=0, step=1, value=0, style={'backgroundColor': '#1f1f1f', 'color': '#e0e0e0', 'border': 'none'})
    ], className="mb-3")

def create_confusion_matrix_figure():
    z = [[196, 3], [7, 33]]
    x = ['Predicted Real', 'Predicted Fake']
    y = ['Actual Real', 'Actual Fake']
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='Viridis'))
    fig.update_layout(title='Confusion Matrix', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#e0e0e0')
    return fig

def create_roc_curve_figure():
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(1 - fpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#bb86fc')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash', color='#6200ee')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#e0e0e0')
    return fig

def create_pr_curve_figure():
    recall = np.linspace(0, 1, 100)
    precision = 1 - np.sqrt(recall)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve', line=dict(color='#bb86fc')))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#e0e0e0')
    return fig

def create_model_comparison_figure():
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracies, marker_color='#bb86fc'))
    fig.add_trace(go.Bar(name='Precision', x=model_names, y=precisions, marker_color='#3700b3'))
    fig.add_trace(go.Bar(name='Recall', x=model_names, y=recalls, marker_color='#03dac6'))
    fig.update_layout(barmode='group', title='Model Comparison', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#e0e0e0')
    return fig

def create_risk_score_meter(score):
    colorscale = [
        [0, "#03dac6"],
        [0.5, "#bb86fc"],
        [1, "#3700b3"]
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

sidebar = html.Div(
    [
        html.H2("Menu", className="text-center", style={"color": "#bb86fc"}),
        html.Hr(style={"borderColor": "#3700b3"}),
        dbc.Nav(
            [
                dbc.NavLink("Dashboard", href="/", active="exact", style={"color": "#e0e0e0"}),
                dbc.NavLink("Model Metrics", href="/model-metrics", active="exact", style={"color": "#e0e0e0"}),
                dbc.NavLink("Risk Score", href="/risk-score", active="exact", style={"color": "#e0e0e0"}),
                dbc.NavLink("Live Monitoring", href="/live-monitoring", active="exact", style={"color": "#e0e0e0"}),
                dbc.NavLink("System Stats", href="/system-stats", active="exact", style={"color": "#e0e0e0"}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

content = html.Div(id="page-content", className="content")

app.layout = html.Div([
    html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap"),
    dcc.Location(id="url"),
    sidebar,
    content
])

def render_dashboard():
    return dbc.Container([
        html.H1("Fake Profile Detection Dashboard", className="text-center my-4"),
        html.P("Welcome to the professional dark-themed dashboard with advanced features.", className="text-center mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Profiles Scanned", className="card-title"),
                        html.H2(f"{system_stats['total_scanned']}", className="card-text"),
                    ])
                ], className="card")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Fake Profiles Detected", className="card-title"),
                        html.H2(f"{system_stats['fake_detected']}", className="card-text"),
                    ])
                ], className="card")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Detection Rate (%)", className="card-title"),
                        html.H2(f"{system_stats['detection_rate']}%", className="card-text"),
                    ])
                ], className="card")
            ], md=3),
        ])
    ], fluid=True)

def render_model_metrics():
    return dbc.Container([
        html.H2("Model Metrics", className="text-center my-4"),
        dcc.Graph(id='accuracy-epochs', figure=go.Figure(data=go.Scatter(x=epochs, y=accuracy_epochs, mode='lines+markers'), layout=go.Layout(title='Accuracy vs Epochs', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#e0e0e0'))),
        dcc.Graph(id='confusion-matrix', figure=create_confusion_matrix_figure()),
        dcc.Graph(id='roc-curve', figure=create_roc_curve_figure()),
        dcc.Graph(id='pr-curve', figure=create_pr_curve_figure()),
        dcc.Graph(id='model-comparison', figure=create_model_comparison_figure()),
    ], fluid=True)

def render_risk_score():
    return dbc.Container([
        html.H2("Risk Score", className="text-center my-4"),
        dcc.Graph(id='risk-score-meter', figure=create_risk_score_meter(42)),
        html.Div("Explainability: Profile flagged due to short bio and low followers.", className="mt-3 p-3 bg-secondary text-white rounded"),
    ], fluid=True)

def render_live_monitoring():
    return dbc.Container([
        html.H2("Live Monitoring", className="text-center my-4"),
        html.Div(id='live-monitoring-output', className="mt-3"),
        dcc.Interval(id='live-monitoring-interval', interval=3000, n_intervals=0)
    ], fluid=True)

def render_system_stats():
    return dbc.Container([
        html.H2("System Stats", className="text-center my-4"),
        html.Div([
            html.P(f"Total Profiles Scanned: {system_stats['total_scanned']}"),
            html.P(f"Fake Profiles Detected: {system_stats['fake_detected']}"),
            html.P(f"Detection Rate: {system_stats['detection_rate']}%"),
            dcc.Graph(
                figure=go.Figure(
                    data=[
                        go.Scatter(x=system_stats['timeline']['time'], y=system_stats['timeline']['scanned'], mode='lines', name='Scanned', line=dict(color='#bb86fc')),
                        go.Scatter(x=system_stats['timeline']['time'], y=system_stats['timeline']['fake_detected'], mode='lines', name='Fake Detected', line=dict(color='#03dac6'))
                    ],
                    layout=go.Layout(title='Scanning Timeline', plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#e0e0e0')
                )
            )
        ], className="p-3")
    ], fluid=True)

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/":
        return render_dashboard()
    elif pathname == "/model-metrics":
        return render_model_metrics()
    elif pathname == "/risk-score":
        return render_risk_score()
    elif pathname == "/live-monitoring":
        return render_live_monitoring()
    elif pathname == "/system-stats":
        return render_system_stats()
    else:
        return dbc.Jumbotron([
            html.H1("404: Not found", className="text-danger"),
            html.P(f"The pathname {pathname} was not recognised...")
        ])

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State(name, "value") for name in feature_names]
)
def predict(n_clicks, *values):
    if not n_clicks:
        return ""
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
            return dbc.Container([
                dbc.Alert(
                    f"Prediction: {'Fake' if prediction == 1 else 'Real'} (Confidence: {confidence:.2f})",
                    color="danger" if prediction == 1 else "success"
                ),
                dcc.Graph(figure=create_risk_score_meter(risk_score)),
                html.Div(explanation_text, className="text-white bg-dark p-2 rounded")
            ])
        else:
            return dbc.Alert(f"Error from prediction API: {response.status_code}", color="warning")
    except Exception as e:
        return dbc.Alert(f"Error calling prediction API: {str(e)}", color="danger")

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
