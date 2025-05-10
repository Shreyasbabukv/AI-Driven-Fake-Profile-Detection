import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import requests

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    return html.Div([
        dbc.Label(label, html_for=name, className="font-weight-bold"),
        dbc.Input(type="number", id=name, min=0, step=1, value=0, className="mb-3")
    ])

app.layout = dbc.Container([
    html.H1("Fake Profile Detector", className="my-4 text-center"),
    dbc.Row([
        dbc.Col([
            dbc.Form(
                [generate_input_field('userFollowerCount', 'Follower Count'),
                 generate_input_field('userFollowingCount', 'Following Count'),
                 generate_input_field('userBiographyLength', 'Biography Length'),
                 generate_input_field('userMediaCount', 'Media Count'),
                 generate_input_field('userHasProfilPic', 'Has Profile Picture (0 or 1)'),
                 generate_input_field('userIsPrivate', 'Is Private Account (0 or 1)'),
                 generate_input_field('usernameDigitCount', 'Username Digit Count'),
                 generate_input_field('usernameLength', 'Username Length'),
                 dbc.Button("Predict", id="predict-button", color="primary", className="mt-3")
                ]
            )
        ], md=6),
        dbc.Col([
            html.Div(id="prediction-output", className="mt-4 p-3 border rounded bg-light")
        ], md=6)
    ])
], fluid=True)

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
            return dbc.Alert(
                f"Prediction: {'Fake' if prediction == 1 else 'Real'}" + (f" (Confidence: {confidence:.2f})" if confidence else ""),
                color="success" if prediction == 0 else "danger"
            )
        else:
            return dbc.Alert(f"Error from prediction API: {response.status_code}", color="warning")
    except Exception as e:
        return dbc.Alert(f"Error calling prediction API: {str(e)}", color="danger")

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
