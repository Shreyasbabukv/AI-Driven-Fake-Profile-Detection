import dash
from dash import dcc, html
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Minimal Dash Plotly Test"),
    dcc.Graph(
        id='test-graph',
        figure=go.Figure(
            data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2], mode='lines+markers')],
            layout=go.Layout(title='Test Plot')
        )
    )
])

if __name__ == '__main__':
    app.run(debug=True)
