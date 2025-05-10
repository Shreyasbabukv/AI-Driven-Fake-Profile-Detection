import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[3,1,2], mode='lines+markers'))
fig.update_layout(title='Debug Test Graph')

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Test Tab', children=[
            dcc.Graph(id='test-graph', figure=fig, style={'height': '400px', 'width': '100%'})
        ])
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
