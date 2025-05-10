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
