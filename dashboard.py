from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import duckdb
import pandas as pd
from environment import EnvVars

cleaned_data = EnvVars.CLEANED_DATA_PATH

signal = 'iforest_score'
threshold = -0.15

def probabilitySignal(signal, threshold):
    db = duckdb.connect()
    # Get Data
    df = db.execute(f"SELECT dt, sensor, event, {signal} FROM read_parquet('{cleaned_data }') ORDER BY dt ASC").df()
    # Get labels for potential anomalies
    anomalies = db.execute(f"SELECT dt, sensor, {signal} FROM read_parquet('{cleaned_data }') WHERE {signal} < {threshold} ORDER BY dt ASC").df()
    db.close()
    df['dt'] = pd.to_datetime(df['dt'])
    anomalies['dt'] = pd.to_datetime(anomalies['dt'])
    df.set_index('dt', inplace=True)

    return df, anomalies

app = Dash()

SIGNAL_CONFIGS = {
    'markov': {'signal': 'markov_prob', 'threshold': 0.01},
    'iforest': {'signal': 'iforest_score', 'threshold': -0.15},
    'eif': {'signal': 'eif_score', 'threshold': -0.15}
}
@app.callback(
    Output('signal-graph', 'figure'),
    Input('signal-selector', 'value')
)
def update_graph(selected_signal):
    config = SIGNAL_CONFIGS[selected_signal]
    signal = config['signal']
    threshold = config['threshold']
    df, anomalies = probabilitySignal(signal, threshold)
    groups = [group for _, group in df.groupby(pd.Grouper(freq='24h')) if not group.empty]
    # Keep only the last 'max_periods' to ensure the final image is readable
    max_periods = 30
    plot_groups = groups[-max_periods:]
    n_plots = len(plot_groups)

    fig = make_subplots(
        rows=n_plots, 
        cols=1, 
        shared_yaxes=True,
        subplot_titles=[group.index[0].strftime('%Y-%m-%d') for group in plot_groups]
    )

    for i, group in enumerate(plot_groups):
        row_idx = i + 1  # Plotly uses 1-based indexing for rows/cols
        day_start = group.index.min()
        day_end = group.index.max()
        # Base Signal Scatter (Blue dots)
        fig.add_trace(
            go.Scatter(
                x=group.index,
                y=group[signal],
                customdata=group[['sensor', 'event']],
                mode='markers',
                marker=dict(color='blue', size=5, opacity=0.6),
                name=f'Signal (Group {row_idx})',
                showlegend=False,
                hovertemplate=(
                "<b>Sensor:</b>%{customdata[0]}<br>"
                "<b>Event:</b>%{customdata[1]}<br>"
                "<b>Time:</b> %{x|%Y-%m-%d %H:%M:%S}<br>"
                "<b>Value:</b> %{y}<extra></extra>" # <extra></extra> hides the secondary trace box
                )
            ),
            row=row_idx, col=1
        )
        mask = (anomalies['dt'] >= day_start) & (anomalies['dt'] <= day_end)
        group_anomalies = anomalies.loc[mask]
        fig.add_trace(
            go.Scatter(
                x=group_anomalies['dt'],
                y=group_anomalies[signal],
                customdata=group[['sensor', 'event']],
                mode='markers',
                marker=dict(color='red', size=5),
                name='Anomaly',
                showlegend=(i == 0),
                hovertemplate=(
                "<b>Sensor:</b>%{customdata[0]}<br>"
                "<b>Event:</b>%{customdata[1]}<br>"
                "<b>Time:</b> %{x|%Y-%m-%d %H:%M:%S}<br>"
                "<b>Value:</b> %{y}<extra></extra>" # <extra></extra> hides the secondary trace box
                )
            ),
            row=row_idx, col=1
        )
        # Add Threshold Horizontal Line
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="black", 
            line_width=1, 
            opacity=0.6,
            row=row_idx, col=1
        )
        
        # Update axes styling (Gridlines and X-axis date formatting)
        fig.update_xaxes(
            tickformat="%H:%M", 
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(0,0,0,0.1)', # Light dashed look
            griddash='dash',
            row=row_idx, col=1
        )
        
        fig.update_yaxes(
            title_text=str(signal),
            showgrid=True, 
            gridwidth=1, 
            gridcolor='rgba(0,0,0,0.1)',
            griddash='dash',
            row=row_idx, col=1
        )

    fig.update_layout(
        height=250 * n_plots, 
        width=1000, 
        plot_bgcolor='white', # Matches Matplotlib default aesthetic
        margin=dict(t=40, b=40, l=60, r=20)
    )
    return fig

app.layout = [
    html.H1(children='Signal Anomalies', style={
            'textAlign': 'center',
        }),
    html.Div(className='row', children=[
    dcc.RadioItems(options=[
                {'label': 'Markov', 'value': 'markov'},
                {'label': 'Isolation Forest', 'value': 'iforest'},
                {'label': 'Extended Isolation Forest', 'value': 'eif'},
        ],
                    inline=True,
                    value='markov',
                    id='signal-selector',)
    ]),
    dcc.Graph(
        id='signal-graph'
    )
]

if __name__ == "__main__":
    app.run(debug=True)
