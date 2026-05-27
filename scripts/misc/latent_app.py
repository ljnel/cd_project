import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np


def create_dash_app(latents, trajs):
    num_trajectories, trajectory_length = trajs.shape

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Latent Space Exploration"),
        html.Div([
            # Left Plot: Latent Space Scatter Plot
            dcc.Graph(
                id='latent-scatter-plot',
                style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'},
                figure={
                    'data': [
                        go.Scatter(
                            x=latents[:, 0],
                            y=latents[:, 1],
                            mode='markers',
                            marker=dict(size=5, opacity=0.7),
                            # Store the index of the trajectory in customdata
                            customdata=np.arange(num_trajectories),
                            hovertemplate="Latent X: %{x:.2f}<br>Latent Y: %{y:.2f}<extra></extra>"
                        )
                    ],
                    'layout': go.Layout(
                        title='2D Latent Space',
                        xaxis_title='Latent Dimension 1',
                        yaxis_title='Latent Dimension 2',
                        hovermode='closest' # Important for plotly_hover event
                    )
                }
            ),
            # Right Plot: Trajectory Plot
            dcc.Graph(
                id='trajectory-line-plot',
                style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'},
                figure={
                    'data': [
                        go.Scatter(
                            x=np.arange(trajectory_length),
                            y=np.zeros(trajectory_length), # Initial empty trajectory
                            mode='lines',
                            name='Original Trajectory'
                        )
                    ],
                    'layout': go.Layout(
                        title='Original Trajectory',
                        xaxis_title='Measurement Index',
                        yaxis_title='Value',
                        uirevision=True # Keeps zoom/pan when data updates
                    )
                }
            )
        ])
    ])

    @app.callback(
        Output('trajectory-line-plot', 'figure'),
        [Input('latent-scatter-plot', 'hoverData')]
    )
    def display_hover_trajectory(hoverData):
        fig = go.Figure()
        
        if hoverData:
            # Get the index of the hovered point from customdata
            point_index = hoverData['points'][0]['customdata']
            
            # Get the original trajectory data using this index
            selected_trajectory = trajs[point_index]
            
            # Create the trace for the selected trajectory
            fig.add_trace(go.Scatter(
                x=np.arange(trajectory_length),
                y=selected_trajectory,
                mode='lines',
                name=f'Trajectory {point_index}'
            ))
            
            # Update layout for the trajectory plot
            fig.update_layout(
                title=f'Original Trajectory for Latent {point_index}',
                xaxis_title='Measurement Index',
                yaxis_title='Value',
                uirevision=True # Important: preserves zoom/pan when updating
            )
        else:
            # If no hover, show a blank or default trajectory
            fig.add_trace(go.Scatter(
                x=np.arange(trajectory_length),
                y=np.zeros(trajectory_length),
                mode='lines',
                name='No Trajectory Selected'
            ))
            fig.update_layout(
                title='Hover over a latent point to see its trajectory',
                xaxis_title='Measurement Index',
                yaxis_title='Value'
            )

        return fig
            
    return app
