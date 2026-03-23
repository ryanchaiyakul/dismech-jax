import numpy as np
import plotly.graph_objects as go
def animate(qs):
    frames = []
    all_coords = np.vstack([qs[:, 0:3], qs[:, 4:7], qs[:, 8:11]])

    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    center = (mins + maxs) / 2

    # Get max range for cube domain
    max_range = np.max(maxs - mins)
    buffer = max_range * 0.1

    # Fixed limits for all frames
    plot_limit = (max_range / 2) + buffer
    x_range = [center[0] - plot_limit, center[0] + plot_limit]
    y_range = [center[1] - plot_limit, center[1] + plot_limit]
    z_range = [center[2] - plot_limit, center[2] + plot_limit]

    # Build frames
    for t in range(len(qs)):
        row = qs[t]
        q_points = [row[0:3], row[4:7], row[8:11]]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=[q_points[0][0], q_points[1][0], q_points[2][0]],
                        y=[q_points[0][1], q_points[1][1], q_points[2][1]],
                        z=[q_points[0][2], q_points[1][2], q_points[2][2]],
                        mode="lines+markers",
                        line=dict(color="black", width=7),
                    ),
                ],
                name=str(t),
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
                aspectmode="cube",  # Forces 1:1:1 scale visuals
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "type": "buttons",
                    "showactive": False,
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ]
                }
            ],
        ),
        frames=frames,
    )
    return fig
