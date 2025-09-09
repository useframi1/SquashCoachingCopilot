import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_ball_trajectory(ball_track, video_name=None, save_path=None, smoothing=0):
    """
    Plot the ball trajectory with time indicated by color gradient.

    Parameters:
    -----------
    ball_track : list of (x, y) tuples
        The ball coordinates over time
    video_name : str, optional
        Name of the video for the plot title
    save_path : str, optional
        Path to save the plot. If None, the plot will be displayed instead
    smoothing : int, optional
        Window size for moving average smoothing. 0 for no smoothing.
    """
    # Filter out None values
    valid_points = [
        (i, point) for i, point in enumerate(ball_track) if point[0] is not None
    ]

    if not valid_points:
        print("No valid ball tracking points found.")
        return

    # Extract frame indices and coordinates
    frame_indices, coordinates = zip(*valid_points)
    x_coords, y_coords = zip(*coordinates)

    # Apply smoothing if requested
    if smoothing > 0 and len(x_coords) > smoothing:
        x_coords = moving_average(x_coords, smoothing)
        y_coords = moving_average(y_coords, smoothing)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a custom colormap that goes from blue to red
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=len(frame_indices)
    )

    # Plot the trajectory with color representing time
    for i in range(len(frame_indices) - 1):
        ax.plot(
            x_coords[i : i + 2],
            y_coords[i : i + 2],
            color=cmap(i / len(frame_indices)),
            linewidth=2,
        )

    # Add colorbar to represent time
    norm = plt.Normalize(frame_indices[0], frame_indices[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # Explicitly specify the axes
    cbar.set_label("Frame Number", fontsize=12)

    # Add start and end points
    ax.scatter(x_coords[0], y_coords[0], color="blue", s=100, label="Start", zorder=5)
    ax.scatter(x_coords[-1], y_coords[-1], color="red", s=100, label="End", zorder=5)

    # Set title and labels
    if video_name:
        ax.set_title(f"Ball Trajectory - {video_name}", fontsize=14)
    else:
        ax.set_title("Ball Trajectory Over Time", fontsize=14)

    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    # Invert Y-axis to match image coordinates (origin at top-left)
    ax.invert_yaxis()

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=10)

    # Set aspect ratio to be equal
    ax.set_aspect("equal")

    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)  # Close the first figure

    # Create an additional plot showing x and y coordinates over time
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot X coordinate over time
    ax1.plot(frame_indices, x_coords, "b-", linewidth=2)
    ax1.set_ylabel("X Coordinate", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot Y coordinate over time
    ax2.plot(frame_indices, y_coords, "r-", linewidth=2)
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Y Coordinate", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    if video_name:
        fig2.suptitle(f"Ball Position Over Time - {video_name}", fontsize=14)
    else:
        fig2.suptitle("Ball Position Over Time", fontsize=14)

    plt.tight_layout()

    # Save or display the position plot
    if save_path:
        position_plot_path = os.path.splitext(save_path)[0] + "_position.png"
        plt.savefig(position_plot_path, dpi=300, bbox_inches="tight")
        print(f"Position plot saved to {position_plot_path}")
    else:
        plt.show()

    plt.close(fig2)  # Close the second figure


def moving_average(data, window_size):
    """Apply moving average smoothing to the data"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode="valid")
