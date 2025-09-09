import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
import cv2
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


def analyze_ball_trajectory(
    ball_track,
    frames=None,
    fps=None,
    video_name=None,
    output_dir="output",
    use_original_resolution=True,
):
    """
    Comprehensive analysis of ball trajectory data.

    Parameters:
    -----------
    ball_track : list of (x, y) tuples
        The ball coordinates over time
    frames : list, optional
        List of video frames (for creating heatmap overlay)
    fps : int, optional
        Frames per second of the video (for velocity calculation)
    video_name : str, optional
        Name of the video for plot titles
    output_dir : str, optional
        Directory to save output files
    use_original_resolution : bool, optional
        If True, coordinates will be scaled to the original video resolution
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get original video resolution
    model_width, model_height = 640, 360  # Model's processing resolution
    postprocess_scale = 2  # The scale factor in postprocess function

    orig_width, orig_height = None, None
    if frames is not None and len(frames) > 0:
        orig_height, orig_width = frames[0].shape[:2]

        # Calculate scaling factors from model to original resolution
        scale_x = (orig_width / model_width) / postprocess_scale
        scale_y = (orig_height / model_height) / postprocess_scale

        print(f"Original video resolution: {orig_width}x{orig_height}")
        print(f"Model resolution: {model_width}x{model_height}")
        print(f"Postprocess scale: {postprocess_scale}")
        print(f"Scaling factors: x={scale_x}, y={scale_y}")

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

    # Convert to numpy arrays for easier manipulation
    frame_indices = np.array(frame_indices)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Scale coordinates to original resolution if requested and if we have the original frames
    if use_original_resolution and orig_width is not None:
        x_coords_scaled = x_coords * scale_x
        y_coords_scaled = y_coords * scale_y
        print("Using original video resolution for analysis")
    else:
        x_coords_scaled = x_coords
        y_coords_scaled = y_coords
        print("Using model's resolution for analysis")

    # 1. Basic statistics
    total_frames = len(ball_track)
    detected_frames = len(valid_points)
    detection_rate = detected_frames / total_frames * 100

    # Print basic statistics
    print(f"Total frames: {total_frames}")
    print(f"Frames with ball detected: {detected_frames}")
    print(f"Detection rate: {detection_rate:.2f}%")

    # Calculate distances between consecutive points
    distances = []
    for i in range(1, len(x_coords_scaled)):
        dx = x_coords_scaled[i] - x_coords_scaled[i - 1]
        dy = y_coords_scaled[i] - y_coords_scaled[i - 1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)

    if distances:
        print(
            f"Average distance between consecutive points: {np.mean(distances):.2f} pixels"
        )
        print(
            f"Maximum distance between consecutive points: {np.max(distances):.2f} pixels"
        )

    # Calculate velocities if fps is provided
    velocities = []
    if fps and len(distances) > 0:
        # Convert pixel distances to pixels per second
        velocities = [
            d * fps / (frame_indices[i] - frame_indices[i - 1])
            for i, d in enumerate(distances, 1)
        ]

        if velocities:
            print(f"Average velocity: {np.mean(velocities):.2f} pixels/second")
            print(f"Maximum velocity: {np.max(velocities):.2f} pixels/second")

    # 2. Generate plots
    # 2.1 Trajectory plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a custom colormap
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=len(frame_indices)
    )

    # Plot trajectory with color gradient
    for i in range(len(frame_indices) - 1):
        ax.plot(
            x_coords_scaled[i : i + 2],
            y_coords_scaled[i : i + 2],
            color=cmap(i / len(frame_indices)),
            linewidth=2,
        )

    # Add colorbar
    norm = plt.Normalize(frame_indices[0], frame_indices[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frame Number", fontsize=12)

    # Mark start and end points
    ax.scatter(
        x_coords_scaled[0],
        y_coords_scaled[0],
        color="blue",
        s=100,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        x_coords_scaled[-1],
        y_coords_scaled[-1],
        color="red",
        s=100,
        label="End",
        zorder=5,
    )

    # Resolution label for the plot title
    res_label = (
        "Original Resolution"
        if use_original_resolution and orig_width is not None
        else "Model Resolution"
    )

    ax.set_title(f"Ball Trajectory ({res_label})", fontsize=14)
    ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    ax.invert_yaxis()  # Invert Y-axis to match image coordinates
    ax.set_aspect("equal")

    # Add resolution info to the plot
    resolution_text = (
        f"Resolution: {int(max(x_coords_scaled))}x{int(max(y_coords_scaled))} (approx.)"
    )
    plt.figtext(0.02, 0.02, resolution_text, fontsize=10)

    plt.tight_layout()
    trajectory_path = os.path.join(output_dir, "ball_trajectory.png")
    plt.savefig(trajectory_path, dpi=300)
    print(f"Saved trajectory plot to {trajectory_path}")
    plt.close()

    # 2.2 Position vs Time plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Apply Savitzky-Golay filter for smoothing if we have enough points
    if len(x_coords_scaled) > 10:
        window_length = min(
            11, len(x_coords_scaled) // 2 * 2 - 1
        )  # Must be odd and less than data length
        if window_length >= 3:
            x_smooth = savgol_filter(x_coords_scaled, window_length, 2)
            y_smooth = savgol_filter(y_coords_scaled, window_length, 2)

            ax1.plot(frame_indices, x_coords_scaled, "b-", alpha=0.3, label="Raw")
            ax1.plot(frame_indices, x_smooth, "b-", linewidth=2, label="Smoothed")

            ax2.plot(frame_indices, y_coords_scaled, "r-", alpha=0.3, label="Raw")
            ax2.plot(frame_indices, y_smooth, "r-", linewidth=2, label="Smoothed")

            ax1.legend()
            ax2.legend()
        else:
            ax1.plot(frame_indices, x_coords_scaled, "b-", linewidth=2)
            ax2.plot(frame_indices, y_coords_scaled, "r-", linewidth=2)
    else:
        ax1.plot(frame_indices, x_coords_scaled, "b-", linewidth=2)
        ax2.plot(frame_indices, y_coords_scaled, "r-", linewidth=2)

    ax1.set_ylabel(f"X Coordinate (pixels, {res_label})", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel(f"Y Coordinate (pixels, {res_label})", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle(f"Ball Position Over Time ({res_label})", fontsize=14)
    plt.tight_layout()

    position_path = os.path.join(output_dir, "ball_position.png")
    plt.savefig(position_path, dpi=300)
    print(f"Saved position plot to {position_path}")
    plt.close()

    # 2.3 Velocity plot (if fps is provided)
    velocity_path = None
    if fps and len(velocities) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate frame indices for velocities
        vel_frame_indices = [
            0.5 * (frame_indices[i] + frame_indices[i - 1])
            for i in range(1, len(frame_indices))
        ]

        ax.plot(vel_frame_indices, velocities, "g-", linewidth=2)

        # Smoothed velocity if we have enough points
        if len(velocities) > 10:
            window_length = min(11, len(velocities) // 2 * 2 - 1)
            if window_length >= 3:
                vel_smooth = savgol_filter(velocities, window_length, 2)
                ax.plot(
                    vel_frame_indices, vel_smooth, "b-", linewidth=2, label="Smoothed"
                )
                ax.legend()

        ax.set_title(f"Ball Velocity Over Time ({res_label})", fontsize=14)
        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Velocity (pixels/second)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        velocity_path = os.path.join(output_dir, "ball_velocity.png")
        plt.savefig(velocity_path, dpi=300)
        print(f"Saved velocity plot to {velocity_path}")
        plt.close()

    # 2.4 Create a heatmap if frames are provided
    heatmap_path = None
    if frames is not None and len(frames) > 0:
        # Use the first frame as the background
        background = frames[0].copy()
        h, w = background.shape[:2]

        # Create heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Add Gaussian blobs for each ball position
        for i, (x, y) in enumerate(coordinates):
            # Scale coordinates to original resolution
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)

            if 0 <= x_orig < w and 0 <= y_orig < h:
                # Create a Gaussian blob around each point
                y_grid, x_grid = np.ogrid[-y_orig : h - y_orig, -x_orig : w - x_orig]
                # Scale sigma based on resolution
                sigma = 15  # Larger sigma for higher resolution
                mask = np.exp(
                    -(x_grid * x_grid + y_grid * y_grid) / (2 * sigma * sigma)
                )
                heatmap += mask

        # Normalize heatmap
        heatmap = heatmap / np.max(heatmap)

        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Blend with background
        alpha = 0.6
        blended = cv2.addWeighted(background, 1 - alpha, heatmap_colored, alpha, 0)

        # Save the heatmap
        heatmap_path = os.path.join(output_dir, "ball_heatmap.png")
        cv2.imwrite(heatmap_path, blended)
        print(f"Saved heatmap to {heatmap_path}")

    # 2.5 Export coordinates to CSV
    csv_path = os.path.join(output_dir, "ball_coordinates.csv")
    with open(csv_path, "w") as f:
        f.write("frame,x,y")
        if use_original_resolution and orig_width is not None:
            f.write(",x_original,y_original\n")
            for i, ((frame_idx, (x, y)), x_orig, y_orig) in enumerate(
                zip(valid_points, x_coords_scaled, y_coords_scaled)
            ):
                f.write(f"{frame_idx},{x},{y},{x_orig},{y_orig}\n")
        else:
            f.write("\n")
            for frame_idx, (x, y) in valid_points:
                f.write(f"{frame_idx},{x},{y}\n")

    print(f"Saved coordinates to {csv_path}")

    return {
        "trajectory_path": trajectory_path,
        "position_path": position_path,
        "velocity_path": velocity_path,
        "heatmap_path": heatmap_path,
        "csv_path": csv_path,
    }
