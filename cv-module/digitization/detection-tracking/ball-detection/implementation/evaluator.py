import cv2
import os
import json
import numpy as np
from ball_tracker import BallTracker
from scipy.spatial import distance
from scipy.signal import medfilt, savgol_filter
import matplotlib.pyplot as plt

from general import load_config


class BallTrackingEvaluator:
    """Evaluates ball tracking performance on videos and generates reports."""

    def __init__(self, config: dict = None):
        """Initialize evaluator with configuration.

        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        if config is None:
            config = load_config("config.json")

        self.config = config

        self.tracker = None
        self.ball_positions = []
        self.velocities = []
        self.video_name = None
        self.fps = None

    def initialize_tracker(self):
        """Initialize the ball tracker."""
        print("Initializing ball tracker...")
        self.tracker = BallTracker(config=self.config)
        print(f"Tracker initialized on device: {self.tracker.device}")

    def process_video(self):
        """Process video and track ball.

        Args:
            video_path: Path to video file. If None, uses config value.
        """
        video_path = self.config["video"]["input_path"]

        self.video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Initialize tracker if not already done
        if self.tracker is None:
            self.initialize_tracker()

        # Open video
        cap = cv2.VideoCapture(video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {self.fps} fps, {total_frames} frames")

        # Setup output video if enabled
        output_dir = self.config["output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        out = None
        if self.config["output"]["save_video"]:
            output_video_path = os.path.join(
                output_dir, f"{self.video_name}_tracked.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (width, height))

        # Reset tracking data
        self.ball_positions = []
        self.velocities = []

        print("Processing video...")
        frame_count = 0
        trace_length = self.config["tracking"]["trace_length"]
        trace_color = tuple(self.config["tracking"]["trace_color"])
        trace_thickness = self.config["tracking"]["trace_thickness"]

        max_frames = self.config["video"]["max_frames"]
        total_frames = max_frames if max_frames else total_frames

        # Main loop - read and process each frame
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Get ball position from tracker
            x, y = self.tracker.process_frame(frame)
            self.ball_positions.append((x, y))

            # Calculate velocity
            if (
                len(self.ball_positions) >= 2
                and x is not None
                and self.ball_positions[-2][0] is not None
            ):
                dist = distance.euclidean((x, y), self.ball_positions[-2])
                velocity = dist * self.fps
            else:
                velocity = 0
            self.velocities.append(velocity)

            # Draw ball trace on frame if saving video
            if out is not None:
                for i in range(trace_length):
                    idx = frame_count - i
                    if idx >= 0 and idx < len(self.ball_positions):
                        pos = self.ball_positions[idx]
                        if pos[0] is not None:
                            cv2.circle(
                                frame,
                                pos,
                                radius=0,
                                color=trace_color,
                                thickness=trace_thickness - i,
                            )

                out.write(frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames")

        cap.release()
        if out is not None:
            out.release()
            print(f"Video saved to {output_video_path}")

    def calculate_metrics(self):
        """Calculate tracking metrics.

        Returns:
            dict: Dictionary containing tracking metrics
        """
        detected_frames = sum(1 for x, y in self.ball_positions if x is not None)
        detection_rate = (
            (detected_frames / len(self.ball_positions) * 100)
            if self.ball_positions
            else 0
        )
        valid_velocities = [v for v in self.velocities if v > 0]

        metrics = {
            "total_frames": len(self.ball_positions),
            "detected_frames": detected_frames,
            "detection_rate_percent": round(detection_rate, 2),
            "avg_velocity_pixels_per_sec": (
                round(np.mean(valid_velocities), 2) if valid_velocities else 0
            ),
            "max_velocity_pixels_per_sec": (
                round(np.max(valid_velocities), 2) if valid_velocities else 0
            ),
            "min_velocity_pixels_per_sec": (
                round(np.min(valid_velocities), 2) if valid_velocities else 0
            ),
        }

        return metrics

    def print_metrics(self, metrics):
        """Print metrics to console."""
        print("\n" + "=" * 50)
        print("BALL TRACKING METRICS")
        print("=" * 50)
        print(f"Total frames:              {metrics['total_frames']}")
        print(f"Detected frames:           {metrics['detected_frames']}")
        print(f"Detection rate:            {metrics['detection_rate_percent']:.2f}%")
        print(
            f"Avg velocity:              {metrics['avg_velocity_pixels_per_sec']:.2f} pixels/sec"
        )
        print(
            f"Max velocity:              {metrics['max_velocity_pixels_per_sec']:.2f} pixels/sec"
        )
        print(
            f"Min velocity:              {metrics['min_velocity_pixels_per_sec']:.2f} pixels/sec"
        )
        print("=" * 50 + "\n")

    def save_position_plot(self):
        """Generate and save position plot."""
        if not self.config["output"]["save_plots"]:
            return

        output_dir = self.config["output"]["output_dir"]
        output_path = os.path.join(output_dir, f"{self.video_name}_positions.png")
        dpi = self.config["output"]["plot_dpi"]

        # Extract frame indices and coordinates
        frames = []
        x_coords = []
        y_coords = []

        for i, (x, y) in enumerate(self.ball_positions):
            if x is not None:
                frames.append(i)
                x_coords.append(x)
                y_coords.append(y)

        if not frames:
            print("No valid positions to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot X coordinate over time
        ax1.plot(frames, x_coords, "b-", linewidth=2)
        ax1.set_ylabel("X Coordinate (pixels)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_title("Ball Position Over Time", fontsize=14)

        # Plot Y coordinate over time
        ax2.plot(frames, y_coords, "r-", linewidth=2)
        ax2.set_xlabel("Frame Number", fontsize=12)
        ax2.set_ylabel("Y Coordinate (pixels)", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved position plot to {output_path}")

    def save_velocity_plot(self):
        """Generate and save velocity plot."""
        if not self.config["output"]["save_plots"]:
            return

        output_dir = self.config["output"]["output_dir"]
        output_path = os.path.join(output_dir, f"{self.video_name}_velocity.png")
        dpi = self.config["output"]["plot_dpi"]

        # Filter valid velocities
        frames = []
        valid_velocities = []

        for i, vel in enumerate(self.velocities):
            if vel > 0:
                frames.append(i)
                valid_velocities.append(vel)

        if not frames:
            print("No valid velocities to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frames, valid_velocities, "g-", linewidth=2)
        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Velocity (pixels/second)", fontsize=12)
        ax.set_title("Ball Velocity Over Time", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved velocity plot to {output_path}")

    def save_metrics_csv(self, metrics):
        """Save metrics to CSV file."""
        if not self.config["output"]["save_metrics"]:
            return

        output_dir = self.config["output"]["output_dir"]
        output_path = os.path.join(output_dir, f"{self.video_name}_metrics.csv")

        with open(output_path, "w") as f:
            f.write("metric,value\n")
            for key, value in metrics.items():
                f.write(f"{key},{value}\n")
        print(f"Saved metrics to {output_path}")

    def smooth_positions(self, median_window=5, savgol_window=9, savgol_poly=3):
        """Apply two-stage smoothing to ball positions.

        Stage 1: Median filter to remove outlier spikes
        Stage 2: Savitzky-Golay filter for smooth curves

        Args:
            median_window: Window size for median filter (must be odd)
            savgol_window: Window size for Savitzky-Golay filter (must be odd)
            savgol_poly: Polynomial order for Savitzky-Golay filter
        """
        if len(self.ball_positions) < savgol_window:
            return

        # Separate x and y coordinates
        x_coords = np.array(
            [pos[0] if pos[0] is not None else np.nan for pos in self.ball_positions]
        )
        y_coords = np.array(
            [pos[1] if pos[1] is not None else np.nan for pos in self.ball_positions]
        )

        # Interpolate missing values (NaNs)
        valid_x_indices = ~np.isnan(x_coords)
        valid_y_indices = ~np.isnan(y_coords)

        if np.sum(valid_x_indices) > 1:
            x_coords_interp = np.interp(
                np.arange(len(x_coords)),
                np.where(valid_x_indices)[0],
                x_coords[valid_x_indices],
            )
        else:
            x_coords_interp = x_coords

        if np.sum(valid_y_indices) > 1:
            y_coords_interp = np.interp(
                np.arange(len(y_coords)),
                np.where(valid_y_indices)[0],
                y_coords[valid_y_indices],
            )
        else:
            y_coords_interp = y_coords

        # Stage 1: Apply median filter to remove outlier spikes
        x_median = medfilt(x_coords_interp, kernel_size=median_window)
        y_median = medfilt(y_coords_interp, kernel_size=median_window)

        # Stage 2: Apply Savitzky-Golay filter for smooth curves
        x_smoothed = savgol_filter(
            x_median, window_length=savgol_window, polyorder=savgol_poly
        )
        y_smoothed = savgol_filter(
            y_median, window_length=savgol_window, polyorder=savgol_poly
        )

        # Convert back to list of tuples
        self.ball_positions = [(int(x), int(y)) for x, y in zip(x_smoothed, y_smoothed)]

    def evaluate(self):
        """Run full evaluation pipeline.

        Args:
            video_path: Path to video file. If None, uses config value.
        """
        # Process video
        self.process_video()

        # Smooth positions before analysis
        print("Applying two-stage smoothing (median + Savitzky-Golay)...")
        self.smooth_positions(median_window=5, savgol_window=9, savgol_poly=3)

        # Calculate metrics
        metrics = self.calculate_metrics()
        self.print_metrics(metrics)

        # Save results
        self.save_position_plot()
        self.save_velocity_plot()
        self.save_metrics_csv(metrics)

        output_dir = self.config["output"]["output_dir"]
        print(f"\nAll results saved to: {output_dir}")
        print("Processing complete!")


def main():
    """Main entry point."""
    evaluator = BallTrackingEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
