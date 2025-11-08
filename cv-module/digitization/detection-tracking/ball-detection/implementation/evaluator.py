import cv2
import os
import json
import numpy as np
from ball_tracker import BallTracker
from scipy.spatial import distance
import matplotlib.pyplot as plt

from utilities.general import load_config
from utilities.postprocessing import postprocess_positions
from utilities.wall_hit_detection import (
    detect_front_wall_hits,
    calculate_hit_statistics,
    save_wall_hits_csv,
)


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
        self.wall_hits = []
        self.video_name = None
        self.fps = None

    def initialize_tracker(self):
        """Initialize the ball tracker."""
        print("Initializing ball tracker...")
        self.tracker = BallTracker(config=self.config)
        print(f"Tracker initialized on device: {self.tracker.device}")

    def process_video(self):
        """Process video and track ball using two-pass approach.

        Pass 1: Detect ball positions in all frames
        Pass 2: Apply smoothing and create annotated video
        """
        video_path = self.config["video"]["input_path"]

        self.video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Initialize tracker if not already done
        if self.tracker is None:
            self.initialize_tracker()

        # Open video for first pass
        cap = cv2.VideoCapture(video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {self.fps} fps, {total_frames} frames")

        # Reset tracking data
        self.ball_positions = []

        max_frames = self.config["video"]["max_frames"]
        total_frames = max_frames if max_frames else total_frames

        # PASS 1: Detect ball positions
        print("Pass 1: Detecting ball positions...")
        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Get ball position from tracker
            x, y = self.tracker.process_frame(frame)
            self.ball_positions.append((x, y))

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Detected {frame_count}/{total_frames} frames")

        cap.release()
        print(f"Detection complete: {len(self.ball_positions)} frames processed")

        # Apply postprocessing pipeline if enabled
        if self.config["postprocessing"]["enabled"]:
            self.ball_positions = postprocess_positions(
                self.ball_positions, self.fps, self.config["postprocessing"]
            )

            # Calculate velocities after postprocessing
            self.velocities = []
            for i in range(len(self.ball_positions)):
                if i >= 1:
                    dist = distance.euclidean(
                        self.ball_positions[i], self.ball_positions[i - 1]
                    )
                    velocity = dist * self.fps
                else:
                    velocity = 0
                self.velocities.append(velocity)

        # Detect wall hits BEFORE creating video (so we can mark them)
        if self.config["wall_hit_detection"]["enabled"]:
            print("\nDetecting front wall hits...")
            wall_hit_config = self.config["wall_hit_detection"]
            self.wall_hits = detect_front_wall_hits(
                self.ball_positions,
                prominence=wall_hit_config["prominence"],
                width=wall_hit_config["width"],
                min_distance=wall_hit_config["min_distance"],
            )
            print(f"  Detected {len(self.wall_hits)} front wall hits")
        else:
            self.wall_hits = []

        # PASS 2: Create annotated video with smoothed positions and wall hits
        if self.config["output"]["save_video"]:
            print("Pass 2: Creating annotated video with smoothed positions...")

            # Setup output directory
            base_output_dir = self.config["output"]["output_dir"]
            output_dir = os.path.join(base_output_dir, self.video_name)
            os.makedirs(output_dir, exist_ok=True)

            output_video_path = os.path.join(
                output_dir, f"{self.video_name}_tracked.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (width, height))

            # Reopen video for second pass
            cap = cv2.VideoCapture(video_path)

            trace_length = self.config["tracking"]["trace_length"]
            trace_color = tuple(self.config["tracking"]["trace_color"])
            trace_thickness = self.config["tracking"]["trace_thickness"]

            # Create a set of wall hit frames for quick lookup
            wall_hit_frames = {hit["frame"]: hit for hit in self.wall_hits}
            # Track which wall hits to keep showing (show for 30 frames after impact)
            wall_hit_display_duration = 30
            active_wall_hits = []  # List of (hit_info, frames_since_impact)

            frame_count = 0
            while cap.isOpened() and frame_count < len(self.ball_positions):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw ball trace using smoothed positions
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

                # Check if current frame is a wall hit
                if frame_count in wall_hit_frames:
                    active_wall_hits.append((wall_hit_frames[frame_count], 0))

                # Draw all active wall hit markers
                hits_to_remove = []
                for i, (hit, frames_since) in enumerate(active_wall_hits):
                    hit_pos = (int(hit["x"]), int(hit["y"]))

                    # Draw impact marker (X shape)
                    marker_size = 20
                    color = (0, 255, 0)  # Green
                    thickness = 3

                    # Draw X
                    cv2.line(
                        frame,
                        (hit_pos[0] - marker_size, hit_pos[1] - marker_size),
                        (hit_pos[0] + marker_size, hit_pos[1] + marker_size),
                        color,
                        thickness,
                    )
                    cv2.line(
                        frame,
                        (hit_pos[0] + marker_size, hit_pos[1] - marker_size),
                        (hit_pos[0] - marker_size, hit_pos[1] + marker_size),
                        color,
                        thickness,
                    )

                    # Draw circle around it
                    cv2.circle(frame, hit_pos, marker_size + 5, color, 2)

                    # Add text label
                    label = f"Hit {len([h for h in self.wall_hits if h['frame'] <= frame_count])}"
                    cv2.putText(
                        frame,
                        label,
                        (hit_pos[0] + 25, hit_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Update frames since impact
                    active_wall_hits[i] = (hit, frames_since + 1)

                    # Mark for removal if too old
                    if frames_since >= wall_hit_display_duration:
                        hits_to_remove.append(i)

                # Remove old wall hits
                for i in reversed(hits_to_remove):
                    active_wall_hits.pop(i)

                out.write(frame)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(
                        f"  Annotated {frame_count}/{len(self.ball_positions)} frames"
                    )

            cap.release()
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

        base_output_dir = self.config["output"]["output_dir"]
        output_dir = os.path.join(base_output_dir, self.video_name)
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

        base_output_dir = self.config["output"]["output_dir"]
        output_dir = os.path.join(base_output_dir, self.video_name)
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

        base_output_dir = self.config["output"]["output_dir"]
        output_dir = os.path.join(base_output_dir, self.video_name)
        output_path = os.path.join(output_dir, f"{self.video_name}_metrics.csv")

        with open(output_path, "w") as f:
            f.write("metric,value\n")
            for key, value in metrics.items():
                f.write(f"{key},{value}\n")
        print(f"Saved metrics to {output_path}")

    def save_wall_hit_results(self):
        """Save wall hit detection results (called after detection in process_video)."""
        if not self.wall_hits:
            return

        print("\nSaving wall hit results...")

        # Calculate and print statistics
        stats = calculate_hit_statistics(self.wall_hits, self.fps)
        if stats["total_hits"] > 0:
            print(
                f"  Average hit interval: {stats['avg_hit_interval_sec']:.2f} seconds"
            )
            print(
                f"  Impact height range: {stats['min_impact_height']:.1f} - {stats['max_impact_height']:.1f} pixels"
            )

        # Save wall hits to CSV
        base_output_dir = self.config["output"]["output_dir"]
        output_dir = os.path.join(base_output_dir, self.video_name)
        output_path = os.path.join(output_dir, f"{self.video_name}_wall_hits.csv")
        save_wall_hits_csv(self.wall_hits, output_path)
        print(f"  Saved wall hits to {output_path}")

        # Save wall hits plot
        self.save_wall_hits_plot()

    def save_wall_hits_plot(self):
        """Generate and save position plot with wall hits marked."""
        if not self.config["output"]["save_plots"] or not self.wall_hits:
            return

        base_output_dir = self.config["output"]["output_dir"]
        output_dir = os.path.join(base_output_dir, self.video_name)
        output_path = os.path.join(output_dir, f"{self.video_name}_wall_hits.png")
        dpi = self.config["output"]["plot_dpi"]

        # Extract all positions
        frames = list(range(len(self.ball_positions)))
        x_coords = [p[0] for p in self.ball_positions]
        y_coords = [p[1] for p in self.ball_positions]

        # Extract wall hit positions
        hit_frames = [hit["frame"] for hit in self.wall_hits]
        hit_y = [hit["y"] for hit in self.wall_hits]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot X coordinate
        ax1.plot(frames, x_coords, "b-", linewidth=1.5, label="X position")
        ax1.set_ylabel("X Coordinate (pixels)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_title(
            "Ball Position Over Time with Front Wall Hits",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(loc="upper right")

        # Plot Y coordinate with wall hits
        ax2.plot(frames, y_coords, "r-", linewidth=1.5, label="Y position")
        ax2.scatter(
            hit_frames,
            hit_y,
            color="green",
            s=100,
            marker="o",
            zorder=5,
            label=f"Wall hits ({len(self.wall_hits)})",
            edgecolors="darkgreen",
            linewidths=2,
        )
        ax2.set_xlabel("Frame Number", fontsize=12)
        ax2.set_ylabel("Y Coordinate (pixels)", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper right")

        # Invert Y axis if needed (lower Y = closer to wall)
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved wall hits plot to {output_path}")

    def evaluate(self):
        """Run full evaluation pipeline.

        Args:
            video_path: Path to video file. If None, uses config value.
        """
        # Process video (includes detection, smoothing, wall hit detection, and video annotation)
        self.process_video()

        # Save wall hit results (if any were detected)
        if self.wall_hits:
            self.save_wall_hit_results()

        # Calculate metrics
        metrics = self.calculate_metrics()
        self.print_metrics(metrics)

        # Save results
        self.save_position_plot()
        self.save_velocity_plot()
        self.save_metrics_csv(metrics)

        base_output_dir = self.config["output"]["output_dir"]
        output_dir = os.path.join(base_output_dir, self.video_name)
        print(f"\nAll results saved to: {output_dir}")
        print("Processing complete!")


def main():
    """Main entry point."""
    evaluator = BallTrackingEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
