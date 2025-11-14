"""
Ball Tracking Evaluator

Evaluates ball tracking performance on videos and generates comprehensive reports
comparing raw and postprocessed results.
"""

import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ball_tracking import BallTracker, WallHitDetector, RacketHitDetector
from court_calibration import CourtCalibrator


def load_config(config_path="config.json"):
    """Load test configuration file."""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(config_dir, config_path)
    with open(full_path, "r") as f:
        return json.load(f)


class BallTrackingEvaluator:
    """Evaluates ball tracking performance on videos."""

    def __init__(self, config=None):
        """Initialize evaluator with configuration."""
        self.config = config if config is not None else load_config()
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.join(
            self.test_dir, self.config["video"]["input_path"]
        )

        # Initialize components
        self.tracker = BallTracker()
        self.wall_hit_detector = WallHitDetector()
        self.racket_hit_detector = RacketHitDetector()
        self.court_calibrator = CourtCalibrator()

        # Video properties (set during processing)
        self.video_name = None
        self.fps = None
        self.width = None
        self.height = None

    def detect_ball_positions(self):
        """Detect ball positions in all frames.

        Returns:
            list: Raw ball positions as (x, y) tuples
        """
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        cap = cv2.VideoCapture(self.video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = self.config["video"].get("max_frames")
        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nVideo: {self.video_name}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Processing {total_frames} frames\n")

        # Detect ball in each frame
        positions = []
        pbar = tqdm(total=total_frames, desc="Detecting ball")

        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count == 0:
                _, keypoints_per_class = self.court_calibrator.process_frame(frame)
                is_white_wall = self.court_calibrator.detect_wall_color(
                    frame=frame, keypoints_per_class=keypoints_per_class
                )["is_white"]
                self.tracker.set_is_black_ball(is_white_wall)

            x, y = self.tracker.process_frame(frame)
            positions.append((x, y))

            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        return positions

    def process_results(self, positions, apply_postprocessing=False):
        """Process positions and detect hits.

        Args:
            positions: List of (x, y) ball positions
            apply_postprocessing: Whether to apply postprocessing

        Returns:
            dict: Results with positions, wall_hits, and racket_hits
        """
        wall_hits = []
        racket_hits = []

        # Apply postprocessing if requested
        if apply_postprocessing:
            processed_positions = self.tracker.postprocess(positions)
            # Detect wall hits
            if self.config.get("hit_detection", {}).get("wall_hits_enabled", True):
                wall_hits = self.wall_hit_detector.detect(processed_positions)

            # Detect racket hits
            if self.config.get("hit_detection", {}).get("racket_hits_enabled", True):
                racket_hits = self.racket_hit_detector.detect(
                    processed_positions, wall_hits
                )
        else:
            processed_positions = positions

        return {
            "positions": processed_positions,
            "wall_hits": wall_hits,
            "racket_hits": racket_hits,
        }

    def calculate_metrics(self, results):
        """Calculate tracking metrics.

        Args:
            results: Results dictionary from process_results

        Returns:
            dict: Metrics dictionary
        """
        positions = results["positions"]
        wall_hits = results["wall_hits"]
        racket_hits = results["racket_hits"]

        # Detection metrics
        total_frames = len(positions)
        detected_frames = sum(1 for x, y in positions if x is not None)
        detection_rate = (
            (detected_frames / total_frames * 100) if total_frames > 0 else 0
        )

        # Hit detection metrics
        wall_hit_stats = WallHitDetector.calculate_statistics(wall_hits, self.fps)
        racket_hit_stats = RacketHitDetector.calculate_statistics(racket_hits, self.fps)

        return {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate_percent": round(detection_rate, 2),
            "wall_hits_total": wall_hit_stats["total_hits"],
            "wall_hits_avg_interval_sec": wall_hit_stats["avg_hit_interval_sec"],
            "racket_hits_total": racket_hit_stats["total_hits"],
            "racket_hits_avg_interval_sec": racket_hit_stats["avg_hit_interval_sec"],
        }

    def save_metrics(self, metrics, output_dir):
        """Save metrics to text file.

        Args:
            metrics: Metrics dictionary
            output_dir: Output directory path
        """
        output_path = os.path.join(output_dir, "metrics.txt")

        with open(output_path, "w") as f:
            f.write("BALL TRACKING METRICS\n")
            f.write("=" * 50 + "\n\n")

            f.write("Detection Performance:\n")
            f.write(f"  Total frames:          {metrics['total_frames']}\n")
            f.write(f"  Detected frames:       {metrics['detected_frames']}\n")
            f.write(
                f"  Detection rate:        {metrics['detection_rate_percent']:.2f}%\n\n"
            )

            f.write("Wall Hit Detection:\n")
            f.write(f"  Total hits:            {metrics['wall_hits_total']}\n")
            f.write(
                f"  Avg interval:          {metrics['wall_hits_avg_interval_sec']:.2f} sec\n\n"
            )

            f.write("Racket Hit Detection:\n")
            f.write(f"  Total hits:            {metrics['racket_hits_total']}\n")
            f.write(
                f"  Avg interval:          {metrics['racket_hits_avg_interval_sec']:.2f} sec\n"
            )

    def save_positions_csv(self, results, output_dir):
        """Save ball positions with hit detections to CSV file.

        Args:
            results: Results dictionary from process_results
            output_dir: Output directory path
        """
        positions = results["positions"]
        wall_hits = results["wall_hits"]
        racket_hits = results["racket_hits"]
        output_path = os.path.join(output_dir, f"{self.video_name}_ball_positions.csv")

        # Create frame-to-hit lookup dictionaries
        wall_hit_by_frame = {hit["frame"]: hit for hit in wall_hits}
        racket_hit_by_frame = {hit["frame"]: hit for hit in racket_hits}

        with open(output_path, "w") as f:
            # Write header
            f.write("frame,x,y,is_wall_hit,is_racket_hit\n")

            # Write positions with hit markers
            for frame_idx, pos in enumerate(positions):
                x = pos[0] if pos[0] is not None else ""
                y = pos[1] if pos[1] is not None else ""

                # Check if this frame has a wall or racket hit
                is_wall_hit = 1 if frame_idx in wall_hit_by_frame else 0
                is_racket_hit = 1 if frame_idx in racket_hit_by_frame else 0

                f.write(f"{frame_idx},{x},{y},{is_wall_hit},{is_racket_hit}\n")

    def save_position_plot(self, results, output_dir):
        """Save position plot with hit detections.

        Args:
            results: Results dictionary from process_results
            output_dir: Output directory path
        """
        positions = results["positions"]
        wall_hits = results["wall_hits"]
        racket_hits = results["racket_hits"]

        output_path = os.path.join(output_dir, "positions.png")
        dpi = self.config["output"]["plot_dpi"]

        # Extract coordinates
        frames = list(range(len(positions)))
        x_coords = [p[0] if p[0] is not None else np.nan for p in positions]
        y_coords = [p[1] if p[1] is not None else np.nan for p in positions]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot X coordinate
        ax1.plot(frames, x_coords, "b-", linewidth=1.5, label="X position", alpha=0.7)
        ax1.set_ylabel("X Coordinate (pixels)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_title("Ball Position Over Time", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")

        # Plot Y coordinate with hit markers
        ax2.plot(frames, y_coords, "r-", linewidth=1.5, label="Y position", alpha=0.7)

        # Add wall hits
        if wall_hits:
            wall_frames = [hit["frame"] for hit in wall_hits]
            wall_y = [hit["y"] for hit in wall_hits]
            ax2.scatter(
                wall_frames,
                wall_y,
                color="green",
                s=120,
                marker="o",
                zorder=5,
                label=f"Wall hits ({len(wall_hits)})",
                edgecolors="darkgreen",
                linewidths=2,
            )

        # Add racket hits
        if racket_hits:
            racket_frames = [hit["frame"] for hit in racket_hits]
            racket_y = [hit["y"] for hit in racket_hits]
            ax2.scatter(
                racket_frames,
                racket_y,
                color="orange",
                s=120,
                marker="*",
                zorder=5,
                label=f"Racket hits ({len(racket_hits)})",
                edgecolors="darkorange",
                linewidths=2,
            )

        ax2.set_xlabel("Frame Number", fontsize=12)
        ax2.set_ylabel("Y Coordinate (pixels)", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")
        ax2.invert_yaxis()  # Lower Y = closer to wall

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    def create_video(self, results, output_dir):
        """Create annotated video with tracking and hit markers.

        Args:
            results: Results dictionary from process_results
            output_dir: Output directory path
        """
        positions = results["positions"]
        wall_hits = results["wall_hits"]
        racket_hits = results["racket_hits"]

        output_path = os.path.join(output_dir, "tracking.mp4")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Open video
        cap = cv2.VideoCapture(self.video_path)

        # Get visualization config
        trace_length = self.config["tracking"]["trace_length"]
        trace_color = tuple(self.config["tracking"]["trace_color"])
        trace_thickness = self.config["tracking"]["trace_thickness"]

        # Create frame-to-hit lookup
        wall_hit_frames = {hit["frame"]: hit for hit in wall_hits}
        racket_hit_frames = {hit["frame"]: hit for hit in racket_hits}

        # Track active hits (show for N frames)
        hit_display_duration = 30
        active_wall_hits = []
        active_racket_hits = []

        # Process frames
        pbar = tqdm(total=len(positions), desc="Creating video")
        frame_idx = 0

        while cap.isOpened() and frame_idx < len(positions):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ball trace
            for i in range(trace_length):
                idx = frame_idx - i
                if 0 <= idx < len(positions):
                    pos = positions[idx]
                    if pos[0] is not None:
                        cv2.circle(
                            frame,
                            pos,
                            radius=0,
                            color=trace_color,
                            thickness=max(1, trace_thickness - i),
                        )

            # Add new hits to active lists
            if frame_idx in wall_hit_frames:
                active_wall_hits.append((wall_hit_frames[frame_idx], 0))
            if frame_idx in racket_hit_frames:
                active_racket_hits.append((racket_hit_frames[frame_idx], 0))

            # Draw active wall hits
            self._draw_hit_markers(
                frame,
                active_wall_hits,
                wall_hits,
                frame_idx,
                color=(0, 255, 0),
                marker_type="X",
                label_prefix="Wall",
            )

            # Draw active racket hits
            self._draw_hit_markers(
                frame,
                active_racket_hits,
                racket_hits,
                frame_idx,
                color=(255, 165, 0),
                marker_type="*",
                label_prefix="Racket",
            )

            # Remove old hits
            active_wall_hits = [
                (h, f + 1) for h, f in active_wall_hits if f < hit_display_duration
            ]
            active_racket_hits = [
                (h, f + 1) for h, f in active_racket_hits if f < hit_display_duration
            ]

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

        cap.release()
        out.release()
        pbar.close()

    def _draw_hit_markers(
        self,
        frame,
        active_hits,
        all_hits,
        current_frame,
        color,
        marker_type,
        label_prefix,
    ):
        """Draw hit markers on frame.

        Args:
            frame: Frame to draw on
            active_hits: List of (hit_dict, frames_since) tuples
            all_hits: List of all hits
            current_frame: Current frame index
            color: Marker color (BGR)
            marker_type: "X" or "*"
            label_prefix: Label text prefix
        """
        marker_size = 20
        thickness = 3

        for hit, _ in active_hits:
            pos = (int(hit["x"]), int(hit["y"]))

            # Draw marker shape
            if marker_type == "X":
                cv2.line(
                    frame,
                    (pos[0] - marker_size, pos[1] - marker_size),
                    (pos[0] + marker_size, pos[1] + marker_size),
                    color,
                    thickness,
                )
                cv2.line(
                    frame,
                    (pos[0] + marker_size, pos[1] - marker_size),
                    (pos[0] - marker_size, pos[1] + marker_size),
                    color,
                    thickness,
                )
            else:  # "*"
                # Draw 4-pointed star
                cv2.line(
                    frame,
                    (pos[0] - marker_size, pos[1] - marker_size),
                    (pos[0] + marker_size, pos[1] + marker_size),
                    color,
                    thickness,
                )
                cv2.line(
                    frame,
                    (pos[0] + marker_size, pos[1] - marker_size),
                    (pos[0] - marker_size, pos[1] + marker_size),
                    color,
                    thickness,
                )
                cv2.line(
                    frame,
                    (pos[0], pos[1] - marker_size),
                    (pos[0], pos[1] + marker_size),
                    color,
                    thickness,
                )
                cv2.line(
                    frame,
                    (pos[0] - marker_size, pos[1]),
                    (pos[0] + marker_size, pos[1]),
                    color,
                    thickness,
                )

            # Draw circle around marker
            cv2.circle(frame, pos, marker_size + 5, color, 2)

            # Add label
            hit_count = len([h for h in all_hits if h["frame"] <= current_frame])
            label = f"{label_prefix} {hit_count}"
            label_pos = (
                pos[0] + 30,
                pos[1] - 10 if label_prefix == "Wall" else pos[1] + 20,
            )
            cv2.putText(
                frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    def save_results(self, results, output_dir, label):
        """Save all results for a processing type.

        Args:
            results: Results dictionary from process_results
            output_dir: Output directory path
            label: Label for console output (e.g., "RAW" or "POSTPROCESSED")
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{label} RESULTS")
        print("=" * 70)

        # Calculate and save metrics
        metrics = self.calculate_metrics(results)
        self.save_metrics(metrics, output_dir)
        print(f"✓ Metrics saved")

        # Save ball positions CSV (only for postprocessed)
        if label == "POSTPROCESSED":
            self.save_positions_csv(results, output_dir)
            print(f"✓ Ball positions CSV saved")

        # Print key metrics
        print(f"  Detection rate: {metrics['detection_rate_percent']:.2f}%")
        print(f"  Wall hits: {metrics['wall_hits_total']}")
        print(f"  Racket hits: {metrics['racket_hits_total']}")

        # Save plot
        if self.config["output"]["save_plots"]:
            self.save_position_plot(results, output_dir)
            print(f"✓ Plot saved")

        # Save video
        if self.config["output"]["save_video"]:
            self.create_video(results, output_dir)
            print(f"✓ Video saved")

    def evaluate(self):
        """Run full evaluation pipeline."""
        print("\n" + "=" * 70)
        print("BALL TRACKING EVALUATION")
        print("=" * 70)

        # Step 1: Detect ball positions (runs once)
        raw_positions = self.detect_ball_positions()

        # Step 2: Process raw results
        print("\nProcessing raw results...")
        raw_results = self.process_results(raw_positions, apply_postprocessing=False)

        # Step 3: Process postprocessed results
        print("Processing postprocessed results...")
        postprocessed_results = self.process_results(
            raw_positions, apply_postprocessing=True
        )

        # Step 4: Save raw results
        base_output_dir = os.path.join(
            self.config["output"]["output_dir"], self.video_name
        )
        raw_output_dir = os.path.join(base_output_dir, "raw")
        self.save_results(raw_results, raw_output_dir, "RAW")

        # Step 5: Save postprocessed results
        postprocessed_output_dir = os.path.join(base_output_dir, "postprocessed")
        self.save_results(
            postprocessed_results, postprocessed_output_dir, "POSTPROCESSED"
        )

        # Done
        print("\n" + "=" * 70)
        print(f"✓ All results saved to: {base_output_dir}")
        print("=" * 70)
        print("Evaluation complete!\n")


def main():
    """Main entry point."""
    evaluator = BallTrackingEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
