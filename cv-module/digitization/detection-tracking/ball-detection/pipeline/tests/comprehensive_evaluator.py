"""
Comprehensive Ball Tracking and Shot Classification Evaluator

Combines ball tracking, hit detection, and shot classification with
comprehensive visualizations including annotated videos and plots.
"""

import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from ball_tracking import (
    BallTracker,
    WallHitDetector,
    RacketHitDetector,
    ShotClassifier,
    ShotType,
)
from court_calibration import CourtCalibrator


def load_config(config_path="config.json"):
    """Load test configuration file"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(config_dir, config_path)
    with open(full_path, "r") as f:
        return json.load(f)


class ComprehensiveEvaluator:
    """Comprehensive evaluator for ball tracking and shot classification"""

    def __init__(self, config=None):
        self.config = config if config is not None else load_config()

        # Get test directory path
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve paths
        self.video_path = os.path.join(
            self.test_dir, self.config["video"]["input_path"]
        )

        # Get video name
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        # Create output directories: tests/outputs/{video_name}/raw and /postprocessed
        base_output_dir = os.path.join(
            self.test_dir, self.config["output"]["output_dir"]
        )
        self.video_output_dir = os.path.join(base_output_dir, self.video_name)
        self.raw_output_dir = os.path.join(self.video_output_dir, "raw")
        self.postprocessed_output_dir = os.path.join(
            self.video_output_dir, "postprocessed"
        )

        # Ensure output directories exist
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.postprocessed_output_dir, exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\nVideo: {self.video_name}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Output directory: {self.video_output_dir}")

        # Initialize components
        print("\nInitializing components...")
        self.ball_tracker = BallTracker()
        self.wall_detector = WallHitDetector()
        self.racket_detector = RacketHitDetector()
        self.shot_classifier = ShotClassifier(fps=self.fps)
        self.court_calibrator = CourtCalibrator()

        print("✓ Ball tracker initialized")
        print("✓ Wall hit detector initialized")
        print("✓ Racket hit detector initialized")
        print("✓ Shot classifier initialized")
        print("✓ Court calibrator initialized")

    def process_video(self):
        """Process video to extract ball trajectory"""
        print("\n" + "=" * 60)
        print("PROCESSING VIDEO")
        print("=" * 60)

        cap = cv2.VideoCapture(self.video_path)
        max_frames = self.config["video"].get("max_frames")

        # Determine total frames for progress bar
        if max_frames:
            total_frames = max_frames
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ball_positions = []

        # Process frames with progress bar
        with tqdm(total=total_frames, desc="Tracking ball", unit="frame") as pbar:
            frame_count = 0

            while cap.isOpened():
                if max_frames and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Detect wall color on first frame to set ball color
                if frame_count == 0:
                    _, keypoints = self.court_calibrator.process_frame(frame)
                    wall_info = self.court_calibrator.detect_wall_color(
                        frame, keypoints
                    )
                    is_black_ball = wall_info["recommended_ball"] == "black"
                    self.ball_tracker.set_is_black_ball(is_black_ball)
                    print(f"\nDetected {wall_info['recommended_ball']} ball")

                # Track ball
                x, y = self.ball_tracker.process_frame(frame)
                ball_positions.append((x, y))

                frame_count += 1
                pbar.update(1)

        cap.release()

        detected_count = sum(1 for pos in ball_positions if pos[0] is not None)
        detection_rate = (detected_count / frame_count * 100) if frame_count > 0 else 0

        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Detected ball in {detected_count} frames ({detection_rate:.1f}%)")

        return ball_positions

    def detect_hits_from_positions(self, positions, label=""):
        """Detect hits from given positions"""
        # Detect wall hits
        wall_hits = []
        if self.config["hit_detection"]["wall_hits_enabled"]:
            wall_hits = self.wall_detector.detect(positions)
            if label:
                print(f"✓ Detected {len(wall_hits)} wall hits ({label})")

        # Detect racket hits
        racket_hits = []
        if self.config["hit_detection"]["racket_hits_enabled"]:
            racket_hits = self.racket_detector.detect(positions, wall_hits)
            if label:
                print(f"✓ Detected {len(racket_hits)} racket hits ({label})")

        return wall_hits, racket_hits

    def postprocess_and_detect_hits(self, ball_positions):
        """Process both raw and postprocessed trajectories"""
        print("\n" + "=" * 60)
        print("PROCESSING RAW & POSTPROCESSED RESULTS")
        print("=" * 60)

        # Raw positions - NO hit detection for raw
        print("\nPreparing raw positions (no hit detection)...")

        # Postprocess trajectory
        print("\nPostprocessing trajectory...")
        processed_positions = self.ball_tracker.postprocess(ball_positions)
        print("✓ Trajectory smoothed and interpolated")

        # Process postprocessed positions - detect hits only here
        print("\nDetecting hits from postprocessed positions...")
        wall_hits, racket_hits = self.detect_hits_from_positions(
            processed_positions, label="postprocessed"
        )

        return {
            "raw": {
                "positions": ball_positions,
                "wall_hits": [],  # No hits for raw
                "racket_hits": [],  # No hits for raw
            },
            "postprocessed": {
                "positions": processed_positions,
                "wall_hits": wall_hits,
                "racket_hits": racket_hits,
            },
        }

    def classify_shots(self, ball_positions, wall_hits, racket_hits):
        """Classify shots using shot classifier"""
        print("\n" + "=" * 60)
        print("CLASSIFYING SHOTS")
        print("=" * 60)

        shots = self.shot_classifier.classify(
            ball_positions=ball_positions,
            wall_hits=wall_hits,
            racket_hits=racket_hits,
        )

        print(f"✓ Classified {len(shots)} shots")

        return shots

    def print_statistics(self, ball_positions, wall_hits, racket_hits, shots):
        """Print comprehensive statistics"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE STATISTICS")
        print("=" * 60)

        # Ball tracking stats
        total_frames = len(ball_positions)
        detected_frames = sum(1 for pos in ball_positions if pos[0] is not None)
        detection_rate = (
            (detected_frames / total_frames * 100) if total_frames > 0 else 0
        )

        print("\nBall Tracking:")
        print(f"   Total frames: {total_frames}")
        print(f"   Detected frames: {detected_frames}")
        print(f"   Detection rate: {detection_rate:.1f}%")

        # Hit detection stats
        print("\nHit Detection:")
        print(f"   Wall hits: {len(wall_hits)}")
        print(f"   Racket hits: {len(racket_hits)}")

        # Shot classification stats
        if shots:
            stats = self.shot_classifier.get_statistics(shots)

            print(f"\nShot Classification:")
            print(f"   Total shots: {stats['total_shots']}")

            print("\n   By Type:")
            for shot_type, count in sorted(stats["by_type"].items()):
                percentage = (count / stats["total_shots"]) * 100
                print(
                    f"      {shot_type.replace('_', ' ').title():25s}: {count:3d} ({percentage:5.1f}%)"
                )

            print("\n   By Direction:")
            for direction, count in sorted(stats["by_direction"].items()):
                percentage = (count / stats["total_shots"]) * 100
                print(
                    f"      {direction.replace('_', ' ').title():25s}: {count:3d} ({percentage:5.1f}%)"
                )

            print("\n   By Depth:")
            for depth, count in sorted(stats["by_depth"].items()):
                percentage = (count / stats["total_shots"]) * 100
                print(f"      {depth.title():25s}: {count:3d} ({percentage:5.1f}%)")

            # Wall hit detection statistics (if available)
            if "wall_hit_detection_rate" in stats:
                print(
                    f"\n   Wall hit detection rate: {stats['wall_hit_detection_rate']*100:.1f}%"
                )
                print(f"   Average vector angle: {stats['average_vector_angle']:.1f}°")
                print(
                    f"   Average rebound distance: {stats['average_rebound_distance']:.1f} px"
                )

        return {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": detection_rate,
            "wall_hits": len(wall_hits),
            "racket_hits": len(racket_hits),
            "shots": len(shots),
        }

    def create_raw_trajectory_plots(self, ball_positions, output_dir):
        """Create simple trajectory plots for raw data (no hits or shots)"""
        if not self.config["output"]["save_plots"]:
            return

        # Extract coordinates
        frames = list(range(len(ball_positions)))
        xs = [
            pos[0] if pos and pos[0] is not None else np.nan for pos in ball_positions
        ]
        ys = [
            pos[1] if pos and pos[1] is not None else np.nan for pos in ball_positions
        ]

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: X trajectory
        axes[0].plot(frames, xs, "b-", linewidth=1.5)
        axes[0].set_ylabel("X Position (pixels)", fontsize=11)
        axes[0].set_title(
            "Ball Trajectory - X Coordinate (Raw)", fontsize=12, fontweight="bold"
        )
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Y trajectory
        axes[1].plot(frames, ys, "r-", linewidth=1.5)
        axes[1].set_xlabel("Frame", fontsize=11)
        axes[1].set_ylabel("Y Position (pixels)", fontsize=11)
        axes[1].set_title(
            "Ball Trajectory - Y Coordinate (Raw)", fontsize=12, fontweight="bold"
        )
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "trajectory_plots.png")
        plt.savefig(
            plot_path, dpi=self.config["output"]["plot_dpi"], bbox_inches="tight"
        )
        print(f"✓ Saved raw trajectory plots: {plot_path}")

        plt.close()

    def create_trajectory_plots(
        self, ball_positions, wall_hits, racket_hits, shots, output_dir, label=""
    ):
        """Create comprehensive trajectory plots"""
        if not self.config["output"]["save_plots"]:
            return

        # Extract coordinates
        frames = list(range(len(ball_positions)))
        xs = [
            pos[0] if pos and pos[0] is not None else np.nan for pos in ball_positions
        ]
        ys = [
            pos[1] if pos and pos[1] is not None else np.nan for pos in ball_positions
        ]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Define colors for shot types
        shot_type_colors = {
            ShotType.STRAIGHT_DRIVE: "#0066CC",
            ShotType.STRAIGHT_DROP: "#66B2FF",
            ShotType.CROSS_COURT_DRIVE: "#CC0000",
            ShotType.CROSS_COURT_DROP: "#FF6666",
            ShotType.DOWN_LINE_DRIVE: "#00CC00",
            ShotType.DOWN_LINE_DROP: "#66FF66",
        }

        # Plot 1: X trajectory
        axes[0].plot(frames, xs, "k-", alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel("X Position (pixels)", fontsize=11)
        axes[0].set_title(
            "Ball Trajectory - X Coordinate with Shot Classifications",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Y trajectory
        axes[1].plot(frames, ys, "k-", alpha=0.3, linewidth=0.5)
        axes[1].set_ylabel("Y Position (pixels)", fontsize=11)
        axes[1].set_title(
            "Ball Trajectory - Y Coordinate with Shot Classifications",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].grid(True, alpha=0.3)

        # Create lookup for next racket hit by finding next hit after current
        racket_hit_frames = sorted([hit["frame"] for hit in racket_hits])

        # Mark shots on both trajectories
        for shot in shots:
            color = shot_type_colors.get(shot.shot_type, "gray")

            # Mark racket hit
            axes[0].axvline(shot.frame, color=color, alpha=0.6, linewidth=2)
            axes[1].axvline(shot.frame, color=color, alpha=0.6, linewidth=2)

            # Find next racket hit frame
            next_racket_frames = [f for f in racket_hit_frames if f > shot.frame]
            if next_racket_frames:
                next_racket_frame = next_racket_frames[0]

                # Draw shot segment (from racket hit to next racket hit)
                segment_frames = range(
                    shot.frame, min(next_racket_frame + 1, len(frames))
                )

                if segment_frames:
                    segment_xs = [xs[f] for f in segment_frames]
                    segment_ys = [ys[f] for f in segment_frames]
                    axes[0].plot(
                        segment_frames,
                        segment_xs,
                        color=color,
                        linewidth=2.5,
                        alpha=0.8,
                    )
                    axes[1].plot(
                        segment_frames,
                        segment_ys,
                        color=color,
                        linewidth=2.5,
                        alpha=0.8,
                    )

        # Plot 3: Shot type timeline
        axes[2].set_ylim(-0.5, 5.5)
        axes[2].set_xlabel("Frame", fontsize=11)
        axes[2].set_ylabel("Shot Type", fontsize=11)
        axes[2].set_title("Shot Type Timeline", fontsize=12, fontweight="bold")
        axes[2].grid(True, alpha=0.3, axis="x")

        # Create shot type index mapping
        shot_type_index = {
            ShotType.STRAIGHT_DRIVE: 0,
            ShotType.STRAIGHT_DROP: 1,
            ShotType.CROSS_COURT_DRIVE: 2,
            ShotType.CROSS_COURT_DROP: 3,
            ShotType.DOWN_LINE_DRIVE: 4,
            ShotType.DOWN_LINE_DROP: 5,
        }

        # Plot each shot as a horizontal bar
        for shot in shots:
            y_pos = shot_type_index.get(shot.shot_type, 0)
            color = shot_type_colors.get(shot.shot_type, "gray")

            # Find next racket hit frame to calculate width
            next_racket_frames = [f for f in racket_hit_frames if f > shot.frame]
            if next_racket_frames:
                width = next_racket_frames[0] - shot.frame

                axes[2].barh(
                    y_pos,
                    width=width,
                    left=shot.frame,
                    height=0.7,
                    color=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Set y-axis labels
        axes[2].set_yticks(list(shot_type_index.values()))
        axes[2].set_yticklabels(
            [st.name.replace("_", " ").title() for st in shot_type_index.keys()],
            fontsize=9,
        )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=color,
                edgecolor="black",
                label=st.name.replace("_", " ").title(),
            )
            for st, color in shot_type_colors.items()
        ]
        axes[2].legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, "comprehensive_plots.png")
        plt.savefig(
            plot_path, dpi=self.config["output"]["plot_dpi"], bbox_inches="tight"
        )
        if label:
            print(f"✓ Saved {label} trajectory plots: {plot_path}")

        plt.close()

    def create_raw_video(self, ball_positions, output_dir):
        """Create simple video with ball trace only (no hits or shots)"""
        if not self.config["output"]["save_video"]:
            return

        output_path = os.path.join(output_dir, "annotated.mp4")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Open video
        cap = cv2.VideoCapture(self.video_path)

        # Get visualization config
        trace_length = self.config["tracking"]["trace_length"]
        trace_color = tuple(self.config["tracking"]["trace_color"])
        trace_thickness = self.config["tracking"]["trace_thickness"]

        # Process frames
        max_frames = self.config["video"].get("max_frames", len(ball_positions))
        with tqdm(total=max_frames, desc="Rendering raw video", unit="frame") as pbar:
            frame_idx = 0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw ball trace only
                for i in range(trace_length):
                    idx = frame_idx - i
                    if 0 <= idx < len(ball_positions):
                        pos = ball_positions[idx]
                        if pos[0] is not None:
                            cv2.circle(
                                frame,
                                (int(pos[0]), int(pos[1])),
                                radius=max(1, trace_thickness - i // 2),
                                color=trace_color,
                                thickness=-1,
                            )

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        print(f"✓ Saved raw video: {output_path}")

    def create_annotated_video(
        self, ball_positions, wall_hits, racket_hits, shots, output_dir, label=""
    ):
        """Create annotated video with ball tracking and shot classifications"""
        if not self.config["output"]["save_video"]:
            return

        output_path = os.path.join(output_dir, "annotated.mp4")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Open video
        cap = cv2.VideoCapture(self.video_path)

        # Get visualization config
        trace_length = self.config["tracking"]["trace_length"]
        trace_color = tuple(self.config["tracking"]["trace_color"])
        trace_thickness = self.config["tracking"]["trace_thickness"]

        # Create frame-to-shot lookup
        # First create racket hit frame lookup
        racket_hit_frames_list = sorted([hit["frame"] for hit in racket_hits])

        shot_by_frame = {}
        for shot in shots:
            # Find next racket hit frame
            next_racket_frames = [f for f in racket_hit_frames_list if f > shot.frame]
            if next_racket_frames:
                end_frame = next_racket_frames[0]
                for frame_idx in range(shot.frame, end_frame + 1):
                    if frame_idx < len(ball_positions):
                        shot_by_frame[frame_idx] = shot

        # Shot type colors (BGR for OpenCV)
        shot_colors = {
            ShotType.STRAIGHT_DRIVE: (204, 102, 0),  # Blue
            ShotType.STRAIGHT_DROP: (255, 178, 102),  # Light blue
            ShotType.CROSS_COURT_DRIVE: (0, 0, 204),  # Red
            ShotType.CROSS_COURT_DROP: (102, 102, 255),  # Light red
            ShotType.DOWN_LINE_DRIVE: (0, 204, 0),  # Green
            ShotType.DOWN_LINE_DROP: (102, 255, 102),  # Light green
        }

        # Create frame-to-hit lookup (with display duration)
        hit_display_duration = (
            30  # Show hit markers for 15 frames (~0.5 seconds at 30fps)
        )

        wall_hit_frames = {}
        for hit in wall_hits:
            for offset in range(hit_display_duration):
                frame = hit["frame"] + offset
                if frame < len(ball_positions):
                    wall_hit_frames[frame] = hit

        racket_hit_frames = {}
        for hit in racket_hits:
            for offset in range(hit_display_duration):
                frame = hit["frame"] + offset
                if frame < len(ball_positions):
                    racket_hit_frames[frame] = hit

        # Process frames
        max_frames = self.config["video"].get("max_frames", len(ball_positions))
        with tqdm(total=max_frames, desc="Rendering video", unit="frame") as pbar:
            frame_idx = 0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw ball trace
                current_shot = shot_by_frame.get(frame_idx)
                if current_shot:
                    trace_col = shot_colors.get(current_shot.shot_type, trace_color)
                else:
                    trace_col = trace_color

                for i in range(trace_length):
                    idx = frame_idx - i
                    if 0 <= idx < len(ball_positions):
                        pos = ball_positions[idx]
                        if pos[0] is not None:
                            cv2.circle(
                                frame,
                                (int(pos[0]), int(pos[1])),
                                radius=max(1, trace_thickness - i // 2),
                                color=trace_col,
                                thickness=-1,
                            )

                # Draw current shot info overlay
                if current_shot:
                    shot_text = current_shot.shot_type.name.replace("_", " ").title()
                    direction_text = (
                        f"Dir: {current_shot.direction.name.replace('_', ' ').title()}"
                    )
                    depth_text = f"Depth: {current_shot.depth.name.title()}"
                    angle_text = (
                        f"Angle: {current_shot.vector_angle_deg:.1f}°"
                        if current_shot.vector_angle_deg is not None
                        else "Angle: N/A"
                    )

                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                    # Draw text
                    color = shot_colors.get(current_shot.shot_type, (255, 255, 255))
                    cv2.putText(
                        frame,
                        shot_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        direction_text,
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        depth_text,
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        angle_text,
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # Draw hit markers
                if frame_idx in wall_hit_frames:
                    hit = wall_hit_frames[frame_idx]
                    pos = (int(hit["x"]), int(hit["y"]))
                    cv2.drawMarker(frame, pos, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
                    cv2.putText(
                        frame,
                        "WALL",
                        (pos[0] + 20, pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                if frame_idx in racket_hit_frames:
                    hit = racket_hit_frames[frame_idx]
                    pos = (int(hit["x"]), int(hit["y"]))
                    cv2.drawMarker(frame, pos, (0, 165, 255), cv2.MARKER_STAR, 30, 3)
                    cv2.putText(
                        frame,
                        "RACKET",
                        (pos[0] + 20, pos[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        if label:
            print(f"✓ Saved {label} annotated video: {output_path}")

    def save_raw_metrics(self, ball_positions, output_dir):
        """Save simple tracking metrics for raw data"""
        if not self.config["output"]["save_metrics"]:
            return

        output_path = os.path.join(output_dir, "metrics.txt")

        total_frames = len(ball_positions)
        detected_frames = sum(1 for pos in ball_positions if pos[0] is not None)
        detection_rate = (
            (detected_frames / total_frames * 100) if total_frames > 0 else 0
        )

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("RAW BALL TRACKING METRICS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video: {self.video_name}\n")
            f.write(f"Resolution: {self.width}x{self.height} @ {self.fps} fps\n\n")

            f.write("TRACKING STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total frames:        {total_frames}\n")
            f.write(f"Detected frames:     {detected_frames}\n")
            f.write(f"Detection rate:      {detection_rate:.1f}%\n\n")

            f.write("NOTE: This is raw tracking data without postprocessing.\n")
            f.write("No hit detection or shot classification performed.\n")

        print(f"✓ Saved raw metrics: {output_path}")

    def save_detailed_report(self, shots, stats, output_dir, label=""):
        """Save detailed text report"""
        if not self.config["output"]["save_metrics"]:
            return

        output_path = os.path.join(output_dir, "report.txt")

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE BALL TRACKING & SHOT CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video: {self.video_name}\n")
            f.write(f"Resolution: {self.width}x{self.height} @ {self.fps} fps\n\n")

            f.write("TRACKING STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total frames:        {stats['total_frames']}\n")
            f.write(f"Detected frames:     {stats['detected_frames']}\n")
            f.write(f"Detection rate:      {stats['detection_rate']:.1f}%\n")
            f.write(f"Wall hits:           {stats['wall_hits']}\n")
            f.write(f"Racket hits:         {stats['racket_hits']}\n")
            f.write(f"Classified shots:    {stats['shots']}\n\n")

            if shots:
                f.write("SHOT-BY-SHOT BREAKDOWN:\n")
                f.write("-" * 80 + "\n\n")

                for i, shot in enumerate(shots, 1):
                    f.write(f"Shot {i}:\n")
                    f.write(f"  Frame:              {shot.frame}\n")
                    f.write(
                        f"  Type:               {shot.shot_type.name.replace('_', ' ').title()}\n"
                    )
                    f.write(
                        f"  Direction:          {shot.direction.name.replace('_', ' ').title()}\n"
                    )
                    f.write(f"  Depth:              {shot.depth.name.title()}\n")
                    f.write(
                        f"  Racket hit pos:     ({shot.racket_hit_pos[0]:.0f}, {shot.racket_hit_pos[1]:.0f})\n"
                    )
                    f.write(
                        f"  Next racket pos:    ({shot.next_racket_hit_pos[0]:.0f}, {shot.next_racket_hit_pos[1]:.0f})\n"
                    )
                    f.write(f"  Confidence:         {shot.confidence:.2f}\n")

                    # Add wall hit information if available
                    if shot.wall_hit_pos is not None:
                        f.write(
                            f"  Wall hit:           ({shot.wall_hit_pos[0]:.0f}, {shot.wall_hit_pos[1]:.0f}) at frame {shot.wall_hit_frame}\n"
                        )
                        f.write(f"  Vector angle:       {shot.vector_angle_deg:.1f}°\n")
                        f.write(
                            f"  Rebound distance:   {shot.rebound_distance:.0f} px\n"
                        )

                    f.write("\n")

        if label:
            print(f"✓ Saved {label} detailed report: {output_path}")

    def run_evaluation(self):
        """Run complete comprehensive evaluation pipeline"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION PIPELINE")
        print("=" * 60)

        # 1. Process video
        ball_positions = self.process_video()

        # 2. Process both raw and postprocessed trajectories
        results = self.postprocess_and_detect_hits(ball_positions)

        # 3. Process RAW results - simple trajectory only, no hits or shots
        print("\n" + "=" * 60)
        print("PROCESSING RAW RESULTS")
        print("=" * 60)
        print(
            "(Raw version: trajectory plots, metrics, and video only - no hit/shot detection)"
        )

        # Calculate basic stats
        total_frames = len(results["raw"]["positions"])
        detected_frames = sum(
            1 for pos in results["raw"]["positions"] if pos[0] is not None
        )
        detection_rate = (
            (detected_frames / total_frames * 100) if total_frames > 0 else 0
        )

        print(f"\nRaw tracking statistics:")
        print(f"  Total frames:     {total_frames}")
        print(f"  Detected frames:  {detected_frames}")
        print(f"  Detection rate:   {detection_rate:.1f}%")

        print("\nCreating RAW visualizations...")
        self.create_raw_trajectory_plots(
            results["raw"]["positions"], self.raw_output_dir
        )
        self.create_raw_video(results["raw"]["positions"], self.raw_output_dir)
        self.save_raw_metrics(results["raw"]["positions"], self.raw_output_dir)

        # 4. Process POSTPROCESSED results
        print("\n" + "=" * 60)
        print("PROCESSING POSTPROCESSED RESULTS")
        print("=" * 60)

        pp_shots = self.classify_shots(
            results["postprocessed"]["positions"],
            results["postprocessed"]["wall_hits"],
            results["postprocessed"]["racket_hits"],
        )

        pp_stats = self.print_statistics(
            results["postprocessed"]["positions"],
            results["postprocessed"]["wall_hits"],
            results["postprocessed"]["racket_hits"],
            pp_shots,
        )

        print("\nCreating POSTPROCESSED visualizations...")
        self.create_trajectory_plots(
            results["postprocessed"]["positions"],
            results["postprocessed"]["wall_hits"],
            results["postprocessed"]["racket_hits"],
            pp_shots,
            self.postprocessed_output_dir,
            label="postprocessed",
        )

        self.create_annotated_video(
            results["postprocessed"]["positions"],
            results["postprocessed"]["wall_hits"],
            results["postprocessed"]["racket_hits"],
            pp_shots,
            self.postprocessed_output_dir,
            label="postprocessed",
        )

        self.save_detailed_report(
            pp_shots, pp_stats, self.postprocessed_output_dir, label="postprocessed"
        )

        # 5. Print comparison
        print("\n" + "=" * 60)
        print("RAW vs POSTPROCESSED COMPARISON")
        print("=" * 60)
        print(f"\nDetection Rate:")
        print(f"  Raw:           {detection_rate:.1f}%")
        print(f"  Postprocessed: {pp_stats['detection_rate']:.1f}%")
        print(f"\nWall Hits:")
        print(f"  Raw:           0 (not computed)")
        print(f"  Postprocessed: {pp_stats['wall_hits']}")
        print(f"\nRacket Hits:")
        print(f"  Raw:           0 (not computed)")
        print(f"  Postprocessed: {pp_stats['racket_hits']}")
        print(f"\nClassified Shots:")
        print(f"  Raw:           0 (not computed)")
        print(f"  Postprocessed: {pp_stats['shots']}")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to:")
        print(f"  Raw:           {self.raw_output_dir}")
        print(f"                 (trajectory plots, metrics, video only)")
        print(f"  Postprocessed: {self.postprocessed_output_dir}")
        print(f"                 (full analysis with hits and shot classification)")

        return {
            "raw": {
                "positions": results["raw"]["positions"],
                "detection_rate": detection_rate,
            },
            "postprocessed": {
                "positions": results["postprocessed"]["positions"],
                "wall_hits": results["postprocessed"]["wall_hits"],
                "racket_hits": results["postprocessed"]["racket_hits"],
                "shots": pp_shots,
                "stats": pp_stats,
            },
        }


if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_evaluation()
