"""
Ball Tracking Evaluator

Evaluates ball tracking performance on videos and generates comprehensive reports
comparing raw and postprocessed results.
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from squashcopilot.modules.ball_tracking import (
    BallTracker,
    WallHitDetector,
    RacketHitDetector,
)
from squashcopilot.modules.court_calibration import CourtCalibrator
from squashcopilot.modules.player_tracking import PlayerTracker

from squashcopilot.common import (
    Frame,
    BallTrackingInput,
    BallPostprocessingInput,
    CourtCalibrationInput,
    WallColorDetectionInput,
    WallHitInput,
    RacketHitInput,
    PlayerTrackingInput,
    PlayerPostprocessingInput,
)
from squashcopilot.common.utils import load_config


class BallTrackingEvaluator:
    """Evaluates ball tracking performance on videos."""

    def __init__(self, config=None):
        """Initialize evaluator with configuration."""
        if config is None:
            # Load the tests section from the ball_tracking config
            full_config = load_config(config_name='ball_tracking')
            config = full_config['tests']
        self.config = config
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.join(
            self.test_dir, self.config["video"]["input_path"]
        )

        # Initialize components
        self.tracker = BallTracker()
        self.wall_hit_detector = WallHitDetector()
        self.racket_hit_detector = RacketHitDetector()
        self.court_calibrator = CourtCalibrator()
        self.player_tracker = PlayerTracker()

        # Video properties (set during processing)
        self.video_name = None
        self.fps = None
        self.width = None
        self.height = None
        self.wall_homography = None
        self.floor_homography = None

    def detect_ball_positions(self):
        """Detect ball positions in all frames.

        Returns:
            list: Raw ball positions as Point2D objects (or None)
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

        # Detect ball and track players in each frame
        positions = []
        player_positions_raw = {1: [], 2: []}
        pbar = tqdm(total=total_frames, desc="Tracking ball and players")

        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Calibrate court and detect wall color on first frame
            if frame_count == 0:
                court_input = CourtCalibrationInput(
                    frame=Frame(image=frame, frame_number=0, timestamp=0.0)
                )
                court_result = self.court_calibrator.process_frame(court_input)

                # Store homographies for hit detection
                self.wall_homography = court_result.get_homography('wall')
                self.floor_homography = court_result.get_homography('floor')

                # Set homography for player tracker
                if self.floor_homography:
                    self.player_tracker.homography = self.floor_homography.matrix

                wall_color_input = WallColorDetectionInput(
                    frame=Frame(image=frame, frame_number=0, timestamp=0.0),
                    keypoints_per_class=court_result.keypoints_per_class,
                )
                wall_color_result = self.court_calibrator.detect_wall_color(
                    wall_color_input
                )
                self.tracker.set_is_black_ball(wall_color_result.is_white)

            # Track ball using new interface
            ball_input = BallTrackingInput(
                frame=Frame(
                    image=frame,
                    frame_number=frame_count,
                    timestamp=frame_count / self.fps,
                )
            )
            ball_result = self.tracker.process_frame(ball_input)
            positions.append(ball_result.position if ball_result.detected else None)

            # Track players
            player_input = PlayerTrackingInput(
                frame=Frame(
                    image=frame,
                    frame_number=frame_count,
                    timestamp=frame_count / self.fps,
                ),
                homography=self.floor_homography
            )
            player_result = self.player_tracker.process_frame(player_input)

            for player_id in [1, 2]:
                player = player_result.get_player(player_id)
                if player:
                    player_positions_raw[player_id].append(player.position)
                else:
                    player_positions_raw[player_id].append(None)

            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        return positions, player_positions_raw

    def process_results(self, positions, player_positions_raw, apply_postprocessing=False):
        """Process positions and detect hits.

        Args:
            positions: List of Point2D ball positions (or None)
            player_positions_raw: Dict mapping player_id to list of positions
            apply_postprocessing: Whether to apply postprocessing

        Returns:
            dict: Results with positions, wall_hits, and racket_hits
        """
        wall_hits = []
        racket_hits = []

        # Apply postprocessing if requested
        if apply_postprocessing:
            # Postprocess ball trajectory
            postprocess_input = BallPostprocessingInput(positions=positions)
            trajectory = self.tracker.postprocess(postprocess_input)
            processed_positions = trajectory.positions

            # Postprocess player positions
            player_postprocess_input = PlayerPostprocessingInput(
                positions_history=player_positions_raw
            )
            player_postprocess_result = self.player_tracker.postprocess(player_postprocess_input)
            player_positions_smooth = {
                player_id: traj.positions
                for player_id, traj in player_postprocess_result.trajectories.items()
            }

            # Detect wall hits
            if self.config.get("hit_detection", {}).get("wall_hits_enabled", True):
                wall_hit_input = WallHitInput(
                    positions=processed_positions,
                    wall_homography=self.wall_homography
                )
                wall_hit_result = self.wall_hit_detector.detect(wall_hit_input)
                wall_hits = wall_hit_result.wall_hits

            # Detect racket hits
            if self.config.get("hit_detection", {}).get("racket_hits_enabled", True):
                racket_hit_input = RacketHitInput(
                    positions=processed_positions,
                    wall_hits=wall_hits,
                    player_positions=player_positions_smooth
                )
                racket_hit_result = self.racket_hit_detector.detect(racket_hit_input)
                racket_hits = racket_hit_result.racket_hits
        else:
            processed_positions = positions
            player_positions_smooth = player_positions_raw

        return {
            "positions": processed_positions,
            "wall_hits": wall_hits,
            "racket_hits": racket_hits,
            "player_positions": player_positions_smooth,
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
        detected_frames = sum(1 for p in positions if p is not None)
        detection_rate = (
            (detected_frames / total_frames * 100) if total_frames > 0 else 0
        )

        return {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate_percent": round(detection_rate, 2),
            "wall_hits_total": len(wall_hits),
            "racket_hits_total": len(racket_hits),
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

            f.write("Racket Hit Detection:\n")
            f.write(f"  Total hits:            {metrics['racket_hits_total']}\n")

    def save_positions_csv(self, results, output_dir):
        """Save ball positions with hit detections to CSV file.

        Args:
            results: Results dictionary from process_results
            output_dir: Output directory path
        """
        positions = results["positions"]
        wall_hits = results["wall_hits"]
        racket_hits = results["racket_hits"]
        player_positions = results.get("player_positions", {})
        output_path = os.path.join(output_dir, f"{self.video_name}_ball_positions.csv")

        # Create frame-to-hit lookup dictionaries
        wall_hit_by_frame = {hit.frame: hit for hit in wall_hits}
        racket_hit_by_frame = {hit.frame: hit for hit in racket_hits}

        with open(output_path, "w") as f:
            # Write header
            f.write("frame,x,y,is_wall_hit,is_racket_hit,wall_hit_x_pixel,wall_hit_y_pixel,wall_hit_x_meter,wall_hit_y_meter,racket_hit_x,racket_hit_y,racket_hit_player_id,racket_hit_player1_x,racket_hit_player1_y,racket_hit_player2_x,racket_hit_player2_y\n")

            # Write positions with hit markers
            for frame_idx, pos in enumerate(positions):
                x = pos.x if pos is not None else ""
                y = pos.y if pos is not None else ""

                # Check if this frame has a wall or racket hit
                is_wall_hit = 1 if frame_idx in wall_hit_by_frame else 0
                is_racket_hit = 1 if frame_idx in racket_hit_by_frame else 0

                # Add wall hit position data if available
                wall_hit_x_pixel = ""
                wall_hit_y_pixel = ""
                wall_hit_x_meter = ""
                wall_hit_y_meter = ""
                if frame_idx in wall_hit_by_frame:
                    wall_hit = wall_hit_by_frame[frame_idx]
                    wall_hit_x_pixel = wall_hit.position.x
                    wall_hit_y_pixel = wall_hit.position.y
                    if wall_hit.position_meter is not None:
                        wall_hit_x_meter = wall_hit.position_meter.x
                        wall_hit_y_meter = wall_hit.position_meter.y

                # Add racket hit position and player data if available
                racket_hit_x = ""
                racket_hit_y = ""
                racket_hit_player_id = ""
                racket_hit_player1_x = ""
                racket_hit_player1_y = ""
                racket_hit_player2_x = ""
                racket_hit_player2_y = ""
                if frame_idx in racket_hit_by_frame:
                    racket_hit = racket_hit_by_frame[frame_idx]
                    racket_hit_x = racket_hit.position.x
                    racket_hit_y = racket_hit.position.y
                    racket_hit_player_id = racket_hit.player_id

                    # Add player positions at the racket hit frame
                    if player_positions:
                        if 1 in player_positions and frame_idx < len(player_positions[1]):
                            player1_pos = player_positions[1][frame_idx]
                            if player1_pos is not None:
                                racket_hit_player1_x = player1_pos.x
                                racket_hit_player1_y = player1_pos.y

                        if 2 in player_positions and frame_idx < len(player_positions[2]):
                            player2_pos = player_positions[2][frame_idx]
                            if player2_pos is not None:
                                racket_hit_player2_x = player2_pos.x
                                racket_hit_player2_y = player2_pos.y

                f.write(f"{frame_idx},{x},{y},{is_wall_hit},{is_racket_hit},{wall_hit_x_pixel},{wall_hit_y_pixel},{wall_hit_x_meter},{wall_hit_y_meter},{racket_hit_x},{racket_hit_y},{racket_hit_player_id},{racket_hit_player1_x},{racket_hit_player1_y},{racket_hit_player2_x},{racket_hit_player2_y}\n")

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
        x_coords = [p.x if p is not None else np.nan for p in positions]
        y_coords = [p.y if p is not None else np.nan for p in positions]

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
            wall_frames = [hit.frame for hit in wall_hits]
            wall_y = [hit.position.y for hit in wall_hits]
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
            racket_frames = [hit.frame for hit in racket_hits]
            racket_y = [hit.position.y for hit in racket_hits]
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
        player_positions = results.get("player_positions", {})

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
        wall_hit_frames = {hit.frame: hit for hit in wall_hits}
        racket_hit_frames = {hit.frame: hit for hit in racket_hits}

        # Track active hits (show for N frames)
        hit_display_duration = 30
        active_wall_hits = []
        active_racket_hits = []

        # Track player hit indicator (show for 1 second = fps frames)
        player_hit_indicator = None  # (player_id, frames_remaining)
        player_hit_duration = int(self.fps)

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
                    if pos is not None:
                        cv2.circle(
                            frame,
                            (int(pos.x), int(pos.y)),
                            radius=0,
                            color=trace_color,
                            thickness=max(1, trace_thickness - i),
                        )

            # Draw player IDs on players
            for player_id in [1, 2]:
                if player_id in player_positions and frame_idx < len(player_positions[player_id]):
                    player_pos = player_positions[player_id][frame_idx]
                    if player_pos is not None:
                        # Player colors: Green for P1, Blue for P2
                        color = (0, 255, 0) if player_id == 1 else (255, 0, 0)

                        # Draw circle at player position
                        center = (int(player_pos.x), int(player_pos.y))
                        cv2.circle(frame, center, radius=25, color=color, thickness=3)

                        # Draw player ID text
                        text = f"P{player_id}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        text_thickness = 2

                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, font, font_scale, text_thickness
                        )

                        # Position text above the circle
                        text_x = center[0] - text_width // 2
                        text_y = center[1] - 35

                        # Draw background rectangle
                        bg_x1 = text_x - 5
                        bg_y1 = text_y - text_height - 5
                        bg_x2 = text_x + text_width + 5
                        bg_y2 = text_y + baseline + 5
                        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

                        # Draw text
                        cv2.putText(
                            frame, text, (text_x, text_y), font, font_scale, color, text_thickness
                        )

            # Add new hits to active lists
            if frame_idx in wall_hit_frames:
                active_wall_hits.append((wall_hit_frames[frame_idx], 0))
            if frame_idx in racket_hit_frames:
                active_racket_hits.append((racket_hit_frames[frame_idx], 0))
                # Set player hit indicator
                racket_hit = racket_hit_frames[frame_idx]
                player_hit_indicator = (racket_hit.player_id, player_hit_duration)

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

            # Draw player hit indicator
            if player_hit_indicator is not None:
                player_id, frames_remaining = player_hit_indicator
                self._draw_player_hit_indicator(frame, player_id)
                # Decrement counter
                frames_remaining -= 1
                if frames_remaining > 0:
                    player_hit_indicator = (player_id, frames_remaining)
                else:
                    player_hit_indicator = None

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
            pos = (int(hit.position.x), int(hit.position.y))

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
            hit_count = len([h for h in all_hits if h.frame <= current_frame])
            label = f"{label_prefix} {hit_count}"
            label_pos = (
                pos[0] + 30,
                pos[1] - 10 if label_prefix == "Wall" else pos[1] + 20,
            )
            cv2.putText(
                frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    def _draw_player_hit_indicator(self, frame, player_id):
        """Draw player hit indicator in top-left corner.

        Args:
            frame: Frame to draw on
            player_id: ID of player who hit (1 or 2)
        """
        text = f"PLAYER {player_id} HIT!"
        position = (20, 40)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        # Choose color based on player
        color = (0, 255, 0) if player_id == 1 else (255, 0, 0)  # Green for P1, Blue for P2

        # Draw background rectangle
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        bg_x1 = position[0] - 10
        bg_y1 = position[1] - text_size[1] - 10
        bg_x2 = position[0] + text_size[0] + 10
        bg_y2 = position[1] + 10

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

        # Draw text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

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

        # Step 1: Detect ball positions and track players (runs once)
        raw_positions, player_positions_raw = self.detect_ball_positions()

        # Step 2: Process raw results
        print("\nProcessing raw results...")
        raw_results = self.process_results(raw_positions, player_positions_raw, apply_postprocessing=False)

        # Step 3: Process postprocessed results
        print("Processing postprocessed results...")
        postprocessed_results = self.process_results(
            raw_positions, player_positions_raw, apply_postprocessing=True
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
