#!/usr/bin/env python3
"""
Manual Annotation Pipeline for Rally Segmentation Analysis

This script allows manual annotation of rally states (start, active, end) while
calculating player distance and intensity metrics for threshold analysis.

Controls:
- s: Mark current frame as START state
- a: Mark current frame as ACTIVE state
- e: Mark current frame as END state
- SPACE: Pause/Resume video
- q: Quit and save annotations
- LEFT/RIGHT arrows: Seek backward/forward by 1 second
- UP/DOWN arrows: Seek backward/forward by 10 seconds

Output: CSV file with frame-level annotations and calculated metrics
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from collections import deque
import argparse
from datetime import datetime

from player_tracker import PlayerTracker
from court_calibrator import CourtCalibrator


class ManualAnnotationPipeline:
    def __init__(self, video_path, config_path="pipeline/config.json", window_size=50):
        self.video_path = video_path
        self.window_size = window_size
        self.video_name = Path(video_path).stem

        # Load configuration
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Current state
        self.current_frame = 0
        self.playing = False
        self.current_rally_state = "unknown"  # Track current rally state

        # Annotation data
        self.annotations = []
        self.frame_buffer = deque(
            maxlen=window_size * 2
        )  # Store extra frames for analysis
        self.auto_save_counter = 0  # Counter for automatic saving

        # Court calibration and player tracking
        self.homography = None
        self.court_calibrator = CourtCalibrator(self.config["court_calibrator"])
        self.player_tracker = None
        self.court_calibrated = False

        # Metrics storage
        self.metrics_history = deque(maxlen=window_size * 2)

        print(f"Loaded video: {video_path}")
        print(
            f"Properties: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.total_frames} frames"
        )
        print(f"Window size for analysis: {window_size} frames")

    def calibrate_court(self, frame):
        """Calibrate court using the first frame"""
        if not self.court_calibrated:
            try:
                print("Attempting court calibration...")
                self.homography = self.court_calibrator.compute_homography(frame)
                self.court_calibrated = True
                print("Court calibration successful!")
                return True
            except Exception as e:
                print(f"Court calibration failed: {e}")
                print("Using identity matrix as fallback")
                self.homography = np.eye(3, dtype=np.float32)
                self.court_calibrated = True
                return False
        return True

    def initialize_player_tracker(self, frame):
        """Initialize player tracker with calibrated homography"""
        if self.player_tracker is None:
            # Ensure court is calibrated first
            self.calibrate_court(frame)

            self.player_tracker = PlayerTracker(
                homography=self.homography, config=self.config["player_tracker"]
            )
            print("Player tracker initialized with homography")

    def calculate_frame_metrics(self, frame):
        """Calculate distance and intensity metrics for current frame"""
        if self.player_tracker is None:
            self.initialize_player_tracker(frame)

        # Process frame to get player positions
        player_coords, player_real_coords = self.player_tracker.process_frame(frame)

        metrics = {
            "frame_number": self.current_frame,
            "player_distance": None,
            "player1_intensity": None,
            "player2_intensity": None,
            "combined_intensity": None,
            "player_count": len(player_coords),
            "player1_x": None,
            "player1_y": None,
            "player2_x": None,
            "player2_y": None,
        }

        # Store current player positions
        player_ids = list(player_real_coords.keys())[:2]
        for i, player_id in enumerate(player_ids, 1):
            if len(player_real_coords[player_id]) > 0:
                pos = player_real_coords[player_id][-1]  # Latest position
                metrics[f"player{i}_x"] = pos[0]
                metrics[f"player{i}_y"] = pos[1]

        # Calculate distance between players if both detected
        if len(player_real_coords) >= 2:
            if (
                len(player_real_coords[player_ids[0]]) > 0
                and len(player_real_coords[player_ids[1]]) > 0
            ):
                pos1 = player_real_coords[player_ids[0]][-1]  # Latest position
                pos2 = player_real_coords[player_ids[1]][-1]

                distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                metrics["player_distance"] = distance

        # Calculate intensity (movement) for each player
        for i, player_id in enumerate(list(player_real_coords.keys())[:2], 1):
            if len(player_real_coords[player_id]) >= 2:
                recent_positions = player_real_coords[player_id][
                    -10:
                ]  # Last 10 positions
                if len(recent_positions) >= 2:
                    # Calculate average movement over recent frames
                    movements = []
                    for j in range(1, len(recent_positions)):
                        prev_pos = recent_positions[j - 1]
                        curr_pos = recent_positions[j]
                        movement = np.sqrt(
                            (curr_pos[0] - prev_pos[0]) ** 2
                            + (curr_pos[1] - prev_pos[1]) ** 2
                        )
                        movements.append(movement)

                    avg_intensity = np.mean(movements) if movements else 0
                    metrics[f"player{i}_intensity"] = avg_intensity

        # Calculate combined intensity
        if (
            metrics["player1_intensity"] is not None
            and metrics["player2_intensity"] is not None
        ):
            metrics["combined_intensity"] = (
                metrics["player1_intensity"] + metrics["player2_intensity"]
            )
        elif metrics["player1_intensity"] is not None:
            metrics["combined_intensity"] = metrics["player1_intensity"]
        elif metrics["player2_intensity"] is not None:
            metrics["combined_intensity"] = metrics["player2_intensity"]

        return metrics

    def calculate_window_statistics(self, state):
        """Calculate mean and median statistics for the analysis window"""
        if len(self.metrics_history) < self.window_size:
            print(
                f"Warning: Only {len(self.metrics_history)} frames available for analysis"
            )
            return None

        # Get the last window_size frames for analysis
        window_metrics = list(self.metrics_history)[-self.window_size :]

        # Extract metrics for calculation
        distances = [
            m["player_distance"]
            for m in window_metrics
            if m["player_distance"] is not None
        ]
        player1_intensities = [
            m["player1_intensity"]
            for m in window_metrics
            if m["player1_intensity"] is not None
        ]
        player2_intensities = [
            m["player2_intensity"]
            for m in window_metrics
            if m["player2_intensity"] is not None
        ]
        combined_intensities = [
            m["combined_intensity"]
            for m in window_metrics
            if m["combined_intensity"] is not None
        ]

        # Extract player positions for averaging
        player1_x_positions = [
            m["player1_x"] for m in window_metrics if m["player1_x"] is not None
        ]
        player1_y_positions = [
            m["player1_y"] for m in window_metrics if m["player1_y"] is not None
        ]
        player2_x_positions = [
            m["player2_x"] for m in window_metrics if m["player2_x"] is not None
        ]
        player2_y_positions = [
            m["player2_y"] for m in window_metrics if m["player2_y"] is not None
        ]

        stats = {
            "video_name": self.video_name,
            "frame_number": self.current_frame,
            "state": state,
            "window_size": len(window_metrics),
            "mean_distance": np.mean(distances) if distances else None,
            "median_distance": np.median(distances) if distances else None,
            "mean_player1_intensity": (
                np.mean(player1_intensities) if player1_intensities else None
            ),
            "median_player1_intensity": (
                np.median(player1_intensities) if player1_intensities else None
            ),
            "mean_player2_intensity": (
                np.mean(player2_intensities) if player2_intensities else None
            ),
            "median_player2_intensity": (
                np.median(player2_intensities) if player2_intensities else None
            ),
            "mean_combined_intensity": (
                np.mean(combined_intensities) if combined_intensities else None
            ),
            "median_combined_intensity": (
                np.median(combined_intensities) if combined_intensities else None
            ),
            # Add median player positions (more robust than averages)
            "median_player1_x": (
                np.median(player1_x_positions) if player1_x_positions else None
            ),
            "median_player1_y": (
                np.median(player1_y_positions) if player1_y_positions else None
            ),
            "median_player2_x": (
                np.median(player2_x_positions) if player2_x_positions else None
            ),
            "median_player2_y": (
                np.median(player2_y_positions) if player2_y_positions else None
            ),
            # Also keep averages for comparison if needed
            "avg_player1_x": (
                np.mean(player1_x_positions) if player1_x_positions else None
            ),
            "avg_player1_y": (
                np.mean(player1_y_positions) if player1_y_positions else None
            ),
            "avg_player2_x": (
                np.mean(player2_x_positions) if player2_x_positions else None
            ),
            "avg_player2_y": (
                np.mean(player2_y_positions) if player2_y_positions else None
            ),
        }

        return stats

    def annotate_state(self, state):
        """Annotate current frame with given state and calculate metrics"""
        print(f"Annotating frame {self.current_frame} as {state.upper()}")

        # Update current rally state
        self.current_rally_state = state

        # Calculate statistics for the window
        stats = self.calculate_window_statistics(state)

        if stats:
            self.annotations.append(stats)
            print(f"Added annotation: {state} at frame {self.current_frame}")

            # Print current statistics
            if stats["mean_distance"] is not None:
                print(
                    f"  Distance - Mean: {stats['mean_distance']:.3f}, Median: {stats['median_distance']:.3f}"
                )
            if stats["mean_combined_intensity"] is not None:
                print(
                    f"  Combined Intensity - Mean: {stats['mean_combined_intensity']:.6f}, Median: {stats['median_combined_intensity']:.6f}"
                )
            if (
                stats["median_player1_x"] is not None
                and stats["median_player1_y"] is not None
            ):
                print(
                    f"  Player 1 Median Position: ({stats['median_player1_x']:.2f}, {stats['median_player1_y']:.2f})"
                )
            if (
                stats["median_player2_x"] is not None
                and stats["median_player2_y"] is not None
            ):
                print(
                    f"  Player 2 Median Position: ({stats['median_player2_x']:.2f}, {stats['median_player2_y']:.2f})"
                )
        else:
            print(
                f"Could not calculate statistics (need at least {self.window_size} frames)"
            )

    def auto_save_metrics(self):
        """Automatically save metrics every window_size frames"""
        if len(self.metrics_history) >= self.window_size:
            stats = self.calculate_window_statistics(self.current_rally_state)
            if stats:
                self.annotations.append(stats)
                print(
                    f"Auto-saved metrics at frame {self.current_frame} (state: {self.current_rally_state})"
                )
                return True
        return False

    def seek_to_frame(self, frame_num):
        """Seek to specific frame"""
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num

    def draw_interface(self, frame):
        """Draw annotation interface on frame"""
        # Create a copy for drawing
        display_frame = frame.copy()

        # Draw semi-transparent overlay for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Draw text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1

        # Show calibration status
        calibration_status = (
            "yes" if self.court_calibrated and self.homography is not None else "no"
        )

        texts = [
            f"Frame: {self.current_frame}/{self.total_frames}",
            f"Time: {self.current_frame/self.fps:.2f}s",
            f"Court Calibrated: {calibration_status}",
            f"Current State: {self.current_rally_state.upper()}",
            f"Annotations: {len(self.annotations)}",
            f"Auto-save in: {self.window_size - self.auto_save_counter} frames",
            "",
            "Controls:",
            "s: START  a: ACTIVE  e: END",
            "SPACE: Pause  q: Quit & Save",
            "right/left: 1s   up/down: 10s",
        ]

        y_offset = 30
        for text in texts:
            if text:  # Skip empty lines
                cv2.putText(
                    display_frame,
                    text,
                    (20, y_offset),
                    font,
                    font_scale,
                    color,
                    thickness,
                )
            y_offset += 20

        return display_frame

    def save_annotations(self):
        """Save annotations to CSV file"""
        if not self.annotations:
            print("No annotations to save")
            return

        # Create output directory if it doesn't exist
        output_dir = Path("testing/annotations")
        output_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.video_name}_annotations_{timestamp}.csv"
        filepath = output_dir / filename

        # Convert to DataFrame and save
        df = pd.DataFrame(self.annotations)
        df.to_csv(filepath, index=False)

        print(f"Saved {len(self.annotations)} annotations to {filepath}")
        print("\nAnnotation summary:")
        print(df.groupby("state").size())

        return filepath

    def run(self):
        """Run the annotation pipeline"""
        print("\nStarting manual annotation pipeline...")
        print("Loading video and initializing tracking...")

        # Main loop
        while True:
            if self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                # When paused, read the current frame
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Restore position
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

            # Calculate metrics for current frame
            metrics = self.calculate_frame_metrics(frame)
            self.metrics_history.append(metrics)

            # Auto-save metrics every window_size frames
            self.auto_save_counter += 1
            if self.auto_save_counter >= self.window_size:
                self.auto_save_metrics()
                self.auto_save_counter = 0  # Reset counter

            # Draw interface
            display_frame = self.draw_interface(frame)

            # Show frame
            cv2.imshow("Manual Annotation Pipeline", display_frame)

            # Handle input
            key = cv2.waitKey(30 if self.playing else 0) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):  # Space bar
                self.playing = not self.playing
                print(f"Video {'playing' if self.playing else 'paused'}")
            elif key == ord("s"):
                self.annotate_state("start")
            elif key == ord("a"):
                self.annotate_state("active")
            elif key == ord("e"):
                self.annotate_state("end")
            elif key == 81 or key == 2:  # Left arrow
                self.seek_to_frame(self.current_frame - self.fps)  # -1 second
                print(f"Seeked to frame {self.current_frame}")
            elif key == 83 or key == 3:  # Right arrow
                self.seek_to_frame(self.current_frame + self.fps)  # +1 second
                print(f"Seeked to frame {self.current_frame}")
            elif key == 82 or key == 0:  # Up arrow
                self.seek_to_frame(self.current_frame - 10 * self.fps)  # -10 seconds
                print(f"Seeked to frame {self.current_frame}")
            elif key == 84 or key == 1:  # Down arrow
                self.seek_to_frame(self.current_frame + 10 * self.fps)  # +10 seconds
                print(f"Seeked to frame {self.current_frame}")

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

        # Save annotations
        self.save_annotations()


def main():
    parser = argparse.ArgumentParser(
        description="Manual Rally State Annotation Pipeline"
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--config", default="pipeline/config.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Number of frames to analyze for metrics calculation",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        # Create and run pipeline
        pipeline = ManualAnnotationPipeline(
            video_path=args.video_path,
            config_path=args.config,
            window_size=args.window_size,
        )
        pipeline.run()
        return 0

    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
