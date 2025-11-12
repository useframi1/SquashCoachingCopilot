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
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from player_tracking import PlayerTracker
from court_calibration import CourtCalibrator
from rally_state_detection.utilities.metrics_aggregator import MetricsAggregator
from rally_state_detection.utilities.general import load_config


class AnnotationPipeline:
    def __init__(self, video_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the annotation pipeline.

        Args:
            video_path: Path to video file. If None, loads from config.
            config: Configuration dictionary. If None, loads from default config.
        """
        self.config = config if config else load_config()

        # Get video path
        if video_path:
            self.video_path = video_path
        elif "annotations" in self.config and "video_path" in self.config["annotations"]:
            self.video_path = self.config["annotations"]["video_path"]
        else:
            raise ValueError("video_path must be provided either as parameter or in config")

        self.video_name = Path(self.video_path).stem

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Current state
        self.current_frame = 0
        self.playing = False
        self.current_rally_state = "start"

        # Annotation data - store frame-by-frame metrics
        self.annotations = []

        # Initialize court calibrator and player tracker
        self.court_calibrator = CourtCalibrator()
        self.homography = None
        self.court_calibrated = False

        self.player_tracker = None  # Will be initialized after court calibration

        # Initialize metrics aggregator (for calculating distances, not for aggregation)
        # Pass config for feature engineering parameters (not used in annotation but required by signature)
        self.metrics_aggregator = MetricsAggregator(window_size=1, config=self.config)

        print(f"Loaded video: {self.video_path}")
        print(
            f"Properties: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.total_frames} frames"
        )

    def calibrate_court(self, frame):
        """Calibrate court using first frame."""
        try:
            homographies, keypoints = self.court_calibrator.process_frame(frame)
            # Use 'floor' homography as default
            self.homography = homographies.get('floor')
            if self.homography is not None:
                self.court_calibrated = True
                print("Court calibrated successfully")
            else:
                print("Warning: Could not calibrate court")
        except Exception as e:
            print(f"Court calibration failed: {e}")

    def initialize_tracker(self, frame):
        """Initialize player tracker after court calibration."""
        if self.homography is None:
            print("Warning: Initializing tracker without court calibration")

        self.player_tracker = PlayerTracker(homography=self.homography)
        print("Player tracker initialized")

    def calculate_frame_metrics(self, frame, frame_num: int) -> Dict[str, Any]:
        """
        Calculate metrics for current frame without aggregation.

        Args:
            frame: Current video frame
            frame_num: Current frame number

        Returns:
            Dictionary with frame metrics
        """
        if self.player_tracker is None:
            return None

        # Process frame with player tracker
        tracking_results = self.player_tracker.process_frame(frame)

        # Extract player positions in real-world coordinates
        player_real_coords = {}
        for player_id in [1, 2]:
            if tracking_results[player_id]["real_position"] is not None:
                real_pos = tracking_results[player_id]["real_position"]
                player_real_coords[player_id] = tuple(real_pos)

        # Calculate frame metrics using metrics aggregator
        metrics = self.metrics_aggregator.calculate_frame_metrics(player_real_coords)

        # Add frame number
        metrics["frame"] = frame_num
        metrics["timestamp"] = frame_num / self.fps

        return metrics

    def annotate_state(self, state: str):
        """Annotate current frame with given state."""
        print(f"Annotating frame {self.current_frame} as {state.upper()}")

        # Update current rally state
        self.current_rally_state = state

    def save_frame_annotation(self, frame, frame_num: int):
        """
        Save metrics for current frame without aggregation.

        Args:
            frame: Current video frame
            frame_num: Current frame number
        """
        metrics = self.calculate_frame_metrics(frame, frame_num)

        if metrics:
            # Add state annotation
            metrics["state"] = self.current_rally_state
            metrics["video_name"] = self.video_name

            self.annotations.append(metrics)
        else:
            print(f"Could not calculate metrics for frame {frame_num}")

    def seek_to_frame(self, frame_num: int):
        """Seek to specific frame."""
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num

    def draw_interface(self, frame):
        """Draw annotation interface on frame."""
        display_frame = frame.copy()

        # Create semi-transparent overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Draw text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1

        # Calibration status
        calibration_status = "yes" if self.court_calibrated else "no"

        texts = [
            f"Frame: {self.current_frame}/{self.total_frames}",
            f"Time: {self.current_frame/self.fps:.2f}s",
            f"Court Calibrated: {calibration_status}",
            f"Current State: {self.current_rally_state.upper()}",
            f"Annotations: {len(self.annotations)}",
            "",
            "Controls:",
            "s: START  a: ACTIVE  e: END",
            "SPACE: Pause  q: Quit & Save",
            "right/left: 1s   up/down: 10s",
        ]

        y_offset = 30
        for text in texts:
            if text:
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
        """Save annotations to CSV file."""
        if not self.annotations:
            print("No annotations to save")
            return

        # Create output directory
        output_dir = Path(self.config.get("annotations", {}).get("data_path", "data/annotations"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.video_name}_annotations_{timestamp}.csv"
        filepath = output_dir / filename

        # Save to CSV
        df = pd.DataFrame(self.annotations)

        # Reorder columns for better readability
        column_order = ["frame", "timestamp", "state", "video_name", "player_distance",
                       "player1_x", "player1_y", "player2_x", "player2_y"]
        df = df[[col for col in column_order if col in df.columns]]

        df.to_csv(filepath, index=False)

        print(f"\nSaved {len(self.annotations)} annotations to {filepath}")
        print("\nAnnotation summary:")
        print(df.groupby("state").size())

        # Print statistics
        print("\nMetrics summary:")
        if "player_distance" in df.columns:
            print(f"Mean player distance: {df['player_distance'].mean():.2f}m")
            print(f"Min player distance: {df['player_distance'].min():.2f}m")
            print(f"Max player distance: {df['player_distance'].max():.2f}m")

        return filepath

    def run(self):
        """Run the annotation pipeline."""
        print("\nStarting manual annotation pipeline...")
        print("Loading video and initializing tracking...")

        # Initialize on first frame
        ret, first_frame = self.cap.read()
        if ret:
            self.calibrate_court(first_frame)
            self.initialize_tracker(first_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Error: Could not read first frame")
            return

        # Main loop
        while True:
            if self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                # When paused, read current frame
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

            # Save frame-by-frame metrics with current state
            self.save_frame_annotation(frame, self.current_frame)

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
                self.seek_to_frame(self.current_frame - self.fps)
                print(f"Seeked to frame {self.current_frame}")
            elif key == 83 or key == 3:  # Right arrow
                self.seek_to_frame(self.current_frame + self.fps)
                print(f"Seeked to frame {self.current_frame}")
            elif key == 82 or key == 0:  # Up arrow
                self.seek_to_frame(self.current_frame - 10 * self.fps)
                print(f"Seeked to frame {self.current_frame}")
            elif key == 84 or key == 1:  # Down arrow
                self.seek_to_frame(self.current_frame + 10 * self.fps)
                print(f"Seeked to frame {self.current_frame}")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Save annotations
        self.save_annotations()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual annotation pipeline for rally state detection")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)

    pipeline = AnnotationPipeline(video_path=args.video, config=config)
    pipeline.run()
