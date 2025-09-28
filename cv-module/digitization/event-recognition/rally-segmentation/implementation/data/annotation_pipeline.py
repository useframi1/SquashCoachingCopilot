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
from pathlib import Path
from datetime import datetime

from utilities.metrics_aggregator import MetricsAggregator
from config import CONFIG


class AnnotationPipeline:
    def __init__(self):
        # self.config = load_config()["annotations"]

        self.video_path = CONFIG["annotations"]["video_path"]
        self.window_size = CONFIG["window_size"]
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

        # Annotation data
        self.annotations = []
        self.auto_save_counter = 0

        # Initialize metrics aggregator
        self.metrics_aggregator = MetricsAggregator(window_size=self.window_size)

        print(f"Loaded video: {self.video_path}")
        print(
            f"Properties: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.total_frames} frames"
        )
        print(f"Window size for analysis: {self.window_size} frames")

    def annotate_state(self, state: str):
        """Annotate current frame with given state and calculate metrics."""
        print(f"Annotating frame {self.current_frame} as {state.upper()}")

        # Update current rally state
        self.current_rally_state = state

        # Get aggregated statistics
        stats = self.metrics_aggregator.get_aggregated_metrics(
            additional_data={
                "video_name": self.video_name,
                "state": state,
            }
        )

        if stats:
            self.annotations.append(stats)
            print(f"Added annotation: {state} at frame {self.current_frame}")

        else:
            print(
                f"Could not calculate statistics "
                f"(need at least {self.window_size} frames)"
            )

    def auto_save_metrics(self):
        """Automatically save metrics every window_size frames."""
        if self.metrics_aggregator.has_full_window():
            stats = self.metrics_aggregator.get_aggregated_metrics(
                additional_data={
                    "video_name": self.video_name,
                    "state": self.current_rally_state,
                }
            )
            if stats:
                self.annotations.append(stats)
                print(
                    f"Auto-saved metrics at frame {self.current_frame} "
                    f"(state: {self.current_rally_state})"
                )
                return True
        return False

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
        calibration_status = (
            "yes"
            if self.metrics_aggregator.court_calibrated
            and self.metrics_aggregator.homography is not None
            else "no"
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
        output_dir = Path(CONFIG["annotations"]["data_path"])
        output_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.video_name}_annotations_{timestamp}.csv"
        filepath = output_dir / filename

        # Save to CSV
        df = pd.DataFrame(self.annotations)
        df.to_csv(filepath, index=False)

        print(f"Saved {len(self.annotations)} annotations to {filepath}")
        print("\nAnnotation summary:")
        print(df.groupby("state").size())

        return filepath

    def run(self):
        """Run the annotation pipeline."""
        print("\nStarting manual annotation pipeline...")
        print("Loading video and initializing tracking...")

        # Initialize on first frame
        ret, first_frame = self.cap.read()
        if ret:
            self.metrics_aggregator.calibrate_court(first_frame)
            self.metrics_aggregator.initialize_tracker(first_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

            # Update metrics using aggregator
            self.metrics_aggregator.update_metrics(frame, self.current_frame)

            # Auto-save metrics every window_size frames
            self.auto_save_counter += 1
            if self.auto_save_counter >= self.window_size:
                self.auto_save_metrics()
                self.auto_save_counter = 0

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
    pipeline = AnnotationPipeline()
    pipeline.run()
