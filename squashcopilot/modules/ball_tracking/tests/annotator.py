"""
Ball Tracking Annotator

Interactive tool for annotating ground truth ball positions in squash videos.
Displays tracker predictions and allows users to click on the actual ball position
or skip frames where the prediction is correct.

Features:
- Frame-by-frame navigation with tracker predictions displayed
- Left-click to set ground truth ball position
- Right-click to mark "no ball visible"
- Skip (Space/Enter) to accept tracker's prediction as ground truth
- Auto-saves progress periodically and on quit
- Resume from existing annotations
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from squashcopilot.common.utils import load_config
from squashcopilot.common.types.base import Frame
from squashcopilot.common.models import BallTrackingInput
from squashcopilot.modules.ball_tracking import BallTracker
from squashcopilot.modules.court_calibration import CourtCalibrator
from squashcopilot.common.models import CourtCalibrationInput


class BallAnnotator:
    """Interactive annotator for ball position ground truth."""

    def __init__(self, config_name: str = "ball_tracking"):
        """Initialize the annotator.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        full_config = load_config(config_name=config_name)
        self.config = full_config["annotation"]

        # Get project root
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Setup paths
        video_dir = project_root / self.config["video_dir"]
        self.video_name = self.config["video_name"]
        self.video_path = video_dir / f"{self.video_name}.mp4"

        output_dir = project_root / self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = output_dir / f"{self.video_name}_ball_annotations.csv"

        # Configuration
        self.autosave_interval = self.config.get("autosave_interval", 50)
        self.show_tracker_prediction = self.config.get("show_tracker_prediction", True)

        # Initialize tracker
        self.tracker = BallTracker()
        self.court_calibrator = CourtCalibrator()

        # Video properties
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0

        # Annotation state
        self.annotations: dict = {}  # frame_number -> (ball_x, ball_y, has_ball)
        self.current_frame: int = 0
        self.current_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None  # Preprocessed image for display
        self.tracker_prediction: Optional[Tuple[float, float]] = None
        self.click_position: Optional[Tuple[int, int]] = None
        self.frames_since_save: int = 0
        self.is_black_ball: bool = False

        # Window name
        self.window_name = "Ball Annotator"

    def _load_video(self) -> None:
        """Load the video file."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _load_annotations(self) -> None:
        """Load existing annotations from CSV if available."""
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                frame_num = int(row["frame_number"])
                has_ball = row["has_ball"]
                if has_ball:
                    self.annotations[frame_num] = (
                        float(row["ball_x"]),
                        float(row["ball_y"]),
                        True
                    )
                else:
                    self.annotations[frame_num] = (None, None, False)
            print(f"Loaded {len(self.annotations)} existing annotations from {self.csv_path}")

    def _save_annotations(self) -> None:
        """Save annotations to CSV file."""
        if not self.annotations:
            print("No annotations to save!")
            return

        rows = []
        for frame_num in sorted(self.annotations.keys()):
            ball_x, ball_y, has_ball = self.annotations[frame_num]
            rows.append({
                "frame_number": frame_num,
                "ball_x": ball_x if has_ball else "",
                "ball_y": ball_y if has_ball else "",
                "has_ball": has_ball
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        print(f"Saved {len(self.annotations)} annotations to {self.csv_path}")
        self.frames_since_save = 0

    def _seek_frame(self, frame_number: int) -> bool:
        """Seek to a specific frame in the video.

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if successful, False otherwise
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return False

        self.current_frame = frame_number
        self.current_image = frame
        self.click_position = None

        # Apply preprocessing for display if black ball mode
        if self.is_black_ball:
            self.display_image = self.tracker.preprocess_frame(frame)
        else:
            self.display_image = frame

        # Get tracker prediction for this frame
        if self.show_tracker_prediction:
            ball_input = BallTrackingInput(
                frame=Frame(
                    image=frame,
                    frame_number=frame_number,
                    timestamp=frame_number / self.fps
                )
            )
            ball_output = self.tracker.process_frame(ball_input)
            if ball_output.ball_detected:
                self.tracker_prediction = (ball_output.ball_x, ball_output.ball_y)
            else:
                self.tracker_prediction = None
        else:
            self.tracker_prediction = None

        return True

    def _find_next_unannotated_frame(self) -> int:
        """Find the next frame that hasn't been annotated yet.

        Returns:
            Frame number of next unannotated frame, or current frame if all are annotated
        """
        for i in range(self.current_frame, self.total_frames):
            if i not in self.annotations:
                return i
        # If all remaining frames are annotated, stay at current
        return self.current_frame

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Mouse callback for click handling.

        Args:
            event: OpenCV mouse event type
            x: X coordinate of click
            y: Y coordinate of click
            flags: Additional flags
            param: User data
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: set ball position
            self.click_position = (x, y)
            self.annotations[self.current_frame] = (float(x), float(y), True)
            self._on_annotation_made()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: mark no ball visible
            self.click_position = None
            self.annotations[self.current_frame] = (None, None, False)
            self._on_annotation_made()

    def _on_annotation_made(self) -> None:
        """Handle post-annotation logic (autosave, advance frame)."""
        self.frames_since_save += 1

        # Autosave periodically
        if self.frames_since_save >= self.autosave_interval:
            self._save_annotations()

    def _draw_frame(self) -> np.ndarray:
        """Draw the current frame with overlays.

        Returns:
            Frame with annotations drawn
        """
        if self.display_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        display = self.display_image.copy()
        height, width = display.shape[:2]

        # Draw tracker prediction (red circle)
        if self.tracker_prediction is not None:
            pred_x, pred_y = int(self.tracker_prediction[0]), int(self.tracker_prediction[1])
            cv2.circle(display, (pred_x, pred_y), 15, (0, 0, 255), 2)
            cv2.circle(display, (pred_x, pred_y), 3, (0, 0, 255), -1)

        # Draw existing annotation or click position (green crosshair)
        annotation = self.annotations.get(self.current_frame)
        if annotation is not None:
            ball_x, ball_y, has_ball = annotation
            if has_ball and ball_x is not None and ball_y is not None:
                ax, ay = int(ball_x), int(ball_y)
                # Draw crosshair
                cv2.line(display, (ax - 20, ay), (ax + 20, ay), (0, 255, 0), 2)
                cv2.line(display, (ax, ay - 20), (ax, ay + 20), (0, 255, 0), 2)
                cv2.circle(display, (ax, ay), 5, (0, 255, 0), -1)
        elif self.click_position is not None:
            cx, cy = self.click_position
            cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
            cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)

        # Draw info overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # Calculate progress
        annotated_count = len(self.annotations)
        progress_pct = (annotated_count / self.total_frames * 100) if self.total_frames > 0 else 0

        # Add text
        y_offset = 35
        texts = [
            f"Frame: {self.current_frame}/{self.total_frames - 1}",
            f"Annotated: {annotated_count} ({progress_pct:.1f}%)",
            "",
            "Controls:",
            "  Left-click: Set ball position",
            "  Right-click: No ball visible",
            "  Space/Enter: Accept tracker prediction",
            "  B: Go back one frame | Q/Esc: Quit & Save",
        ]

        for i, text in enumerate(texts):
            color = (255, 255, 255)
            if i == 0:  # Frame number
                color = (255, 255, 0)
            elif i == 1:  # Progress
                color = (0, 255, 255)
            cv2.putText(
                display, text, (20, y_offset + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Draw annotation status
        status_y = height - 30
        if self.current_frame in self.annotations:
            _, _, has_ball = self.annotations[self.current_frame]
            if has_ball:
                status = "ANNOTATED (ball visible)"
                color = (0, 255, 0)
            else:
                status = "ANNOTATED (no ball)"
                color = (0, 165, 255)
        else:
            status = "NOT ANNOTATED"
            color = (0, 0, 255)

        cv2.putText(display, status, (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Legend
        legend_x = width - 250
        cv2.circle(display, (legend_x, 30), 8, (0, 0, 255), 2)
        cv2.putText(display, "= Tracker prediction", (legend_x + 15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(display, (legend_x - 10, 55), (legend_x + 10, 55), (0, 255, 0), 2)
        cv2.putText(display, "= Ground truth", (legend_x + 15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def run(self) -> None:
        """Run the interactive annotator."""
        print("=" * 60)
        print("Ball Position Annotator")
        print("=" * 60)
        print(f"Video: {self.video_path}")
        print(f"Output: {self.csv_path}")
        print(f"Autosave interval: {self.autosave_interval} frames")
        print("=" * 60)

        # Load video
        print("\nLoading video...")
        self._load_video()
        print(f"Video: {self.width}x{self.height} @ {self.fps:.2f} FPS, {self.total_frames} frames")

        # Load existing annotations
        self._load_annotations()

        # Calibrate court on first frame for black ball detection
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.cap.read()
        if ret:
            court_input = CourtCalibrationInput(
                frame=Frame(image=first_frame, frame_number=0, timestamp=0.0)
            )
            calibration = self.court_calibrator.process_frame(court_input)
            self.is_black_ball = calibration.is_black_ball
            self.tracker.set_is_black_ball(self.is_black_ball)
            print(f"Black ball detection: {self.is_black_ball}")

        # Find starting frame (first unannotated or 0)
        if self.annotations:
            start_frame = self._find_next_unannotated_frame()
            print(f"Resuming from frame {start_frame}")
        else:
            start_frame = 0

        # Reset tracker state before starting
        self.tracker.reset()

        # Pre-process frames to warm up TrackNet (needs 3 frames)
        for i in range(min(3, start_frame)):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret:
                ball_input = BallTrackingInput(
                    frame=Frame(image=frame, frame_number=i, timestamp=i / self.fps)
                )
                self.tracker.process_frame(ball_input)

        # Seek to starting frame
        if not self._seek_frame(start_frame):
            print("Failed to seek to starting frame!")
            return

        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print("\nStarting annotation... Press 'Q' or 'Esc' to quit and save.\n")

        try:
            while True:
                # Draw and display frame
                display = self._draw_frame()
                cv2.imshow(self.window_name, display)

                # Handle key presses
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q') or key == 27:  # Q or Esc
                    print("\nQuitting and saving...")
                    break

                elif key == ord(' ') or key == 13:  # Space or Enter
                    # Accept tracker prediction as ground truth
                    if self.tracker_prediction is not None:
                        self.annotations[self.current_frame] = (
                            self.tracker_prediction[0],
                            self.tracker_prediction[1],
                            True
                        )
                    else:
                        # No prediction means no ball
                        self.annotations[self.current_frame] = (None, None, False)
                    self._on_annotation_made()

                    # Advance to next frame
                    if self.current_frame < self.total_frames - 1:
                        self._seek_frame(self.current_frame + 1)

                elif key == ord('b') or key == ord('B'):  # Go back
                    if self.current_frame > 0:
                        self._seek_frame(self.current_frame - 1)

                elif key == ord('n') or key == ord('N'):  # Next frame (without annotating)
                    if self.current_frame < self.total_frames - 1:
                        self._seek_frame(self.current_frame + 1)

                elif key == ord('j'):  # Jump to specific frame
                    # This could be enhanced with a dialog, for now just skip 10 frames
                    target = min(self.current_frame + 10, self.total_frames - 1)
                    self._seek_frame(target)

                elif key == ord('k'):  # Jump back 10 frames
                    target = max(self.current_frame - 10, 0)
                    self._seek_frame(target)

        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
        finally:
            # Always save on exit
            self._save_annotations()

            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

        print("\nAnnotation session complete!")
        print(f"Total annotations: {len(self.annotations)}")


def main():
    """Main entry point for the annotator."""
    annotator = BallAnnotator()
    annotator.run()


if __name__ == "__main__":
    main()
