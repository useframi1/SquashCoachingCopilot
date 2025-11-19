"""
Rally State Annotator

This module provides an interactive video annotator for labeling rally states (start/end).
Users can play a video and press 's' for start or 'e' for end to annotate frames.
The annotations are saved to a CSV file with the selected features and labels.
"""

import cv2
import pandas as pd
from pathlib import Path

from squashcopilot.common.utils import load_config


class RallyStateAnnotator:
    """Interactive annotator for rally state detection."""

    def __init__(self, config_name: str = "rally_state_detection"):
        """
        Initialize the annotator.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        self.config = load_config(config_name=config_name)

        # Get project root (parent of squashcopilot/)
        project_root = Path(__file__).parent.parent.parent.parent
        self.annotations_dir = (
            project_root / self.config["annotation"]["annotations_dir"]
        )
        self.video_name = self.config["annotation"]["video_name"]
        self.features = self.config["annotation"]["features"]
        self.label_column = self.config["annotation"]["label_column"]
        self.labels = self.config["annotation"]["labels"]

        # Build paths
        # CSV path from annotations directory
        csv_dir = self.annotations_dir / self.video_name
        self.csv_path = csv_dir / f"{self.video_name}_annotations.csv"

        # Video path from video directory
        video_base_dir = project_root / self.config["annotation"]["video_dir"]
        self.video_path = video_base_dir / f"{self.video_name}.mp4"

        # Output path from config
        output_base_dir = project_root / self.config["annotation"]["output_dir"]
        output_dir = output_base_dir / self.video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"{self.video_name}_annotations.csv"

        # Annotation state
        self.current_label = self.labels["start"]  # Default to 'start'
        self.frame_annotations = {}  # frame_number -> label
        self.paused = False
        self.quit_requested = False

        # Load CSV data
        self.df = None
        self.cap = None

    def _load_csv(self) -> pd.DataFrame:
        """Load the annotations CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        return pd.read_csv(self.csv_path)

    def _load_video(self) -> cv2.VideoCapture:
        """Load the video file."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        return cap

    def _draw_overlay(self, frame, frame_number: int) -> None:
        """
        Draw annotation information overlay on the frame.

        Args:
            frame: The video frame
            frame_number: Current frame number
        """
        height, width = frame.shape[:2]

        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Add text information
        y_offset = 35
        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y_offset += 30
        cv2.putText(
            frame,
            f"Current Label: {self.current_label}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self.current_label == "start" else (0, 0, 255),
            2,
        )
        y_offset += 30
        cv2.putText(
            frame,
            "Press 's' for START | 'e' for END",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 25
        cv2.putText(
            frame,
            "Press 'p' to PAUSE | 'q' to QUIT & SAVE",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def _save_annotations(self) -> None:
        """Save annotations to CSV file."""
        if not self.frame_annotations:
            print("No annotations to save!")
            return

        # Create output dataframe
        output_rows = []

        for frame_num in sorted(self.frame_annotations.keys()):
            row_data = {"frame": frame_num}

            # Add features from original CSV
            if frame_num < len(self.df):
                for feature in self.features:
                    if feature in self.df.columns:
                        row_data[feature] = self.df.loc[frame_num, feature]
                    else:
                        print(f"Warning: Feature '{feature}' not found in CSV")
                        row_data[feature] = None

            # Add label
            row_data[self.label_column] = self.frame_annotations[frame_num]

            output_rows.append(row_data)

        # Create DataFrame and save
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(self.output_path, index=False)
        print(f"\nAnnotations saved to: {self.output_path}")
        print(f"Total frames annotated: {len(self.frame_annotations)}")

    def run(self) -> None:
        """Run the interactive annotator."""
        print("=" * 60)
        print("Rally State Annotator")
        print("=" * 60)
        print(f"Video: {self.video_path}")
        print(f"CSV: {self.csv_path}")
        print(f"Output: {self.output_path}")
        print(f"Features: {', '.join(self.features)}")
        print("=" * 60)
        print("\nControls:")
        print("  's' - Switch to START labeling mode")
        print("  'e' - Switch to END labeling mode")
        print("  'p' - Pause/Resume video")
        print("  'q' - Quit and save annotations")
        print("=" * 60)

        # Load data
        print("\nLoading CSV data...")
        self.df = self._load_csv()
        print(f"Loaded {len(self.df)} frames from CSV")

        print("Loading video...")
        self.cap = self._load_video()
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Loaded video: {total_frames} frames @ {fps:.2f} FPS")

        print("\nStarting annotation... Press 'q' to quit and save.\n")

        frame_number = 0
        window_name = "Rally State Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Resize window to larger size (1280x720)
        cv2.resizeWindow(window_name, 1280, 720)

        # Calculate delay for proper FPS playback (in milliseconds)
        # Use a minimal delay of 1ms to allow video to play at natural speed
        delay = 1

        while True:
            if not self.paused:
                ret, frame = self.cap.read()

                if not ret:
                    print("\nEnd of video reached.")
                    break

                # Annotate current frame with current label
                self.frame_annotations[frame_number] = self.current_label

                # Draw overlay
                self._draw_overlay(frame, frame_number)

                # Display frame
                cv2.imshow(window_name, frame)

                frame_number += 1

            # Handle key presses with minimal delay for smooth playback
            key = cv2.waitKey(delay if not self.paused else 30) & 0xFF

            if key == ord("s"):
                self.current_label = self.labels["start"]
                print(f"[Frame {frame_number}] Switched to START mode")

            elif key == ord("e"):
                self.current_label = self.labels["end"]
                print(f"[Frame {frame_number}] Switched to END mode")

            elif key == ord("p"):
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RESUMED"
                print(f"[Frame {frame_number}] {status}")

            elif key == ord("q"):
                print(f"\n[Frame {frame_number}] Quit requested.")
                self.quit_requested = True
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Save annotations
        self._save_annotations()


def main():
    """Main entry point for the annotator."""
    annotator = RallyStateAnnotator()
    annotator.run()


if __name__ == "__main__":
    main()
