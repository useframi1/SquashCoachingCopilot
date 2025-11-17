"""
Stroke Annotator Tool
Allows manual annotation of stroke types (forehand/backhand) for detected racket hits.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from squashcopilot.common import KEYPOINT_NAMES
from squashcopilot.common.utils import load_config


class StrokeAnnotator:
    """Interactive tool to annotate stroke types from detected racket hits."""

    def __init__(self, annotation_csv: Path, video_path: Path, frame_window: int = 10):
        """
        Initialize the stroke annotator.

        Args:
            annotation_csv: Path to the CSV file with annotations
            video_path: Path to the corresponding video file
            frame_window: Number of frames before and after racket hit to show (default: 10)
        """
        self.annotation_csv = annotation_csv
        self.video_path = video_path
        self.frame_window = frame_window

        # Load annotations
        self.df = pd.read_csv(annotation_csv)

        # Find all racket hits
        self.racket_hits = self.df[self.df["is_racket_hit"] == True].copy()
        self.current_hit_index = 0

        # Store annotated strokes
        self.annotated_strokes = []

        # Video capture
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Loaded {len(self.racket_hits)} racket hits from {annotation_csv.name}")
        print(f"Video: {video_path.name} ({self.fps} fps)")

    def get_frame_range(self, hit_frame: int) -> Tuple[int, int]:
        """Get the frame range around a racket hit."""
        start_frame = max(0, hit_frame - self.frame_window)
        end_frame = min(len(self.df) - 1, hit_frame + self.frame_window)
        return start_frame, end_frame

    def extract_player_keypoints(
        self, player_id: int, start_frame: int, end_frame: int
    ) -> pd.DataFrame:
        """
        Extract keypoints for a specific player across a frame range.

        Args:
            player_id: Player ID (1 or 2)
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            DataFrame with frame number and keypoints for the selected player
        """
        # Get the frame range
        frame_data = self.df.iloc[start_frame : end_frame + 1].copy()

        # Build column names for this player using KEYPOINT_NAMES from common
        player_prefix = f"player_{player_id}_"
        keypoint_cols = []
        for kp_name in KEYPOINT_NAMES:
            keypoint_cols.append(f"{player_prefix}kp_{kp_name}_x")
            keypoint_cols.append(f"{player_prefix}kp_{kp_name}_y")

        # Select frame and keypoint columns
        result = frame_data[["frame"] + keypoint_cols].copy()

        # Rename columns to remove player prefix for cleaner output
        rename_map = {}
        for kp_name in KEYPOINT_NAMES:
            rename_map[f"{player_prefix}kp_{kp_name}_x"] = f"kp_{kp_name}_x"
            rename_map[f"{player_prefix}kp_{kp_name}_y"] = f"kp_{kp_name}_y"

        result.rename(columns=rename_map, inplace=True)

        return result

    def play_clip(
        self, start_frame: int, end_frame: int, hit_frame: int
    ) -> Optional[str]:
        """
        Play a video clip and get user annotation.

        Args:
            start_frame: Starting frame
            end_frame: Ending frame
            hit_frame: Frame where racket hit occurred

        Returns:
            Annotation string or None if skipped
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        paused = False

        while frame_idx <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Add frame information overlay
            display_frame = frame.copy()

            # Highlight the hit frame
            if frame_idx == hit_frame:
                cv2.putText(
                    display_frame,
                    "RACKET HIT!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                )
                cv2.rectangle(
                    display_frame,
                    (0, 0),
                    (display_frame.shape[1], display_frame.shape[0]),
                    (0, 0, 255),
                    10,
                )

            # Frame counter
            cv2.putText(
                display_frame,
                f"Frame: {frame_idx}/{end_frame}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Instructions
            instructions = [
                "Controls:",
                "1 - Player 1 Forehand",
                "2 - Player 1 Backhand",
                "3 - Player 2 Forehand",
                "4 - Player 2 Backhand",
                "s - Skip",
                "r - Replay",
                "SPACE - Pause/Resume",
                "q - Quit",
            ]

            y_offset = display_frame.shape[0] - 280
            for i, text in enumerate(instructions):
                cv2.putText(
                    display_frame,
                    text,
                    (50, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Stroke Annotator", display_frame)

            # Handle key press
            key = cv2.waitKey(int(1000 / self.fps) if not paused else 0) & 0xFF

            if key == ord("q"):
                return "quit"
            elif key == ord("1"):
                return "player_1_forehand"
            elif key == ord("2"):
                return "player_1_backhand"
            elif key == ord("3"):
                return "player_2_forehand"
            elif key == ord("4"):
                return "player_2_backhand"
            elif key == ord("s"):
                return "skip"
            elif key == ord("r"):
                # Replay - reset to start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_idx = start_frame
                continue
            elif key == ord(" "):
                paused = not paused
                continue

            if not paused:
                frame_idx += 1

        # If we reach the end without a key press, loop the clip
        return self.play_clip(start_frame, end_frame, hit_frame)

    def annotate_all(self) -> List[dict]:
        """
        Iterate through all racket hits and annotate them.

        Returns:
            List of annotated stroke dictionaries
        """
        total_hits = len(self.racket_hits)

        for idx, (_, hit_row) in enumerate(self.racket_hits.iterrows()):
            hit_frame = int(hit_row["frame"])
            start_frame, end_frame = self.get_frame_range(hit_frame)

            print(f"\n{'='*60}")
            print(f"Racket Hit {idx + 1}/{total_hits}")
            print(f"Frame: {hit_frame} (showing frames {start_frame}-{end_frame})")
            print(f"{'='*60}")

            # Play the clip and get annotation
            annotation = self.play_clip(start_frame, end_frame, hit_frame)

            if annotation == "quit":
                print("\nQuitting annotation session...")
                break

            if annotation == "skip":
                print("Skipped.")
                continue

            # Parse annotation
            parts = annotation.split("_")
            player_id = int(parts[1])
            stroke_type = parts[2]

            # Extract keypoints for this player
            keypoints_df = self.extract_player_keypoints(
                player_id, start_frame, end_frame
            )

            # Create annotation record for each frame in the window
            for _, kp_row in keypoints_df.iterrows():
                stroke_record = {
                    "hit_frame": hit_frame,
                    "frame": kp_row["frame"],
                    "player_id": player_id,
                    "stroke_type": stroke_type,
                }

                # Add all keypoints
                for kp_name in KEYPOINT_NAMES:
                    stroke_record[f"kp_{kp_name}_x"] = kp_row[f"kp_{kp_name}_x"]
                    stroke_record[f"kp_{kp_name}_y"] = kp_row[f"kp_{kp_name}_y"]

                self.annotated_strokes.append(stroke_record)

            print(f"Annotated as: Player {player_id} - {stroke_type.capitalize()}")

        cv2.destroyAllWindows()
        return self.annotated_strokes

    def save_annotations(self, output_path: Path):
        """Save annotated strokes to a CSV file."""
        if not self.annotated_strokes:
            print("No annotations to save.")
            return

        df = pd.DataFrame(self.annotated_strokes)
        df.to_csv(output_path, index=False)
        print(
            f"\nSaved {len(self.annotated_strokes)} annotated frames to {output_path}"
        )
        print(f"Total strokes annotated: {df['hit_frame'].nunique()}")

    def __del__(self):
        """Clean up video capture."""
        if hasattr(self, "cap"):
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point for the annotator tool."""
    # Load configuration
    config = load_config(config_name="stroke_detection")
    annotation_config = config.get("annotation", {})

    # Get settings from config
    video_name = annotation_config.get("video_name", "video-1")
    annotation_dir = Path(
        annotation_config.get("annotation_dir", "squashcopilot/annotation/annotations")
    )
    output_dir = Path(
        annotation_config.get(
            "output_dir", "squashcopilot/modules/stroke_detection/data"
        )
    )
    frame_window = annotation_config.get("frame_window", 10)

    print(f"Stroke Annotator")
    print(f"{'='*60}")
    print(f"Video: {video_name}")
    print(f"Frame window: ±{frame_window} frames")
    print(f"{'='*60}\n")

    # Construct paths
    video_folder = annotation_dir / video_name
    annotation_csv = video_folder / f"{video_name}_annotations.csv"
    video_path = video_folder / f"{video_name}_annotated.mp4"

    # Verify files exist
    if not annotation_csv.exists():
        print(f"Error: Annotation CSV not found: {annotation_csv}")
        print(f"Please check the 'video_name' setting in configs/stroke_detection.yaml")
        return

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        print(f"Please check the 'video_name' setting in configs/stroke_detection.yaml")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize annotator
    annotator = StrokeAnnotator(annotation_csv, video_path, frame_window)

    # Run annotation
    annotator.annotate_all()

    # Save results
    output_csv = output_dir / f"{video_name}_strokes_annotated_2.csv"
    annotator.save_annotations(output_csv)


if __name__ == "__main__":
    main()
