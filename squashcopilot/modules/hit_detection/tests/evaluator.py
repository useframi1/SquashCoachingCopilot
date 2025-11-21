"""
Hit Detection Evaluator

Evaluates wall and racket hit detection by loading data from pipeline outputs.
Visualizes detected hits on video and generates metrics.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, List, Optional
import tqdm

from squashcopilot.modules.hit_detection import WallHitDetector, RacketHitDetector
from squashcopilot.common.utils import load_config
from squashcopilot.common.models import (
    RallySegment,
    WallHitDetectionInput,
    RacketHitDetectionInput,
)


class HitDetectionEvaluator:
    """Evaluator for wall and racket hit detection.

    Loads ball coordinates and player positions from pipeline output CSV files,
    runs hit detection, and generates visualizations and metrics.
    """

    def __init__(self, config: dict = None):
        """Initialize the hit detection evaluator.

        Args:
            config: Configuration dictionary. If None, loads from hit_detection.yaml
        """
        if config is None:
            full_config = load_config(config_name="hit_detection")
            self.detection_config = full_config
            config = full_config.get("tests", {})
        else:
            self.detection_config = config

        self.config = config

        # Get test directory path
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Get project root directory (parent of squashcopilot package)
        # Navigate up from: hit_detection/tests/ -> hit_detection -> modules -> squashcopilot -> project_root
        project_root = self.test_dir.parent.parent.parent.parent

        # Resolve directories (relative to project root)
        data_config = self.config.get("data", {})

        # Annotations directory (where CSV files are stored)
        annotations_dir_rel = data_config.get(
            "annotations_dir", "squashcopilot/annotation/annotations"
        )
        self.annotations_dir = project_root / annotations_dir_rel

        # Video directory (where video files are stored)
        video_dir_rel = data_config.get("video_dir", "squashcopilot/videos")
        self.video_base_dir = project_root / video_dir_rel

        # Video name to evaluate (from config)
        self.video_name = data_config.get("video", "video-3")

        # Paths for the specific video
        self.video_dir = self.annotations_dir / self.video_name
        self.csv_path = self.video_dir / f"{self.video_name}_annotations.csv"
        self.video_path = self.video_base_dir / f"{self.video_name}.mp4"

        # Output directory
        output_config = self.config.get("output", {})
        output_dir_name = output_config.get("output_dir", "outputs")
        self.output_dir = self.test_dir / output_dir_name / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization config
        self.vis_config = self.config.get("visualization", {})
        self.wall_hit_color = tuple(self.vis_config.get("wall_hit_color", [0, 255, 0]))
        self.racket_hit_color = tuple(
            self.vis_config.get("racket_hit_color", [255, 0, 0])
        )
        self.ball_color = tuple(self.vis_config.get("ball_color", [0, 0, 255]))
        self.trace_length = self.vis_config.get("trace_length", 10)
        self.marker_radius = self.vis_config.get("marker_radius", 15)
        self.marker_thickness = self.vis_config.get("marker_thickness", 3)

        # Initialize detectors
        self.wall_hit_detector = WallHitDetector(self.detection_config)
        self.racket_hit_detector = RacketHitDetector(self.detection_config)

        # Validate paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required input files exist."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Analysis CSV not found: {self.csv_path}")

        if not self.video_path.exists():
            # Try without _annotated suffix
            alt_video_path = self.video_dir / f"{self.video_name}.mp4"
            if alt_video_path.exists():
                self.video_path = alt_video_path
            else:
                print(f"Warning: Video not found at {self.video_path}")
                print("Video visualization will be skipped.")
                self.video_path = None

    def load_data(self) -> pd.DataFrame:
        """Load analysis data from pipeline output CSV.

        Returns:
            DataFrame with ball and player positions indexed by frame_number.
        """
        print(f"Loading data from: {self.csv_path}")
        df = pd.read_csv(self.csv_path, index_col="frame")
        print(f"Loaded {len(df)} frames")
        return df

    def extract_rally_segments(self, df: pd.DataFrame) -> List[RallySegment]:
        """Extract rally segments from DataFrame.

        Args:
            df: DataFrame with is_rally_frame and rally_id columns

        Returns:
            List of RallySegment objects
        """
        segments = []

        if "is_rally_frame" not in df.columns or "rally_id" not in df.columns:
            # If no rally info, treat entire video as one rally
            print("No rally segmentation found, treating entire video as one rally")
            segments.append(
                RallySegment(
                    rally_id=0, start_frame=df.index.min(), end_frame=df.index.max()
                )
            )
            return segments

        # Get rally frames
        rally_df = df[df["is_rally_frame"] == True]

        if len(rally_df) == 0:
            print("No rally frames found, treating entire video as one rally")
            segments.append(
                RallySegment(
                    rally_id=0, start_frame=df.index.min(), end_frame=df.index.max()
                )
            )
            return segments

        # Group by rally_id
        for rally_id in rally_df["rally_id"].unique():
            rally_frames = rally_df[rally_df["rally_id"] == rally_id]
            segments.append(
                RallySegment(
                    rally_id=int(rally_id),
                    start_frame=int(rally_frames.index.min()),
                    end_frame=int(rally_frames.index.max()),
                )
            )

        print(f"Found {len(segments)} rally segments")
        return segments

    def run_detection(
        self, df: pd.DataFrame, segments: List[RallySegment]
    ) -> pd.DataFrame:
        """Run wall and racket hit detection.

        Args:
            df: DataFrame with ball and player positions
            segments: List of rally segments

        Returns:
            DataFrame with hit detection columns added
        """
        print("Running wall hit detection...")

        # Create a mock calibration (we don't have full calibration in CSV)
        # Wall hits will still be detected but meter coordinates won't be computed
        calibration = None

        # Wall hit detection
        wall_input = WallHitDetectionInput(
            df=df, segments=segments, calibration=calibration
        )
        wall_output = self.wall_hit_detector.detect_wall_hits(wall_input)
        df = wall_output.df
        print(f"Detected {wall_output.num_wall_hits} wall hits")

        # Racket hit detection (requires wall hits)
        print("Running racket hit detection...")
        racket_input = RacketHitDetectionInput(df=df, segments=segments)
        racket_output = self.racket_hit_detector.detect_racket_hits(racket_input)
        df = racket_output.df
        print(f"Detected {racket_output.num_racket_hits} racket hits")

        return df, wall_output, racket_output

    def visualize_video(self, df: pd.DataFrame, output_path: Optional[Path] = None):
        """Create visualization video with hit markers and player positions.

        Args:
            df: DataFrame with hit detection columns
            output_path: Path to save output video
        """
        if self.video_path is None:
            print("Skipping video visualization (no video file found)")
            return

        if output_path is None:
            output_path = self.output_dir / f"{self.video_name}_hit_detection.mp4"

        print(f"Creating visualization video: {output_path}")

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Only process frames that exist in annotations
        annotation_frames = sorted(df.index.tolist())
        start_frame = annotation_frames[0]
        end_frame = annotation_frames[-1]
        print(f"Processing frames {start_frame} to {end_frame} ({len(annotation_frames)} annotated frames)")

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Setup video writer
        codec = self.config.get("output", {}).get("video_codec", "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Hit marker duration (1 second)
        hit_marker_duration = int(fps)

        # Build hit frame lookup for persistent markers
        wall_hit_frames = (
            df[df["is_wall_hit"] == True].index.tolist()
            if "is_wall_hit" in df.columns
            else []
        )
        racket_hit_frames = (
            df[df["is_racket_hit"] == True].index.tolist()
            if "is_racket_hit" in df.columns
            else []
        )

        # Player colors
        player_colors = {
            1: (0, 255, 0),  # Green for Player 1
            2: (255, 0, 0),  # Blue for Player 2
        }

        for frame_idx in tqdm.tqdm(range(start_frame, end_frame + 1), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx not in df.index:
                continue

            row = df.loc[frame_idx]

            # Draw player positions with IDs
            for player_id in [1, 2]:
                player_x = row.get(f"player_{player_id}_x_pixel")
                player_y = row.get(f"player_{player_id}_y_pixel")

                if pd.notna(player_x) and pd.notna(player_y):
                    pos = (int(player_x), int(player_y))
                    color = player_colors[player_id]

                    # Draw circle at player position
                    cv2.circle(frame, pos, 10, color, -1)

                    # Draw player ID label
                    label = f"P{player_id}"
                    cv2.putText(
                        frame,
                        label,
                        (pos[0] + 15, pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.vis_config.get("font_scale", 0.6),
                        color,
                        self.vis_config.get("font_thickness", 2),
                    )

            # Draw persistent wall hit markers (last 1 second)
            for hit_frame in wall_hit_frames:
                if hit_frame <= frame_idx < hit_frame + hit_marker_duration:
                    hit_row = df.loc[hit_frame]
                    hit_x = hit_row.get("wall_hit_x_pixel", hit_row.get("ball_x"))
                    hit_y = hit_row.get("wall_hit_y_pixel", hit_row.get("ball_y"))
                    if pd.notna(hit_x) and pd.notna(hit_y):
                        hit_pos = (int(hit_x), int(hit_y))
                        # Draw green circle marker for wall hit
                        cv2.circle(
                            frame,
                            hit_pos,
                            self.marker_radius,
                            self.wall_hit_color,
                            self.marker_thickness,
                        )
                        cv2.putText(
                            frame,
                            "WALL",
                            (hit_pos[0] + 20, hit_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.vis_config.get("font_scale", 0.6),
                            self.wall_hit_color,
                            self.vis_config.get("font_thickness", 2),
                        )

            # Draw persistent racket hit markers (last 1 second)
            for hit_frame in racket_hit_frames:
                if hit_frame <= frame_idx < hit_frame + hit_marker_duration:
                    hit_row = df.loc[hit_frame]
                    ball_x = hit_row.get("ball_x")
                    ball_y = hit_row.get("ball_y")
                    if pd.notna(ball_x) and pd.notna(ball_y):
                        hit_pos = (int(ball_x), int(ball_y))
                        player_id = int(hit_row.get("racket_hit_player_id", -1))
                        # Draw red X marker for racket hit
                        cv2.drawMarker(
                            frame,
                            hit_pos,
                            self.racket_hit_color,
                            cv2.MARKER_CROSS,
                            self.marker_radius * 2,
                            self.marker_thickness,
                        )
                        label = f"P{player_id}" if player_id > 0 else "HIT"
                        cv2.putText(
                            frame,
                            label,
                            (hit_pos[0] + 20, hit_pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.vis_config.get("font_scale", 0.6),
                            self.racket_hit_color,
                            self.vis_config.get("font_thickness", 2),
                        )

            video_writer.write(frame)

        cap.release()
        video_writer.release()
        print(f"Video saved: {output_path}")

    def plot_ball_trajectory(
        self, df: pd.DataFrame, output_path: Optional[Path] = None
    ):
        """Plot ball trajectory with hit markers.

        Args:
            df: DataFrame with hit detection columns
            output_path: Path to save plot
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.video_name}_trajectory.png"

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Get frame numbers
        frames = df.index.values
        ball_x = df["ball_x"].values
        ball_y = df["ball_y"].values

        # Plot X coordinate
        ax1 = axes[0]
        ax1.plot(frames, ball_x, "b-", alpha=0.7, label="Ball X", linewidth=0.5)
        ax1.set_ylabel("X Position (pixels)")
        ax1.set_title(f"Ball Trajectory - {self.video_name}")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot Y coordinate
        ax2 = axes[1]
        ax2.plot(frames, ball_y, "b-", alpha=0.7, label="Ball Y", linewidth=0.5)
        ax2.set_ylabel("Y Position (pixels)")
        ax2.set_xlabel("Frame Number")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Mark wall hits
        wall_hits = df[df["is_wall_hit"] == True]
        if len(wall_hits) > 0:
            ax1.scatter(
                wall_hits.index,
                wall_hits["ball_x"],
                c="green",
                s=100,
                marker="o",
                label="Wall Hit",
                zorder=5,
            )
            ax2.scatter(
                wall_hits.index,
                wall_hits["ball_y"],
                c="green",
                s=100,
                marker="o",
                label="Wall Hit",
                zorder=5,
            )
            ax1.legend(loc="upper right")
            ax2.legend(loc="upper right")

        # Mark racket hits
        racket_hits = df[df["is_racket_hit"] == True]
        if len(racket_hits) > 0:
            ax1.scatter(
                racket_hits.index,
                racket_hits["ball_x"],
                c="red",
                s=100,
                marker="x",
                label="Racket Hit",
                zorder=5,
            )
            ax2.scatter(
                racket_hits.index,
                racket_hits["ball_y"],
                c="red",
                s=100,
                marker="x",
                label="Racket Hit",
                zorder=5,
            )
            ax1.legend(loc="upper right")
            ax2.legend(loc="upper right")

        plt.tight_layout()
        dpi = self.config.get("output", {}).get("plot_dpi", 300)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {output_path}")

    def save_metrics(
        self,
        df: pd.DataFrame,
        wall_output,
        racket_output,
        segments: List[RallySegment],
        output_path: Optional[Path] = None,
    ):
        """Save detection metrics to file.

        Args:
            df: DataFrame with hit detection columns
            wall_output: WallHitDetectionOutput
            racket_output: RacketHitDetectionOutput
            segments: List of rally segments
            output_path: Path to save metrics
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.video_name}_metrics.txt"

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"HIT DETECTION RESULTS - {self.video_name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("OVERALL STATISTICS:\n")
            f.write(f"   Total frames: {len(df)}\n")
            f.write(f"   Total wall hits: {wall_output.num_wall_hits}\n")
            f.write(f"   Total racket hits: {racket_output.num_racket_hits}\n\n")

            f.write("PER-RALLY STATISTICS:\n")
            for segment in segments:
                rally_id = segment.rally_id
                wall_hits = wall_output.wall_hits_per_rally.get(rally_id, 0)
                racket_hits = racket_output.racket_hits_per_rally.get(rally_id, 0)
                f.write(f"   Rally {rally_id}:\n")
                f.write(
                    f"      Frames: {segment.start_frame} - {segment.end_frame} ({segment.num_frames} frames)\n"
                )
                f.write(f"      Wall hits: {wall_hits}\n")
                f.write(f"      Racket hits: {racket_hits}\n")

            f.write("\nDETECTOR CONFIGURATION:\n")
            f.write("   Wall Hit Detection:\n")
            f.write(f"      Prominence: {self.wall_hit_detector.prominence}\n")
            f.write(f"      Width: {self.wall_hit_detector.width}\n")
            f.write(f"      Min distance: {self.wall_hit_detector.min_distance}\n")
            f.write("   Racket Hit Detection:\n")
            f.write(f"      Slope window: {self.racket_hit_detector.slope_window}\n")
            f.write(
                f"      Slope threshold: {self.racket_hit_detector.slope_threshold}\n"
            )
            f.write(
                f"      Lookback frames: {self.racket_hit_detector.lookback_frames}\n"
            )
            f.write(f"      Min distance: {self.racket_hit_detector.min_distance}\n")
            f.write(
                f"      Confidence ratio: {self.racket_hit_detector.confidence_ratio}\n"
            )

            # Player attribution breakdown
            f.write("\nPLAYER ATTRIBUTION:\n")
            racket_hits_df = df[df["is_racket_hit"] == True]
            if len(racket_hits_df) > 0:
                p1_hits = len(
                    racket_hits_df[racket_hits_df["racket_hit_player_id"] == 1]
                )
                p2_hits = len(
                    racket_hits_df[racket_hits_df["racket_hit_player_id"] == 2]
                )
                f.write(f"   Player 1 racket hits: {p1_hits}\n")
                f.write(f"   Player 2 racket hits: {p2_hits}\n")

        print(f"Metrics saved: {output_path}")

    def run_evaluation(self):
        """Run complete hit detection evaluation pipeline."""
        print("\n" + "=" * 60)
        print(f"HIT DETECTION EVALUATION - {self.video_name}")
        print("=" * 60 + "\n")

        # Load data
        df = self.load_data()

        # Extract rally segments
        segments = self.extract_rally_segments(df)

        # Run detection
        df, wall_output, racket_output = self.run_detection(df, segments)

        # Generate outputs
        output_config = self.config.get("output", {})

        if output_config.get("save_video", True):
            self.visualize_video(df)

        if output_config.get("save_plots", True):
            self.plot_ball_trajectory(df)

        if output_config.get("save_metrics", True):
            self.save_metrics(df, wall_output, racket_output, segments)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}")

        return {
            "df": df,
            "wall_output": wall_output,
            "racket_output": racket_output,
            "segments": segments,
        }


if __name__ == "__main__":
    evaluator = HitDetectionEvaluator()
    evaluator.run_evaluation()
