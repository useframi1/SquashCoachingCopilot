"""
Shot Type Classification Evaluator

Evaluates shot classification performance using annotation module data.
Generates comprehensive reports with visualizations and statistics.
"""

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Dict, Any
from pathlib import Path

from squashcopilot.modules.shot_type_classification import ShotClassifier
from squashcopilot.common.utils import load_config
from squashcopilot.common import (
    ShotClassificationInput,
    ShotClassificationOutput,
    RallySegment,
)


class ShotClassificationEvaluator:
    """Evaluator for shot type classification using annotation module data."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize evaluator with configuration."""
        if config is None:
            # Load the tests section from the shot_type_classification config
            full_config = load_config(config_name="shot_type_classification")
            config = full_config["tests"]
        self.config = config

        # Get project root (parent of squashcopilot/)
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Get video name from config
        self.video_name = self.config["test_video"]

        # Annotations directory (where CSV files are stored)
        annotations_dir = project_root / self.config["annotations_dir"]
        self.data_dir = annotations_dir / self.video_name
        self.csv_path = self.data_dir / f"{self.video_name}_annotations.csv"

        # Video directory (where video files are stored)
        video_dir = project_root / self.config["video_dir"]
        self.video_path = video_dir / f"{self.video_name}.mp4"

        # Create output directory in shot_type_classification module
        test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        base_output_dir = test_dir / self.config["output"]["output_dir"]
        self.output_dir = base_output_dir / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get video properties
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Annotations CSV not found: {self.csv_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\nVideo: {self.video_name}")
        print(f"Annotations CSV: {self.csv_path}")
        print(f"Video path: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Output directory: {self.output_dir}")

        # Initialize shot classifier
        print("\nInitializing shot classifier...")
        self.shot_classifier = ShotClassifier(fps=self.fps)
        print("Shot classifier initialized")

    def load_annotation_data(self) -> pd.DataFrame:
        """Load annotations from CSV file.

        Returns:
            DataFrame with frame as index and all annotation columns
        """
        print("\n" + "=" * 60)
        print("LOADING DATA FROM ANNOTATION MODULE")
        print("=" * 60)

        # Load CSV
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} frames from {self.csv_path}")

        # Set frame as index
        if "frame" in df.columns:
            df = df.set_index("frame")

        # Calculate detection statistics
        total_frames = len(df)

        p1_detected = df["player_1_x_meter"].notna().sum()
        p2_detected = df["player_2_x_meter"].notna().sum()
        wall_hits = df["is_wall_hit"].sum() if "is_wall_hit" in df.columns else 0
        racket_hits = df["is_racket_hit"].sum() if "is_racket_hit" in df.columns else 0

        print(f"\nLoaded {total_frames} frames")
        print(
            f"Player 1 detected in {p1_detected} frames ({p1_detected/total_frames*100:.1f}%)"
        )
        print(
            f"Player 2 detected in {p2_detected} frames ({p2_detected/total_frames*100:.1f}%)"
        )
        print(f"Wall hits: {int(wall_hits)}")
        print(f"Racket hits: {int(racket_hits)}")

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
                    rally_id=0, start_frame=int(df.index.min()), end_frame=int(df.index.max())
                )
            )
            return segments

        # Get rally frames
        rally_df = df[df["is_rally_frame"] == True]

        if len(rally_df) == 0:
            print("No rally frames found, treating entire video as one rally")
            segments.append(
                RallySegment(
                    rally_id=0, start_frame=int(df.index.min()), end_frame=int(df.index.max())
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

    def classify_shots(
        self, df: pd.DataFrame, segments: List[RallySegment]
    ) -> ShotClassificationOutput:
        """Classify shots using shot classifier with DataFrame input.

        Args:
            df: DataFrame with player positions and hit detection columns
            segments: List of rally segments

        Returns:
            ShotClassificationOutput with updated DataFrame
        """
        print("\n" + "=" * 60)
        print("CLASSIFYING SHOTS")
        print("=" * 60)

        # Create ShotClassificationInput with DataFrame
        input_data = ShotClassificationInput(
            df=df,
            segments=segments,
        )

        # Classify shots
        result = self.shot_classifier.classify_shots(input_data)

        print(f"Classified {result.num_shots} shots")

        return result

    def print_statistics(
        self, df: pd.DataFrame, result: ShotClassificationOutput
    ) -> Dict[str, Any]:
        """Print comprehensive statistics."""
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)

        # Player tracking stats
        total_frames = len(df)
        p1_detected = df["player_1_x_meter"].notna().sum()
        p2_detected = df["player_2_x_meter"].notna().sum()

        print("\nPlayer Tracking:")
        print(f"   Total frames: {total_frames}")
        print(
            f"   Player 1 detected: {p1_detected} ({p1_detected/total_frames*100:.1f}%)"
        )
        print(
            f"   Player 2 detected: {p2_detected} ({p2_detected/total_frames*100:.1f}%)"
        )

        # Hit detection stats
        wall_hits = df["is_wall_hit"].sum() if "is_wall_hit" in df.columns else 0
        racket_hits = df["is_racket_hit"].sum() if "is_racket_hit" in df.columns else 0

        print("\nHit Detection:")
        print(f"   Wall hits: {int(wall_hits)}")
        print(f"   Racket hits: {int(racket_hits)}")

        # Shot classification stats
        print(f"\nShot Classification:")
        print(f"   Total shots: {result.num_shots}")

        if result.shot_counts:
            print("\n   By Type:")
            for shot_type, count in sorted(result.shot_counts.items()):
                percentage = (count / result.num_shots) * 100 if result.num_shots > 0 else 0
                print(
                    f"      {shot_type.replace('_', ' ').title():25s}: {count:3d} ({percentage:5.1f}%)"
                )

        return {
            "total_frames": total_frames,
            "p1_detected": int(p1_detected),
            "p2_detected": int(p2_detected),
            "wall_hits": int(wall_hits),
            "racket_hits": int(racket_hits),
            "shots": result.num_shots,
            "shot_counts": result.shot_counts,
        }

    def create_plots(self, result: ShotClassificationOutput) -> None:
        """Create comprehensive shot type distribution plots."""
        if not self.config["output"]["save_plots"]:
            return

        print("\nCreating shot type distribution plots...")

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Shot Type Classification - {self.video_name}", fontsize=16)

        # Plot 1: Shot type distribution
        ax1 = axes[0]
        if result.shot_counts:
            shot_types = list(result.shot_counts.keys())
            shot_counts = list(result.shot_counts.values())

            # Shorten names for better display
            short_names = [st.replace("_", "\n") for st in shot_types]

            ax1.bar(range(len(shot_types)), shot_counts, color="steelblue")
            ax1.set_xticks(range(len(shot_types)))
            ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
            ax1.set_ylabel("Count")
            ax1.set_title("Shot Type Distribution")
            ax1.grid(True, alpha=0.3, axis="y")
        else:
            ax1.text(0.5, 0.5, "No shots classified", ha="center", va="center")
            ax1.set_title("Shot Type Distribution")

        # Plot 2: Direction distribution (from shot types)
        ax2 = axes[1]
        if result.shot_counts:
            # Aggregate by direction
            direction_counts = {"straight": 0, "cross_court": 0}
            for shot_type, count in result.shot_counts.items():
                if "cross_court" in shot_type.lower():
                    direction_counts["cross_court"] += count
                else:
                    direction_counts["straight"] += count

            directions = list(direction_counts.keys())
            dir_counts = list(direction_counts.values())

            ax2.bar(directions, dir_counts, color="coral")
            ax2.set_ylabel("Count")
            ax2.set_title("Shot Direction Distribution")
            ax2.grid(True, alpha=0.3, axis="y")
        else:
            ax2.text(0.5, 0.5, "No shots classified", ha="center", va="center")
            ax2.set_title("Shot Direction Distribution")

        plt.tight_layout()
        output_path = self.output_dir / "shot_type_plots.png"
        plt.savefig(
            output_path, dpi=self.config["output"]["plot_dpi"], bbox_inches="tight"
        )
        plt.close()

        print(f"Saved shot type plots: {output_path}")

    def create_annotated_video(
        self, df: pd.DataFrame, result: ShotClassificationOutput
    ) -> None:
        """Create annotated video with player positions and shot classifications."""
        if not self.config["output"]["save_video"]:
            return

        print("\nCreating annotated video...")

        output_path = self.output_dir / "annotated.mp4"

        # Get frame range from DataFrame
        start_frame = int(df.index.min())
        end_frame = int(df.index.max())
        num_frames = end_frame - start_frame + 1

        print(f"Processing frames {start_frame} to {end_frame} ({num_frames} frames)")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (self.width, self.height)
        )

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Get result DataFrame with shot classifications
        result_df = result.df

        # Shot type colors (BGR for OpenCV)
        shot_colors = {
            "straight_drive": (204, 102, 0),
            "straight_drop": (255, 178, 102),
            "cross_court_drive": (0, 0, 204),
            "cross_court_drop": (102, 102, 255),
        }

        # Create frame-to-shot lookup (show shot info for frames after racket hit)
        shot_display_duration = int(2 * self.fps)  # 2 seconds
        shot_by_frame = {}

        # Find frames with shots classified
        shots_df = result_df[result_df["shot_type"] != ""]
        for shot_frame in shots_df.index:
            shot_type = result_df.loc[shot_frame, "shot_type"]
            shot_direction = result_df.loc[shot_frame, "shot_direction"]
            shot_depth = result_df.loc[shot_frame, "shot_depth"]

            for offset in range(shot_display_duration):
                display_frame = shot_frame + offset
                if display_frame <= end_frame:
                    shot_by_frame[display_frame] = {
                        "type": shot_type,
                        "direction": shot_direction,
                        "depth": shot_depth,
                        "frame": shot_frame,
                    }

        # Create hit markers lookup
        hit_display_duration = 30  # frames

        wall_hit_frames = {}
        if "is_wall_hit" in df.columns:
            wall_hits_df = df[df["is_wall_hit"] == True]
            for hit_frame in wall_hits_df.index:
                for offset in range(hit_display_duration):
                    display_frame = hit_frame + offset
                    if display_frame <= end_frame:
                        wall_hit_frames[display_frame] = {
                            "frame": hit_frame,
                            "x": df.loc[hit_frame, "wall_hit_x_pixel"]
                            if "wall_hit_x_pixel" in df.columns
                            else df.loc[hit_frame, "ball_x"],
                            "y": df.loc[hit_frame, "wall_hit_y_pixel"]
                            if "wall_hit_y_pixel" in df.columns
                            else df.loc[hit_frame, "ball_y"],
                        }

        racket_hit_frames = {}
        if "is_racket_hit" in df.columns:
            racket_hits_df = df[df["is_racket_hit"] == True]
            for hit_frame in racket_hits_df.index:
                for offset in range(hit_display_duration):
                    display_frame = hit_frame + offset
                    if display_frame <= end_frame:
                        racket_hit_frames[display_frame] = {
                            "frame": hit_frame,
                            "x": df.loc[hit_frame, "ball_x"],
                            "y": df.loc[hit_frame, "ball_y"],
                            "player_id": df.loc[hit_frame, "racket_hit_player_id"]
                            if "racket_hit_player_id" in df.columns
                            else None,
                        }

        # Build shot vectors lookup (player->wall and wall->player)
        shot_vectors_by_frame = {}
        shots_df = result_df[result_df["shot_type"] != ""]
        racket_hit_list = df[df["is_racket_hit"] == True].index.tolist() if "is_racket_hit" in df.columns else []

        for shot_frame in shots_df.index:
            # Find the next racket hit after this shot
            next_hits = [f for f in racket_hit_list if f > shot_frame]
            if not next_hits:
                continue
            next_hit_frame = next_hits[0]

            # Get hitting player position (pixel coordinates)
            hitting_player_id = df.loc[shot_frame, "racket_hit_player_id"] if "racket_hit_player_id" in df.columns else None
            if pd.isna(hitting_player_id):
                continue
            hitting_player_id = int(hitting_player_id)

            hitting_x = df.loc[shot_frame, f"player_{hitting_player_id}_x_pixel"]
            hitting_y = df.loc[shot_frame, f"player_{hitting_player_id}_y_pixel"]

            # Get receiving player position (pixel coordinates at next hit)
            receiving_player_id = 2 if hitting_player_id == 1 else 1
            receiving_x = df.loc[next_hit_frame, f"player_{receiving_player_id}_x_pixel"]
            receiving_y = df.loc[next_hit_frame, f"player_{receiving_player_id}_y_pixel"]

            # Find wall hit between these frames
            wall_hit_x, wall_hit_y = None, None
            if "is_wall_hit" in df.columns:
                wall_hits_between = df[(df.index > shot_frame) & (df.index < next_hit_frame) & (df["is_wall_hit"] == True)]
                if len(wall_hits_between) > 0:
                    wall_frame = wall_hits_between.index[0]
                    wall_hit_x = df.loc[wall_frame, "wall_hit_x_pixel"] if "wall_hit_x_pixel" in df.columns else df.loc[wall_frame, "ball_x"]
                    wall_hit_y = df.loc[wall_frame, "wall_hit_y_pixel"] if "wall_hit_y_pixel" in df.columns else df.loc[wall_frame, "ball_y"]

            # Store vectors for display duration
            if all(pd.notna([hitting_x, hitting_y, wall_hit_x, wall_hit_y, receiving_x, receiving_y])):
                for offset in range(shot_display_duration):
                    display_frame = shot_frame + offset
                    if display_frame <= end_frame:
                        shot_vectors_by_frame[display_frame] = {
                            "hitting_pos": (int(hitting_x), int(hitting_y)),
                            "wall_pos": (int(wall_hit_x), int(wall_hit_y)),
                            "receiving_pos": (int(receiving_x), int(receiving_y)),
                        }

        # Process frames
        with tqdm(total=num_frames, desc="Rendering video", unit="frame") as pbar:
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw player positions
                if frame_idx in df.index:
                    row = df.loc[frame_idx]

                    # Player 1
                    p1_x = row.get("player_1_x_pixel")
                    p1_y = row.get("player_1_y_pixel")
                    if pd.notna(p1_x) and pd.notna(p1_y):
                        cv2.circle(
                            frame, (int(p1_x), int(p1_y)), 15, (255, 0, 0), -1
                        )
                        cv2.putText(
                            frame,
                            "P1",
                            (int(p1_x) - 10, int(p1_y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2,
                        )

                    # Player 2
                    p2_x = row.get("player_2_x_pixel")
                    p2_y = row.get("player_2_y_pixel")
                    if pd.notna(p2_x) and pd.notna(p2_y):
                        cv2.circle(
                            frame, (int(p2_x), int(p2_y)), 15, (0, 0, 255), -1
                        )
                        cv2.putText(
                            frame,
                            "P2",
                            (int(p2_x) - 10, int(p2_y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                # Draw current shot info overlay
                current_shot = shot_by_frame.get(frame_idx)
                if current_shot:
                    shot_text = current_shot["type"].replace("_", " ").title()
                    direction_text = f"Dir: {current_shot['direction'].replace('_', ' ').title()}"
                    depth_text = f"Depth: {current_shot['depth'].title()}"

                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                    # Draw text
                    color = shot_colors.get(current_shot["type"], (255, 255, 255))
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
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # Draw shot vectors (player->wall and wall->player)
                if frame_idx in shot_vectors_by_frame:
                    vectors = shot_vectors_by_frame[frame_idx]
                    hitting_pos = vectors["hitting_pos"]
                    wall_pos = vectors["wall_pos"]
                    receiving_pos = vectors["receiving_pos"]

                    # Draw player->wall vector (cyan)
                    cv2.arrowedLine(
                        frame,
                        hitting_pos,
                        wall_pos,
                        (255, 255, 0),  # Cyan
                        3,
                        tipLength=0.03,
                    )

                    # Draw wall->player vector (magenta)
                    cv2.arrowedLine(
                        frame,
                        wall_pos,
                        receiving_pos,
                        (255, 0, 255),  # Magenta
                        3,
                        tipLength=0.03,
                    )

                # Draw hit markers
                if frame_idx in wall_hit_frames:
                    hit = wall_hit_frames[frame_idx]
                    if pd.notna(hit["x"]) and pd.notna(hit["y"]):
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
                    if pd.notna(hit["x"]) and pd.notna(hit["y"]):
                        pos = (int(hit["x"]), int(hit["y"]))
                        cv2.drawMarker(frame, pos, (0, 165, 255), cv2.MARKER_STAR, 30, 3)
                        player_id = hit.get("player_id")
                        player_text = f"RACKET (P{int(player_id)})" if pd.notna(player_id) else "RACKET"
                        cv2.putText(
                            frame,
                            player_text,
                            (pos[0] + 20, pos[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )

                # Draw frame number
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx}",
                    (10, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Write frame
                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()

        print(f"Saved annotated video: {output_path}")

    def save_detailed_report(
        self, stats: Dict[str, Any], result: ShotClassificationOutput
    ) -> None:
        """Save detailed text report with shot statistics."""
        if not self.config["output"]["save_metrics"]:
            return

        print("\nSaving detailed report...")

        output_path = self.output_dir / "report.txt"

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("SHOT TYPE CLASSIFICATION - DETAILED REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Video: {self.video_name}\n")
            f.write(f"Total frames: {stats['total_frames']}\n")
            f.write(f"Player 1 detected: {stats['p1_detected']}\n")
            f.write(f"Player 2 detected: {stats['p2_detected']}\n")
            f.write(f"Wall hits: {stats['wall_hits']}\n")
            f.write(f"Racket hits: {stats['racket_hits']}\n")
            f.write(f"Classified shots: {stats['shots']}\n\n")

            if result.shot_counts:
                f.write("SHOT STATISTICS:\n")
                f.write("-" * 60 + "\n\n")

                f.write("By Type:\n")
                for shot_type, count in sorted(result.shot_counts.items()):
                    percentage = (count / result.num_shots) * 100 if result.num_shots > 0 else 0
                    f.write(f"  {shot_type:30s}: {count:3d} ({percentage:5.1f}%)\n")

                # Aggregate by direction
                f.write("\nBy Direction:\n")
                direction_counts = {"straight": 0, "cross_court": 0}
                for shot_type, count in result.shot_counts.items():
                    if "cross_court" in shot_type.lower():
                        direction_counts["cross_court"] += count
                    else:
                        direction_counts["straight"] += count

                for direction, count in sorted(direction_counts.items()):
                    percentage = (count / result.num_shots) * 100 if result.num_shots > 0 else 0
                    f.write(f"  {direction:30s}: {count:3d} ({percentage:5.1f}%)\n")

                # Aggregate by depth
                f.write("\nBy Depth:\n")
                depth_counts = {"drive": 0, "drop": 0}
                for shot_type, count in result.shot_counts.items():
                    if "drop" in shot_type.lower():
                        depth_counts["drop"] += count
                    else:
                        depth_counts["drive"] += count

                for depth, count in sorted(depth_counts.items()):
                    percentage = (count / result.num_shots) * 100 if result.num_shots > 0 else 0
                    f.write(f"  {depth:30s}: {count:3d} ({percentage:5.1f}%)\n")

            # Shot-by-shot breakdown
            f.write("\n" + "=" * 60 + "\n")
            f.write("SHOT-BY-SHOT BREAKDOWN\n")
            f.write("=" * 60 + "\n\n")

            shots_df = result.df[result.df["shot_type"] != ""]
            for i, (frame_idx, row) in enumerate(shots_df.iterrows(), 1):
                f.write(f"Shot #{i}\n")
                f.write(f"  Frame:     {frame_idx}\n")
                f.write(f"  Type:      {row['shot_type']}\n")
                f.write(f"  Direction: {row['shot_direction']}\n")
                f.write(f"  Depth:     {row['shot_depth']}\n")
                f.write("\n")

        print(f"Saved detailed report: {output_path}")

    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        print("\n" + "=" * 60)
        print("SHOT TYPE CLASSIFICATION EVALUATION")
        print("=" * 60)

        # Step 1: Load annotation data
        df = self.load_annotation_data()

        # Step 2: Extract rally segments
        segments = self.extract_rally_segments(df)

        # Step 3: Classify shots
        result = self.classify_shots(df, segments)

        # Step 4: Print statistics
        stats = self.print_statistics(df, result)

        # Step 5: Create visualizations
        if self.config["output"]["save_plots"]:
            self.create_plots(result)

        if self.config["output"]["save_video"]:
            self.create_annotated_video(df, result)

        if self.config["output"]["save_metrics"]:
            self.save_detailed_report(stats, result)

        # Done
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}")

        return {
            "stats": stats,
            "result": result,
            "segments": segments,
        }


if __name__ == "__main__":
    evaluator = ShotClassificationEvaluator()
    evaluator.run_evaluation()
