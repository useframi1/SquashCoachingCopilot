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
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from squashcopilot.modules.shot_type_classification import ShotClassifier
from squashcopilot.common.utils import load_config
from squashcopilot.common import (
    Point2D,
    WallHit,
    RacketHit,
    ShotType,
    ShotClassificationInput,
    ShotResult,
    ShotStatistics,
)


class ShotClassificationEvaluator:
    """Evaluator for shot type classification using annotation module data"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            # Load the tests section from the shot_type_classification config
            full_config = load_config(config_name="shot_type_classification")
            config = full_config["tests"]
        self.config = config

        # Get project root (parent of squashcopilot/)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        self.annotations_dir = project_root / self.config["annotations_dir"]

        # Get video name from config
        self.video_name = self.config["test_video"]

        self.video_dir = self.annotations_dir / self.video_name
        self.csv_path = self.video_dir / f"{self.video_name}_annotations.csv"
        self.video_path = self.video_dir / f"{self.video_name}_annotated.mp4"

        # Create output directory in shot_type_classification module
        test_dir = os.path.dirname(os.path.abspath(__file__))
        base_output_dir = os.path.join(test_dir, self.config["output"]["output_dir"])
        self.output_dir = os.path.join(base_output_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Get video properties
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Annotations CSV not found: {self.csv_path}")

        cap = cv2.VideoCapture(self.video_path)
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
        print("✓ Shot classifier initialized")

    def load_annotation_data(
        self,
    ) -> Tuple[
        List[Optional[Point2D]],
        List[Optional[Point2D]],
        List[WallHit],
        List[RacketHit],
    ]:
        """Load player positions and hits from annotation CSV file"""
        print("\n" + "=" * 60)
        print("LOADING DATA FROM ANNOTATION MODULE")
        print("=" * 60)

        # Load CSV using pandas for proper data type handling
        df = pd.read_csv(self.csv_path)

        player1_positions_meter = []
        player2_positions_meter = []
        wall_hits = []
        racket_hits = []

        for idx, row in df.iterrows():
            frame = int(row["frame"])

            # Parse player 1 position in meters
            p1_x = (
                row["player_1_x_meter"] if pd.notna(row["player_1_x_meter"]) else None
            )
            p1_y = (
                row["player_1_y_meter"] if pd.notna(row["player_1_y_meter"]) else None
            )

            if p1_x is not None and p1_y is not None:
                player1_positions_meter.append(Point2D(x=float(p1_x), y=float(p1_y)))
            else:
                player1_positions_meter.append(None)

            # Parse player 2 position in meters
            p2_x = (
                row["player_2_x_meter"] if pd.notna(row["player_2_x_meter"]) else None
            )
            p2_y = (
                row["player_2_y_meter"] if pd.notna(row["player_2_y_meter"]) else None
            )

            if p2_x is not None and p2_y is not None:
                player2_positions_meter.append(Point2D(x=float(p2_x), y=float(p2_y)))
            else:
                player2_positions_meter.append(None)

            # Parse wall hits
            if row["is_wall_hit"] == 1:
                wall_x_meter = (
                    row["wall_hit_x_meter"]
                    if pd.notna(row["wall_hit_x_meter"])
                    else None
                )
                wall_y_meter = (
                    row["wall_hit_y_meter"]
                    if pd.notna(row["wall_hit_y_meter"])
                    else None
                )
                wall_x_pixel = (
                    row["wall_hit_x_pixel"]
                    if pd.notna(row["wall_hit_x_pixel"])
                    else None
                )
                wall_y_pixel = (
                    row["wall_hit_y_pixel"]
                    if pd.notna(row["wall_hit_y_pixel"])
                    else None
                )

                if wall_x_meter is not None and wall_x_pixel is not None:
                    wall_hits.append(
                        WallHit(
                            frame=frame,
                            position=Point2D(
                                x=float(wall_x_pixel), y=float(wall_y_pixel)
                            ),
                            position_meter=Point2D(
                                x=float(wall_x_meter), y=float(wall_y_meter)
                            ),
                            prominence=0.0,
                        )
                    )

            # Parse racket hits
            if row["is_racket_hit"] == 1:
                player_id = (
                    int(row["racket_hit_player_id"])
                    if pd.notna(row["racket_hit_player_id"])
                    else None
                )
                ball_x = row["ball_x"] if pd.notna(row["ball_x"]) else None
                ball_y = row["ball_y"] if pd.notna(row["ball_y"]) else None

                if player_id is not None and ball_x is not None and ball_y is not None:
                    racket_hits.append(
                        RacketHit(
                            frame=frame,
                            position=Point2D(x=float(ball_x), y=float(ball_y)),
                            slope=0.0,
                            player_id=player_id,
                        )
                    )

        # Calculate detection statistics
        p1_detected = sum(1 for pos in player1_positions_meter if pos is not None)
        p2_detected = sum(1 for pos in player2_positions_meter if pos is not None)
        total_frames = len(player1_positions_meter)

        print(f"\n✓ Loaded {total_frames} frames")
        print(
            f"✓ Player 1 detected in {p1_detected} frames ({p1_detected/total_frames*100:.1f}%)"
        )
        print(
            f"✓ Player 2 detected in {p2_detected} frames ({p2_detected/total_frames*100:.1f}%)"
        )
        print(f"✓ Loaded {len(wall_hits)} wall hits")
        print(f"✓ Loaded {len(racket_hits)} racket hits")

        return player1_positions_meter, player2_positions_meter, wall_hits, racket_hits

    def classify_shots(
        self,
        player1_positions_meter: List[Optional[Point2D]],
        player2_positions_meter: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
    ) -> List[ShotResult]:
        """Classify shots using shot classifier with player position data"""
        print("\n" + "=" * 60)
        print("CLASSIFYING SHOTS")
        print("=" * 60)

        # Create ShotClassificationInput with player positions
        input_data = ShotClassificationInput(
            player1_positions_meter=player1_positions_meter,
            player2_positions_meter=player2_positions_meter,
            wall_hits=wall_hits,
            racket_hits=racket_hits,
        )

        # Classify shots
        result = self.shot_classifier.classify(input_data)
        shots = result.shots

        print(f"✓ Classified {len(shots)} shots")

        return shots

    def print_statistics(
        self,
        player1_positions_meter: List[Optional[Point2D]],
        player2_positions_meter: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        shots: List[ShotResult],
    ) -> Dict[str, Any]:
        """Print comprehensive statistics"""
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)

        # Player tracking stats
        total_frames = len(player1_positions_meter)
        p1_detected = sum(1 for pos in player1_positions_meter if pos is not None)
        p2_detected = sum(1 for pos in player2_positions_meter if pos is not None)

        print("\nPlayer Tracking:")
        print(f"   Total frames: {total_frames}")
        print(
            f"   Player 1 detected: {p1_detected} ({p1_detected/total_frames*100:.1f}%)"
        )
        print(
            f"   Player 2 detected: {p2_detected} ({p2_detected/total_frames*100:.1f}%)"
        )

        # Hit detection stats
        print("\nHit Detection:")
        print(f"   Wall hits: {len(wall_hits)}")
        print(f"   Racket hits: {len(racket_hits)}")

        # Shot classification stats
        if shots:
            # Compute statistics using ShotStatistics.from_shots
            shot_stats = ShotStatistics.from_shots(shots)

            print(f"\nShot Classification:")
            print(f"   Total shots: {shot_stats.total_shots}")

            print("\n   By Type:")
            for shot_type, count in sorted(shot_stats.by_type.items()):
                percentage = (count / shot_stats.total_shots) * 100
                print(
                    f"      {shot_type.replace('_', ' ').title():25s}: {count:3d} ({percentage:5.1f}%)"
                )

            print("\n   By Direction:")
            for direction, count in sorted(shot_stats.by_direction.items()):
                percentage = (count / shot_stats.total_shots) * 100
                print(
                    f"      {direction.replace('_', ' ').title():25s}: {count:3d} ({percentage:5.1f}%)"
                )

            print("\n   By Depth:")
            for depth, count in sorted(shot_stats.by_depth.items()):
                percentage = (count / shot_stats.total_shots) * 100
                print(f"      {depth.title():25s}: {count:3d} ({percentage:5.1f}%)")

            # Wall hit detection statistics
            print(
                f"\n   Wall hit detection rate: {shot_stats.wall_hit_detection_rate*100:.1f}%"
            )
            if shot_stats.average_rebound_distance is not None:
                print(
                    f"   Average rebound distance: {shot_stats.average_rebound_distance:.2f}m"
                )

        return {
            "total_frames": total_frames,
            "p1_detected": p1_detected,
            "p2_detected": p2_detected,
            "wall_hits": len(wall_hits),
            "racket_hits": len(racket_hits),
            "shots": len(shots),
        }

    def create_plots(self, shots: List[ShotResult]) -> None:
        """Create comprehensive shot type distribution plots"""
        if not self.config["output"]["save_plots"]:
            return

        print("\nCreating shot type distribution plots...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Shot Type Classification - {self.video_name}", fontsize=16)

        # Plot 2: Shot type distribution
        ax2 = axes[0, 1]
        if shots:
            shot_stats = ShotStatistics.from_shots(shots)
            shot_types = list(shot_stats.by_type.keys())
            shot_counts = list(shot_stats.by_type.values())

            # Shorten names for better display
            short_names = [st.replace("_", "\n") for st in shot_types]

            ax2.bar(range(len(shot_types)), shot_counts, color="steelblue")
            ax2.set_xticks(range(len(shot_types)))
            ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Count")
            ax2.set_title("Shot Type Distribution")
            ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Direction distribution
        ax3 = axes[1, 0]
        if shots:
            directions = list(shot_stats.by_direction.keys())
            dir_counts = list(shot_stats.by_direction.values())

            ax3.bar(directions, dir_counts, color="coral")
            ax3.set_ylabel("Count")
            ax3.set_title("Shot Direction Distribution")
            ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: Depth distribution
        ax4 = axes[1, 1]
        if shots:
            depths = list(shot_stats.by_depth.keys())
            depth_counts = list(shot_stats.by_depth.values())

            ax4.bar(depths, depth_counts, color="lightgreen")
            ax4.set_ylabel("Count")
            ax4.set_title("Shot Depth Distribution")
            ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "shot_type_plots.png")
        plt.savefig(
            output_path, dpi=self.config["output"]["plot_dpi"], bbox_inches="tight"
        )
        plt.close()

        print(f"✓ Saved trajectory plots: {output_path}")

    def create_annotated_video(
        self,
        player1_positions_meter: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        shots: List[ShotResult],
    ) -> None:
        """Create annotated video with player positions, vectors, and shot classifications"""
        if not self.config["output"]["save_video"]:
            return

        print("\nCreating annotated video...")

        output_path = os.path.join(self.output_dir, "annotated.mp4")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Open video
        cap = cv2.VideoCapture(self.video_path)

        # Load player pixel positions for visualization
        # We need to read the CSV again to get pixel coordinates
        df = pd.read_csv(self.csv_path)

        player1_positions_pixel = []
        player2_positions_pixel = []

        for _, row in df.iterrows():
            # Player 1 pixel position
            p1_x_px = (
                row["player_1_x_pixel"] if pd.notna(row["player_1_x_pixel"]) else None
            )
            p1_y_px = (
                row["player_1_y_pixel"] if pd.notna(row["player_1_y_pixel"]) else None
            )

            if p1_x_px is not None and p1_y_px is not None:
                player1_positions_pixel.append(
                    Point2D(x=float(p1_x_px), y=float(p1_y_px))
                )
            else:
                player1_positions_pixel.append(None)

            # Player 2 pixel position
            p2_x_px = (
                row["player_2_x_pixel"] if pd.notna(row["player_2_x_pixel"]) else None
            )
            p2_y_px = (
                row["player_2_y_pixel"] if pd.notna(row["player_2_y_pixel"]) else None
            )

            if p2_x_px is not None and p2_y_px is not None:
                player2_positions_pixel.append(
                    Point2D(x=float(p2_x_px), y=float(p2_y_px))
                )
            else:
                player2_positions_pixel.append(None)

        # Create frame-to-shot lookup
        racket_hit_frames_list = sorted([hit.frame for hit in racket_hits])

        shot_by_frame = {}
        for shot in shots:
            # Find next racket hit frame
            next_racket_frames = [f for f in racket_hit_frames_list if f > shot.frame]
            if next_racket_frames:
                end_frame = next_racket_frames[0]
                for frame_idx in range(shot.frame, end_frame + 1):
                    if frame_idx < len(player1_positions_meter):
                        shot_by_frame[frame_idx] = shot

        # Shot type colors (BGR for OpenCV)
        shot_colors = {
            ShotType.STRAIGHT_DRIVE: (204, 102, 0),
            ShotType.STRAIGHT_DROP: (255, 178, 102),
            ShotType.CROSS_COURT_DRIVE: (0, 0, 204),
            ShotType.CROSS_COURT_DROP: (102, 102, 255),
        }

        # Create frame-to-hit lookup (with display duration)
        hit_display_duration = 30

        wall_hit_frames = {}
        for hit in wall_hits:
            for offset in range(hit_display_duration):
                frame = hit.frame + offset
                if frame < len(player1_positions_meter):
                    wall_hit_frames[frame] = hit

        racket_hit_frames = {}
        for hit in racket_hits:
            for offset in range(hit_display_duration):
                frame = hit.frame + offset
                if frame < len(player1_positions_meter):
                    racket_hit_frames[frame] = hit

        # Process frames
        max_frames = len(player1_positions_meter)
        with tqdm(total=max_frames, desc="Rendering video", unit="frame") as pbar:
            frame_idx = 0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw player positions
                p1_pos = player1_positions_pixel[frame_idx]
                p2_pos = player2_positions_pixel[frame_idx]

                if p1_pos is not None:
                    cv2.circle(
                        frame, (int(p1_pos.x), int(p1_pos.y)), 15, (255, 0, 0), -1
                    )
                    cv2.putText(
                        frame,
                        "P1",
                        (int(p1_pos.x) - 10, int(p1_pos.y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                if p2_pos is not None:
                    cv2.circle(
                        frame, (int(p2_pos.x), int(p2_pos.y)), 15, (0, 0, 255), -1
                    )
                    cv2.putText(
                        frame,
                        "P2",
                        (int(p2_pos.x) - 10, int(p2_pos.y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                # Draw current shot info overlay
                current_shot = shot_by_frame.get(frame_idx)
                if current_shot:
                    shot_text = current_shot.shot_type.name.replace("_", " ").title()
                    direction_text = (
                        f"Dir: {current_shot.direction.name.replace('_', ' ').title()}"
                    )
                    depth_text = f"Depth: {current_shot.depth.name.title()}"
                    distance_text = (
                        f"Rebound: {current_shot.rebound_distance:.2f}m"
                        if current_shot.rebound_distance is not None
                        else "Rebound: N/A"
                    )

                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (450, 155), (0, 0, 0), -1)
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
                        distance_text,
                        (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # Draw hit markers
                if frame_idx in wall_hit_frames:
                    hit = wall_hit_frames[frame_idx]
                    pos = (int(hit.position.x), int(hit.position.y))
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
                    pos = (int(hit.position.x), int(hit.position.y))
                    cv2.drawMarker(frame, pos, (0, 165, 255), cv2.MARKER_STAR, 30, 3)
                    player_text = f"RACKET (P{hit.player_id})"
                    cv2.putText(
                        frame,
                        player_text,
                        (pos[0] + 20, pos[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                # Draw shot vectors (show for 2 seconds after racket hit)
                # Need to convert meter positions to pixel positions for visualization
                # We'll use a simple approach: find the hitting/receiving player pixel positions
                if current_shot and current_shot.wall_hit_pos:
                    vector_display_duration = int(2 * self.fps)  # 2 seconds
                    frames_since_shot = frame_idx - current_shot.frame

                    if frames_since_shot <= vector_display_duration:
                        # Get the hitting player ID from racket hits
                        hitting_racket = next(
                            (h for h in racket_hits if h.frame == current_shot.frame),
                            None,
                        )
                        if hitting_racket:
                            hitting_player_id = hitting_racket.player_id

                            # Get player pixel positions at shot frame
                            if hitting_player_id == 1:
                                hitting_player_pixel = player1_positions_pixel[
                                    current_shot.frame
                                ]
                            else:
                                hitting_player_pixel = player2_positions_pixel[
                                    current_shot.frame
                                ]

                            # Get receiving player at end frame
                            next_racket = next(
                                (
                                    h
                                    for h in racket_hits
                                    if h.frame > current_shot.frame
                                ),
                                None,
                            )
                            if next_racket:
                                receiving_player_id = next_racket.player_id
                                if receiving_player_id == 1:
                                    receiving_player_pixel = player1_positions_pixel[
                                        next_racket.frame
                                    ]
                                else:
                                    receiving_player_pixel = player2_positions_pixel[
                                        next_racket.frame
                                    ]

                                wall_pos_pixel = (
                                    int(current_shot.wall_hit_pos.x),
                                    int(current_shot.wall_hit_pos.y),
                                )

                                # Draw attack vector: Hitting Player → Wall (cyan)
                                if hitting_player_pixel:
                                    hitting_pos = (
                                        int(hitting_player_pixel.x),
                                        int(hitting_player_pixel.y),
                                    )
                                    cv2.arrowedLine(
                                        frame,
                                        hitting_pos,
                                        wall_pos_pixel,
                                        (255, 255, 0),  # Cyan
                                        3,
                                        tipLength=0.3,
                                    )
                                    cv2.putText(
                                        frame,
                                        "Attack",
                                        (hitting_pos[0] + 10, hitting_pos[1] + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 0),
                                        2,
                                    )

                                # Draw rebound vector: Wall → Receiving Player (magenta)
                                if receiving_player_pixel:
                                    receiving_pos = (
                                        int(receiving_player_pixel.x),
                                        int(receiving_player_pixel.y),
                                    )
                                    cv2.arrowedLine(
                                        frame,
                                        wall_pos_pixel,
                                        receiving_pos,
                                        (255, 0, 255),  # Magenta
                                        3,
                                        tipLength=0.3,
                                    )
                                    cv2.putText(
                                        frame,
                                        "Rebound",
                                        (
                                            wall_pos_pixel[0] + 10,
                                            wall_pos_pixel[1] + 30,
                                        ),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 0, 255),
                                        2,
                                    )

                # Write frame
                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        print(f"✓ Saved annotated video: {output_path}")

    def save_detailed_report(
        self, shots: List[ShotResult], stats: Dict[str, Any]
    ) -> None:
        """Save detailed text report with shot-by-shot breakdown"""
        if not self.config["output"]["save_metrics"]:
            return

        print("\nSaving detailed report...")

        output_path = os.path.join(self.output_dir, "report.txt")

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

            if shots:
                shot_stats = ShotStatistics.from_shots(shots)

                f.write("SHOT STATISTICS:\n")
                f.write("-" * 60 + "\n\n")

                f.write("By Type:\n")
                for shot_type, count in sorted(shot_stats.by_type.items()):
                    percentage = (count / shot_stats.total_shots) * 100
                    f.write(f"  {shot_type:30s}: {count:3d} ({percentage:5.1f}%)\n")

                f.write("\nBy Direction:\n")
                for direction, count in sorted(shot_stats.by_direction.items()):
                    percentage = (count / shot_stats.total_shots) * 100
                    f.write(f"  {direction:30s}: {count:3d} ({percentage:5.1f}%)\n")

                f.write("\nBy Depth:\n")
                for depth, count in sorted(shot_stats.by_depth.items()):
                    percentage = (count / shot_stats.total_shots) * 100
                    f.write(f"  {depth:30s}: {count:3d} ({percentage:5.1f}%)\n")

                f.write(
                    f"\nWall hit detection rate: {shot_stats.wall_hit_detection_rate*100:.1f}%\n"
                )
                if shot_stats.average_rebound_distance is not None:
                    f.write(
                        f"Average rebound distance: {shot_stats.average_rebound_distance:.2f}m\n"
                    )

                f.write("\n" + "=" * 60 + "\n")
                f.write("SHOT-BY-SHOT BREAKDOWN\n")
                f.write("=" * 60 + "\n\n")

                for i, shot in enumerate(shots, 1):
                    f.write(f"Shot #{i}\n")
                    f.write(f"  Frame:              {shot.frame}\n")
                    f.write(f"  Type:               {shot.shot_type.name}\n")
                    f.write(f"  Direction:          {shot.direction.name}\n")
                    f.write(f"  Depth:              {shot.depth.name}\n")
                    f.write(
                        f"  Racket hit pos:     ({shot.racket_hit_pos.x:.0f}, {shot.racket_hit_pos.y:.0f})\n"
                    )
                    f.write(
                        f"  Next racket pos:    ({shot.next_racket_hit_pos.x:.0f}, {shot.next_racket_hit_pos.y:.0f})\n"
                    )
                    f.write(f"  Confidence:         {shot.confidence:.2f}\n")

                    if shot.wall_hit_pos is not None:
                        f.write(
                            f"  Wall hit:           ({shot.wall_hit_pos.x:.0f}, {shot.wall_hit_pos.y:.0f}) at frame {shot.wall_hit_frame}\n"
                        )
                        f.write(f"  Rebound distance:   {shot.rebound_distance:.2f}m\n")

                    f.write("\n")

        print(f"✓ Saved detailed report: {output_path}")

    def evaluate(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        print("\n" + "=" * 60)
        print("SHOT TYPE CLASSIFICATION EVALUATION")
        print("=" * 60)

        # Step 1: Load annotation data from annotation module
        player1_positions_meter, player2_positions_meter, wall_hits, racket_hits = (
            self.load_annotation_data()
        )

        # Step 2: Classify shots
        shots = self.classify_shots(
            player1_positions_meter, player2_positions_meter, wall_hits, racket_hits
        )

        # Step 3: Print statistics
        stats = self.print_statistics(
            player1_positions_meter,
            player2_positions_meter,
            wall_hits,
            racket_hits,
            shots,
        )

        # Step 4: Create visualizations
        if self.config["output"]["save_plots"]:
            self.create_plots(shots)

        if self.config["output"]["save_video"]:
            self.create_annotated_video(
                player1_positions_meter,
                wall_hits,
                racket_hits,
                shots,
            )

        if self.config["output"]["save_metrics"]:
            self.save_detailed_report(shots, stats)

        # Done
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}")


if __name__ == "__main__":
    evaluator = ShotClassificationEvaluator()
    evaluator.evaluate()
