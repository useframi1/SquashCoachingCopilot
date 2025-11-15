"""
Shot Type Classification Evaluator

Evaluates shot classification performance using ball tracking CSV results.
Generates comprehensive reports with visualizations and statistics.
"""

import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple, Any

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
    """Evaluator for shot type classification using ball tracking CSV"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            # Load the tests section from the shot_type_classification config
            full_config = load_config(config_name="shot_type_classification")
            config = full_config["tests"]
        self.config = config

        # Get test directory path
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Get video name from config
        self.video_name = self.config["test_video"]

        # Construct paths based on video name
        video_data_dir = os.path.join(self.test_dir, "data", self.video_name)
        self.csv_path = os.path.join(
            video_data_dir, f"{self.video_name}_ball_positions.csv"
        )
        self.video_path = os.path.join(video_data_dir, f"{self.video_name}.mp4")

        # Create output directory
        base_output_dir = os.path.join(
            self.test_dir, self.config["output"]["output_dir"]
        )
        self.output_dir = os.path.join(base_output_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\nVideo: {self.video_name}")
        print(f"CSV: {os.path.basename(self.csv_path)}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Output directory: {self.output_dir}")

        # Initialize shot classifier
        print("\nInitializing shot classifier...")
        self.shot_classifier = ShotClassifier(fps=self.fps)
        print("✓ Shot classifier initialized")

    def load_csv_data(
        self,
    ) -> Tuple[List[Optional[Point2D]], List[WallHit], List[RacketHit]]:
        """Load ball positions and hits from CSV file using common data models"""
        print("\n" + "=" * 60)
        print("LOADING BALL TRACKING DATA FROM CSV")
        print("=" * 60)

        ball_positions = []
        wall_hits = []
        racket_hits = []

        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])

                # Parse ball position using Point2D model
                x = float(row["x"]) if row["x"] else None
                y = float(row["y"]) if row["y"] else None

                if x is not None and y is not None:
                    ball_positions.append(Point2D(x=x, y=y))
                else:
                    ball_positions.append(None)

                # Parse wall hit using WallHit model
                if int(row["is_wall_hit"]) == 1 and x is not None and y is not None:
                    wall_hits.append(
                        WallHit(
                            frame=frame,
                            position=Point2D(x=x, y=y),
                            prominence=0.0,  # Not available in CSV
                        )
                    )

                # Parse racket hit using RacketHit model
                if int(row["is_racket_hit"]) == 1 and x is not None and y is not None:
                    racket_hits.append(
                        RacketHit(
                            frame=frame,
                            position=Point2D(x=x, y=y),
                            slope=0.0,  # Not available in CSV
                        )
                    )

        detected_count = sum(1 for pos in ball_positions if pos is not None)
        detection_rate = (
            (detected_count / len(ball_positions) * 100) if ball_positions else 0
        )

        print(f"\n✓ Loaded {len(ball_positions)} frames")
        print(f"✓ Ball detected in {detected_count} frames ({detection_rate:.1f}%)")
        print(f"✓ Loaded {len(wall_hits)} wall hits")
        print(f"✓ Loaded {len(racket_hits)} racket hits")

        return ball_positions, wall_hits, racket_hits

    def classify_shots(
        self,
        ball_positions: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
    ) -> List[ShotResult]:
        """Classify shots using shot classifier with proper input model"""
        print("\n" + "=" * 60)
        print("CLASSIFYING SHOTS")
        print("=" * 60)

        # Create ShotClassificationInput with the loaded data
        input_data = ShotClassificationInput(
            ball_positions=ball_positions,
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
        ball_positions: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        shots: List[ShotResult],
    ) -> Dict[str, Any]:
        """Print comprehensive statistics"""
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)

        # Ball tracking stats
        total_frames = len(ball_positions)
        detected_frames = sum(1 for pos in ball_positions if pos is not None)
        detection_rate = (
            (detected_frames / total_frames * 100) if total_frames > 0 else 0
        )

        print("\nBall Tracking:")
        print(f"   Total frames: {total_frames}")
        print(f"   Detected frames: {detected_frames}")
        print(f"   Detection rate: {detection_rate:.1f}%")

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
            if shot_stats.average_vector_angle is not None:
                print(
                    f"   Average vector angle: {shot_stats.average_vector_angle:.1f}°"
                )
            if shot_stats.average_rebound_distance is not None:
                print(
                    f"   Average rebound distance: {shot_stats.average_rebound_distance:.1f} px"
                )

        return {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": detection_rate,
            "wall_hits": len(wall_hits),
            "racket_hits": len(racket_hits),
            "shots": len(shots),
        }

    def create_trajectory_plots(
        self,
        ball_positions: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        shots: List[ShotResult],
    ) -> None:
        """Create comprehensive trajectory plots"""
        if not self.config["output"]["save_plots"]:
            return

        print("\nCreating trajectory plots...")

        # Extract coordinates from Point2D objects
        frames = list(range(len(ball_positions)))
        xs = [pos.x if pos is not None else np.nan for pos in ball_positions]
        ys = [pos.y if pos is not None else np.nan for pos in ball_positions]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Define colors for shot types
        shot_type_colors = {
            ShotType.STRAIGHT_DRIVE: "#0066CC",
            ShotType.STRAIGHT_DROP: "#66B2FF",
            ShotType.CROSS_COURT_DRIVE: "#CC0000",
            ShotType.CROSS_COURT_DROP: "#FF6666",
            ShotType.DOWN_LINE_DRIVE: "#00CC00",
            ShotType.DOWN_LINE_DROP: "#66FF66",
        }

        # Plot 1: X trajectory
        axes[0].plot(frames, xs, "k-", alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel("X Position (pixels)", fontsize=11)
        axes[0].set_title(
            "Ball Trajectory - X Coordinate with Shot Classifications",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Y trajectory
        axes[1].plot(frames, ys, "k-", alpha=0.3, linewidth=0.5)
        axes[1].set_ylabel("Y Position (pixels)", fontsize=11)
        axes[1].set_title(
            "Ball Trajectory - Y Coordinate with Shot Classifications",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].grid(True, alpha=0.3)

        # Create lookup for next racket hit
        racket_hit_frames = sorted([hit.frame for hit in racket_hits])

        # Mark shots on both trajectories
        for shot in shots:
            color = shot_type_colors.get(shot.shot_type, "gray")

            # Mark racket hit
            axes[0].axvline(shot.frame, color=color, alpha=0.6, linewidth=2)
            axes[1].axvline(shot.frame, color=color, alpha=0.6, linewidth=2)

            # Find next racket hit frame
            next_racket_frames = [f for f in racket_hit_frames if f > shot.frame]
            if next_racket_frames:
                next_racket_frame = next_racket_frames[0]

                # Draw shot segment
                segment_frames = range(
                    shot.frame, min(next_racket_frame + 1, len(frames))
                )

                if segment_frames:
                    segment_xs = [xs[f] for f in segment_frames]
                    segment_ys = [ys[f] for f in segment_frames]
                    axes[0].plot(
                        segment_frames,
                        segment_xs,
                        color=color,
                        linewidth=2.5,
                        alpha=0.8,
                    )
                    axes[1].plot(
                        segment_frames,
                        segment_ys,
                        color=color,
                        linewidth=2.5,
                        alpha=0.8,
                    )

        # Plot 3: Shot type timeline
        axes[2].set_ylim(-0.5, 5.5)
        axes[2].set_xlabel("Frame", fontsize=11)
        axes[2].set_ylabel("Shot Type", fontsize=11)
        axes[2].set_title("Shot Type Timeline", fontsize=12, fontweight="bold")
        axes[2].grid(True, alpha=0.3, axis="x")

        # Create shot type index mapping
        shot_type_index = {
            ShotType.STRAIGHT_DRIVE: 0,
            ShotType.STRAIGHT_DROP: 1,
            ShotType.CROSS_COURT_DRIVE: 2,
            ShotType.CROSS_COURT_DROP: 3,
            ShotType.DOWN_LINE_DRIVE: 4,
            ShotType.DOWN_LINE_DROP: 5,
        }

        # Plot each shot as a horizontal bar
        for shot in shots:
            y_pos = shot_type_index.get(shot.shot_type, 0)
            color = shot_type_colors.get(shot.shot_type, "gray")

            # Find next racket hit frame to calculate width
            next_racket_frames = [f for f in racket_hit_frames if f > shot.frame]
            if next_racket_frames:
                width = next_racket_frames[0] - shot.frame

                axes[2].barh(
                    y_pos,
                    width=width,
                    left=shot.frame,
                    height=0.7,
                    color=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Set y-axis labels
        axes[2].set_yticks(list(shot_type_index.values()))
        axes[2].set_yticklabels(
            [st.name.replace("_", " ").title() for st in shot_type_index.keys()],
            fontsize=9,
        )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=color,
                edgecolor="black",
                label=st.name.replace("_", " ").title(),
            )
            for st, color in shot_type_colors.items()
        ]
        axes[2].legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, "trajectory_plots.png")
        plt.savefig(
            plot_path, dpi=self.config["output"]["plot_dpi"], bbox_inches="tight"
        )
        print(f"✓ Saved trajectory plots: {plot_path}")

        plt.close()

    def create_annotated_video(
        self,
        ball_positions: List[Optional[Point2D]],
        wall_hits: List[WallHit],
        racket_hits: List[RacketHit],
        shots: List[ShotResult],
    ) -> None:
        """Create annotated video with ball tracking and shot classifications"""
        if not self.config["output"]["save_video"]:
            return

        print("\nCreating annotated video...")

        output_path = os.path.join(self.output_dir, "annotated.mp4")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Open video
        cap = cv2.VideoCapture(self.video_path)

        # Get visualization config
        trace_length = self.config["tracking"]["trace_length"]
        trace_color = tuple(self.config["tracking"]["trace_color"])
        trace_thickness = self.config["tracking"]["trace_thickness"]

        # Create frame-to-shot lookup
        racket_hit_frames_list = sorted([hit.frame for hit in racket_hits])

        shot_by_frame = {}
        for shot in shots:
            # Find next racket hit frame
            next_racket_frames = [f for f in racket_hit_frames_list if f > shot.frame]
            if next_racket_frames:
                end_frame = next_racket_frames[0]
                for frame_idx in range(shot.frame, end_frame + 1):
                    if frame_idx < len(ball_positions):
                        shot_by_frame[frame_idx] = shot

        # Shot type colors (BGR for OpenCV)
        shot_colors = {
            ShotType.STRAIGHT_DRIVE: (204, 102, 0),
            ShotType.STRAIGHT_DROP: (255, 178, 102),
            ShotType.CROSS_COURT_DRIVE: (0, 0, 204),
            ShotType.CROSS_COURT_DROP: (102, 102, 255),
            ShotType.DOWN_LINE_DRIVE: (0, 204, 0),
            ShotType.DOWN_LINE_DROP: (102, 255, 102),
        }

        # Create frame-to-hit lookup (with display duration)
        hit_display_duration = 30

        wall_hit_frames = {}
        for hit in wall_hits:
            for offset in range(hit_display_duration):
                frame = hit.frame + offset
                if frame < len(ball_positions):
                    wall_hit_frames[frame] = hit

        racket_hit_frames = {}
        for hit in racket_hits:
            for offset in range(hit_display_duration):
                frame = hit.frame + offset
                if frame < len(ball_positions):
                    racket_hit_frames[frame] = hit

        # Process frames (process all frames from CSV)
        max_frames = len(ball_positions)
        with tqdm(total=max_frames, desc="Rendering video", unit="frame") as pbar:
            frame_idx = 0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw ball trace
                current_shot = shot_by_frame.get(frame_idx)
                if current_shot:
                    trace_col = shot_colors.get(current_shot.shot_type, trace_color)
                else:
                    trace_col = trace_color

                for i in range(trace_length):
                    idx = frame_idx - i
                    if 0 <= idx < len(ball_positions):
                        pos = ball_positions[idx]
                        if pos is not None:
                            cv2.circle(
                                frame,
                                (int(pos.x), int(pos.y)),
                                radius=max(1, trace_thickness - i // 2),
                                color=trace_col,
                                thickness=-1,
                            )

                # Draw current shot info overlay
                if current_shot:
                    shot_text = current_shot.shot_type.name.replace("_", " ").title()
                    direction_text = (
                        f"Dir: {current_shot.direction.name.replace('_', ' ').title()}"
                    )
                    depth_text = f"Depth: {current_shot.depth.name.title()}"
                    angle_text = (
                        f"Angle: {current_shot.vector_angle_deg:.1f}°"
                        if current_shot.vector_angle_deg is not None
                        else "Angle: N/A"
                    )

                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
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
                        angle_text,
                        (20, 120),
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
                    cv2.putText(
                        frame,
                        "RACKET",
                        (pos[0] + 20, pos[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                # Draw shot vectors (show for 2 seconds after racket hit)
                if current_shot:
                    vector_display_duration = int(2 * self.fps)  # 2 seconds
                    frames_since_shot = frame_idx - current_shot.frame

                    if (
                        frames_since_shot <= vector_display_duration
                        and current_shot.wall_hit_pos
                    ):
                        # Draw vector 1: Racket → Wall (attack vector) in cyan
                        racket_pos = (
                            int(current_shot.racket_hit_pos.x),
                            int(current_shot.racket_hit_pos.y),
                        )
                        wall_pos = (
                            int(current_shot.wall_hit_pos.x),
                            int(current_shot.wall_hit_pos.y),
                        )
                        cv2.arrowedLine(
                            frame,
                            racket_pos,
                            wall_pos,
                            (255, 255, 0),  # Cyan
                            3,
                            tipLength=0.2,
                        )

                        # Draw vector 2: Wall → Next Racket (rebound vector) in magenta
                        next_racket_pos = (
                            int(current_shot.next_racket_hit_pos.x),
                            int(current_shot.next_racket_hit_pos.y),
                        )
                        cv2.arrowedLine(
                            frame,
                            wall_pos,
                            next_racket_pos,
                            (255, 0, 255),  # Magenta
                            3,
                            tipLength=0.2,
                        )

                        # Add vector labels
                        cv2.putText(
                            frame,
                            "Attack",
                            (
                                int((racket_pos[0] + wall_pos[0]) / 2),
                                int((racket_pos[1] + wall_pos[1]) / 2) - 10,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Rebound",
                            (
                                int((wall_pos[0] + next_racket_pos[0]) / 2),
                                int((wall_pos[1] + next_racket_pos[1]) / 2) - 10,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            2,
                        )

                out.write(frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        print(f"✓ Saved annotated video: {output_path}")

    def save_detailed_report(
        self, shots: List[ShotResult], stats: Dict[str, Any]
    ) -> None:
        """Save detailed text report"""
        if not self.config["output"]["save_metrics"]:
            return

        print("\nSaving detailed report...")

        output_path = os.path.join(self.output_dir, "report.txt")

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SHOT TYPE CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video: {self.video_name}\n")
            f.write(f"Resolution: {self.width}x{self.height} @ {self.fps} fps\n\n")

            f.write("TRACKING STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total frames:        {stats['total_frames']}\n")
            f.write(f"Detected frames:     {stats['detected_frames']}\n")
            f.write(f"Detection rate:      {stats['detection_rate']:.1f}%\n")
            f.write(f"Wall hits:           {stats['wall_hits']}\n")
            f.write(f"Racket hits:         {stats['racket_hits']}\n")
            f.write(f"Classified shots:    {stats['shots']}\n\n")

            if shots:
                f.write("SHOT-BY-SHOT BREAKDOWN:\n")
                f.write("-" * 80 + "\n\n")

                for i, shot in enumerate(shots, 1):
                    f.write(f"Shot {i}:\n")
                    f.write(f"  Frame:              {shot.frame}\n")
                    f.write(
                        f"  Type:               {shot.shot_type.name.replace('_', ' ').title()}\n"
                    )
                    f.write(
                        f"  Direction:          {shot.direction.name.replace('_', ' ').title()}\n"
                    )
                    f.write(f"  Depth:              {shot.depth.name.title()}\n")
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
                        f.write(f"  Vector angle:       {shot.vector_angle_deg:.1f}°\n")
                        f.write(
                            f"  Rebound distance:   {shot.rebound_distance:.0f} px\n"
                        )

                    f.write("\n")

        print(f"✓ Saved detailed report: {output_path}")

    def evaluate(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        print("\n" + "=" * 60)
        print("SHOT TYPE CLASSIFICATION EVALUATION")
        print("=" * 60)

        # Step 1: Load ball tracking data from CSV
        ball_positions, wall_hits, racket_hits = self.load_csv_data()

        # Step 2: Classify shots
        shots = self.classify_shots(ball_positions, wall_hits, racket_hits)

        # Step 3: Print statistics
        stats = self.print_statistics(ball_positions, wall_hits, racket_hits, shots)

        # Step 4: Create visualizations
        if self.config["output"]["save_plots"]:
            self.create_trajectory_plots(ball_positions, wall_hits, racket_hits, shots)

        if self.config["output"]["save_video"]:
            self.create_annotated_video(ball_positions, wall_hits, racket_hits, shots)

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
