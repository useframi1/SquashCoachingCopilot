"""
Point Win Detection Evaluator

Runs point detection and displays results with video playback.
NO LOGIC - only visualization and output generation.
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List

from squashcopilot.modules.point_win_detection import PointWinDetector
from squashcopilot.common.utils import load_config


class PointWinEvaluator:
    """Evaluator for point win detection - DISPLAY ONLY."""

    def __init__(self, config: dict = None):
        """Initialize the evaluator."""
        if config is None:
            full_config = load_config(config_name="point_win_detection")
            self.detection_config = full_config
            config = full_config.get("tests", {})
        else:
            self.detection_config = config

        self.config = config

        # Get paths
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = self.test_dir.parent.parent.parent.parent
        data_config = self.config.get("data", {})

        # Main annotations directory
        main_annotations_dir_rel = data_config.get(
            "main_annotations_dir", "squashcopilot/annotation/annotations"
        )
        self.main_annotations_dir = project_root / main_annotations_dir_rel

        # Ground truth rally state directory
        ground_truth_dir_rel = data_config.get(
            "ground_truth_dir",
            "squashcopilot/modules/rally_state_detection/tests/data"
        )
        self.ground_truth_dir = project_root / ground_truth_dir_rel

        # Video directory
        video_dir_rel = data_config.get("video_dir", "squashcopilot/videos")
        self.video_base_dir = project_root / video_dir_rel

        # Video name
        self.video_name = data_config.get("video", "video-3")

        # Specific file paths
        self.main_csv_path = (
            self.main_annotations_dir / self.video_name / f"{self.video_name}_annotations.csv"
        )
        self.ground_truth_csv_path = (
            self.ground_truth_dir / self.video_name / f"{self.video_name}_annotations.csv"
        )
        self.video_path = self.video_base_dir / f"{self.video_name}.mp4"

        # Output directory
        output_config = self.config.get("output", {})
        output_dir_name = output_config.get("output_dir", "outputs")
        self.output_dir = self.test_dir / output_dir_name / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization config
        self.vis_config = self.config.get("visualization", {})
        self.font_scale = self.vis_config.get("font_scale", 1.2)
        self.font_thickness = self.vis_config.get("font_thickness", 2)
        self.text_color = tuple(self.vis_config.get("text_color", [255, 255, 255]))
        self.bg_color = tuple(self.vis_config.get("bg_color", [0, 0, 0]))
        self.bg_alpha = self.vis_config.get("bg_alpha", 0.7)
        self.rally_info_x = self.vis_config.get("rally_info_x", 30)
        self.rally_info_y = self.vis_config.get("rally_info_y", 50)
        self.line_spacing = self.vis_config.get("line_spacing", 40)
        self.playback_speed = self.vis_config.get("playback_speed", 1.0)

        # Initialize detector
        self.detector = PointWinDetector(self.detection_config)

        # Validate paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required input files exist."""
        if not self.main_csv_path.exists():
            raise FileNotFoundError(f"Main annotations CSV not found: {self.main_csv_path}")

        if not self.ground_truth_csv_path.exists():
            print(f"Warning: Ground truth CSV not found: {self.ground_truth_csv_path}")
            print("Will attempt to use rally_state from main CSV")
            self.ground_truth_csv_path = None

        if not self.video_path.exists():
            print(f"Warning: Video not found at {self.video_path}")
            print("Video playback will be skipped.")
            self.video_path = None

    def load_full_data(self) -> pd.DataFrame:
        """Load full CSV data with player positions for visualization."""
        print(f"\nLoading full CSV for visualization: {self.main_csv_path}")
        df_full = pd.read_csv(self.main_csv_path)
        
        if "frame" in df_full.columns and df_full.index.name != "frame":
            df_full = df_full.set_index("frame")
        
        return df_full

    def play_video_with_overlay(
        self,
        df_full: pd.DataFrame,
        rallies: List[Dict],
        point_winners: Dict[int, Dict]
    ):
        """Play full video with rally information overlaid."""
        if self.video_path is None:
            print("Video path not available, skipping playback")
            return

        print("\n" + "=" * 60)
        print("PLAYING VIDEO WITH OVERLAYS")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  Q: Quit")
        print("=" * 60)

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int((1000 / fps) / self.playback_speed)

        cv2.namedWindow("Point Win Detection", cv2.WINDOW_NORMAL)

        # Build frame to rally mapping
        frame_to_rally = {}
        for rally in rallies:
            rally_id = rally["rally_id"]
            for frame in range(rally["start_frame"], rally["end_frame"] + 1):
                frame_to_rally[frame] = rally_id

        frame_idx = 0
        paused = False
        current_rally_id = None
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            rally_id = frame_to_rally.get(frame_idx)
            
            if rally_id != current_rally_id and rally_id is not None:
                current_rally_id = rally_id
                winner_info = point_winners.get(rally_id, {})
                winner = winner_info.get("winner", -1)
                method = winner_info.get("method", "unknown")
                reason = winner_info.get("reason", "Unknown")
                
                winner_str = "Let/Replay" if winner == 0 else (f"Player {winner}" if winner > 0 else "Unknown")
                print(f"\nRally {rally_id} (Frame {frame_idx})")
                print(f"  Winner: {winner_str} [{method}]")
                print(f"  Reason: {reason}")

            overlay = frame.copy()
            
            # Draw player positions as circles on the video
            if frame_idx in df_full.index:
                # Player 1 - Blue circle
                player_1_x_pixel = df_full.loc[frame_idx, "player_1_x_pixel"] if "player_1_x_pixel" in df_full.columns else None
                player_1_y_pixel = df_full.loc[frame_idx, "player_1_y_pixel"] if "player_1_y_pixel" in df_full.columns else None
                
                if player_1_x_pixel is not None and not np.isnan(player_1_x_pixel):
                    p1_pos = (int(player_1_x_pixel), int(player_1_y_pixel))
                    cv2.circle(overlay, p1_pos, 20, (255, 0, 0), -1)  # Blue filled circle
                    cv2.circle(overlay, p1_pos, 22, (255, 255, 255), 2)  # White border
                    cv2.putText(overlay, "P1", (p1_pos[0] - 15, p1_pos[1] + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Player 2 - Red circle
                player_2_x_pixel = df_full.loc[frame_idx, "player_2_x_pixel"] if "player_2_x_pixel" in df_full.columns else None
                player_2_y_pixel = df_full.loc[frame_idx, "player_2_y_pixel"] if "player_2_y_pixel" in df_full.columns else None
                
                if player_2_x_pixel is not None and not np.isnan(player_2_x_pixel):
                    p2_pos = (int(player_2_x_pixel), int(player_2_y_pixel))
                    cv2.circle(overlay, p2_pos, 20, (0, 0, 255), -1)  # Red filled circle
                    cv2.circle(overlay, p2_pos, 22, (255, 255, 255), 2)  # White border
                    cv2.putText(overlay, "P2", (p2_pos[0] - 15, p2_pos[1] + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if rally_id is not None:
                winner_info = point_winners.get(rally_id, {})
                winner = winner_info.get("winner", -1)
                method = winner_info.get("method", "unknown")
                end_frame = winner_info.get("end_frame", -1)
                deciding_frame = winner_info.get("deciding_frame", -1)
                
                is_deciding_frame = " <- DECIDING FRAME" if frame_idx == deciding_frame else ""
                winner_display = "Let/Replay" if winner == 0 else (f"Player {winner}" if winner > 0 else "Unknown")
                
                texts = [
                    f"Rally {rally_id} [{method}]",
                    f"Winner: {winner_display}",
                    f"Frame: {frame_idx}/{end_frame}{is_deciding_frame}",
                ]
                
                max_width = 0
                for text in texts:
                    if text:
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
                        )
                        max_width = max(max_width, text_width)
                
                bg_height = len(texts) * self.line_spacing + 20
                bg_width = max_width + 40
                bg_rect = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
                bg_rect[:] = self.bg_color
                
                y1 = self.rally_info_y - 10
                y2 = min(y1 + bg_height, overlay.shape[0])
                x1 = self.rally_info_x - 10
                x2 = min(x1 + bg_width, overlay.shape[1])
                bg_rect = bg_rect[:y2-y1, :x2-x1]
                
                roi = overlay[y1:y2, x1:x2]
                overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - self.bg_alpha, bg_rect, self.bg_alpha, 0)
                
                y_offset = self.rally_info_y
                for text in texts:
                    if text:
                        cv2.putText(overlay, text, (self.rally_info_x, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                                  self.text_color, self.font_thickness)
                    y_offset += self.line_spacing

            cv2.imshow("Point Win Detection", overlay)
            
            key = cv2.waitKey(frame_delay if not paused else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo playback complete")

    def save_results(
        self,
        df: pd.DataFrame,
        rallies: List[Dict],
        point_winners: Dict[int, Dict]
    ):
        """Save detection results to files."""
        output_config = self.config.get("output", {})

        # Save CSV
        if output_config.get("save_csv", True):
            csv_path = self.output_dir / f"{self.video_name}_point_winners.csv"
            output_df = df.copy()
            output_df["rally_id"] = -1
            output_df["point_winner"] = -1
            
            for rally in rallies:
                rally_id = rally["rally_id"]
                start_frame = rally["start_frame"]
                end_frame = rally["end_frame"]
                winner = point_winners[rally_id]["winner"]
                
                output_df.loc[start_frame:end_frame, "rally_id"] = rally_id
                output_df.loc[start_frame:end_frame, "point_winner"] = winner
            
            output_df.to_csv(csv_path)
            print(f"\nCSV saved: {csv_path}")

        # Save summary
        if output_config.get("save_summary", True):
            summary_path = self.output_dir / f"{self.video_name}_summary.txt"
            
            with open(summary_path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write(f"POINT WIN DETECTION SUMMARY - {self.video_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total Rallies: {len(rallies)}\n\n")
                
                player_1_wins = sum(1 for w in point_winners.values() if w["winner"] == 1)
                player_2_wins = sum(1 for w in point_winners.values() if w["winner"] == 2)
                lets = sum(1 for w in point_winners.values() if w["winner"] == 0)
                unknown = sum(1 for w in point_winners.values() if w["winner"] == -1)
                
                f.write("OVERALL STATISTICS:\n")
                f.write(f"  Player 1 wins: {player_1_wins}\n")
                f.write(f"  Player 2 wins: {player_2_wins}\n")
                f.write(f"  Lets/Replays: {lets}\n")
                f.write(f"  Unknown: {unknown}\n\n")
                
                f.write("RALLY-BY-RALLY BREAKDOWN:\n")
                for rally in rallies:
                    rally_id = rally["rally_id"]
                    winner_info = point_winners[rally_id]
                    winner = winner_info["winner"]
                    reason = winner_info["reason"]
                    start_frame = winner_info["start_frame"]
                    end_frame = winner_info["end_frame"]
                    num_frames = winner_info["num_frames"]
                    deciding_frame = winner_info.get("deciding_frame", -1)
                    method = winner_info.get("method", "unknown")
                    
                    winner_str = "Let/Replay" if winner == 0 else (f"Player {winner}" if winner > 0 else "Unknown")
                    
                    f.write(f"\n  Rally {rally_id}:\n")
                    f.write(f"    Frames: {start_frame} - {end_frame} ({num_frames} frames)\n")
                    f.write(f"    Method: {method}\n")
                    f.write(f"    Deciding frame: {deciding_frame}\n")
                    f.write(f"    Winner: {winner_str}\n")
                    f.write(f"    Reason: {reason}\n")
            
            print(f"Summary saved: {summary_path}")

        # Save metrics
        if output_config.get("save_metrics", True):
            metrics_path = self.output_dir / f"{self.video_name}_metrics.txt"
            
            with open(metrics_path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write(f"POINT WIN DETECTION METRICS - {self.video_name}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("DETECTION STATISTICS:\n")
                f.write(f"  Total rallies: {len(rallies)}\n")
                
                if len(rallies) > 0:
                    rally_lengths = [r["num_frames"] for r in rallies]
                    f.write(f"  Average rally length: {np.mean(rally_lengths):.1f} frames\n")
                    f.write(f"  Shortest rally: {min(rally_lengths)} frames\n")
                    f.write(f"  Longest rally: {max(rally_lengths)} frames\n")
                f.write("\n")
                
                player_1_wins = sum(1 for w in point_winners.values() if w["winner"] == 1)
                player_2_wins = sum(1 for w in point_winners.values() if w["winner"] == 2)
                lets = sum(1 for w in point_winners.values() if w["winner"] == 0)
                unknown = sum(1 for w in point_winners.values() if w["winner"] == -1)
                total = len(rallies)
                
                f.write("WIN STATISTICS:\n")
                if total > 0:
                    f.write(f"  Player 1: {player_1_wins} ({player_1_wins/total*100:.1f}%)\n")
                    f.write(f"  Player 2: {player_2_wins} ({player_2_wins/total*100:.1f}%)\n")
                    f.write(f"  Lets/Replays: {lets} ({lets/total*100:.1f}%)\n")
                    f.write(f"  Unknown: {unknown} ({unknown/total*100:.1f}%)\n")
                f.write("\n")
                
                f.write("DETECTOR CONFIGURATION:\n")
                f.write(f"  Tin height: {self.detector.tin_height_y}m\n")
                f.write(f"  Out line height: {self.detector.out_line_height_y}m\n")
                f.write(f"  Court X range: {self.detector.left_wall_min_x}m - {self.detector.right_wall_max_x}m\n")
                f.write(f"  Court Y range: {self.detector.front_wall_min_y}m - {self.detector.front_wall_max_y}m\n")
            
            print(f"Metrics saved: {metrics_path}")

    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "=" * 60)
        print(f"POINT WIN DETECTION EVALUATION - {self.video_name}")
        print("=" * 60 + "\n")

        # Run detector
        df, rallies, point_winners = self.detector.run_detection(
            self.main_csv_path,
            self.ground_truth_csv_path
        )

        # Save results
        self.save_results(df, rallies, point_winners)

        # Play video
        if self.video_path is not None:
            play_video = input("\nPlay video with overlays? (y/n): ").lower().strip()
            if play_video == 'y':
                df_full = self.load_full_data()
                self.play_video_with_overlay(df_full, rallies, point_winners)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {self.output_dir}")

        return {"df": df, "rallies": rallies, "point_winners": point_winners}


if __name__ == "__main__":
    evaluator = PointWinEvaluator()
    evaluator.run_evaluation()