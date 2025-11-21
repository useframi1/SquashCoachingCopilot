"""
Stroke Detection Evaluator

Evaluates stroke detection performance against ground truth annotations.
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm

from squashcopilot.common.utils import load_config
from squashcopilot.common.constants import KEYPOINT_NAMES
from squashcopilot.common import (
    StrokeClassificationInput,
    StrokeClassificationOutput,
)
from squashcopilot.modules.stroke_detection import StrokeDetector


class StrokeDetectionEvaluator:
    """Evaluates and visualizes stroke detection performance."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize evaluator with configuration.

        Args:
            config: Configuration dictionary. If None, loads from stroke_detection.yaml
        """
        if config is None:
            full_config = load_config(config_name="stroke_detection")
            self.config = full_config["tests"]
            self.full_config = full_config
        else:
            self.config = config.get("tests", config)
            self.full_config = config

        self.video_name = self.config["video_name"]

        # Get project root and build paths
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Data directory (where test annotations are) - tests/data/
        data_base_dir = project_root / self.config["data_dir"]
        self.data_dir = data_base_dir / self.video_name

        # Video directory (where video files are stored)
        video_dir_rel = self.config.get("video_dir", "squashcopilot/videos")
        video_dir = project_root / video_dir_rel
        self.video_path = video_dir / f"{self.video_name}.mp4"

        # Output directory (where results will be saved)
        output_base_dir = project_root / self.config["output_dir"]
        self.output_dir = output_base_dir / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detector
        self.detector = StrokeDetector(config=self.full_config)

        print(f"\n{'=' * 70}")
        print(f"STROKE DETECTION EVALUATOR - {self.video_name}")
        print(f"{'=' * 70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Video path: {self.video_path}")
        print(f"Output directory: {self.output_dir}")

    def _convert_windowed_to_raw(self, windowed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert windowed sequence format to raw annotations format.

        Windowed format: hit_frame, frame, player_id, stroke_type, kp_*
        Raw format: frame, player_1_kp_*, player_2_kp_*, is_racket_hit, racket_hit_player_id

        Args:
            windowed_df: DataFrame in windowed sequence format

        Returns:
            DataFrame in raw annotations format
        """
        # Get all unique frames
        all_frames = sorted(windowed_df["frame"].dropna().unique())

        # Build map of hit frames to player_id and stroke_type
        hit_frames_map = {}
        for hit_frame in windowed_df["hit_frame"].unique():
            hit_data = windowed_df[windowed_df["hit_frame"] == hit_frame].iloc[0]
            hit_frames_map[hit_frame] = {
                "player_id": int(hit_data["player_id"]),
                "stroke_type": hit_data["stroke_type"],
            }

        # Group windowed data by frame and player_id
        frame_player_data = {}
        for _, row in windowed_df.iterrows():
            frame = row["frame"]
            player_id = int(row["player_id"])
            key = (frame, player_id)
            frame_player_data[key] = row

        # Build raw annotations format
        raw_rows = []
        for frame in all_frames:
            row_data = {"frame": int(frame)}

            # Check if this frame is a hit frame
            row_data["is_racket_hit"] = frame in hit_frames_map
            if frame in hit_frames_map:
                row_data["racket_hit_player_id"] = hit_frames_map[frame]["player_id"]
                row_data["stroke_type"] = hit_frames_map[frame]["stroke_type"]
            else:
                row_data["racket_hit_player_id"] = None

            # Add keypoints for both players
            for player_id in [1, 2]:
                key = (frame, player_id)
                if key in frame_player_data:
                    player_data = frame_player_data[key]
                    for kp_name in KEYPOINT_NAMES:
                        row_data[f"player_{player_id}_kp_{kp_name}_x"] = (
                            player_data.get(f"kp_{kp_name}_x", 0.0)
                        )
                        row_data[f"player_{player_id}_kp_{kp_name}_y"] = (
                            player_data.get(f"kp_{kp_name}_y", 0.0)
                        )
                else:
                    # Fill with zeros if player data not available for this frame
                    for kp_name in KEYPOINT_NAMES:
                        row_data[f"player_{player_id}_kp_{kp_name}_x"] = 0.0
                        row_data[f"player_{player_id}_kp_{kp_name}_y"] = 0.0

            raw_rows.append(row_data)

        raw_df = pd.DataFrame(raw_rows)
        print(
            f"Converted {len(windowed_df)} windowed rows to {len(raw_df)} raw annotation frames"
        )
        return raw_df

    def load_annotations(self) -> pd.DataFrame:
        """
        Load annotations from CSV file.

        The CSV can be in two formats:
        1. Raw annotations: frame, player_1_kp_*, player_2_kp_*, is_racket_hit, racket_hit_player_id
        2. Windowed sequences: hit_frame, frame, player_id, stroke_type, kp_*

        For windowed format, adds is_racket_hit column where frame == hit_frame.

        Returns:
            DataFrame with frame as index and player keypoints, racket hits, etc.
        """
        csv_path = self.data_dir / f"{self.video_name}_annotations.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} rows from {csv_path.name}")

        # Check if this is windowed sequence format (has 'hit_frame' column)
        if "hit_frame" in df.columns:
            print(
                "Detected windowed sequence format - converting to raw annotations format..."
            )
            df = self._convert_windowed_to_raw(df)

        # Set frame as index
        if "frame" in df.columns:
            df = df.set_index("frame")

        # Check for racket hits
        racket_hits = df[df["is_racket_hit"] == True]
        print(f"Found {len(racket_hits)} racket hits")

        return df

    def extract_player_keypoints(self, df: pd.DataFrame) -> Dict[int, List]:
        """
        Extract player keypoints from DataFrame.

        Args:
            df: DataFrame with player keypoint columns

        Returns:
            Dictionary mapping player_id to list of keypoint arrays
        """
        player_keypoints = {1: [], 2: []}

        for frame_idx in df.index:
            row = df.loc[frame_idx]

            for player_id in [1, 2]:
                keypoints = []
                has_keypoints = False

                for kp_name in KEYPOINT_NAMES:
                    x_col = f"player_{player_id}_kp_{kp_name}_x"
                    y_col = f"player_{player_id}_kp_{kp_name}_y"

                    x = row.get(x_col, 0.0)
                    y = row.get(y_col, 0.0)

                    if pd.notna(x) and pd.notna(y) and (x != 0 or y != 0):
                        has_keypoints = True

                    keypoints.append(
                        [
                            float(x) if pd.notna(x) else 0.0,
                            float(y) if pd.notna(y) else 0.0,
                        ]
                    )

                if has_keypoints:
                    player_keypoints[player_id].append(np.array(keypoints))
                else:
                    player_keypoints[player_id].append(None)

        return player_keypoints

    def load_ground_truth(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Load ground truth stroke labels.

        This can come from either:
        1. A separate ground truth CSV file (if configured)
        2. A 'stroke_type' column in the annotations CSV

        Args:
            df: Annotations DataFrame

        Returns:
            Dictionary mapping frame number to ground truth stroke type string
        """
        ground_truth = {}

        # Check if annotations DataFrame has stroke_type column
        if "stroke_type" in df.columns:
            print("\nUsing stroke_type column from annotations")
            racket_hits_df = df[df["is_racket_hit"] == True].copy()

            for frame_idx, row in racket_hits_df.iterrows():
                if pd.notna(row.get("stroke_type")) and row.get("stroke_type") != "":
                    ground_truth[frame_idx] = row["stroke_type"]

            print(f"Found {len(ground_truth)} ground truth stroke labels")

        else:
            print("\nWARNING: No ground truth stroke labels found!")
            print("  - Add 'stroke_type' column to annotations CSV, or")
            print("  - Specify 'ground_truth_path' in config")

        return ground_truth

    def compute_metrics(
        self,
        result_df: pd.DataFrame,
        ground_truth: Dict[int, str],
    ) -> Dict:
        """
        Compute evaluation metrics.

        Args:
            result_df: DataFrame with stroke predictions
            ground_truth: Dictionary mapping frame to ground truth stroke type

        Returns:
            Dictionary with accuracy, precision, recall, F1, and confusion matrix
        """
        if len(ground_truth) == 0:
            print("\nCannot compute metrics without ground truth labels")
            return {}

        # Match predictions to ground truth
        y_true = []
        y_pred = []

        for frame, gt_stroke_type in ground_truth.items():
            # Get prediction for this frame
            if frame in result_df.index:
                pred_stroke_type = result_df.loc[frame, "stroke_type"]
                if pd.isna(pred_stroke_type) or pred_stroke_type == "":
                    pred_stroke_type = "neither"
            else:
                pred_stroke_type = "neither"

            y_true.append(gt_stroke_type)
            y_pred.append(pred_stroke_type)

        # Get unique labels
        labels = sorted(set(y_true + y_pred))

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        # Per-class metrics
        per_class_precision = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
            "labels": labels,
            "per_class_precision": dict(zip(labels, per_class_precision)),
            "per_class_recall": dict(zip(labels, per_class_recall)),
            "per_class_f1": dict(zip(labels, per_class_f1)),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def plot_confusion_matrix(self, metrics: Dict) -> None:
        """Plot and save confusion matrix."""
        if "confusion_matrix" not in metrics:
            return

        conf_matrix = metrics["confusion_matrix"]
        labels = metrics["labels"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
        plt.title(f"Stroke Detection Confusion Matrix - {self.video_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{self.video_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved confusion matrix to {plot_path.name}")

    def create_annotated_video(
        self,
        df: pd.DataFrame,
        result: StrokeClassificationOutput,
    ) -> None:
        """
        Create video with predicted strokes overlaid for evaluated frames only.

        Args:
            df: Original annotations DataFrame (for frame range)
            result: StrokeClassificationOutput with predictions
        """
        if not self.video_path.exists():
            print(f"\nVideo not found: {self.video_path}")
            return

        print(f"\nCreating annotated video...")

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"  Could not open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get frame range from annotations
        start_frame = int(df.index.min())
        end_frame = int(df.index.max())
        num_frames = end_frame - start_frame + 1

        print(f"Processing frames {start_frame} to {end_frame} ({num_frames} frames)")

        # Setup video writer
        output_video_path = self.output_dir / f"{self.video_name}_strokes_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        # Create lookup dictionary for stroke annotations (display for 1 second)
        display_duration = int(fps)  # Number of frames to display annotation (1 second)
        result_df = result.df

        # Build dictionary that maps frame -> stroke info
        stroke_display_dict = {}

        # Find frames with stroke predictions
        strokes_df = result_df[result_df["stroke_type"] != ""]
        for stroke_frame in strokes_df.index:
            stroke_type = result_df.loc[stroke_frame, "stroke_type"]
            confidence = result_df.loc[stroke_frame, "stroke_confidence"]
            player_id = result_df.loc[stroke_frame, "racket_hit_player_id"]

            for offset in range(display_duration + 1):
                display_frame = stroke_frame + offset
                if display_frame <= end_frame:
                    stroke_display_dict[display_frame] = {
                        "stroke_type": stroke_type,
                        "confidence": confidence,
                        "player_id": player_id,
                        "frame": stroke_frame,
                    }

        # Player colors (BGR)
        player_colors = {
            1: (255, 0, 0),  # Blue for Player 1
            2: (0, 0, 255),  # Red for Player 2
        }

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Process frames
        with tqdm(total=num_frames, desc="Processing frames") as pbar:
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = frame.copy()

                # Draw player IDs at their positions
                if frame_idx in df.index:
                    row = df.loc[frame_idx]

                    for player_id in [1, 2]:
                        # Get hip center as player position
                        left_hip_x = row.get(f"player_{player_id}_kp_left_hip_x", 0)
                        left_hip_y = row.get(f"player_{player_id}_kp_left_hip_y", 0)
                        right_hip_x = row.get(f"player_{player_id}_kp_right_hip_x", 0)
                        right_hip_y = row.get(f"player_{player_id}_kp_right_hip_y", 0)

                        # Calculate center position
                        if (
                            pd.notna(left_hip_x)
                            and pd.notna(right_hip_x)
                            and (left_hip_x != 0 or right_hip_x != 0)
                        ):
                            center_x = int((left_hip_x + right_hip_x) / 2)
                            center_y = int((left_hip_y + right_hip_y) / 2)

                            color = player_colors[player_id]

                            # Draw player ID label
                            cv2.putText(
                                display_frame,
                                f"P{player_id}",
                                (center_x - 15, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2,
                            )

                # Check if this frame should display stroke annotation
                if frame_idx in stroke_display_dict:
                    stroke_info = stroke_display_dict[frame_idx]

                    # Create text
                    player_id = stroke_info["player_id"]
                    if pd.notna(player_id):
                        stroke_text = (
                            f"P{int(player_id)}: {stroke_info['stroke_type'].upper()} "
                            f"({stroke_info['confidence']:.2f})"
                        )
                    else:
                        stroke_text = (
                            f"{stroke_info['stroke_type'].upper()} "
                            f"({stroke_info['confidence']:.2f})"
                        )

                    # Get text size to center it
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size = cv2.getTextSize(
                        stroke_text, font, font_scale, thickness
                    )[0]

                    # Calculate position for top center
                    text_x = (width - text_size[0]) // 2
                    text_y = 50

                    # Draw background rectangle
                    padding = 10
                    cv2.rectangle(
                        display_frame,
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (0, 0, 0),
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        display_frame,
                        stroke_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                    )

                # Draw frame number
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_idx}",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                out.write(display_frame)
                pbar.update(1)

        cap.release()
        out.release()

        print(f"  Saved annotated video to {output_video_path.name}")

    def print_metrics(self, metrics: Dict) -> None:
        """Print evaluation metrics in a formatted way."""
        if not metrics:
            return

        print(f"\n{'=' * 70}")
        print("EVALUATION METRICS")
        print(f"{'=' * 70}")

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        print(f"\nPer-Class Metrics:")
        for label in metrics["labels"]:
            print(f"\n  {label.upper()}:")
            print(f"    Precision: {metrics['per_class_precision'][label]:.4f}")
            print(f"    Recall:    {metrics['per_class_recall'][label]:.4f}")
            print(f"    F1 Score:  {metrics['per_class_f1'][label]:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  Labels: {metrics['labels']}")
        print(metrics["confusion_matrix"])

    def save_results(
        self,
        result: StrokeClassificationOutput,
        metrics: Optional[Dict] = None,
    ) -> None:
        """Save predictions and metrics to files."""
        result_df = result.df

        # Save predictions to CSV
        strokes_df = result_df[result_df["stroke_type"] != ""][
            ["stroke_type", "stroke_confidence", "racket_hit_player_id"]
        ].copy()

        predictions_path = self.output_dir / f"{self.video_name}_predictions.csv"
        strokes_df.to_csv(predictions_path)
        print(f"\n  Saved predictions to {predictions_path.name}")

        # Save metrics to text file if available
        if metrics:
            metrics_path = self.output_dir / f"{self.video_name}_metrics.txt"
            with open(metrics_path, "w") as f:
                f.write(f"Stroke Detection Evaluation - {self.video_name}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Overall Metrics:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:  {metrics['f1']:.4f}\n\n")
                f.write(f"Per-Class Metrics:\n")
                for label in metrics["labels"]:
                    f.write(f"\n  {label.upper()}:\n")
                    f.write(
                        f"    Precision: {metrics['per_class_precision'][label]:.4f}\n"
                    )
                    f.write(
                        f"    Recall:    {metrics['per_class_recall'][label]:.4f}\n"
                    )
                    f.write(f"    F1 Score:  {metrics['per_class_f1'][label]:.4f}\n")

            print(f"  Saved metrics to {metrics_path.name}")

    def run_evaluation(self) -> Dict:
        """Run the complete evaluation pipeline."""
        print(f"\n{'=' * 70}")
        print("STARTING EVALUATION")
        print(f"{'=' * 70}")

        # Load annotations
        df = self.load_annotations()

        # Extract player keypoints from DataFrame
        print("\nExtracting player keypoints...")
        player_keypoints = self.extract_player_keypoints(df)
        print(
            f"  Player 1: {sum(1 for kp in player_keypoints[1] if kp is not None)} frames with keypoints"
        )
        print(
            f"  Player 2: {sum(1 for kp in player_keypoints[2] if kp is not None)} frames with keypoints"
        )

        # Load ground truth
        ground_truth = self.load_ground_truth(df)

        # Run detection
        print(f"\n{'=' * 70}")
        print("RUNNING STROKE DETECTION")
        print(f"{'=' * 70}")

        # Create input for detector
        input_data = StrokeClassificationInput(
            df=df,
            player_keypoints=player_keypoints,
        )

        result = self.detector.detect_strokes(input_data)
        print(f"Detected {result.num_strokes} strokes")
        print(f"Stroke counts: {result.stroke_counts}")

        # Compute metrics if we have ground truth
        metrics = None
        if ground_truth:
            print(f"\n{'=' * 70}")
            print("COMPUTING METRICS")
            print(f"{'=' * 70}")
            metrics = self.compute_metrics(result.df, ground_truth)
            self.print_metrics(metrics)
            self.plot_confusion_matrix(metrics)

        # Save results
        print(f"\n{'=' * 70}")
        print("SAVING RESULTS")
        print(f"{'=' * 70}")
        self.save_results(result, metrics)

        # Create annotated video
        if self.config.get("create_video", True):
            self.create_annotated_video(df, result)

        print(f"\n{'=' * 70}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Results saved to: {self.output_dir}")

        return {
            "result": result,
            "metrics": metrics,
            "ground_truth": ground_truth,
        }


if __name__ == "__main__":
    evaluator = StrokeDetectionEvaluator()
    evaluator.run_evaluation()
