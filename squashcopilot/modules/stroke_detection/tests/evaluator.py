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
from squashcopilot.common.types import StrokeType
from squashcopilot.common.models.stroke import StrokeResult
from squashcopilot.modules.stroke_detection import StrokeDetector


class StrokeDetectionEvaluator:
    """Evaluates and visualizes stroke detection performance."""

    def __init__(self, config_name: str = "stroke_detection"):
        """
        Initialize evaluator with configuration.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        full_config = load_config(config_name=config_name)
        self.config = full_config["evaluator"]
        self.video_name = self.config["video_name"]

        # Get project root and build paths
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Data directory (where test annotations are) - tests/data/
        data_base_dir = project_root / self.config["data_dir"]
        self.data_dir = data_base_dir / self.video_name

        # Video path - look for annotated video in annotation module
        # (used only for creating annotated output video)
        annotations_dir = project_root / "squashcopilot/annotation/annotations"
        self.video_path = (
            annotations_dir / self.video_name / f"{self.video_name}_annotated.mp4"
        )

        # Output directory (where results will be saved)
        output_base_dir = project_root / self.config["output_dir"]
        self.output_dir = output_base_dir / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ground truth path (optional - can be provided externally)
        self.ground_truth_path = self.config.get("ground_truth_path")
        if self.ground_truth_path:
            self.ground_truth_path = project_root / self.ground_truth_path

        # Initialize detector
        self.detector = StrokeDetector(config=full_config)

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
        from squashcopilot.common.constants import KEYPOINT_NAMES

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
            DataFrame with frame, player keypoints, racket hits, etc.
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

        # Check for racket hits
        racket_hits = df[df["is_racket_hit"] == True]
        print(f"Found {len(racket_hits)} racket hits")

        return df

    def load_ground_truth(self, df: pd.DataFrame) -> Dict[int, StrokeType]:
        """
        Load ground truth stroke labels.

        This can come from either:
        1. A separate ground truth CSV file (if configured)
        2. A 'stroke_type' column in the annotations CSV
        3. Manual labels stored elsewhere

        Args:
            df: Annotations DataFrame

        Returns:
            Dictionary mapping frame number to ground truth StrokeType
        """
        ground_truth = {}

        # Try to load from separate ground truth file if specified
        if self.ground_truth_path and self.ground_truth_path.exists():
            print(f"\nLoading ground truth from {self.ground_truth_path.name}")
            gt_df = pd.read_csv(self.ground_truth_path)

            for _, row in gt_df.iterrows():
                frame = row["frame"]
                stroke_type_str = row["stroke_type"]
                ground_truth[frame] = StrokeType.from_string(stroke_type_str)

            print(f"Loaded {len(ground_truth)} ground truth stroke labels")

        # Otherwise, check if annotations DataFrame has stroke_type column
        elif "stroke_type" in df.columns:
            print("\nUsing stroke_type column from annotations")
            racket_hits_df = df[df["is_racket_hit"] == True].copy()

            for _, row in racket_hits_df.iterrows():
                if pd.notna(row.get("stroke_type")):
                    frame = row["frame"]
                    stroke_type_str = row["stroke_type"]
                    ground_truth[frame] = StrokeType.from_string(stroke_type_str)

            print(f"Found {len(ground_truth)} ground truth stroke labels")

        else:
            print("\n⚠ WARNING: No ground truth stroke labels found!")
            print("  - Add 'stroke_type' column to annotations CSV, or")
            print("  - Specify 'ground_truth_path' in config")

        return ground_truth

    def compute_metrics(
        self,
        predicted_strokes: List[StrokeResult],
        ground_truth: Dict[int, StrokeType],
    ) -> Dict:
        """
        Compute evaluation metrics.

        Args:
            predicted_strokes: List of predicted stroke results
            ground_truth: Dictionary mapping frame to ground truth stroke type

        Returns:
            Dictionary with accuracy, precision, recall, F1, and confusion matrix
        """
        if len(ground_truth) == 0:
            print("\n⚠ Cannot compute metrics without ground truth labels")
            return {}

        # Match predictions to ground truth
        y_true = []
        y_pred = []

        for frame, gt_stroke_type in ground_truth.items():
            # Find prediction for this frame
            pred_stroke = None
            for stroke in predicted_strokes:
                if stroke.frame == frame:
                    pred_stroke = stroke
                    break

            if pred_stroke:
                y_true.append(str(gt_stroke_type))
                y_pred.append(str(pred_stroke.stroke_type))
            else:
                # No prediction for this frame - treat as "neither" or skip
                y_true.append(str(gt_stroke_type))
                y_pred.append(str(StrokeType.NEITHER))

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
        """
        Plot and save confusion matrix.

        Args:
            metrics: Dictionary with confusion matrix and labels
        """
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
        predicted_strokes: List[StrokeResult],
    ) -> None:
        """
        Create video with predicted strokes overlaid.

        Args:
            predicted_strokes: List of predicted strokes
        """
        if not self.video_path.exists():
            print(f"\n⚠ Video not found: {self.video_path}")
            return

        print(f"\nCreating annotated video...")

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"  ⚠ Could not open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        output_video_path = self.output_dir / f"{self.video_name}_strokes_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        # Create lookup dictionary for stroke annotations (display for 1 second)
        display_duration = int(fps)  # Number of frames to display annotation (1 second)

        # Build dictionary that maps frame -> stroke info
        # This will show the stroke annotation starting at the hit frame for 1 second
        stroke_display_dict = {}  # frame -> StrokeResult

        for stroke in predicted_strokes:
            hit_frame = stroke.frame
            # Display from hit_frame (inclusive) for display_duration frames
            for offset in range(display_duration + 1):
                display_frame_num = hit_frame + offset
                stroke_display_dict[display_frame_num] = stroke

        # Process frames
        frame_idx = 0
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = frame.copy()

                # Check if this frame should display stroke annotation
                if frame_idx in stroke_display_dict:
                    stroke = stroke_display_dict[frame_idx]

                    # Create text: "P{player_id}: {STROKE_TYPE} ({confidence:.2f})"
                    stroke_text = (
                        f"P{stroke.player_id}: {str(stroke.stroke_type).upper()} "
                        f"({stroke.confidence:.2f})"
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
                    text_y = 50  # 50 pixels from top

                    # Draw text with background for better visibility
                    # Draw background rectangle
                    padding = 10
                    cv2.rectangle(
                        display_frame,
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (0, 0, 0),  # Black background
                        -1,
                    )

                    # Draw text
                    cv2.putText(
                        display_frame,
                        stroke_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 255, 0),  # Green text
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
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

        print(f"  ✓ Saved annotated video to {output_video_path.name}")

    def print_metrics(self, metrics: Dict) -> None:
        """
        Print evaluation metrics in a formatted way.

        Args:
            metrics: Dictionary with all computed metrics
        """
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
        predicted_strokes: List[StrokeResult],
        metrics: Optional[Dict] = None,
    ) -> None:
        """
        Save predictions and metrics to files.

        Args:
            predicted_strokes: List of predicted strokes
            metrics: Optional metrics dictionary
        """
        # Save predictions to CSV
        predictions_data = []
        for stroke in predicted_strokes:
            predictions_data.append(
                {
                    "frame": stroke.frame,
                    "player_id": stroke.player_id,
                    "stroke_type": str(stroke.stroke_type),
                    "confidence": stroke.confidence,
                }
            )

        predictions_df = pd.DataFrame(predictions_data)
        predictions_path = self.output_dir / f"{self.video_name}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\n  ✓ Saved predictions to {predictions_path.name}")

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

            print(f"  ✓ Saved metrics to {metrics_path.name}")

    def run(self) -> None:
        """Run the complete evaluation pipeline."""
        print(f"\n{'=' * 70}")
        print("STARTING EVALUATION")
        print(f"{'=' * 70}")

        # Load annotations
        df = self.load_annotations()

        # Load ground truth
        ground_truth = self.load_ground_truth(df)

        # Run detection
        print(f"\n{'=' * 70}")
        print("RUNNING STROKE DETECTION")
        print(f"{'=' * 70}")
        result = self.detector.detect_from_dataframe(df, video_name=self.video_name)
        predicted_strokes = result.strokes

        # Compute metrics if we have ground truth
        metrics = None
        if ground_truth:
            print(f"\n{'=' * 70}")
            print("COMPUTING METRICS")
            print(f"{'=' * 70}")
            metrics = self.compute_metrics(predicted_strokes, ground_truth)
            self.print_metrics(metrics)
            self.plot_confusion_matrix(metrics)

        # Save results
        print(f"\n{'=' * 70}")
        print("SAVING RESULTS")
        print(f"{'=' * 70}")
        self.save_results(predicted_strokes, metrics)

        # Create annotated video
        if self.config.get("create_video", True):
            self.create_annotated_video(predicted_strokes)

        print(f"\n{'=' * 70}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    evaluator = StrokeDetectionEvaluator()
    evaluator.run()
