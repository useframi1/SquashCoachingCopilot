"""
Ball Tracking Evaluator

Evaluates ball tracking performance by comparing predictions from the annotation
module against ground truth annotations. Generates metrics, plots, and annotated videos.

Outputs:
    outputs/{video-name}/
        - ball_positions.png
        - ball_annotated.mp4
        - metrics.txt
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

from squashcopilot.common.utils import load_config


class BallTrackingEvaluator:
    """Evaluates ball tracking performance using pre-computed predictions."""

    def __init__(self):
        """Initialize evaluator with configuration."""
        self.full_config = load_config(config_name="ball_tracking")
        self.config = self.full_config["tests"]

        # Get project root
        self.project_root = Path(__file__).parent.parent.parent.parent.parent

        # Video settings
        self.video_name = self.config["video"]["video_name"]
        video_dir = self.project_root / self.config["video"]["video_dir"]
        self.video_path = video_dir / f"{self.video_name}.mp4"

        # Predictions and ground truth paths
        predictions_dir = self.project_root / self.config["predictions_dir"]
        self.predictions_path = predictions_dir / self.video_name / f"{self.video_name}_annotations.csv"

        gt_dir = self.project_root / self.config["ground_truth_dir"]
        self.ground_truth_path = gt_dir / f"{self.video_name}_ball_annotations.csv"

        # Evaluation settings
        self.pixel_tolerance = self.config.get("pixel_tolerance", 10)
        self.use_ground_truth = self.config.get("use_ground_truth", True)

        # Output settings - save to ball_tracking/tests/outputs/
        test_dir = Path(__file__).parent
        output_base = test_dir / self.config["output"]["output_dir"]
        self.output_dir = output_base / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization settings
        self.trace_length = self.config["tracking"]["trace_length"]
        self.trace_color = tuple(self.config["tracking"]["trace_color"])
        self.trace_thickness = self.config["tracking"]["trace_thickness"]
        self.plot_dpi = self.config["output"]["plot_dpi"]

        # Video properties (set during processing)
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    def load_predictions(self) -> Optional[pd.DataFrame]:
        """Load ball predictions from the annotation module CSV.

        Returns:
            DataFrame with predictions or None if not found
        """
        if not self.predictions_path.exists():
            print(f"Predictions file not found: {self.predictions_path}")
            return None

        df = pd.read_csv(self.predictions_path)

        # Check for required columns
        if "ball_x" not in df.columns or "ball_y" not in df.columns:
            print("Predictions missing ball_x/ball_y columns")
            return None

        # Rename 'frame' to 'frame_number' if needed
        if "frame" in df.columns and "frame_number" not in df.columns:
            df = df.rename(columns={"frame": "frame_number"})

        # Create ball_detected column based on whether ball_x/ball_y are valid
        df["ball_detected"] = df["ball_x"].notna() & df["ball_y"].notna()

        # Set frame_number as index
        df = df.set_index("frame_number")

        return df

    def load_ground_truth(self) -> Optional[pd.DataFrame]:
        """Load ground truth annotations from CSV.

        Returns:
            DataFrame with ground truth or None if not found
        """
        if not self.ground_truth_path.exists():
            print(f"Ground truth file not found: {self.ground_truth_path}")
            return None

        df = pd.read_csv(self.ground_truth_path)

        # Validate required columns
        required_cols = ["frame_number", "ball_x", "ball_y", "has_ball"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Ground truth missing columns: {missing}")
            return None

        # Convert has_ball to boolean if needed
        df["has_ball"] = df["has_ball"].astype(bool)

        # Set frame_number as index
        df = df.set_index("frame_number")

        return df

    def evaluate_against_ground_truth(
        self,
        predictions_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> Dict:
        """Compare predictions to ground truth with pixel tolerance.

        Args:
            predictions_df: DataFrame with predictions (indexed by frame_number)
            ground_truth_df: DataFrame with ground truth (indexed by frame_number)

        Returns:
            Dictionary with evaluation metrics
        """
        # Get common frames
        common_frames = predictions_df.index.intersection(ground_truth_df.index)
        if len(common_frames) == 0:
            print("No common frames between predictions and ground truth!")
            return {}

        print(f"Evaluating on {len(common_frames)} annotated frames")

        # Initialize counters
        tp = 0  # True Positive: correct detection within tolerance
        fp = 0  # False Positive: wrong detection or outside tolerance
        fn = 0  # False Negative: missed detection
        tn = 0  # True Negative: correctly no detection

        distances = []  # For MAE/MSE calculation

        for frame_num in common_frames:
            pred_row = predictions_df.loc[frame_num]
            gt_row = ground_truth_df.loc[frame_num]

            pred_detected = pred_row["ball_detected"]
            gt_has_ball = gt_row["has_ball"]

            if gt_has_ball and pred_detected:
                # Both say ball exists - check position accuracy
                pred_x, pred_y = pred_row["ball_x"], pred_row["ball_y"]
                gt_x, gt_y = gt_row["ball_x"], gt_row["ball_y"]

                # Handle NaN values
                if pd.isna(pred_x) or pd.isna(pred_y) or pd.isna(gt_x) or pd.isna(gt_y):
                    fp += 1
                    continue

                distance = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                distances.append(distance)

                if distance <= self.pixel_tolerance:
                    tp += 1
                else:
                    fp += 1  # Detection exists but position is wrong

            elif gt_has_ball and not pred_detected:
                fn += 1

            elif not gt_has_ball and pred_detected:
                fp += 1

            else:
                tn += 1

        # Calculate metrics
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # Position error metrics
        mae = np.mean(distances) if distances else 0.0
        mse = np.mean(np.array(distances) ** 2) if distances else 0.0
        rmse = np.sqrt(mse)

        # Percentage of detections within tolerance
        within_tolerance = sum(1 for d in distances if d <= self.pixel_tolerance)
        within_tolerance_pct = (within_tolerance / len(distances) * 100) if distances else 0.0

        metrics = {
            "total_frames": len(common_frames),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "pixel_tolerance": self.pixel_tolerance,
            "mae_pixels": round(mae, 2),
            "mse_pixels": round(mse, 2),
            "rmse_pixels": round(rmse, 2),
            "within_tolerance_pct": round(within_tolerance_pct, 2),
        }

        return metrics

    def save_metrics(self, metrics: Dict) -> None:
        """Save evaluation metrics to file."""
        output_path = self.output_dir / "metrics.txt"

        with open(output_path, "w") as f:
            f.write("BALL TRACKING EVALUATION\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Video: {self.video_name}\n")
            f.write(f"Evaluated Frames: {metrics['total_frames']}\n")
            f.write(f"Pixel Tolerance: {metrics['pixel_tolerance']} pixels\n\n")

            f.write("Detection Metrics:\n")
            f.write(f"  True Positives:   {metrics['true_positives']}\n")
            f.write(f"  False Positives:  {metrics['false_positives']}\n")
            f.write(f"  False Negatives:  {metrics['false_negatives']}\n")
            f.write(f"  True Negatives:   {metrics['true_negatives']}\n\n")

            f.write("Performance Metrics:\n")
            f.write(f"  Precision:        {metrics['precision']:.4f}\n")
            f.write(f"  Recall:           {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:         {metrics['f1_score']:.4f}\n")
            f.write(f"  Accuracy:         {metrics['accuracy']:.4f}\n\n")

            f.write("Position Error (for correct detections):\n")
            f.write(f"  MAE:              {metrics['mae_pixels']:.2f} pixels\n")
            f.write(f"  RMSE:             {metrics['rmse_pixels']:.2f} pixels\n")
            f.write(f"  Within Tolerance: {metrics['within_tolerance_pct']:.2f}%\n")

        print(f"Metrics saved to: {output_path}")

    def print_metrics(self, metrics: Dict) -> None:
        """Print evaluation metrics to console."""
        print("\nEVALUATION RESULTS")
        print("-" * 50)
        print(f"Frames evaluated: {metrics['total_frames']}")
        print(f"Pixel tolerance:  {metrics['pixel_tolerance']} pixels")
        print(f"\nTP: {metrics['true_positives']} | FP: {metrics['false_positives']} | "
              f"FN: {metrics['false_negatives']} | TN: {metrics['true_negatives']}")
        print(f"\nPrecision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"\nPosition MAE:  {metrics['mae_pixels']:.2f} pixels")
        print(f"Position RMSE: {metrics['rmse_pixels']:.2f} pixels")
        print(f"Within {metrics['pixel_tolerance']}px: {metrics['within_tolerance_pct']:.2f}%")

    def save_position_plot(self, predictions_df: pd.DataFrame, eval_frames: Optional[list] = None) -> None:
        """Save ball position plot.

        Args:
            predictions_df: DataFrame with ball predictions
            eval_frames: List of frame numbers to include. If None, includes all frames.
        """
        output_path = self.output_dir / "ball_positions.png"

        # Filter to evaluation frames if provided
        if eval_frames is not None:
            plot_df = predictions_df.loc[predictions_df.index.isin(eval_frames)]
        else:
            plot_df = predictions_df

        # Extract coordinates
        frames = plot_df.index.values
        x_coords = plot_df["ball_x"].values
        y_coords = plot_df["ball_y"].values

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot X coordinate
        ax1.plot(frames, x_coords, "b-", linewidth=0.5, label="X position", alpha=0.7)
        ax1.set_ylabel("X Coordinate (pixels)", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_title(f"Ball Position Over Time - {self.video_name}", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")

        # Plot Y coordinate
        ax2.plot(frames, y_coords, "r-", linewidth=0.5, label="Y position", alpha=0.7)
        ax2.set_xlabel("Frame Number", fontsize=12)
        ax2.set_ylabel("Y Coordinate (pixels)", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close()
        print(f"Position plot saved to: {output_path}")

    def create_annotated_video(self, predictions_df: pd.DataFrame, eval_frames: Optional[list] = None) -> None:
        """Create annotated video with ball tracking predictions.

        Args:
            predictions_df: DataFrame with ball predictions
            eval_frames: List of frame numbers to include. If None, includes all frames.
        """
        output_path = self.output_dir / "ball_annotated.mp4"

        if not self.video_path.exists():
            print(f"Video file not found: {self.video_path}")
            return

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine which frames to include
        if eval_frames is not None:
            frames_to_include = sorted(eval_frames)
        else:
            frames_to_include = sorted(predictions_df.index.tolist())

        if not frames_to_include:
            print("No frames to include in video")
            return

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))

        # Process only the evaluation frames
        pbar = tqdm(total=len(frames_to_include), desc="Creating annotated video")

        for frame_idx in frames_to_include:
            # Seek to the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Draw ball trace
            for i in range(self.trace_length):
                idx = frame_idx - i
                if idx >= 0 and idx in predictions_df.index:
                    row = predictions_df.loc[idx]
                    if row["ball_detected"] and pd.notna(row["ball_x"]) and pd.notna(row["ball_y"]):
                        pos = (int(row["ball_x"]), int(row["ball_y"]))
                        cv2.circle(
                            frame,
                            pos,
                            radius=max(2, 8 - i),
                            color=self.trace_color,
                            thickness=max(1, self.trace_thickness - i),
                        )

            # Draw current ball position with larger marker
            if frame_idx in predictions_df.index:
                row = predictions_df.loc[frame_idx]
                if row["ball_detected"] and pd.notna(row["ball_x"]) and pd.notna(row["ball_y"]):
                    pos = (int(row["ball_x"]), int(row["ball_y"]))
                    cv2.circle(frame, pos, radius=10, color=self.trace_color, thickness=-1)

            # Add frame number overlay
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            pbar.update(1)

        cap.release()
        out.release()
        pbar.close()
        print(f"Annotated video saved to: {output_path} ({len(frames_to_include)} frames)")

    def evaluate(self) -> None:
        """Run the evaluation pipeline."""
        print("\n" + "=" * 70)
        print("BALL TRACKING EVALUATION")
        print("=" * 70)
        print(f"Video: {self.video_name}")
        print(f"Output: {self.output_dir}")

        # Load predictions
        print(f"\nLoading predictions from: {self.predictions_path}")
        predictions_df = self.load_predictions()
        if predictions_df is None:
            print("Cannot proceed without predictions. Run annotation pipeline first.")
            return

        print(f"Loaded {len(predictions_df)} prediction frames")

        # Track evaluation frames for video generation
        eval_frames = None

        # Evaluate against ground truth if enabled
        if self.use_ground_truth:
            print(f"\nLoading ground truth from: {self.ground_truth_path}")
            gt_df = self.load_ground_truth()
            if gt_df is not None:
                print(f"Loaded {len(gt_df)} ground truth frames")
                # Get the common frames for evaluation
                eval_frames = list(predictions_df.index.intersection(gt_df.index))
                metrics = self.evaluate_against_ground_truth(predictions_df, gt_df)
                if metrics:
                    self.print_metrics(metrics)
                    self.save_metrics(metrics)
            else:
                print("Ground truth not found. Skipping evaluation metrics.")

        # Generate outputs
        if self.config["output"]["save_plots"]:
            print("\nGenerating position plot...")
            self.save_position_plot(predictions_df, eval_frames)

        if self.config["output"]["save_video"]:
            print("\nGenerating annotated video...")
            self.create_annotated_video(predictions_df, eval_frames)

        # Done
        print("\n" + "=" * 70)
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 70)
        print("Evaluation complete!\n")


if __name__ == "__main__":
    evaluator = BallTrackingEvaluator()
    evaluator.evaluate()
