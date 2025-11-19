"""
Rally State Detection Evaluator

Evaluates rally segmentation performance against ground truth annotations.
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from squashcopilot.common.utils import load_config
from squashcopilot.common.models.rally import (
    RallySegmentationInput,
    RallySegment,
)
from squashcopilot.modules.rally_state_detection import RallyStateDetector


class RallyStateEvaluator:
    """Evaluates and visualizes rally state detection performance."""

    def __init__(self, config_name: str = "rally_state_detection"):
        """
        Initialize evaluator with configuration.

        Args:
            config_name: Name of the configuration file (without .yaml extension)
        """
        full_config = load_config(config_name=config_name)
        self.config = full_config["evaluator"]
        self.annotation_config = full_config["annotation"]
        self.video_name = self.config["video_name"]
        self.boundary_tolerance = self.config["boundary_tolerance"]

        # Get project root and build paths
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # Data directory (where annotations are saved)
        data_base_dir = project_root / self.config["data_dir"]
        self.data_dir = data_base_dir / self.video_name

        # Video directory (where video files are stored)
        video_base_dir = project_root / self.config["video_dir"]
        self.video_path = video_base_dir / f"{self.video_name}.mp4"

        # Output directory (where plots will be saved)
        output_base_dir = project_root / self.config["output_dir"]
        self.output_dir = output_base_dir / self.video_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detector
        self.detector = RallyStateDetector(config=full_config)

    def load_annotations(self) -> pd.DataFrame:
        """
        Load rally state annotations from CSV file.

        Returns:
            DataFrame with frame, ball_y, and rally state labels
        """
        csv_path = self.data_dir / f"{self.video_name}_annotations.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} annotated frames from {csv_path}")

        return df

    def extract_ground_truth_segments(self, df: pd.DataFrame) -> List[RallySegment]:
        """
        Extract ground truth rally segments from annotations.

        Args:
            df: DataFrame with frame and rally_state columns

        Returns:
            List of RallySegment objects representing ground truth
        """
        label_col = self.annotation_config["label_column"]
        segments = []
        rally_id = 0
        in_rally = False
        rally_start = None

        for i, row in df.iterrows():
            frame = row["frame"]
            state = row[label_col]

            if state == "start" and not in_rally:
                # Start of rally
                rally_start = frame
                in_rally = True

            elif state == "end" and in_rally:
                # End of rally
                rally_end = frame
                segments.append(
                    RallySegment(
                        rally_id=rally_id,
                        start_frame=rally_start,
                        end_frame=rally_end,
                    )
                )
                rally_id += 1
                in_rally = False
                rally_start = None

        # Handle case where annotation ends during a rally
        if in_rally and rally_start is not None:
            segments.append(
                RallySegment(
                    rally_id=rally_id,
                    start_frame=rally_start,
                    end_frame=df["frame"].iloc[-1],
                )
            )

        return segments

    def compute_metrics(
        self,
        predicted_segments: List[RallySegment],
        ground_truth_segments: List[RallySegment],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics with tolerance for boundary matching.

        Args:
            predicted_segments: List of predicted rally segments
            ground_truth_segments: List of ground truth rally segments

        Returns:
            Dictionary with precision, recall, F1, and boundary accuracy metrics
        """
        tolerance = self.boundary_tolerance

        # Convert segments to sets of frame ranges for easier comparison
        def segment_to_frames(segment):
            return set(range(segment.start_frame, segment.end_frame + 1))

        # Track matches
        gt_matched = [False] * len(ground_truth_segments)
        pred_matched = [False] * len(predicted_segments)

        # Count boundary matches
        start_boundary_matches = 0
        end_boundary_matches = 0
        total_boundaries = len(ground_truth_segments) * 2  # start + end for each rally

        # For each predicted segment, find best matching ground truth segment
        for pred_idx, pred_seg in enumerate(predicted_segments):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_seg in enumerate(ground_truth_segments):
                # Compute IoU (Intersection over Union)
                pred_frames = segment_to_frames(pred_seg)
                gt_frames = segment_to_frames(gt_seg)

                intersection = len(pred_frames & gt_frames)
                union = len(pred_frames | gt_frames)
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Consider it a match if IoU > 0.5
            if best_iou > 0.5:
                pred_matched[pred_idx] = True
                gt_matched[best_gt_idx] = True

                # Check boundary accuracy with tolerance
                gt_seg = ground_truth_segments[best_gt_idx]
                if abs(pred_seg.start_frame - gt_seg.start_frame) <= tolerance:
                    start_boundary_matches += 1
                if abs(pred_seg.end_frame - gt_seg.end_frame) <= tolerance:
                    end_boundary_matches += 1

        # Compute metrics
        true_positives = sum(pred_matched)
        false_positives = len(predicted_segments) - true_positives
        false_negatives = len(ground_truth_segments) - sum(gt_matched)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        boundary_accuracy = (
            start_boundary_matches + end_boundary_matches
        ) / total_boundaries

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "boundary_accuracy": boundary_accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "predicted_rallies": len(predicted_segments),
            "ground_truth_rallies": len(ground_truth_segments),
        }

    def plot_comparison(
        self,
        df: pd.DataFrame,
        predicted_segments: List[RallySegment],
        ground_truth_segments: List[RallySegment],
    ):
        """
        Plot comparison between predicted and ground truth rally segments.

        Args:
            df: DataFrame with annotations
            predicted_segments: Predicted rally segments
            ground_truth_segments: Ground truth segments
            preprocessed_trajectory: Smoothed ball trajectory
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

        frames = df["frame"].values

        # Top plot: Ball trajectory with ground truth
        ax1.plot(
            frames,
            df["ball_y"].values,
            linewidth=1,
            color="lightblue",
            alpha=0.5,
            label="Ball Y (Original)",
        )

        # Highlight ground truth rally segments
        for segment in ground_truth_segments:
            ax1.axvspan(
                segment.start_frame,
                segment.end_frame,
                alpha=0.3,
                color="green",
                label="Ground Truth Rally" if segment.rally_id == 0 else "",
            )

        ax1.set_ylabel("Ball Y Coordinate", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Ground Truth Rally Segments - {self.video_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax1.legend(loc="upper right", fontsize=10)

        # Bottom plot: Ball trajectory with predictions
        ax2.plot(
            frames,
            df["ball_y"].values,
            linewidth=1,
            color="lightblue",
            alpha=0.5,
            label="Ball Y (Original)",
        )

        # Highlight predicted rally segments
        for segment in predicted_segments:
            ax2.axvspan(
                segment.start_frame,
                segment.end_frame,
                alpha=0.3,
                color="orange",
                label="Predicted Rally" if segment.rally_id == 0 else "",
            )

        ax2.set_xlabel("Frame Number", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Ball Y Coordinate", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Predicted Rally Segments - {self.video_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        ax2.legend(loc="upper right", fontsize=10)

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f"{self.video_name}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Comparison plot saved to: {output_path}")

    def create_annotated_video(
        self,
        predicted_segments: List[RallySegment],
    ):
        """
        Create an annotated video showing rally states.

        Args:
            predicted_segments: Predicted rally segments
        """
        # Use the video path from config
        if not self.video_path.exists():
            print(
                f"Warning: Video file not found at {self.video_path}, skipping video annotation"
            )
            return

        print(f"Reading video from: {self.video_path}")

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Create output video writer
        output_path = self.output_dir / f"{self.video_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Create a mapping from frame number to rally state
        frame_to_state = {}
        for segment in predicted_segments:
            for frame_num in range(segment.start_frame, segment.end_frame + 1):
                frame_to_state[frame_num] = "Rally Active"

        # Process each frame with progress bar
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Creating annotated video", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get rally state for this frame
            rally_state = frame_to_state.get(frame_idx, "Rally Inactive")

            # Set color based on state
            if rally_state == "Rally Active":
                color = (0, 255, 0)  # Green for active
                text = "RALLY ACTIVE"
            else:
                color = (0, 0, 255)  # Red for inactive
                text = "RALLY INACTIVE"

            # Add text overlay
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1.5
            thickness = 3

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Position text at top center
            text_x = (width - text_width) // 2
            text_y = 60

            # Draw semi-transparent background rectangle
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (text_x - 10, text_y - text_height - 10),
                (text_x + text_width + 10, text_y + baseline + 10),
                (0, 0, 0),
                -1,
            )
            # Blend the overlay with the frame
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw text
            cv2.putText(
                frame, text, (text_x, text_y), font, font_scale, color, thickness
            )

            # Draw frame number in bottom left
            frame_text = f"Frame: {frame_idx}"
            cv2.putText(
                frame,
                frame_text,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Write frame to output video
            out.write(frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

        # Release resources
        cap.release()
        out.release()

        print(f"Annotated video saved to: {output_path}")

    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics.

        Args:
            metrics: Dictionary with evaluation metrics
        """
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        print(f"Video: {self.video_name}")
        print(f"Boundary Tolerance: ±{self.boundary_tolerance} frames")
        print()
        print(f"Ground Truth Rallies: {metrics['ground_truth_rallies']}")
        print(f"Predicted Rallies: {metrics['predicted_rallies']}")
        print()
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print()
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print()
        print(f"Boundary Accuracy: {metrics['boundary_accuracy']:.4f}")
        print(
            f"  (Percentage of rally boundaries within ±{self.boundary_tolerance} frames)"
        )
        print("=" * 70)

    def run(self):
        """Run the complete evaluation pipeline."""
        print("=" * 70)
        print("RALLY STATE DETECTION EVALUATOR")
        print("=" * 70)

        # Load annotations
        print(f"\nLoading annotations for {self.video_name}...")
        df = self.load_annotations()

        # Extract ground truth segments
        print(f"Extracting ground truth rally segments...")
        ground_truth_segments = self.extract_ground_truth_segments(df)
        print(f"Found {len(ground_truth_segments)} ground truth rallies")

        # Prepare input for detector
        print(f"\nRunning rally segmentation...")
        input_data = RallySegmentationInput(
            ball_positions=df["ball_y"].tolist(),
            frame_numbers=df["frame"].tolist(),
            player_1_x=df["player_1_x_meter"].tolist(),
            player_1_y=df["player_1_y_meter"].tolist(),
            player_2_x=df["player_2_x_meter"].tolist(),
            player_2_y=df["player_2_y_meter"].tolist(),
        )

        # Run detector
        result = self.detector.segment_rallies(input_data)
        print(f"Detected {len(result.segments)} rallies")

        # Compute metrics
        print(f"\nComputing evaluation metrics...")
        metrics = self.compute_metrics(result.segments, ground_truth_segments)

        # Print metrics
        self.print_metrics(metrics)

        # Plot comparison
        print(f"\nGenerating comparison plot...")
        self.plot_comparison(df, result.segments, ground_truth_segments)

        # Create annotated video
        print(f"\nCreating annotated video...")
        self.create_annotated_video(result.segments)

        print("\n" + "=" * 70)
        print("✓ Evaluation complete!")
        print(f"✓ Output saved to: {self.output_dir}")
        print("=" * 70)

        return metrics


def main():
    """Main entry point."""
    evaluator = RallyStateEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
