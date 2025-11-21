"""
Ball Tracking Evaluator

Evaluates ball tracking performance on videos and generates comprehensive reports
comparing raw and postprocessed results.
"""

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from squashcopilot.modules.ball_tracking import BallTracker
from squashcopilot.modules.court_calibration import CourtCalibrator

from squashcopilot.common import (
    Frame,
    BallTrackingInput,
    BallPostprocessingInput,
    BallPostprocessingOutput,
    CourtCalibrationInput,
    ball_tracking_outputs_to_dataframe,
)
from squashcopilot.common.utils import load_config


class BallTrackingEvaluator:
    """Evaluates ball tracking performance on videos."""

    def __init__(self, config=None):
        """Initialize evaluator with configuration."""
        if config is None:
            # Load the tests section from the ball_tracking config
            full_config = load_config(config_name="ball_tracking")
            config = full_config["tests"]
        self.config = config
        self.test_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Resolve video path from config
        project_root = Path(__file__).parent.parent.parent.parent.parent
        video_dir_rel = self.config["video"].get("video_dir", "squashcopilot/videos")
        self.video_name = self.config["video"]["video_name"]
        self.video_path = project_root / video_dir_rel / f"{self.video_name}.mp4"

        # Initialize components
        self.tracker = BallTracker()
        self.court_calibrator = CourtCalibrator()

        # Video properties (set during processing)
        self.video_name = None
        self.fps = None
        self.width = None
        self.height = None
        self.calibration = None

    def detect_ball_positions(self) -> pd.DataFrame:
        """Detect ball positions in all frames.

        Returns:
            DataFrame with ball tracking results indexed by frame_number
        """
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = self.config["video"].get("max_frames")
        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nVideo: {self.video_name}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Processing {total_frames} frames\n")

        # Detect ball in each frame
        outputs = []
        pbar = tqdm(total=total_frames, desc="Tracking ball")

        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame_img = cap.read()
            if not ret:
                break

            # Calibrate court on first frame
            if frame_count == 0:
                court_input = CourtCalibrationInput(
                    frame=Frame(image=frame_img, frame_number=0, timestamp=0.0)
                )
                self.calibration = self.court_calibrator.process_frame(court_input)

                # Set ball color based on wall color
                self.tracker.set_is_black_ball(self.calibration.is_black_ball)

            # Track ball using new interface
            ball_input = BallTrackingInput(
                frame=Frame(
                    image=frame_img,
                    frame_number=frame_count,
                    timestamp=frame_count / self.fps,
                )
            )
            ball_output = self.tracker.process_frame(ball_input)
            outputs.append(ball_output)

            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        # Convert to DataFrame
        df = ball_tracking_outputs_to_dataframe(outputs)
        return df

    def apply_postprocessing(self, df: pd.DataFrame) -> BallPostprocessingOutput:
        """Apply postprocessing to ball tracking results.

        Args:
            df: DataFrame with raw ball tracking results

        Returns:
            BallPostprocessingOutput with processed DataFrame
        """
        print("Applying postprocessing (outlier removal + interpolation)...")

        postprocess_input = BallPostprocessingInput(df=df)
        postprocess_output = self.tracker.postprocess(postprocess_input)

        print(
            f"Postprocessing complete. Outliers removed: {postprocess_output.num_ball_outliers}, "
            f"Gaps filled: {postprocess_output.num_ball_gaps_filled}"
        )
        return postprocess_output

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate tracking metrics.

        Args:
            df: DataFrame with ball tracking results

        Returns:
            dict: Metrics dictionary
        """
        total_frames = len(df)
        detected_frames = df["ball_detected"].sum()
        detection_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0

        # Calculate average confidence for detected frames
        detected_df = df[df["ball_detected"]]
        avg_confidence = detected_df["ball_confidence"].mean() if len(detected_df) > 0 else 0

        return {
            "total_frames": total_frames,
            "detected_frames": int(detected_frames),
            "detection_rate_percent": round(detection_rate, 2),
            "average_confidence": round(avg_confidence, 3),
        }

    def save_metrics(self, metrics: dict, output_dir: Path):
        """Save metrics to text file.

        Args:
            metrics: Metrics dictionary
            output_dir: Output directory path
        """
        output_path = output_dir / "metrics.txt"

        with open(output_path, "w") as f:
            f.write("BALL TRACKING METRICS\n")
            f.write("=" * 50 + "\n\n")

            f.write("Detection Performance:\n")
            f.write(f"  Total frames:          {metrics['total_frames']}\n")
            f.write(f"  Detected frames:       {metrics['detected_frames']}\n")
            f.write(f"  Detection rate:        {metrics['detection_rate_percent']:.2f}%\n")
            f.write(f"  Average confidence:    {metrics['average_confidence']:.3f}\n")

    def save_positions_csv(self, df: pd.DataFrame, output_dir: Path):
        """Save ball positions to CSV file.

        Args:
            df: DataFrame with ball tracking results
            output_dir: Output directory path
        """
        output_path = output_dir / f"{self.video_name}_ball_positions.csv"
        df.to_csv(output_path)
        print(f"Ball positions saved to: {output_path}")

    def save_position_plot(self, df: pd.DataFrame, output_dir: Path):
        """Save position plot.

        Args:
            df: DataFrame with ball tracking results
            output_dir: Output directory path
        """
        output_path = output_dir / "positions.png"
        dpi = self.config["output"]["plot_dpi"]

        # Extract coordinates
        frames = df.index.values
        x_coords = df["ball_x"].values
        y_coords = df["ball_y"].values

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
        ax2.invert_yaxis()  # Lower Y = closer to wall

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Position plot saved to: {output_path}")

    def create_video(self, df: pd.DataFrame, output_dir: Path):
        """Create annotated video with ball tracking.

        Args:
            df: DataFrame with ball tracking results
            output_dir: Output directory path
        """
        output_path = output_dir / "tracking.mp4"

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))

        # Get visualization config
        trace_length = self.config["tracking"]["trace_length"]
        trace_color = tuple(self.config["tracking"]["trace_color"])
        trace_thickness = self.config["tracking"]["trace_thickness"]

        # Process frames
        pbar = tqdm(total=len(df), desc="Creating video")
        frame_idx = 0

        while cap.isOpened() and frame_idx < len(df):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ball trace
            for i in range(trace_length):
                idx = frame_idx - i
                if idx >= 0 and idx in df.index:
                    row = df.loc[idx]
                    if row["ball_detected"] and pd.notna(row["ball_x"]) and pd.notna(row["ball_y"]):
                        pos = (int(row["ball_x"]), int(row["ball_y"]))
                        cv2.circle(
                            frame,
                            pos,
                            radius=max(2, 8 - i),
                            color=trace_color,
                            thickness=max(1, trace_thickness - i),
                        )

            # Draw current ball position with larger marker
            if frame_idx in df.index:
                row = df.loc[frame_idx]
                if row["ball_detected"] and pd.notna(row["ball_x"]) and pd.notna(row["ball_y"]):
                    pos = (int(row["ball_x"]), int(row["ball_y"]))
                    cv2.circle(frame, pos, radius=10, color=trace_color, thickness=-1)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

        cap.release()
        out.release()
        pbar.close()
        print(f"Video saved to: {output_path}")

    def save_results(self, df: pd.DataFrame, output_dir: Path, label: str):
        """Save all results for a processing type.

        Args:
            df: DataFrame with ball tracking results
            output_dir: Output directory path
            label: Label for console output (e.g., "RAW" or "POSTPROCESSED")
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{label} RESULTS")
        print("=" * 70)

        # Calculate and save metrics
        metrics = self.calculate_metrics(df)
        self.save_metrics(metrics, output_dir)
        print(f"Metrics saved")

        # Print key metrics
        print(f"  Detection rate: {metrics['detection_rate_percent']:.2f}%")
        print(f"  Average confidence: {metrics['average_confidence']:.3f}")

        # Save ball positions CSV
        self.save_positions_csv(df, output_dir)

        # Save plot
        if self.config["output"]["save_plots"]:
            self.save_position_plot(df, output_dir)

        # Save video
        if self.config["output"]["save_video"]:
            self.create_video(df, output_dir)

    def evaluate(self):
        """Run full evaluation pipeline."""
        print("\n" + "=" * 70)
        print("BALL TRACKING EVALUATION")
        print("=" * 70)

        # Step 1: Detect ball positions
        df_raw = self.detect_ball_positions()

        # Step 2: Save raw results
        base_output_dir = self.test_dir / self.config["output"]["output_dir"] / self.video_name
        raw_output_dir = base_output_dir / "raw"
        self.save_results(df_raw, raw_output_dir, "RAW")

        # Step 3: Apply postprocessing
        postprocess_output = self.apply_postprocessing(df_raw)
        df_postprocessed = postprocess_output.df

        # Step 4: Save postprocessed results
        postprocessed_output_dir = base_output_dir / "postprocessed"
        self.save_results(df_postprocessed, postprocessed_output_dir, "POSTPROCESSED")

        # Done
        print("\n" + "=" * 70)
        print(f"All results saved to: {base_output_dir}")
        print("=" * 70)
        print("Evaluation complete!\n")


if __name__ == "__main__":
    evaluator = BallTrackingEvaluator()
    evaluator.evaluate()
