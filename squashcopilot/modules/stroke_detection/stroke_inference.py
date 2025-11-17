"""
Stroke Type Inference Script

This script performs real-time stroke type prediction on annotated squash videos.
It loads a trained LSTM model and predicts forehand/backhand strokes at racket hit frames.

Features:
- Loads video and annotation CSV
- Detects racket hits from annotations
- Extracts 31-frame sequences around hits
- Uses raw keypoint coordinates (no normalization)
- Predicts stroke type using trained LSTM model
- Displays video with predictions overlaid
- Saves predictions to CSV
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import yaml

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from common.constants import KEYPOINT_NAMES
from common.utils import load_config


# ======================== LSTM Model Definition ========================


class LSTMStrokeClassifier(nn.Module):
    """
    LSTM-based stroke classifier.
    Must match the architecture used during training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super(LSTMStrokeClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out


# ======================== Normalization ========================


def normalize_keypoints_for_player(
    df: pd.DataFrame,
    player_id: int,
    keypoint_names: List[str],
    min_torso_length: float = 1e-6,
) -> pd.DataFrame:
    """
    Normalize keypoints for a specific player relative to hip center and torso length.

    Args:
        df: DataFrame with player keypoint columns
        player_id: Player ID (1 or 2)
        keypoint_names: List of keypoint names
        min_torso_length: Minimum torso length to avoid division by zero

    Returns:
        DataFrame with added normalized keypoint columns
    """
    df = df.copy()
    prefix = f"player_{player_id}_"

    # Calculate hip center
    hip_center_x = (df[f"{prefix}kp_left_hip_x"] + df[f"{prefix}kp_right_hip_x"]) / 2
    hip_center_y = (df[f"{prefix}kp_left_hip_y"] + df[f"{prefix}kp_right_hip_y"]) / 2

    # Calculate shoulder center
    shoulder_center_x = (
        df[f"{prefix}kp_left_shoulder_x"] + df[f"{prefix}kp_right_shoulder_x"]
    ) / 2
    shoulder_center_y = (
        df[f"{prefix}kp_left_shoulder_y"] + df[f"{prefix}kp_right_shoulder_y"]
    ) / 2

    # Calculate torso length
    torso_length = np.sqrt(
        (shoulder_center_x - hip_center_x) ** 2
        + (shoulder_center_y - hip_center_y) ** 2
    )

    # Prevent division by zero
    torso_length = np.where(torso_length < min_torso_length, 1.0, torso_length)

    # Normalize all keypoints
    for kp_name in keypoint_names:
        df[f"norm_{kp_name}_x"] = (
            df[f"{prefix}kp_{kp_name}_x"] - hip_center_x
        ) / torso_length
        df[f"norm_{kp_name}_y"] = (
            df[f"{prefix}kp_{kp_name}_y"] - hip_center_y
        ) / torso_length

    return df


# ======================== Sequence Extraction ========================


def extract_sequence_for_hit(
    df: pd.DataFrame,
    hit_frame: int,
    player_id: int,
    sequence_length: int,
    keypoint_names: List[str],
) -> Optional[np.ndarray]:
    """
    Extract a normalized keypoint sequence around a racket hit frame.

    Args:
        df: DataFrame with normalized keypoints
        hit_frame: Frame number where racket hit occurred (actual frame number, not row index)
        player_id: Player ID who made the hit
        sequence_length: Number of frames in sequence (e.g., 31 = 15 before + hit + 15 after)
        keypoint_names: List of keypoint names

    Returns:
        Sequence array of shape (sequence_length, num_features) or None if not enough frames
    """
    # Calculate frame range (centered on hit frame NUMBER, not row index)
    half_window = sequence_length // 2
    start_frame_num = hit_frame - half_window
    end_frame_num = hit_frame + half_window + 1  # +1 to get exactly sequence_length frames

    # Filter by FRAME NUMBER, not by iloc (row index)!
    # This is critical - we need frames by their 'frame' column value, not their position in the DataFrame
    frame_slice = df[(df['frame'] >= start_frame_num) & (df['frame'] < end_frame_num)].copy()

    # Check if we got the right sequence length
    if len(frame_slice) != sequence_length:
        return None

    # Sort by frame to ensure temporal order
    frame_slice = frame_slice.sort_values('frame')

    # Build feature columns
    feature_cols = []
    for kp_name in keypoint_names:
        feature_cols.append(f"norm_{kp_name}_x")
        feature_cols.append(f"norm_{kp_name}_y")

    # Extract features
    sequence = frame_slice[feature_cols].values

    return sequence.astype(np.float32)


# ======================== Inference ========================


class StrokeInference:
    """Handles stroke type inference on video."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the inference system.

        Args:
            config_path: Path to config YAML file
        """
        # Load config
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent
                / "configs"
                / "stroke_detection.yaml"
            )

        self.config = load_config(config_name="stroke_detection")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model, self.label_encoder, self.model_config = self._load_model()

        # Sequence length from model config
        self.sequence_length = self.model_config["sequence_length"]

        # Keypoint names
        self.keypoint_names = KEYPOINT_NAMES

        # Normalization settings
        self.min_torso_length = self.config["normalization"]["min_torso_length"]

        # Prediction display settings
        self.prediction_display_frames = 30  # Show prediction for 30 frames after hit
        self.current_predictions = (
            {}
        )  # {frame_num: (player_id, stroke_type, confidence)}

    def _load_model(self) -> Tuple[nn.Module, object, Dict]:
        """Load trained LSTM model from checkpoint."""
        model_path = Path(__file__).parent / self.config["model"]["model_path"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model config
        model_config = checkpoint["config"]

        # Initialize model
        model = LSTMStrokeClassifier(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        ).to(self.device)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Get label encoder
        label_encoder = checkpoint["label_encoder"]

        print(f"✓ Model loaded successfully")
        print(f"  Classes: {checkpoint['label_classes']}")

        return model, label_encoder, model_config

    def predict_stroke(self, sequence: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict stroke type for a sequence.

        Args:
            sequence: Array of shape (sequence_length, num_features)

        Returns:
            predicted_label: Stroke type ('forehand' or 'backhand')
            confidence: Confidence score (0-1)
            all_probs: Probabilities for all classes
        """
        with torch.no_grad():
            # Add batch dimension and convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Forward pass
            output = self.model(sequence_tensor)

            # Get probabilities
            probs = torch.softmax(output, dim=1)

            # Get prediction
            confidence, predicted = torch.max(probs, 1)

            # Decode label
            predicted_label = self.label_encoder.inverse_transform([predicted.item()])[
                0
            ]
            confidence_score = confidence.item()
            all_probs = probs.cpu().numpy()[0]

        return predicted_label, confidence_score, all_probs

    def process_video(
        self,
        video_path: Path,
        annotations_path: Path,
        output_dir: Path,
        visualize: bool = True,
    ) -> pd.DataFrame:
        """
        Process video and predict stroke types at racket hit frames.

        Args:
            video_path: Path to video file
            annotations_path: Path to annotations CSV
            output_dir: Directory to save output
            visualize: Whether to show visualization

        Returns:
            DataFrame with predictions
        """
        print(f"\n{'='*70}")
        print("PROCESSING VIDEO")
        print(f"{'='*70}")
        print(f"Video: {video_path}")
        print(f"Annotations: {annotations_path}")

        # Load annotations
        print("\nLoading annotations...")
        df = pd.read_csv(annotations_path)
        print(f"✓ Loaded {len(df)} frames")

        # Find racket hits
        racket_hits = df[df["is_racket_hit"] == True].copy()
        print(f"✓ Found {len(racket_hits)} racket hits")

        if len(racket_hits) == 0:
            print("⚠ No racket hits found in annotations!")
            return pd.DataFrame()

        # Normalize keypoints for both players
        print("\nNormalizing keypoints...")
        df = normalize_keypoints_for_player(
            df, 1, self.keypoint_names, self.min_torso_length
        )
        df = normalize_keypoints_for_player(
            df, 2, self.keypoint_names, self.min_torso_length
        )
        print("✓ Normalization complete")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nVideo info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # Setup video writer if visualizing
        video_writer = None
        if visualize and self.config["inference"].get("save_video", False):
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = output_dir / f"{video_path.stem}_predictions.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )
            print(f"\nSaving output video to: {output_video_path}")

        # Process racket hits and make predictions
        print(f"\n{'='*70}")
        print("PREDICTING STROKES")
        print(f"{'='*70}")

        predictions = []

        for idx, (_, hit_row) in enumerate(racket_hits.iterrows()):
            hit_frame = int(hit_row["frame"])
            player_id = int(hit_row["racket_hit_player_id"])

            print(
                f"\nRacket hit {idx+1}/{len(racket_hits)}: Frame {hit_frame}, Player {player_id}"
            )

            # Extract sequence
            sequence = extract_sequence_for_hit(
                df, hit_frame, player_id, self.sequence_length, self.keypoint_names
            )

            if sequence is None:
                print(f"  ⚠ Skipped (not enough frames)")
                continue

            # Predict
            stroke_type, confidence, probs = self.predict_stroke(sequence)

            print(
                f"  ✓ Predicted: {stroke_type.upper()} (confidence: {confidence:.3f})"
            )

            # Store prediction
            predictions.append(
                {
                    "frame": hit_frame,
                    "player_id": player_id,
                    "stroke_type": stroke_type,
                    "confidence": confidence,
                }
            )

            # Store for display
            for i in range(self.prediction_display_frames):
                display_frame = hit_frame + i
                if display_frame < total_frames:
                    self.current_predictions[display_frame] = (
                        player_id,
                        stroke_type,
                        confidence,
                    )

        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Save predictions to CSV
        if self.config["inference"].get("save_predictions", True):
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / f"{video_path.stem}_predictions.csv"
            predictions_df.to_csv(csv_path, index=False)
            print(f"\n✓ Predictions saved to: {csv_path}")

        # Play video with predictions
        if visualize:
            print(f"\n{'='*70}")
            print("PLAYING VIDEO WITH PREDICTIONS")
            print(f"{'='*70}")
            print("\nControls:")
            print("  SPACE - Pause/Resume")
            print("  q - Quit")
            print("  → - Next frame (when paused)")
            print("  ← - Previous frame (when paused)")

            self._play_video_with_predictions(cap, df, video_writer)

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        return predictions_df

    def _play_video_with_predictions(
        self,
        cap: cv2.VideoCapture,
        df: pd.DataFrame,
        video_writer: Optional[cv2.VideoWriter],
    ):
        """Play video with stroke predictions overlaid."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        paused = False
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                # When paused, read the current frame
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

            # Create display frame
            display_frame = frame.copy()

            # Draw player positions and keypoints
            if frame_idx < len(df):
                row = df.iloc[frame_idx]

                for player_id in [1, 2]:
                    prefix = f"player_{player_id}_"

                    # Draw player position
                    x_pixel = row[f"{prefix}x_pixel"]
                    y_pixel = row[f"{prefix}y_pixel"]

                    if not pd.isna(x_pixel) and not pd.isna(y_pixel):
                        color = (0, 255, 0) if player_id == 1 else (255, 0, 0)
                        cv2.circle(
                            display_frame, (int(x_pixel), int(y_pixel)), 8, color, -1
                        )
                        cv2.putText(
                            display_frame,
                            f"P{player_id}",
                            (int(x_pixel) - 15, int(y_pixel) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            # Draw prediction if available
            if frame_idx in self.current_predictions:
                player_id, stroke_type, confidence = self.current_predictions[frame_idx]

                # Draw prediction box
                color = (0, 255, 0) if player_id == 1 else (255, 0, 0)

                text = f"P{player_id}: {stroke_type.upper()} ({confidence:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]

                # Draw background box
                box_coords = ((10, 10), (20 + text_size[0], 50 + text_size[1]))
                cv2.rectangle(
                    display_frame, box_coords[0], box_coords[1], (0, 0, 0), -1
                )
                cv2.rectangle(display_frame, box_coords[0], box_coords[1], color, 2)

                # Draw text
                cv2.putText(
                    display_frame,
                    text,
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    3,
                )

            # Draw frame number
            cv2.putText(
                display_frame,
                f"Frame: {frame_idx}",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Draw pause indicator
            if paused:
                cv2.putText(
                    display_frame,
                    "PAUSED",
                    (display_frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

            # Show frame
            cv2.imshow("Stroke Inference", display_frame)

            # Write to output video if enabled
            if video_writer:
                video_writer.write(display_frame)

            # Handle key press
            wait_time = 1 if paused else int(1000 / fps)
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == 81 or key == 2:  # Left arrow
                if paused and frame_idx > 0:
                    frame_idx -= 1
            elif key == 83 or key == 3:  # Right arrow
                if paused and frame_idx < len(df) - 1:
                    frame_idx += 1


# ======================== Main ========================


def main():
    """Main entry point for inference."""
    print("=" * 70)
    print("STROKE TYPE INFERENCE")
    print("=" * 70)

    # Initialize inference system
    inference = StrokeInference()

    # Get paths from config (resolve relative to SquashCoachingCopilot root)
    # Since we're in squashcopilot/modules/stroke_detection/, go up 3 levels
    project_root = Path(__file__).parent.parent.parent.parent
    video_path = project_root / inference.config["inference"]["video_path"]

    # Derive annotations path from video path
    # Assumes annotations are in same directory as video with naming pattern:
    # video-1_annotated.mp4 -> video-1_annotations.csv
    annotations_path = (
        video_path.parent
        / f"{video_path.stem.replace('_annotated', '')}_annotations.csv"
    )

    output_dir = project_root / inference.config["inference"]["output_dir"]
    visualize = inference.config["inference"].get("visualize", True)

    # Verify files exist
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    if not annotations_path.exists():
        print(f"Error: Annotations not found: {annotations_path}")
        return

    # Process video
    predictions_df = inference.process_video(
        video_path, annotations_path, output_dir, visualize=visualize
    )

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal predictions: {len(predictions_df)}")

    if len(predictions_df) > 0:
        print("\nPredictions by stroke type:")
        print(predictions_df["stroke_type"].value_counts())

        print("\nPredictions by player:")
        print(predictions_df["player_id"].value_counts())

        print(f"\nAverage confidence: {predictions_df['confidence'].mean():.3f}")

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
