"""
Stroke Detector

This module provides LSTM-based stroke type detection (forehand/backhand) for squash videos.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

from squashcopilot.common.utils import load_config
from squashcopilot.common.constants import KEYPOINT_NAMES
from squashcopilot.common.types import StrokeType
from squashcopilot.common.models.stroke import (
    StrokeDetectionInput,
    StrokeDetectionResult,
    StrokeResult,
)
from squashcopilot.modules.stroke_detection.model.lstm_classifier import (
    LSTMStrokeClassifier,
)


class StrokeDetector:
    """
    LSTM-based stroke detector.

    Uses a trained LSTM model to detect stroke types (forehand/backhand) from player keypoints
    around racket hit frames.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the stroke detector.

        Args:
            config: Configuration dictionary. If None, loads from stroke_detection.yaml
        """
        # Load configuration
        self.config = config if config else load_config(config_name="stroke_detection")

        # Get inference configuration
        self.inference_config = self.config["inference"]

        # Get sequence length from training config
        self.sequence_length = self.config["training"]["sequence_length"]
        self.window_size = (
            self.sequence_length - 1
        ) // 2  # Calculate window size from sequence length

        # Get project root and model path
        project_root = Path(__file__).parent.parent.parent.parent
        model_path = project_root / self.inference_config["model_path"]

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keypoint names
        self.keypoint_names = KEYPOINT_NAMES

        # Normalization settings
        self.min_torso_length = self.config["normalization"]["min_torso_length"]

        # Load model (this will also set model config from checkpoint)
        self.model, self.label_encoder = self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """
        Load trained LSTM model from checkpoint.

        Args:
            model_path: Path to model checkpoint file

        Returns:
            Tuple of (model, label_encoder)

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please train a model first."
            )

        try:
            # Load checkpoint
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Get model configuration
            model_config = checkpoint["config"]

            # Store model configuration
            self.model_config = model_config

            # Verify sequence length matches
            if model_config["sequence_length"] != self.sequence_length:
                print(
                    f"  ⚠ Warning: Model was trained with sequence_length={model_config['sequence_length']}, but config specifies {self.sequence_length}"
                )

            # Initialize model
            model = LSTMStrokeClassifier(
                input_size=model_config["input_size"],
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                num_classes=model_config["num_classes"],
                dropout=model_config["dropout"],
            )

            # Load weights
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(self.device)

            # Get label encoder
            label_encoder = checkpoint.get("label_encoder")

            print(f"✓ Model loaded from {model_path}")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Classes: {checkpoint.get('label_classes', 'N/A')}")

            return model, label_encoder

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def _normalize_keypoints_array(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints relative to hip center and torso length.

        Args:
            keypoints: Array of shape (sequence_length, num_features)
                      where num_features = num_keypoints * 2 (x1, y1, x2, y2, ...)
                      Keypoints are in the order of KEYPOINT_NAMES

        Returns:
            Normalized keypoints array of same shape
        """
        # Keypoints order matches KEYPOINT_NAMES
        # Indices: left_shoulder(0,1), right_shoulder(2,3), ..., left_hip(12,13), right_hip(14,15), ...

        # Extract hip keypoints (indices 6 and 7 in KEYPOINT_NAMES)
        left_hip_idx = self.keypoint_names.index("left_hip")
        right_hip_idx = self.keypoint_names.index("right_hip")

        # Extract shoulder keypoints (indices 0 and 1 in KEYPOINT_NAMES)
        left_shoulder_idx = self.keypoint_names.index("left_shoulder")
        right_shoulder_idx = self.keypoint_names.index("right_shoulder")

        # Calculate hip center for each frame
        hip_center_x = (
            keypoints[:, left_hip_idx * 2] + keypoints[:, right_hip_idx * 2]
        ) / 2
        hip_center_y = (
            keypoints[:, left_hip_idx * 2 + 1] + keypoints[:, right_hip_idx * 2 + 1]
        ) / 2

        # Calculate shoulder center for each frame
        shoulder_center_x = (
            keypoints[:, left_shoulder_idx * 2] + keypoints[:, right_shoulder_idx * 2]
        ) / 2
        shoulder_center_y = (
            keypoints[:, left_shoulder_idx * 2 + 1]
            + keypoints[:, right_shoulder_idx * 2 + 1]
        ) / 2

        # Calculate torso length for each frame
        torso_length = np.sqrt(
            (shoulder_center_x - hip_center_x) ** 2
            + (shoulder_center_y - hip_center_y) ** 2
        )

        # Prevent division by zero
        torso_length = np.where(
            torso_length < self.min_torso_length,
            1.0,
            torso_length,
        )

        # Normalize all keypoints
        normalized = keypoints.copy()
        for i in range(0, keypoints.shape[1], 2):  # Iterate over x coordinates
            normalized[:, i] = (keypoints[:, i] - hip_center_x) / torso_length  # x
            normalized[:, i + 1] = (
                keypoints[:, i + 1] - hip_center_y
            ) / torso_length  # y

        return normalized

    def _predict_stroke(self, sequence: np.ndarray) -> tuple:
        """
        Predict stroke type for a sequence.

        Args:
            sequence: Array of shape (sequence_length, num_features)

        Returns:
            Tuple of (stroke_type_str, confidence)
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
            if self.label_encoder is not None:
                predicted_label = self.label_encoder.inverse_transform(
                    [predicted.item()]
                )[0]
            else:
                predicted_label = str(predicted.item())

            confidence_score = confidence.item()

        return predicted_label, confidence_score

    def detect(self, input_data: StrokeDetectionInput) -> StrokeDetectionResult:
        """
        Detect stroke types for all racket hits in the input data.

        This is the main detection method. It processes all racket hits in the input,
        extracts windowed sequences around each hit, and predicts the stroke type.

        Args:
            input_data: StrokeDetectionInput with player keypoints and racket hits

        Returns:
            StrokeDetectionResult with list of detected strokes
        """
        strokes = []

        # Process each racket hit
        for i, hit_frame in enumerate(input_data.racket_hits):
            player_id = input_data.racket_hit_player_ids[i]

            # 1. Get the window where there is a racket hit
            half_window = self.window_size
            start_frame = hit_frame - half_window
            end_frame = (
                hit_frame + half_window + 1
            )  # +1 to include hit_frame + window_size

            # Find indices in frame_numbers list
            start_idx = None
            end_idx = None
            for idx, frame_num in enumerate(input_data.frame_numbers):
                if frame_num == start_frame:
                    start_idx = idx
                if frame_num == end_frame - 1:  # end_frame is exclusive
                    end_idx = idx + 1
                    break

            # Check if we have enough frames
            if (
                start_idx is None
                or end_idx is None
                or (end_idx - start_idx) != self.sequence_length
            ):
                # Not enough frames - skip this hit
                continue

            # 2. Get the keypoints of the player who hit the ball
            player_keypoints = input_data.player_keypoints[player_id]
            window_keypoints = player_keypoints[
                start_idx:end_idx
            ]  # (sequence_length, num_keypoints, 2)

            # Reshape to (sequence_length, num_features) where num_features = num_keypoints * 2
            num_frames, num_keypoints, _ = window_keypoints.shape
            window_keypoints_flat = window_keypoints.reshape(
                num_frames, num_keypoints * 2
            )

            # 3. Normalize these keypoints
            normalized_keypoints = self._normalize_keypoints_array(
                window_keypoints_flat
            )

            # 4. Run the inference
            stroke_type_str, confidence = self._predict_stroke(normalized_keypoints)

            # Create stroke result
            stroke = StrokeResult(
                frame=hit_frame,
                player_id=player_id,
                stroke_type=StrokeType.from_string(stroke_type_str),
                confidence=confidence,
            )

            strokes.append(stroke)

        return StrokeDetectionResult(strokes=strokes)

    def detect_from_dataframe(
        self, df: pd.DataFrame, video_name: Optional[str] = None
    ) -> StrokeDetectionResult:
        """
        Convenience method to detect strokes directly from an annotations DataFrame.

        This is useful for evaluation when you have a CSV file loaded.

        Args:
            df: DataFrame with columns:
                - frame
                - player_{1,2}_kp_{keypoint}_x/y
                - is_racket_hit
                - racket_hit_player_id
            video_name: Optional video name for logging

        Returns:
            StrokeDetectionResult with detected strokes
        """
        if video_name:
            print(f"\nDetecting strokes for {video_name}...")

        # Extract racket hits
        racket_hits_df = df[df["is_racket_hit"] == True].copy()
        racket_hits = racket_hits_df["frame"].tolist()
        racket_hit_player_ids = (
            racket_hits_df["racket_hit_player_id"].astype(int).tolist()
        )

        if len(racket_hits) == 0:
            print("  ⚠ No racket hits found")
            return StrokeDetectionResult(strokes=[])

        print(f"  Found {len(racket_hits)} racket hits")

        # Extract player keypoints arrays
        # Build arrays: player_id -> (num_frames, num_keypoints, 2)
        player_keypoints = {}
        for player_id in [1, 2]:
            keypoints_list = []
            for _, row in df.iterrows():
                kpts = []
                for kp_name in self.keypoint_names:
                    x = row.get(f"player_{player_id}_kp_{kp_name}_x", 0.0)
                    y = row.get(f"player_{player_id}_kp_{kp_name}_y", 0.0)
                    kpts.append([x, y])
                keypoints_list.append(kpts)

            player_keypoints[player_id] = np.array(keypoints_list, dtype=np.float32)

        # Create input
        input_data = StrokeDetectionInput(
            player_keypoints=player_keypoints,
            racket_hits=racket_hits,
            racket_hit_player_ids=racket_hit_player_ids,
            frame_numbers=df["frame"].tolist(),
        )

        # Detect strokes
        result = self.detect(input_data)

        print(f"  ✓ Detected {len(result.strokes)} strokes")
        return result
