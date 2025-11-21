"""
Stroke Detector

This module provides LSTM-based stroke type detection (forehand/backhand) for squash videos.
Uses DataFrame-based pipeline architecture.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

from squashcopilot.common.utils import load_config
from squashcopilot.common.constants import KEYPOINT_NAMES
from squashcopilot.common.types import StrokeType
from squashcopilot.common.models import (
    StrokeClassificationInput,
    StrokeClassificationOutput,
)
from squashcopilot.modules.stroke_detection.model.lstm_classifier import (
    LSTMStrokeClassifier,
)


class StrokeDetector:
    """
    LSTM-based stroke detector.

    Uses a trained LSTM model to detect stroke types (forehand/backhand) from player keypoints
    around racket hit frames. Works with DataFrame-based pipeline architecture.
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
        self.window_size = (self.sequence_length - 1) // 2

        # Get project root and model path
        project_root = Path(__file__).parent.parent.parent.parent
        model_path = project_root / self.inference_config["model_path"]

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keypoint names
        self.keypoint_names = KEYPOINT_NAMES

        # Normalization settings
        self.min_torso_length = self.config["normalization"]["min_torso_length"]

        # Load model
        self.model, self.label_encoder = self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load trained LSTM model from checkpoint."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please train a model first."
            )

        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            model_config = checkpoint["config"]
            self.model_config = model_config

            if model_config["sequence_length"] != self.sequence_length:
                print(
                    f"  Warning: Model was trained with sequence_length={model_config['sequence_length']}, "
                    f"but config specifies {self.sequence_length}"
                )

            model = LSTMStrokeClassifier(
                input_size=model_config["input_size"],
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                num_classes=model_config["num_classes"],
                dropout=model_config["dropout"],
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(self.device)

            label_encoder = checkpoint.get("label_encoder")

            return model, label_encoder

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def _normalize_keypoints_array(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints relative to hip center and torso length.

        Args:
            keypoints: Array of shape (sequence_length, num_features)

        Returns:
            Normalized keypoints array of same shape
        """
        left_hip_idx = self.keypoint_names.index("left_hip")
        right_hip_idx = self.keypoint_names.index("right_hip")
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

        torso_length = np.where(
            torso_length < self.min_torso_length,
            1.0,
            torso_length,
        )

        # Normalize all keypoints
        normalized = keypoints.copy()
        for i in range(0, keypoints.shape[1], 2):
            normalized[:, i] = (keypoints[:, i] - hip_center_x) / torso_length
            normalized[:, i + 1] = (keypoints[:, i + 1] - hip_center_y) / torso_length

        return normalized

    def _predict_stroke(self, sequence: np.ndarray) -> tuple:
        """Predict stroke type for a sequence."""
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            output = self.model(sequence_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

            if self.label_encoder is not None:
                predicted_label = self.label_encoder.inverse_transform(
                    [predicted.item()]
                )[0]
            else:
                predicted_label = str(predicted.item())

            confidence_score = confidence.item()

        return predicted_label, confidence_score

    def detect_strokes(
        self,
        input_data: StrokeClassificationInput,
    ) -> StrokeClassificationOutput:
        """
        Detect stroke types for all racket hits and add columns to DataFrame.

        Args:
            input_data: StrokeClassificationInput with df and player_keypoints

        Returns:
            StrokeClassificationOutput with df containing added columns:
            stroke_type, stroke_confidence
        """
        df = input_data.df
        player_keypoints = input_data.player_keypoints

        df = df.copy()
        df["stroke_type"] = ""
        df["stroke_confidence"] = 0.0

        # Get racket hit frames
        racket_hit_frames = df[df["is_racket_hit"]].index.tolist()
        frame_numbers = df.index.tolist()

        stroke_counts = {"forehand": 0, "backhand": 0, "neither": 0}

        for hit_frame in racket_hit_frames:
            player_id = int(df.loc[hit_frame, "racket_hit_player_id"])

            if player_id not in [1, 2]:
                continue

            # Get window around hit frame
            half_window = self.window_size
            start_frame = hit_frame - half_window
            end_frame = hit_frame + half_window

            # Find indices in frame_numbers list
            try:
                start_idx = frame_numbers.index(start_frame)
                end_idx = frame_numbers.index(end_frame) + 1
            except ValueError:
                # Not enough frames - skip this hit
                continue

            # Check we have the right number of frames
            if (end_idx - start_idx) != self.sequence_length:
                continue

            # Get player keypoints for this window
            kp_list = player_keypoints.get(player_id, [])
            if len(kp_list) < end_idx:
                continue

            window_keypoints = []
            valid_window = True
            for idx in range(start_idx, end_idx):
                kp = kp_list[idx]
                if kp is None:
                    valid_window = False
                    break
                window_keypoints.append(kp)

            if not valid_window:
                continue

            # Stack and reshape: (sequence_length, num_keypoints, 2) -> (sequence_length, num_features)
            window_keypoints = np.array(window_keypoints)
            num_frames = window_keypoints.shape[0]
            num_keypoints = window_keypoints.shape[1]
            window_keypoints_flat = window_keypoints.reshape(
                num_frames, num_keypoints * 2
            )

            # Normalize keypoints
            normalized_keypoints = self._normalize_keypoints_array(
                window_keypoints_flat
            )

            # Predict stroke type
            stroke_type_str, confidence = self._predict_stroke(normalized_keypoints)

            # Update DataFrame
            df.loc[hit_frame, "stroke_type"] = stroke_type_str
            df.loc[hit_frame, "stroke_confidence"] = confidence

            # Track counts
            stroke_counts[stroke_type_str] = stroke_counts.get(stroke_type_str, 0) + 1

        # Compute stats
        stats = self.get_stroke_stats(df)

        return StrokeClassificationOutput(
            df=df,
            num_strokes=stats["num_strokes"],
            stroke_counts=stats["stroke_counts"],
        )

    def get_stroke_stats(self, df: pd.DataFrame) -> Dict:
        """Get stroke detection statistics."""
        stroke_type_counts = (
            df[df["is_racket_hit"]]["stroke_type"].value_counts().to_dict()
        )
        num_strokes = int(df["is_racket_hit"].sum())

        return {
            "num_strokes": num_strokes,
            "stroke_counts": stroke_type_counts,
        }
