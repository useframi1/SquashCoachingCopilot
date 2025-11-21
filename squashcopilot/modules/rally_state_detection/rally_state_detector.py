"""
Rally State Detector

This module provides LSTM-based rally state detection for squash videos.
Uses DataFrame-based input/output for pipeline integration.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

from squashcopilot.common.utils import load_config
from squashcopilot.common.models import (
    RallySegmentationInput,
    RallySegmentationOutput,
    RallySegment,
    CourtCalibrationOutput,
)
from squashcopilot.modules.rally_state_detection.model.lstm_model import (
    RallyStateLSTM,
)


class RallyStateDetector:
    """
    LSTM-based rally state detector.

    Uses a trained LSTM model to detect rally states from ball and player trajectories.
    Works with DataFrame-based pipeline architecture.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the rally state detector.

        Args:
            config: Configuration dictionary. If None, loads from rally_state_detection.yaml
        """
        # Load configuration
        self.config = (
            config if config else load_config(config_name="rally_state_detection")
        )

        # Get inference configuration
        self.inference_config = self.config["inference"]
        self.min_segment_length = self.inference_config["min_segment_length"]
        self.merge_gap_threshold = self.inference_config["merge_gap_threshold"]

        # Get project root and model path
        project_root = Path(__file__).parent.parent.parent.parent
        model_path = project_root / self.inference_config["model_path"]

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model (this will also set sequence_length and features from checkpoint)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path) -> RallyStateLSTM:
        """
        Load trained LSTM model from checkpoint.

        Args:
            model_path: Path to model checkpoint file

        Returns:
            Loaded LSTM model in evaluation mode
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please train a model first using RallyStateTrainer."
            )

        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            model_config = checkpoint["config"]

            # Set sequence_length and features from checkpoint
            self.sequence_length = model_config["sequence_length"]
            self.features = model_config["features"]

            # Initialize model
            model = RallyStateLSTM(
                input_size=model_config["input_size"],
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"],
                bidirectional=model_config["bidirectional"],
            ).to(self.device)

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def _extract_features_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature array from DataFrame.

        Args:
            df: DataFrame with tracking data

        Returns:
            Feature array of shape (num_frames, num_features)
        """

        return df[self.features].to_numpy()

    def _predict_frames(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Run LSTM inference on feature array to get frame-by-frame predictions.

        Args:
            features: Feature array of shape (num_frames, num_features)
            batch_size: Number of sequences to process in parallel

        Returns:
            Binary predictions array of shape (num_frames,)
        """
        num_frames = len(features)
        predictions = np.zeros(num_frames, dtype=int)

        if num_frames < self.sequence_length:
            return predictions

        num_windows = num_frames - self.sequence_length + 1
        middle_idx = self.sequence_length // 2

        with torch.no_grad():
            num_batches = (num_windows + batch_size - 1) // batch_size
            pbar = tqdm(total=num_batches, desc="Predicting rally states", unit="batch")

            for batch_start in range(0, num_windows, batch_size):
                batch_end = min(batch_start + batch_size, num_windows)

                batch_sequences = []
                for i in range(batch_start, batch_end):
                    sequence = features[i : i + self.sequence_length]
                    batch_sequences.append(sequence)

                batch_tensor = torch.FloatTensor(np.stack(batch_sequences)).to(
                    self.device
                )
                outputs = self.model(batch_tensor)

                for idx, i in enumerate(range(batch_start, batch_end)):
                    middle_pred = (outputs[idx, middle_idx, 0] > 0.5).int().item()
                    frame_idx = i + middle_idx
                    predictions[frame_idx] = middle_pred

                pbar.update(1)

            pbar.close()

            # Handle edge frames
            if self.sequence_length // 2 > 0:
                first_pred_idx = self.sequence_length // 2
                predictions[:first_pred_idx] = predictions[first_pred_idx]

            last_pred_idx = (
                num_frames - self.sequence_length + self.sequence_length // 2
            )
            if last_pred_idx < num_frames:
                predictions[last_pred_idx:] = predictions[last_pred_idx - 1]

        return predictions

    def _predictions_to_segments(
        self, predictions: np.ndarray, frame_numbers: List[int]
    ) -> List[RallySegment]:
        """
        Convert binary frame predictions to rally segments.

        Args:
            predictions: Binary array with 0s and 1s
            frame_numbers: List of frame numbers

        Returns:
            List of RallySegment objects
        """
        segments = []
        rally_id = 0
        in_rally = False
        rally_start = None

        for pred, frame_num in zip(predictions, frame_numbers):
            if pred == 1 and not in_rally:
                in_rally = True
                rally_start = frame_num
            elif pred == 0 and in_rally:
                rally_end = frame_num
                segments.append(
                    RallySegment(
                        rally_id=rally_id,
                        start_frame=rally_start,
                        end_frame=rally_end,
                    )
                )
                rally_id += 1
                in_rally = False

        if in_rally and rally_start is not None:
            segments.append(
                RallySegment(
                    rally_id=rally_id,
                    start_frame=rally_start,
                    end_frame=frame_numbers[-1],
                )
            )

        return segments

    def _filter_short_segments(
        self, segments: List[RallySegment]
    ) -> List[RallySegment]:
        """Remove segments shorter than minimum segment length."""
        filtered = [
            seg for seg in segments if seg.num_frames >= self.min_segment_length
        ]

        for i, seg in enumerate(filtered):
            seg.rally_id = i

        return filtered

    def _merge_nearby_segments(
        self, segments: List[RallySegment]
    ) -> List[RallySegment]:
        """Merge rally segments that are close together."""
        if not segments:
            return []

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            gap = next_seg.start_frame - current.end_frame

            if gap <= self.merge_gap_threshold:
                current = RallySegment(
                    rally_id=current.rally_id,
                    start_frame=current.start_frame,
                    end_frame=next_seg.end_frame,
                )
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)

        for i, seg in enumerate(merged):
            seg.rally_id = i

        return merged

    def segment_rallies(
        self, input_data: RallySegmentationInput
    ) -> RallySegmentationOutput:
        """
        Detect rally segments from tracking DataFrame using LSTM.

        Args:
            input_data: RallySegmentationInput containing DataFrame and calibration

        Returns:
            RallySegmentationOutput with DataFrame (adds rally_id, is_rally_frame columns)
            and segment list
        """
        df = input_data.df.copy()
        frame_numbers = list(df.index)

        # Extract features from DataFrame
        try:
            features = self._extract_features_from_df(df)
        except ValueError as e:
            raise ValueError(
                f"Failed to extract features from DataFrame: {str(e)}\n"
                f"Required features: {self.features}"
            )

        # Handle NaN values in features (replace with interpolated values or 0)
        features = np.nan_to_num(features, nan=0.0)

        # Run LSTM inference
        try:
            predictions = self._predict_frames(features)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {str(e)}")

        # Convert predictions to segments
        segments = self._predictions_to_segments(predictions, frame_numbers)
        segments = self._filter_short_segments(segments)
        segments = self._merge_nearby_segments(segments)

        # Add rally columns to DataFrame
        df["is_rally_frame"] = predictions.astype(bool)
        df["rally_id"] = -1  # Default to -1 (not in rally)

        # Assign rally IDs to frames
        for segment in segments:
            mask = (df.index >= segment.start_frame) & (df.index <= segment.end_frame)
            df.loc[mask, "rally_id"] = segment.rally_id

        # Calculate statistics
        rally_frame_count = int(df["is_rally_frame"].sum())

        return RallySegmentationOutput(
            df=df,
            segments=segments,
            total_frames=len(frame_numbers),
            num_rallies=len(segments),
            rally_frame_count=rally_frame_count,
        )
