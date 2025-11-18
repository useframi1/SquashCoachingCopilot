"""
Rally State Detector

This module provides LSTM-based rally state detection for squash videos.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

from squashcopilot.common.utils import load_config
from squashcopilot.common.models.rally import (
    RallySegmentationInput,
    RallySegmentationResult,
    RallySegment,
)
from squashcopilot.modules.rally_state_detection.models.lstm_model import (
    RallyStateLSTM,
)


class RallyStateDetector:
    """
    LSTM-based rally state detector.

    Uses a trained LSTM model to detect rally states from ball and player trajectories.
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

        Sets self.sequence_length and self.features from the checkpoint config.

        Args:
            model_path: Path to model checkpoint file

        Returns:
            Loaded LSTM model in evaluation mode

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                f"Please train a model first using RallyStateTrainer."
            )

        try:
            # Load checkpoint
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Get model configuration
            model_config = checkpoint["config"]

            # Set sequence_length and features from checkpoint
            # These must match what the model was trained with
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

            # Load weights
            model.load_state_dict(checkpoint["model_state_dict"])

            # Set to evaluation mode
            model.eval()

            print(f"Loaded LSTM model from: {model_path}")
            print(f"Model features: {model_config['features']}")
            print(f"Sequence length: {model_config['sequence_length']}")

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def _predict_frames(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Run LSTM inference on feature array to get frame-by-frame predictions.

        Uses sliding window approach with batched processing for efficiency.

        Args:
            features: Feature array of shape (num_frames, num_features)
            batch_size: Number of sequences to process in parallel (default: 32)

        Returns:
            Binary predictions array of shape (num_frames,) with values 0 or 1

        Note:
            For frames at the beginning/end where full sequence isn't available,
            uses padding or nearest available prediction.
        """
        num_frames = len(features)
        predictions = np.zeros(num_frames, dtype=int)

        # Handle case where video is shorter than sequence length
        if num_frames < self.sequence_length:
            # For very short videos, predict all as inactive (0)
            return predictions

        # Total number of sliding windows
        num_windows = num_frames - self.sequence_length + 1
        middle_idx = self.sequence_length // 2

        # Sliding window inference with batching
        with torch.no_grad():
            # Create progress bar
            num_batches = (num_windows + batch_size - 1) // batch_size
            pbar = tqdm(
                total=num_batches,
                desc="Predicting rally states",
                unit="batch"
            )

            for batch_start in range(0, num_windows, batch_size):
                batch_end = min(batch_start + batch_size, num_windows)

                # Extract batch of sequences
                batch_sequences = []
                for i in range(batch_start, batch_end):
                    sequence = features[i : i + self.sequence_length]
                    batch_sequences.append(sequence)

                # Stack into batch tensor
                batch_tensor = torch.FloatTensor(np.stack(batch_sequences)).to(self.device)

                # Get model predictions for the batch
                outputs = self.model(batch_tensor)  # shape: (batch_size, seq_len, 1)

                # Extract middle predictions for each sequence in batch
                for idx, i in enumerate(range(batch_start, batch_end)):
                    middle_pred = (outputs[idx, middle_idx, 0] > 0.5).int().item()
                    frame_idx = i + middle_idx
                    predictions[frame_idx] = middle_pred

                pbar.update(1)

            pbar.close()

            # Handle edge frames at the beginning
            if self.sequence_length // 2 > 0:
                # Use first available prediction for frames before middle of first window
                first_pred_idx = self.sequence_length // 2
                predictions[:first_pred_idx] = predictions[first_pred_idx]

            # Handle edge frames at the end
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

        A rally segment is defined from the first frame with label 1 (start)
        to the first frame with label 0 (end).

        Args:
            predictions: Binary array of shape (num_frames,) with 0s and 1s
            frame_numbers: List of frame numbers corresponding to predictions

        Returns:
            List of RallySegment objects
        """
        segments = []
        rally_id = 0
        in_rally = False
        rally_start = None

        for pred, frame_num in zip(predictions, frame_numbers):
            if pred == 1 and not in_rally:
                # Rally starts
                in_rally = True
                rally_start = frame_num

            elif pred == 0 and in_rally:
                # Rally ends
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

        # Handle case where video ends during a rally
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
        """
        Remove segments shorter than minimum segment length.

        Args:
            segments: List of rally segments

        Returns:
            Filtered list of segments
        """
        filtered = [
            seg for seg in segments if seg.duration_frames >= self.min_segment_length
        ]

        # Reassign rally IDs
        for i, seg in enumerate(filtered):
            seg.rally_id = i

        return filtered

    def _merge_nearby_segments(
        self, segments: List[RallySegment]
    ) -> List[RallySegment]:
        """
        Merge rally segments that are close together.

        Segments separated by less than merge_gap_threshold frames are merged.

        Args:
            segments: List of rally segments (must be sorted by start_frame)

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            gap = next_seg.start_frame - current.end_frame

            if gap <= self.merge_gap_threshold:
                # Merge segments
                current = RallySegment(
                    rally_id=current.rally_id,
                    start_frame=current.start_frame,
                    end_frame=next_seg.end_frame,
                )
            else:
                # Save current and move to next
                merged.append(current)
                current = next_seg

        # Don't forget the last segment
        merged.append(current)

        # Reassign rally IDs
        for i, seg in enumerate(merged):
            seg.rally_id = i

        return merged

    def segment_rallies(
        self, input_data: RallySegmentationInput
    ) -> RallySegmentationResult:
        """
        Detect rally segments from ball and player trajectories using LSTM.

        Args:
            input_data: RallySegmentationInput containing ball positions and
                       optional player positions

        Returns:
            RallySegmentationResult with detected rally segments

        Raises:
            ValueError: If required features are not provided in input
            RuntimeError: If model prediction fails

        Example:
            >>> detector = RallyStateDetector()
            >>> input_data = RallySegmentationInput(
            ...     ball_positions=[245.3, 248.1, ...],
            ...     frame_numbers=[0, 1, 2, ...]
            ... )
            >>> result = detector.segment_rallies(input_data)
            >>> print(f"Detected {result.num_rallies} rallies")
        """
        # Extract features using the new get_features_array method
        try:
            features = input_data.get_features_array(self.features)
        except ValueError as e:
            raise ValueError(
                f"Failed to extract features from input: {str(e)}\n"
                f"Required features: {self.features}"
            )

        # Run LSTM inference
        try:
            predictions = self._predict_frames(features)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {str(e)}")

        # Convert predictions to segments
        segments = self._predictions_to_segments(predictions, input_data.frame_numbers)

        # Filter short segments
        segments = self._filter_short_segments(segments)

        # Merge nearby segments
        segments = self._merge_nearby_segments(segments)

        # Create result
        result = RallySegmentationResult(
            segments=segments,
            total_frames=len(input_data.frame_numbers),
        )

        return result
