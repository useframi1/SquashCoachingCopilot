import numpy as np
import torch
from typing import Dict
import os

from .model.lstm_classifier import LSTMClassifier
from squashcopilot.common.utils import load_config, get_package_dir
from .utils import (
    extract_keypoints,
    normalize_keypoints,
    prepare_features,
)

from squashcopilot.common import (
    StrokeType,
    StrokeDetectionInput,
    StrokeResult,
    StrokeDetectionResult,
)


class StrokeDetector:
    """
    LSTM-based Stroke Detection Pipeline
    Processes player keypoints and predicts stroke types using a sliding window approach.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the stroke detector

        Args:
            model_path: Path to trained PyTorch model (.pt file). If None, uses path from config
            config_path: Path to config.json. If None, uses default location
        """
        # Load configuration
        self.config = config if config else load_config(config_name='stroke_detection')

        # Model configuration
        self.model_path = os.path.join(
            get_package_dir(__file__), self.config["model"]["model_path"]
        )

        # Detection parameters
        self.window_size = self.config["detection"]["window_size"]
        self.confidence_threshold = self.config["detection"]["confidence_threshold"]
        self.cooldown_frames = self.config["detection"]["cooldown_frames"]

        # Keypoint configuration
        self.relevant_indices = self.config["keypoints"]["relevant_indices"]
        self.relevant_names = self.config["keypoints"]["relevant_names"]

        # Normalization configuration
        self.min_torso_length = self.config["normalization"]["min_torso_length"]

        # Load model
        self._load_model()

        # Player tracking
        self.player_buffers = {}  # {player_id: [normalized_keypoints, ...]}
        self.cooldown_state = {}  # {player_id: (last_frame, last_stroke)}
        self.frame_counter = 0

    def _load_model(self):
        """Load the PyTorch LSTM model"""
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # Extract model configuration
        model_config = checkpoint["model_config"]

        # Initialize model
        self.model = LSTMClassifier(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )

        # Load weights and label encoder
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.label_encoder = checkpoint.get("label_encoder", None)

        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)

    def _predict_stroke(self, player_id: int) -> tuple:
        """
        Predict stroke for a player from their buffer

        Args:
            player_id: Player ID

        Returns:
            tuple: (stroke_type, confidence)
        """
        # Get the sliding window (last N frames)
        window = self.player_buffers[player_id][-self.window_size :]

        # Prepare features
        features = prepare_features(window, self.relevant_names, self.window_size)

        # Get prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        # Get predicted class and confidence
        prediction = np.argmax(probabilities)
        confidence = float(probabilities[prediction])

        # Convert to stroke name
        if self.label_encoder is not None:
            stroke_type = self.label_encoder.inverse_transform([prediction])[0]
        else:
            stroke_type = str(prediction)

        return stroke_type, confidence

    def _check_cooldown(self, player_id: int, stroke_type: str) -> bool:
        """
        Check if player is in cooldown period for this stroke

        Args:
            player_id: Player ID
            stroke_type: Predicted stroke type

        Returns:
            True if in cooldown period
        """
        if player_id not in self.cooldown_state:
            return False

        last_frame, last_stroke = self.cooldown_state[player_id]
        frames_since_last = self.frame_counter - last_frame

        # In cooldown if same stroke and within cooldown period
        return frames_since_last < self.cooldown_frames and stroke_type == last_stroke

    def _update_cooldown(self, player_id: int, stroke_type: str):
        """Update cooldown state for player"""
        self.cooldown_state[player_id] = (self.frame_counter, stroke_type)

    def process_frame(self, input_data: StrokeDetectionInput) -> StrokeDetectionResult:
        """
        Process keypoints for multiple players and return stroke predictions

        Args:
            input_data: StrokeDetectionInput with player keypoints

        Returns:
            StrokeDetectionResult with structured stroke predictions
        """
        frame_number = input_data.frame_number
        self.frame_counter += 1

        strokes = {}

        for player_id, kpts in input_data.player_keypoints.items():
            try:
                # Initialize buffer for new player
                if player_id not in self.player_buffers:
                    self.player_buffers[player_id] = []

                # Convert PlayerKeypointsData to legacy format for processing
                keypoints_data = None
                if kpts is not None:
                    keypoints_data = {"xy": kpts.xy, "conf": kpts.conf}

                if keypoints_data is not None:
                    # Extract relevant keypoints
                    keypoints = extract_keypoints(
                        keypoints_data, self.relevant_indices, self.relevant_names
                    )

                    # Normalize keypoints
                    normalized_keypoints = normalize_keypoints(
                        keypoints, self.relevant_names, self.min_torso_length
                    )
                else:
                    # No keypoints detected, use last valid keypoints
                    normalized_keypoints = self.player_buffers.get(player_id)[-1]

                # Add to buffer
                self.player_buffers[player_id].append(normalized_keypoints)

                # Check if buffer has enough frames for prediction
                if len(self.player_buffers[player_id]) >= self.window_size:
                    # Predict stroke
                    stroke_type, confidence = self._predict_stroke(player_id)

                    # Check cooldown
                    in_cooldown = self._check_cooldown(player_id, stroke_type)

                    if in_cooldown:
                        # Still in cooldown, return last prediction
                        strokes[player_id] = StrokeResult(
                            player_id=player_id,
                            stroke_type=StrokeType.from_string(stroke_type),
                            confidence=confidence,
                            frame_number=frame_number,
                            in_cooldown=True,
                        )
                    else:
                        # Check if prediction meets threshold
                        if (
                            stroke_type != "neither"
                            and confidence > self.confidence_threshold
                        ):
                            # Valid prediction, update cooldown
                            self._update_cooldown(player_id, stroke_type)
                            strokes[player_id] = StrokeResult(
                                player_id=player_id,
                                stroke_type=StrokeType.from_string(stroke_type),
                                confidence=confidence,
                                frame_number=frame_number,
                                in_cooldown=False,
                            )
                        else:
                            # Low confidence or "neither"
                            strokes[player_id] = StrokeResult(
                                player_id=player_id,
                                stroke_type=StrokeType.NEITHER,
                                confidence=confidence,
                                frame_number=frame_number,
                                in_cooldown=False,
                            )
                else:
                    # Not enough frames yet
                    strokes[player_id] = StrokeResult(
                        player_id=player_id,
                        stroke_type=StrokeType.NEITHER,
                        confidence=0.0,
                        frame_number=frame_number,
                        in_cooldown=False,
                    )

            except Exception as e:
                print(
                    f"Error processing player {player_id} at frame {self.frame_counter}: {e}"
                )
                strokes[player_id] = StrokeResult(
                    player_id=player_id,
                    stroke_type=StrokeType.NEITHER,
                    confidence=0.0,
                    frame_number=frame_number,
                    in_cooldown=False,
                )

        return StrokeDetectionResult(strokes=strokes, frame_number=frame_number)
