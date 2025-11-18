"""
Data models for rally state detection.

This module defines input and output models for the rally-segmentation module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from squashcopilot.common.types import Config


# ============================================================================
# Rally Segmentation Input Models
# ============================================================================


@dataclass
class RallySegmentationInput:
    """
    Input for rally state detection.

    Supports multiple features for LSTM-based detection while maintaining
    backward compatibility with ball-only detection.

    Attributes:
        ball_positions: List of ball y-coordinates per frame (required)
        frame_numbers: List of frame numbers corresponding to positions (required)
        player_1_x: Optional list of player 1 x-coordinates in meters
        player_1_y: Optional list of player 1 y-coordinates in meters
        player_2_x: Optional list of player 2 x-coordinates in meters
        player_2_y: Optional list of player 2 y-coordinates in meters
        config: Optional configuration
    """

    ball_positions: List[float]
    frame_numbers: List[int]
    player_1_x: Optional[List[float]] = None
    player_1_y: Optional[List[float]] = None
    player_2_x: Optional[List[float]] = None
    player_2_y: Optional[List[float]] = None
    config: Optional[Config] = None

    def __post_init__(self):
        """Validate input."""
        num_frames = len(self.ball_positions)

        # Validate frame numbers match
        if len(self.frame_numbers) != num_frames:
            raise ValueError(
                f"Mismatch between ball_positions ({num_frames}) "
                f"and frame_numbers ({len(self.frame_numbers)})"
            )

        # Validate optional features have same length if provided
        optional_features = [
            ("player_1_x", self.player_1_x),
            ("player_1_y", self.player_1_y),
            ("player_2_x", self.player_2_x),
            ("player_2_y", self.player_2_y),
        ]

        for feature_name, feature_values in optional_features:
            if feature_values is not None and len(feature_values) != num_frames:
                raise ValueError(
                    f"Mismatch between ball_positions ({num_frames}) "
                    f"and {feature_name} ({len(feature_values)})"
                )

    def get_features_array(self, feature_names: List[str]) -> np.ndarray:
        """
        Get feature array for specified features.

        Args:
            feature_names: List of feature names to extract.
                          Valid names: 'ball_y', 'player_1_x_meter', 'player_1_y_meter',
                                      'player_2_x_meter', 'player_2_y_meter'

        Returns:
            Numpy array of shape (num_frames, num_features)

        Raises:
            ValueError: If feature not available or feature name invalid

        Example:
            >>> features = input_data.get_features_array(['ball_y', 'player_1_x_meter'])
            >>> features.shape
            (10000, 2)
        """
        # Map feature names to attributes
        feature_map = {
            "ball_y": self.ball_positions,
            "player_1_x_meter": self.player_1_x,
            "player_1_y_meter": self.player_1_y,
            "player_2_x_meter": self.player_2_x,
            "player_2_y_meter": self.player_2_y,
        }

        # Extract features
        feature_arrays = []
        for feature_name in feature_names:
            if feature_name not in feature_map:
                raise ValueError(
                    f"Invalid feature name: {feature_name}. "
                    f"Valid names: {list(feature_map.keys())}"
                )

            feature_data = feature_map[feature_name]
            if feature_data is None:
                raise ValueError(
                    f"Feature '{feature_name}' was requested but not provided in input"
                )

            feature_arrays.append(feature_data)

        # Stack into array (num_frames, num_features)
        return np.column_stack(feature_arrays)


# ============================================================================
# Rally Segmentation Output Models
# ============================================================================


@dataclass
class RallySegment:
    """
    A single rally segment.

    Attributes:
        rally_id: Unique rally identifier
        start_frame: Frame where rally starts
        end_frame: Frame where rally ends
        duration_frames: Number of frames in rally
    """

    rally_id: int
    start_frame: int
    end_frame: int
    duration_frames: int = field(init=False)

    def __post_init__(self):
        """Calculate duration."""
        self.duration_frames = self.end_frame - self.start_frame + 1

    def contains_frame(self, frame_number: int) -> bool:
        """
        Check if a frame is within this rally.

        Args:
            frame_number: Frame index

        Returns:
            True if frame is in rally
        """
        return self.start_frame <= frame_number <= self.end_frame

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rally_id": self.rally_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
        }


@dataclass
class RallySegmentationResult:
    """
    Complete rally segmentation result.

    Attributes:
        segments: List of detected rally segments
        total_frames: Total number of frames processed
    """

    segments: List[RallySegment]
    total_frames: int

    @property
    def num_rallies(self) -> int:
        """Get number of rallies."""
        return len(self.segments)

    def get_rally_at_frame(self, frame_number: int) -> Optional[RallySegment]:
        """
        Get the rally containing a specific frame.

        Args:
            frame_number: Frame index

        Returns:
            RallySegment or None if frame not in any rally
        """
        for rally in self.segments:
            if rally.contains_frame(frame_number):
                return rally
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "segments": [seg.to_dict() for seg in self.segments],
            "total_frames": self.total_frames,
            "num_rallies": self.num_rallies,
        }

        return result
