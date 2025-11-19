"""
Data models for stroke detection.

This module defines input and output models for the stroke-detection module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from squashcopilot.common.types import StrokeType


# ============================================================================
# Stroke Detection Models
# ============================================================================


@dataclass
class StrokeDetectionInput:
    """
    Input for stroke detection across multiple frames.

    This model is designed for evaluation where we have racket hit frames
    and need to predict stroke types using a window of keypoints around each hit.

    Attributes:
        player_keypoints: Dictionary mapping player_id to array of keypoints (frames x keypoints x 2)
        racket_hits: List of frame numbers where racket hits occurred
        racket_hit_player_ids: List of player IDs corresponding to each racket hit
        frame_numbers: List of all frame numbers in the sequence
    """

    player_keypoints: Dict[
        int, np.ndarray
    ]  # player_id -> (num_frames, num_keypoints, 2)
    racket_hits: List[int]  # Frame numbers of racket hits
    racket_hit_player_ids: List[int]  # Player ID for each racket hit
    frame_numbers: List[int]  # All frame numbers in sequence

    def __post_init__(self):
        """Validate input data consistency."""
        if len(self.racket_hits) != len(self.racket_hit_player_ids):
            raise ValueError(
                f"Mismatch between racket_hits ({len(self.racket_hits)}) "
                f"and racket_hit_player_ids ({len(self.racket_hit_player_ids)})"
            )

        # Validate player IDs in racket hits exist in player_keypoints
        for player_id in self.racket_hit_player_ids:
            if player_id not in self.player_keypoints:
                raise ValueError(
                    f"Player ID {player_id} in racket_hit_player_ids "
                    f"not found in player_keypoints keys: {list(self.player_keypoints.keys())}"
                )

    def get_racket_hit_at_frame(self, frame: int) -> Optional[int]:
        """
        Get the player ID who hit at a specific frame.

        Args:
            frame: Frame number

        Returns:
            Player ID if there's a racket hit at this frame, None otherwise
        """
        for i, hit_frame in enumerate(self.racket_hits):
            if hit_frame == frame:
                return self.racket_hit_player_ids[i]
        return None


@dataclass
class StrokeResult:
    """
    Result of stroke detection for a single racket hit.

    Similar to ShotResult in shot type classification.

    Attributes:
        frame: Frame number where racket hit occurred
        player_id: Player identifier who made the stroke
        stroke_type: Detected stroke type (forehand/backhand)
        confidence: Detection confidence (0.0-1.0)
    """

    frame: int
    player_id: int
    stroke_type: StrokeType
    confidence: float

    def is_valid_stroke(self) -> bool:
        """
        Check if this is a valid stroke (not 'neither').

        Returns:
            True if stroke is forehand or backhand
        """
        return self.stroke_type in (StrokeType.FOREHAND, StrokeType.BACKHAND)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame": self.frame,
            "player_id": self.player_id,
            "stroke_type": str(self.stroke_type),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrokeResult":
        """
        Create StrokeResult from dictionary.

        Args:
            d: Dictionary with all fields

        Returns:
            StrokeResult instance
        """
        return cls(
            frame=d["frame"],
            player_id=d["player_id"],
            stroke_type=StrokeType.from_string(d["stroke_type"]),
            confidence=d.get("confidence", 0.0),
        )


@dataclass
class StrokeDetectionResult:
    """
    Result of stroke detection containing all detected strokes.

    Similar to ShotClassificationResult in shot type classification.

    Attributes:
        strokes: List of all detected stroke results
    """

    strokes: List[StrokeResult] = field(default_factory=list)

    def get_stroke_at_frame(self, frame: int) -> Optional[StrokeResult]:
        """
        Get stroke result for a specific frame.

        Args:
            frame: Frame number

        Returns:
            StrokeResult or None if not found
        """
        for stroke in self.strokes:
            if stroke.frame == frame:
                return stroke
        return None

    def get_strokes_for_player(self, player_id: int) -> List[StrokeResult]:
        """
        Get all stroke results for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            List of StrokeResult for that player
        """
        return [stroke for stroke in self.strokes if stroke.player_id == player_id]

    def get_valid_strokes(self) -> List[StrokeResult]:
        """
        Get all valid strokes (excluding 'neither').

        Returns:
            List of StrokeResult with actual strokes (forehand/backhand)
        """
        return [stroke for stroke in self.strokes if stroke.is_valid_stroke()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strokes": [stroke.to_dict() for stroke in self.strokes],
            "total_strokes": len(self.strokes),
            "valid_strokes": len(self.get_valid_strokes()),
        }

    @property
    def total_strokes(self) -> int:
        """Get total number of stroke detections."""
        return len(self.strokes)

    @property
    def stroke_count_by_type(self) -> Dict[StrokeType, int]:
        """
        Count strokes by type.

        Returns:
            Dictionary mapping stroke type to count
        """
        counts: Dict[StrokeType, int] = {}
        for stroke in self.strokes:
            counts[stroke.stroke_type] = counts.get(stroke.stroke_type, 0) + 1
        return counts

    @property
    def stroke_count_by_player(self) -> Dict[int, int]:
        """
        Count strokes by player.

        Returns:
            Dictionary mapping player_id to count
        """
        counts: Dict[int, int] = {}
        for stroke in self.strokes:
            counts[stroke.player_id] = counts.get(stroke.player_id, 0) + 1
        return counts
