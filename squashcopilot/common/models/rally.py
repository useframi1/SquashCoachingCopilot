"""
Data models for rally state detection.

This module defines input and output models for the rally-segmentation module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from squashcopilot.common.types import Config


# ============================================================================
# Rally Segmentation Input Models
# ============================================================================


@dataclass
class RallySegmentationInput:
    """
    Input for rally state detection.

    Attributes:
        ball_positions: List of ball y-coordinates per frame
        frame_numbers: List of frame numbers corresponding to ball positions
        config: Optional configuration
    """

    ball_positions: List[float]
    frame_numbers: List[int]
    config: Optional[Config] = None

    def __post_init__(self):
        """Validate input."""
        if len(self.ball_positions) != len(self.frame_numbers):
            raise ValueError(
                f"Mismatch between ball_positions ({len(self.ball_positions)}) "
                f"and frame_numbers ({len(self.frame_numbers)})"
            )


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
        preprocessed_trajectory: Optional smoothed ball trajectory after Savgol filtering
    """

    segments: List[RallySegment]
    total_frames: int
    preprocessed_trajectory: Optional[List[float]] = None

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
        if self.preprocessed_trajectory:
            result["preprocessed_trajectory"] = self.preprocessed_trajectory
        return result
