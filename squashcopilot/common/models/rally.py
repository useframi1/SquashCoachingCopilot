"""
Data models for rally state detection.

This module defines input and output models for the rally-segmentation module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from squashcopilot.common.types import Point2D, Config, RallyState


# ============================================================================
# Rally Metrics Models
# ============================================================================


@dataclass
class PlayerMetrics:
    """
    Player position metrics for a single frame.

    Attributes:
        frame_number: Frame index
        player1_position: Player 1 position in real-world coordinates
        player2_position: Player 2 position in real-world coordinates
        player_distance: Distance between players
    """

    frame_number: int
    player1_position: Point2D
    player2_position: Point2D
    player_distance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "player1_x": self.player1_position.x,
            "player1_y": self.player1_position.y,
            "player2_x": self.player2_position.x,
            "player2_y": self.player2_position.y,
            "player_distance": self.player_distance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerMetrics":
        """
        Create PlayerMetrics from dictionary.

        Args:
            d: Dictionary with frame_number and player positions

        Returns:
            PlayerMetrics instance
        """
        return cls(
            frame_number=d["frame_number"],
            player1_position=Point2D(x=d["player1_x"], y=d["player1_y"]),
            player2_position=Point2D(x=d["player2_x"], y=d["player2_y"]),
            player_distance=d.get("player_distance", 0.0),
        )


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics over a time window.

    Attributes:
        frame_number: Representative frame number (typically window center)
        mean_distance: Mean distance between players
        median_player1_position: Median position of player 1
        median_player2_position: Median position of player 2
    """

    frame_number: int
    mean_distance: float
    median_player1_position: Point2D
    median_player2_position: Point2D

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_number": self.frame_number,
            "mean_distance": self.mean_distance,
            "median_player1_x": self.median_player1_position.x,
            "median_player1_y": self.median_player1_position.y,
            "median_player2_x": self.median_player2_position.x,
            "median_player2_y": self.median_player2_position.y,
        }


# ============================================================================
# Rally State Detection Models
# ============================================================================


@dataclass
class RallyStateInput:
    """
    Input for rally state detection.

    Attributes:
        metrics: List of player metrics per frame
        aggregated: Whether metrics are already aggregated
        config: Optional configuration
    """

    metrics: List[PlayerMetrics]
    aggregated: bool = False
    config: Optional[Config] = None


@dataclass
class RallyStateResult:
    """
    Result of rally state detection for a single frame.

    Attributes:
        frame_number: Frame index
        state: Predicted rally state
        confidence: Prediction confidence (0.0-1.0)
        features: Optional dictionary of engineered features used for prediction
    """

    frame_number: int
    state: RallyState
    confidence: float
    features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "frame_number": self.frame_number,
            "state": str(self.state),
            "confidence": self.confidence,
        }
        if self.features:
            result["features"] = self.features
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RallyStateResult":
        """
        Create RallyStateResult from dictionary.

        Args:
            d: Dictionary with frame_number, state, confidence

        Returns:
            RallyStateResult instance
        """
        return cls(
            frame_number=d["frame_number"],
            state=RallyState.from_string(d["state"]),
            confidence=d.get("confidence", 1.0),
            features=d.get("features"),
        )


@dataclass
class RallyStateSequence:
    """
    Sequence of rally state predictions.

    Attributes:
        results: List of state results per frame
        raw_predictions: Optional raw predictions before postprocessing
        postprocessed: Whether predictions have been postprocessed
    """

    results: List[RallyStateResult]
    raw_predictions: Optional[List[RallyStateResult]] = None
    postprocessed: bool = False

    def get_state_at_frame(self, frame_number: int) -> Optional[RallyState]:
        """
        Get rally state at a specific frame.

        Args:
            frame_number: Frame index

        Returns:
            RallyState or None if frame not found
        """
        for result in self.results:
            if result.frame_number == frame_number:
                return result.state
        return None

    def get_state_changes(self) -> List[Tuple[int, RallyState, RallyState]]:
        """
        Get all state transitions.

        Returns:
            List of (frame_number, old_state, new_state) tuples
        """
        changes = []
        prev_state = None

        for result in self.results:
            if prev_state is not None and result.state != prev_state:
                changes.append((result.frame_number, prev_state, result.state))
            prev_state = result.state

        return changes

    def get_segments_by_state(self, state: RallyState) -> List[Tuple[int, int]]:
        """
        Get frame ranges for all segments with a specific state.

        Args:
            state: Rally state to search for

        Returns:
            List of (start_frame, end_frame) tuples
        """
        segments = []
        start_frame = None

        for result in self.results:
            if result.state == state:
                if start_frame is None:
                    start_frame = result.frame_number
            else:
                if start_frame is not None:
                    segments.append((start_frame, result.frame_number - 1))
                    start_frame = None

        # Handle case where sequence ends in the target state
        if start_frame is not None:
            segments.append((start_frame, self.results[-1].frame_number))

        return segments

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "postprocessed": self.postprocessed,
            "num_frames": len(self.results),
        }


# ============================================================================
# Rally Segmentation Models
# ============================================================================


@dataclass
class Rally:
    """
    A single rally segment.

    Attributes:
        rally_id: Unique rally identifier
        start_frame: Frame where rally starts
        end_frame: Frame where rally ends
        duration_frames: Number of frames in rally
        state_sequence: Optional state sequence for this rally
    """

    rally_id: int
    start_frame: int
    end_frame: int
    duration_frames: int = field(init=False)
    state_sequence: Optional[List[RallyStateResult]] = None

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
        result = {
            "rally_id": self.rally_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
        }
        if self.state_sequence:
            result["state_sequence"] = [s.to_dict() for s in self.state_sequence]
        return result


@dataclass
class RallySegmentation:
    """
    Complete rally segmentation for a video.

    Attributes:
        rallies: List of detected rallies
        total_frames: Total number of frames in video
    """

    rallies: List[Rally]
    total_frames: int

    @property
    def num_rallies(self) -> int:
        """Get number of rallies."""
        return len(self.rallies)

    def get_rally_at_frame(self, frame_number: int) -> Optional[Rally]:
        """
        Get the rally containing a specific frame.

        Args:
            frame_number: Frame index

        Returns:
            Rally or None if frame not in any rally
        """
        for rally in self.rallies:
            if rally.contains_frame(frame_number):
                return rally
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rallies": [r.to_dict() for r in self.rallies],
            "total_frames": self.total_frames,
            "num_rallies": self.num_rallies,
        }
