"""
Data models for stroke detection.

This module defines input and output models for the stroke-detection module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from squashcopilot.common.types import StrokeType
from squashcopilot.common.models.player import PlayerKeypointsData


# ============================================================================
# Stroke Detection Models
# ============================================================================

@dataclass
class StrokeDetectionInput:
    """
    Input for stroke detection on a single frame.

    Attributes:
        player_keypoints: Dictionary mapping player_id to keypoints data
        frame_number: Frame index
    """
    player_keypoints: Dict[int, PlayerKeypointsData]
    frame_number: int


@dataclass
class StrokeResult:
    """
    Result of stroke detection for a single player.

    Attributes:
        player_id: Player identifier
        stroke_type: Detected stroke type
        confidence: Detection confidence (0.0-1.0)
        frame_number: Frame where stroke was detected
        in_cooldown: Whether detector is in cooldown period
    """
    player_id: int
    stroke_type: StrokeType
    confidence: float
    frame_number: int
    in_cooldown: bool = False

    def is_stroke_detected(self) -> bool:
        """
        Check if an actual stroke was detected (not 'neither').

        Returns:
            True if stroke is forehand or backhand
        """
        return self.stroke_type in (StrokeType.FOREHAND, StrokeType.BACKHAND)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'player_id': self.player_id,
            'stroke': str(self.stroke_type),
            'confidence': self.confidence,
            'frame_number': self.frame_number,
            'in_cooldown': self.in_cooldown
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StrokeResult':
        """
        Create StrokeResult from dictionary.

        Args:
            d: Dictionary with all fields

        Returns:
            StrokeResult instance
        """
        return cls(
            player_id=d['player_id'],
            stroke_type=StrokeType.from_string(d['stroke']),
            confidence=d.get('confidence', 0.0),
            frame_number=d.get('frame_number', 0),
            in_cooldown=d.get('in_cooldown', False)
        )

    @classmethod
    def no_stroke(cls, player_id: int, frame_number: int) -> 'StrokeResult':
        """
        Create a result for when no stroke is detected.

        Args:
            player_id: Player identifier
            frame_number: Frame index

        Returns:
            StrokeResult with stroke_type=NEITHER
        """
        return cls(
            player_id=player_id,
            stroke_type=StrokeType.NEITHER,
            confidence=0.0,
            frame_number=frame_number,
            in_cooldown=False
        )


@dataclass
class StrokeDetectionResult:
    """
    Result of stroke detection for all players in a frame.

    Attributes:
        strokes: Dictionary mapping player_id to stroke result
        frame_number: Frame index
    """
    strokes: Dict[int, StrokeResult]
    frame_number: int

    def get_stroke(self, player_id: int) -> Optional[StrokeResult]:
        """
        Get stroke result for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            StrokeResult or None if not found
        """
        return self.strokes.get(player_id)

    def has_any_stroke(self) -> bool:
        """
        Check if any player performed a stroke.

        Returns:
            True if at least one player has a detected stroke
        """
        return any(stroke.is_stroke_detected() for stroke in self.strokes.values())

    def get_detected_strokes(self) -> List[StrokeResult]:
        """
        Get all detected strokes (excluding 'neither').

        Returns:
            List of StrokeResult with actual strokes
        """
        return [
            stroke for stroke in self.strokes.values()
            if stroke.is_stroke_detected()
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strokes': {
                player_id: stroke.to_dict()
                for player_id, stroke in self.strokes.items()
            },
            'frame_number': self.frame_number,
            'has_stroke': self.has_any_stroke()
        }

    @classmethod
    def no_strokes(cls, player_ids: List[int], frame_number: int) -> 'StrokeDetectionResult':
        """
        Create a result for when no strokes are detected.

        Args:
            player_ids: List of player identifiers
            frame_number: Frame index

        Returns:
            StrokeDetectionResult with NEITHER for all players
        """
        return cls(
            strokes={
                pid: StrokeResult.no_stroke(pid, frame_number)
                for pid in player_ids
            },
            frame_number=frame_number
        )


# ============================================================================
# Stroke Sequence Models
# ============================================================================

@dataclass
class StrokeEvent:
    """
    A detected stroke event with timing information.

    Attributes:
        player_id: Player who performed the stroke
        stroke_type: Type of stroke
        frame_number: Frame where stroke occurred
        timestamp: Timestamp in seconds (if available)
        confidence: Detection confidence
    """
    player_id: int
    stroke_type: StrokeType
    frame_number: int
    timestamp: Optional[float] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'player_id': self.player_id,
            'stroke_type': str(self.stroke_type),
            'frame_number': self.frame_number,
            'confidence': self.confidence
        }
        if self.timestamp is not None:
            result['timestamp'] = self.timestamp
        return result


@dataclass
class StrokeSequence:
    """
    Sequence of stroke events over time.

    Attributes:
        events: List of stroke events in chronological order
    """
    events: List[StrokeEvent] = field(default_factory=list)

    def add_event(self, event: StrokeEvent) -> None:
        """Add a stroke event to the sequence."""
        self.events.append(event)

    def get_events_for_player(self, player_id: int) -> List[StrokeEvent]:
        """
        Get all stroke events for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            List of StrokeEvent for that player
        """
        return [e for e in self.events if e.player_id == player_id]

    def get_events_in_range(
        self,
        start_frame: int,
        end_frame: int
    ) -> List[StrokeEvent]:
        """
        Get stroke events within a frame range.

        Args:
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)

        Returns:
            List of StrokeEvent in range
        """
        return [
            e for e in self.events
            if start_frame <= e.frame_number <= end_frame
        ]

    def get_stroke_count_by_type(self) -> Dict[StrokeType, int]:
        """
        Count strokes by type.

        Returns:
            Dictionary mapping stroke type to count
        """
        counts = {}
        for event in self.events:
            counts[event.stroke_type] = counts.get(event.stroke_type, 0) + 1
        return counts

    def get_stroke_count_by_player(self) -> Dict[int, int]:
        """
        Count strokes by player.

        Returns:
            Dictionary mapping player_id to count
        """
        counts = {}
        for event in self.events:
            counts[event.player_id] = counts.get(event.player_id, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'events': [e.to_dict() for e in self.events],
            'total_strokes': len(self.events),
            'by_type': {str(k): v for k, v in self.get_stroke_count_by_type().items()},
            'by_player': self.get_stroke_count_by_player()
        }

    def __len__(self) -> int:
        """Get number of stroke events."""
        return len(self.events)
