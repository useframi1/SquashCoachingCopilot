"""
Data models for ball detection and tracking.

This module defines input and output models for the ball-detection module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from squashcopilot.common.types import Frame, Config, Point2D


# ============================================================================
# Ball Tracking Models
# ============================================================================

@dataclass
class BallTrackingInput:
    """
    Input for ball tracking on a single frame.

    Attributes:
        frame: Video frame to process
    """
    frame: Frame


@dataclass
class BallDetectionResult:
    """
    Result of ball detection on a single frame.

    Attributes:
        position: Ball position in pixels (None if not detected)
        confidence: Detection confidence (0.0-1.0)
        frame_number: Frame index
        detected: Whether ball was detected
    """
    position: Optional[Point2D]
    confidence: float
    frame_number: int
    detected: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position.to_dict() if self.position else None,
            'confidence': self.confidence,
            'frame_number': self.frame_number,
            'detected': self.detected
        }

    @classmethod
    def not_detected(cls, frame_number: int) -> 'BallDetectionResult':
        """
        Create a result for when ball is not detected.

        Args:
            frame_number: Frame index

        Returns:
            BallDetectionResult with detected=False
        """
        return cls(
            position=None,
            confidence=0.0,
            frame_number=frame_number,
            detected=False
        )


@dataclass
class BallPostprocessingInput:
    """
    Input for ball trajectory postprocessing.

    Attributes:
        positions: List of ball positions (can contain None for missing detections)
        config: Optional configuration for postprocessing
    """
    positions: List[Optional[Point2D]]
    config: Optional[Config] = None


@dataclass
class BallTrajectory:
    """
    Processed ball trajectory with cleaned positions.

    Attributes:
        positions: List of smoothed ball positions (gaps filled)
        original_positions: Original positions before postprocessing
        outliers_removed: Number of outliers removed
        gaps_filled: Number of gaps interpolated
    """
    positions: List[Point2D]
    original_positions: List[Optional[Point2D]]
    outliers_removed: int = 0
    gaps_filled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'positions': [p.to_dict() for p in self.positions],
            'original_positions': [
                p.to_dict() if p else None
                for p in self.original_positions
            ],
            'outliers_removed': self.outliers_removed,
            'gaps_filled': self.gaps_filled
        }


# ============================================================================
# Wall Hit Detection Models
# ============================================================================

@dataclass
class WallHitInput:
    """
    Input for wall hit detection.

    Attributes:
        positions: Smoothed ball trajectory
        config: Optional configuration for detection
    """
    positions: List[Point2D]
    config: Optional[Config] = None


@dataclass
class WallHit:
    """
    Detected wall hit event.

    Attributes:
        frame: Frame index where wall hit occurred
        position: Ball position at wall hit
        prominence: Valley prominence (measure of hit strength)
    """
    frame: int
    position: Point2D
    prominence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frame': self.frame,
            'position': self.position.to_dict(),
            'prominence': self.prominence
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WallHit':
        """
        Create WallHit from dictionary.

        Args:
            d: Dictionary with frame, position (x, y), prominence

        Returns:
            WallHit instance
        """
        if isinstance(d.get('position'), dict):
            position = Point2D(**d['position'])
        else:
            # Legacy format: x and y as separate keys
            position = Point2D(x=d['x'], y=d['y'])

        return cls(
            frame=d['frame'],
            position=position,
            prominence=d.get('prominence', 0.0)
        )


@dataclass
class WallHitDetectionResult:
    """
    Result of wall hit detection on a trajectory.

    Attributes:
        wall_hits: List of detected wall hits
        num_hits: Number of hits detected
    """
    wall_hits: List[WallHit]
    num_hits: int = field(init=False)

    def __post_init__(self):
        """Calculate num_hits."""
        self.num_hits = len(self.wall_hits)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'wall_hits': [hit.to_dict() for hit in self.wall_hits],
            'num_hits': self.num_hits
        }


# ============================================================================
# Racket Hit Detection Models
# ============================================================================

@dataclass
class RacketHitInput:
    """
    Input for racket hit detection.

    Attributes:
        positions: Smoothed ball trajectory
        wall_hits: Detected wall hits
        config: Optional configuration for detection
    """
    positions: List[Point2D]
    wall_hits: List[WallHit]
    config: Optional[Config] = None


@dataclass
class RacketHit:
    """
    Detected racket hit event.

    Attributes:
        frame: Frame index where racket hit occurred
        position: Ball position at racket hit
        slope: Negative slope value indicating hit
    """
    frame: int
    position: Point2D
    slope: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frame': self.frame,
            'position': self.position.to_dict(),
            'slope': self.slope
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RacketHit':
        """
        Create RacketHit from dictionary.

        Args:
            d: Dictionary with frame, position (x, y), slope

        Returns:
            RacketHit instance
        """
        if isinstance(d.get('position'), dict):
            position = Point2D(**d['position'])
        else:
            # Legacy format: x and y as separate keys
            position = Point2D(x=d['x'], y=d['y'])

        return cls(
            frame=d['frame'],
            position=position,
            slope=d.get('slope', 0.0)
        )


@dataclass
class RacketHitDetectionResult:
    """
    Result of racket hit detection on a trajectory.

    Attributes:
        racket_hits: List of detected racket hits
        num_hits: Number of hits detected
    """
    racket_hits: List[RacketHit]
    num_hits: int = field(init=False)

    def __post_init__(self):
        """Calculate num_hits."""
        self.num_hits = len(self.racket_hits)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'racket_hits': [hit.to_dict() for hit in self.racket_hits],
            'num_hits': self.num_hits
        }
