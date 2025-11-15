"""
Data models for shot type classification.

This module defines input and output models for the shot-type-classification module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from squashcopilot.common.types import Point2D, ShotDirection, ShotDepth, ShotType
from squashcopilot.common.models.ball import WallHit, RacketHit


# ============================================================================
# Shot Classification Models
# ============================================================================

@dataclass
class ShotClassificationInput:
    """
    Input for shot type classification.

    Attributes:
        ball_positions: Ball trajectory as sequence of positions
        wall_hits: Detected wall hit events
        racket_hits: Detected racket hit events
    """
    ball_positions: List[Optional[Point2D]]
    wall_hits: List[WallHit]
    racket_hits: List[RacketHit]


@dataclass
class ShotResult:
    """
    Result of shot type classification for a single shot.

    Attributes:
        frame: Frame number where shot occurred (racket hit frame)
        direction: Shot direction classification
        depth: Shot depth classification
        shot_type: Combined shot type
        racket_hit_pos: Position where racket hit the ball
        next_racket_hit_pos: Position of next racket hit (if available)
        wall_hit_pos: Position where ball hit the wall (if detected)
        wall_hit_frame: Frame number of wall hit (if detected)
        vector_angle_deg: Angle between attack and rebound vectors
        rebound_distance: Distance of ball rebound in pixels
        confidence: Classification confidence (0.0-1.0)
    """
    frame: int
    direction: ShotDirection
    depth: ShotDepth
    shot_type: ShotType
    racket_hit_pos: Point2D
    next_racket_hit_pos: Optional[Point2D]
    wall_hit_pos: Optional[Point2D]
    wall_hit_frame: Optional[int]
    vector_angle_deg: Optional[float]
    rebound_distance: Optional[float]
    confidence: float

    def has_wall_hit(self) -> bool:
        """Check if shot has an associated wall hit."""
        return self.wall_hit_pos is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'frame': self.frame,
            'direction': str(self.direction),
            'depth': str(self.depth),
            'shot_type': str(self.shot_type),
            'racket_hit_pos': self.racket_hit_pos.to_dict(),
            'next_racket_hit_pos': self.next_racket_hit_pos.to_dict() if self.next_racket_hit_pos else None,
            'wall_hit_pos': self.wall_hit_pos.to_dict() if self.wall_hit_pos else None,
            'wall_hit_frame': self.wall_hit_frame,
            'vector_angle_deg': self.vector_angle_deg,
            'rebound_distance': self.rebound_distance,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ShotResult':
        """
        Create ShotResult from dictionary.

        Args:
            d: Dictionary with all fields

        Returns:
            ShotResult instance
        """
        return cls(
            frame=d['frame'],
            direction=ShotDirection.from_string(d['direction']),
            depth=ShotDepth.from_string(d['depth']),
            shot_type=ShotType.from_string(d['shot_type']),
            racket_hit_pos=Point2D(**d['racket_hit_pos']),
            next_racket_hit_pos=Point2D(**d['next_racket_hit_pos']) if d.get('next_racket_hit_pos') else None,
            wall_hit_pos=Point2D(**d['wall_hit_pos']) if d.get('wall_hit_pos') else None,
            wall_hit_frame=d.get('wall_hit_frame'),
            vector_angle_deg=d.get('vector_angle_deg'),
            rebound_distance=d.get('rebound_distance'),
            confidence=d.get('confidence', 1.0)
        )


@dataclass
class ShotClassificationResult:
    """
    Result of shot classification for an entire trajectory.

    Attributes:
        shots: List of classified shots
        num_shots: Number of shots classified
        wall_hit_detection_rate: Percentage of shots with detected wall hits
    """
    shots: List[ShotResult]
    num_shots: int = field(init=False)
    wall_hit_detection_rate: float = field(init=False)

    def __post_init__(self):
        """Calculate derived statistics."""
        self.num_shots = len(self.shots)
        if self.num_shots > 0:
            wall_hits = sum(1 for shot in self.shots if shot.has_wall_hit())
            self.wall_hit_detection_rate = wall_hits / self.num_shots
        else:
            self.wall_hit_detection_rate = 0.0

    def get_shot_at_frame(self, frame: int) -> Optional[ShotResult]:
        """
        Get shot at a specific frame.

        Args:
            frame: Frame number

        Returns:
            ShotResult or None if not found
        """
        for shot in self.shots:
            if shot.frame == frame:
                return shot
        return None

    def get_shots_by_type(self, shot_type: ShotType) -> List[ShotResult]:
        """
        Get all shots of a specific type.

        Args:
            shot_type: Shot type to filter by

        Returns:
            List of ShotResult matching the type
        """
        return [shot for shot in self.shots if shot.shot_type == shot_type]

    def get_shots_by_direction(self, direction: ShotDirection) -> List[ShotResult]:
        """
        Get all shots with a specific direction.

        Args:
            direction: Direction to filter by

        Returns:
            List of ShotResult matching the direction
        """
        return [shot for shot in self.shots if shot.direction == direction]

    def get_shots_by_depth(self, depth: ShotDepth) -> List[ShotResult]:
        """
        Get all shots with a specific depth.

        Args:
            depth: Depth to filter by

        Returns:
            List of ShotResult matching the depth
        """
        return [shot for shot in self.shots if shot.depth == depth]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'shots': [shot.to_dict() for shot in self.shots],
            'num_shots': self.num_shots,
            'wall_hit_detection_rate': self.wall_hit_detection_rate
        }


# ============================================================================
# Shot Statistics Models
# ============================================================================

@dataclass
class ShotStatistics:
    """
    Statistical summary of shots.

    Attributes:
        total_shots: Total number of shots
        by_type: Count of shots by combined type
        by_direction: Count of shots by direction
        by_depth: Count of shots by depth
        wall_hit_detection_rate: Percentage with detected wall hits
        average_vector_angle: Average angle between attack/rebound vectors
        average_rebound_distance: Average rebound distance in pixels
    """
    total_shots: int
    by_type: Dict[str, int]
    by_direction: Dict[str, int]
    by_depth: Dict[str, int]
    wall_hit_detection_rate: float
    average_vector_angle: Optional[float]
    average_rebound_distance: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_shots': self.total_shots,
            'by_type': self.by_type,
            'by_direction': self.by_direction,
            'by_depth': self.by_depth,
            'wall_hit_detection_rate': self.wall_hit_detection_rate,
            'average_vector_angle': self.average_vector_angle,
            'average_rebound_distance': self.average_rebound_distance
        }

    @classmethod
    def from_shots(cls, shots: List[ShotResult]) -> 'ShotStatistics':
        """
        Compute statistics from a list of shots.

        Args:
            shots: List of ShotResult

        Returns:
            ShotStatistics instance
        """
        total_shots = len(shots)

        if total_shots == 0:
            return cls(
                total_shots=0,
                by_type={},
                by_direction={},
                by_depth={},
                wall_hit_detection_rate=0.0,
                average_vector_angle=None,
                average_rebound_distance=None
            )

        # Count by type
        by_type = {}
        for shot in shots:
            type_str = str(shot.shot_type)
            by_type[type_str] = by_type.get(type_str, 0) + 1

        # Count by direction
        by_direction = {}
        for shot in shots:
            dir_str = str(shot.direction)
            by_direction[dir_str] = by_direction.get(dir_str, 0) + 1

        # Count by depth
        by_depth = {}
        for shot in shots:
            depth_str = str(shot.depth)
            by_depth[depth_str] = by_depth.get(depth_str, 0) + 1

        # Wall hit detection rate
        wall_hits = sum(1 for shot in shots if shot.has_wall_hit())
        wall_hit_rate = wall_hits / total_shots

        # Average vector angle
        angles = [shot.vector_angle_deg for shot in shots if shot.vector_angle_deg is not None]
        avg_angle = sum(angles) / len(angles) if angles else None

        # Average rebound distance
        distances = [shot.rebound_distance for shot in shots if shot.rebound_distance is not None]
        avg_distance = sum(distances) / len(distances) if distances else None

        return cls(
            total_shots=total_shots,
            by_type=by_type,
            by_direction=by_direction,
            by_depth=by_depth,
            wall_hit_detection_rate=wall_hit_rate,
            average_vector_angle=avg_angle,
            average_rebound_distance=avg_distance
        )
