"""
Shot type definitions and data structures for shot classification.

This module defines the enums and dataclasses used to represent
classified shots in squash. All measurements are in pixel coordinates.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class ShotDirection(Enum):
    """Direction of shot based on lateral ball movement"""

    STRAIGHT = "straight"
    CROSS_COURT = "cross_court"
    DOWN_THE_LINE = "down_the_line"

    def __str__(self):
        return self.value.replace("_", " ").title()


class ShotDepth(Enum):
    """Depth classification of shot"""

    DROP = "drop"
    LONG = "long"

    def __str__(self):
        return self.value.title()


class ShotType(Enum):
    """Complete shot type combining direction and depth"""

    STRAIGHT_DRIVE = "straight_drive"
    STRAIGHT_DROP = "straight_drop"
    CROSS_COURT_DRIVE = "cross_court_drive"
    CROSS_COURT_DROP = "cross_court_drop"
    DOWN_LINE_DRIVE = "down_line_drive"
    DOWN_LINE_DROP = "down_line_drop"

    def __str__(self):
        return self.value.replace("_", " ").title()

    @property
    def direction(self) -> ShotDirection:
        """Get the direction component of this shot type"""
        if "straight" in self.value:
            return ShotDirection.STRAIGHT
        elif "cross" in self.value:
            return ShotDirection.CROSS_COURT
        elif "down_line" in self.value:
            return ShotDirection.DOWN_THE_LINE

    @property
    def depth(self) -> ShotDepth:
        """Get the depth component of this shot type"""
        if "drop" in self.value:
            return ShotDepth.DROP
        elif "drive" in self.value:
            return ShotDepth.LONG


@dataclass
class Shot:
    """
    Represents a classified shot with all relevant features.

    All spatial measurements are in pixel coordinates.
    All temporal measurements are in frames.
    Shot spans from one racket hit to the next racket hit (player exchange).
    Uses vector-based classification: racket→wall and wall→next racket.
    """

    # Identification
    frame: int  # Frame number of racket hit (start of shot)

    # Classification results
    direction: ShotDirection
    depth: ShotDepth
    shot_type: ShotType

    # Position features (pixels)
    racket_hit_pos: Tuple[float, float]  # (x, y) at initial racket contact
    next_racket_hit_pos: Tuple[float, float]  # (x, y) at next racket contact

    # Wall hit information (front wall only, optional)
    wall_hit_pos: Optional[Tuple[float, float]] = None  # (x, y) on front wall
    wall_hit_frame: Optional[int] = None  # Frame when ball hit front wall

    # Vector-based features
    vector_angle_deg: Optional[float] = None  # Angle between attack and rebound vectors
    rebound_distance: Optional[float] = None  # Length of wall→next racket vector (px)

    # Optional metadata
    confidence: float = 1.0  # Classification confidence (0-1)

    def __str__(self):
        return (
            f"Shot(frame={self.frame}, "
            f"type={self.shot_type}, angle={self.vector_angle_deg:.1f}°)"
        )

    def __repr__(self):
        return self.__str__()

    def summary(self) -> str:
        """Return a detailed string summary of the shot"""
        summary_lines = [
            f"Shot at frame {self.frame}",
            f"  Type: {self.shot_type}",
            f"  Direction: {self.direction}",
            f"  Depth: {self.depth}",
            f"  Racket hit: ({self.racket_hit_pos[0]:.0f}, {self.racket_hit_pos[1]:.0f})",
            f"  Next racket hit: ({self.next_racket_hit_pos[0]:.0f}, {self.next_racket_hit_pos[1]:.0f})",
        ]

        # Add wall hit information if available
        if self.wall_hit_pos is not None:
            summary_lines.extend([
                f"  Wall hit: ({self.wall_hit_pos[0]:.0f}, {self.wall_hit_pos[1]:.0f}) at frame {self.wall_hit_frame}",
                f"  Vector angle: {self.vector_angle_deg:.1f}°",
                f"  Rebound distance: {self.rebound_distance:.0f} px",
            ])

        summary_lines.append(f"  Confidence: {self.confidence:.2f}")

        return "\n".join(summary_lines)
