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
    """

    # Identification
    frame: int  # Frame number of racket hit
    player_id: Optional[int]  # Which player hit (1 or 2, None if unknown)

    # Classification results
    direction: ShotDirection
    depth: ShotDepth
    shot_type: ShotType

    # Position features (pixels)
    racket_hit_pos: Tuple[float, float]  # (x, y) at racket contact
    wall_hit_pos: Tuple[float, float]  # (x, y) at wall impact

    # Trajectory features (pixels)
    lateral_displacement: float  # Horizontal displacement (Δx) in pixels
    distance_to_wall: float  # Euclidean distance from racket to wall in pixels
    velocity_px_per_frame: float  # Average velocity in pixels/frame

    # Temporal features (frames)
    time_to_wall_frames: int  # Number of frames from racket to wall hit

    # Optional metadata
    confidence: float = 1.0  # Classification confidence (0-1)

    def __str__(self):
        return (
            f"Shot(frame={self.frame}, player={self.player_id}, "
            f"type={self.shot_type}, velocity={self.velocity_px_per_frame:.1f} px/f)"
        )

    def __repr__(self):
        return self.__str__()

    def summary(self) -> str:
        """Return a detailed string summary of the shot"""
        return (
            f"Shot at frame {self.frame}\n"
            f"  Player: {self.player_id if self.player_id else 'Unknown'}\n"
            f"  Type: {self.shot_type}\n"
            f"  Direction: {self.direction}\n"
            f"  Depth: {self.depth}\n"
            f"  Lateral displacement: {self.lateral_displacement:.0f} px\n"
            f"  Distance to wall: {self.distance_to_wall:.0f} px\n"
            f"  Velocity: {self.velocity_px_per_frame:.1f} px/frame\n"
            f"  Time to wall: {self.time_to_wall_frames} frames"
        )
