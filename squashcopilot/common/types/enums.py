"""
Enumerations for the squash coaching copilot system.

This module defines enum types for categorical values used across modules.
"""

from enum import Enum
from typing import List


class RallyState(str, Enum):
    """
    Rally state classification.

    Represents the current state of a rally in a squash match.
    """

    START = "start"  # Rally is starting (players getting ready)
    END = "end"  # Rally has ended (point scored or let/stroke)

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "RallyState":
        """
        Convert string to RallyState.

        Args:
            s: String representation

        Returns:
            RallyState enum value
        """
        s_lower = s.lower()
        for state in cls:
            if state.value == s_lower:
                return state
        raise ValueError(f"Invalid rally state: {s}")

    @classmethod
    def all_values(cls) -> List[str]:
        """Get all possible state values."""
        return [state.value for state in cls]


class StrokeType(str, Enum):
    """
    Player stroke classification.

    Represents the type of stroke a player is performing.
    """

    FOREHAND = "forehand"
    BACKHAND = "backhand"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "StrokeType":
        """
        Convert string to StrokeType.

        Args:
            s: String representation

        Returns:
            StrokeType enum value
        """
        s_lower = s.lower()
        for stroke in cls:
            if stroke.value == s_lower:
                return stroke
        raise ValueError(f"Invalid stroke type: {s}")

    @classmethod
    def all_values(cls) -> List[str]:
        """Get all possible stroke values."""
        return [stroke.value for stroke in cls]


class ShotDirection(str, Enum):
    """
    Shot direction classification.

    Represents the direction of a shot based on ball trajectory.
    """

    STRAIGHT = "straight"
    CROSS_COURT = "cross_court"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "ShotDirection":
        """
        Convert string to ShotDirection.

        Args:
            s: String representation

        Returns:
            ShotDirection enum value
        """
        s_lower = s.lower()
        for direction in cls:
            if direction.value == s_lower:
                return direction
        raise ValueError(f"Invalid shot direction: {s}")

    @classmethod
    def all_values(cls) -> List[str]:
        """Get all possible direction values."""
        return [direction.value for direction in cls]


class ShotDepth(str, Enum):
    """
    Shot depth classification.

    Represents whether a shot is a drop shot or a long/drive shot.
    """

    DROP = "drop"
    LONG = "long"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "ShotDepth":
        """
        Convert string to ShotDepth.

        Args:
            s: String representation

        Returns:
            ShotDepth enum value
        """
        s_lower = s.lower()
        for depth in cls:
            if depth.value == s_lower:
                return depth
        raise ValueError(f"Invalid shot depth: {s}")

    @classmethod
    def all_values(cls) -> List[str]:
        """Get all possible depth values."""
        return [depth.value for depth in cls]


class ShotType(str, Enum):
    """
    Combined shot type classification.

    Represents the full shot type combining direction and depth.
    """

    STRAIGHT_DRIVE = "straight_drive"
    STRAIGHT_DROP = "straight_drop"
    CROSS_COURT_DRIVE = "cross_court_drive"
    CROSS_COURT_DROP = "cross_court_drop"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_direction_and_depth(
        cls, direction: ShotDirection, depth: ShotDepth
    ) -> "ShotType":
        """
        Create ShotType from direction and depth.

        Args:
            direction: Shot direction
            depth: Shot depth

        Returns:
            Combined ShotType
        """
        mapping = {
            (ShotDirection.STRAIGHT, ShotDepth.LONG): cls.STRAIGHT_DRIVE,
            (ShotDirection.STRAIGHT, ShotDepth.DROP): cls.STRAIGHT_DROP,
            (ShotDirection.CROSS_COURT, ShotDepth.LONG): cls.CROSS_COURT_DRIVE,
            (ShotDirection.CROSS_COURT, ShotDepth.DROP): cls.CROSS_COURT_DROP,
        }
        return mapping[(direction, depth)]

    @property
    def direction(self) -> ShotDirection:
        """Extract direction from shot type."""
        if "straight" in self.value:
            return ShotDirection.STRAIGHT
        elif "cross" in self.value:
            return ShotDirection.CROSS_COURT

    @property
    def depth(self) -> ShotDepth:
        """Extract depth from shot type."""
        if "drop" in self.value:
            return ShotDepth.DROP
        else:
            return ShotDepth.LONG

    @classmethod
    def from_string(cls, s: str) -> "ShotType":
        """
        Convert string to ShotType.

        Args:
            s: String representation

        Returns:
            ShotType enum value
        """
        s_lower = s.lower()
        for shot_type in cls:
            if shot_type.value == s_lower:
                return shot_type
        raise ValueError(f"Invalid shot type: {s}")

    @classmethod
    def all_values(cls) -> List[str]:
        """Get all possible shot type values."""
        return [shot_type.value for shot_type in cls]
