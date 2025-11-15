"""
Common types for the squash coaching copilot system.
"""

from .base import Frame, Config
from .geometry import Point2D, BoundingBox, Homography, Keypoints
from .enums import RallyState, StrokeType, ShotDirection, ShotDepth, ShotType

__all__ = [
    # Base types
    'Frame',
    'Config',
    # Geometry types
    'Point2D',
    'BoundingBox',
    'Homography',
    'Keypoints',
    # Enums
    'RallyState',
    'StrokeType',
    'ShotDirection',
    'ShotDepth',
    'ShotType',
]
