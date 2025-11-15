"""Ball Detection Pipeline - A package for detecting the ball in squash videos."""

from .ball_tracker import BallTracker
from .wall_hit_detector import WallHitDetector
from .racket_hit_detector import RacketHitDetector

__version__ = "0.1.1"
__all__ = [
    "BallTracker",
    "WallHitDetector",
    "RacketHitDetector",
]
