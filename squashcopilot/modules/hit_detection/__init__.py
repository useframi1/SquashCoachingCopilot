"""Hit Detection Module - Detects wall and racket hits in squash videos."""

from .wall_hit_detector import WallHitDetector
from .racket_hit_detector import RacketHitDetector

__version__ = "0.1.0"
__all__ = [
    "WallHitDetector",
    "RacketHitDetector",
]
