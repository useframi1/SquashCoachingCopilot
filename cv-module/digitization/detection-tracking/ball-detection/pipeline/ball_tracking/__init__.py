"""Ball Detection Pipeline - A package for detecting the ball in squash videos."""

from ball_tracking.ball_tracker import BallTracker
from ball_tracking.wall_hit_detector import WallHitDetector
from ball_tracking.racket_hit_detector import RacketHitDetector
from ball_tracking.shot_types import Shot, ShotType, ShotDirection, ShotDepth
from ball_tracking.shot_classifier import ShotClassifier
from ball_tracking.shot_features import ShotFeatureExtractor

__version__ = "0.1.1"
__all__ = [
    "BallTracker",
    "WallHitDetector",
    "RacketHitDetector",
    "ShotClassifier",
    "ShotFeatureExtractor",
    "Shot",
    "ShotType",
    "ShotDirection",
    "ShotDepth",
]
