"""
Data models for the squash coaching copilot system.
"""

# Ball detection models
from .ball import (
    BallTrackingInput,
    BallDetectionResult,
    BallPostprocessingInput,
    BallTrajectory,
    WallHitInput,
    WallHit,
    WallHitDetectionResult,
    RacketHitInput,
    RacketHit,
    RacketHitDetectionResult,
)

# Court detection models
from .court import (
    CourtCalibrationInput,
    CourtCalibrationResult,
    WallColorDetectionInput,
    WallColorResult,
)

# Player tracking models
from .player import (
    PlayerKeypointsData,
    PlayerTrackingInput,
    PlayerDetectionResult,
    PlayerTrackingResult,
    PlayerPostprocessingInput,
    PlayerTrajectory,
    PlayerPostprocessingResult,
)

# Rally state models
from .rally import (
    RallySegmentationInput,
    RallySegment,
    RallySegmentationResult,
)

# Stroke detection models
from .stroke import (
    StrokeDetectionInput,
    StrokeResult,
    StrokeDetectionResult,
    StrokeEvent,
    StrokeSequence,
)

# Shot classification models
from .shot import (
    ShotClassificationInput,
    ShotResult,
    ShotClassificationResult,
    ShotStatistics,
)

__all__ = [
    # Ball
    "BallTrackingInput",
    "BallDetectionResult",
    "BallPostprocessingInput",
    "BallTrajectory",
    "WallHitInput",
    "WallHit",
    "WallHitDetectionResult",
    "RacketHitInput",
    "RacketHit",
    "RacketHitDetectionResult",
    # Court
    "CourtCalibrationInput",
    "CourtCalibrationResult",
    "WallColorDetectionInput",
    "WallColorResult",
    # Player
    "PlayerKeypointsData",
    "PlayerTrackingInput",
    "PlayerDetectionResult",
    "PlayerTrackingResult",
    "PlayerPostprocessingInput",
    "PlayerTrajectory",
    "PlayerPostprocessingResult",
    # Rally
    "RallySegmentationInput",
    "RallySegment",
    "RallySegmentationResult",
    # Stroke
    "StrokeDetectionInput",
    "StrokeResult",
    "StrokeDetectionResult",
    "StrokeEvent",
    "StrokeSequence",
    # Shot
    "ShotClassificationInput",
    "ShotResult",
    "ShotClassificationResult",
    "ShotStatistics",
]
