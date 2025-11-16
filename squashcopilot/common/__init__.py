"""
Common package for the squash coaching copilot system.

This package provides shared data models and types for communication
between all modules in the system.
"""

# Re-export utility functions
from .utils import load_config, get_package_dir

# Re-export constants
from .constants import (
    BODY_KEYPOINT_INDICES,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    COCO_KEYPOINT_NAMES_FULL,
)

# Re-export commonly used types
from .types import (
    Frame,
    Config,
    Point2D,
    BoundingBox,
    Homography,
    Keypoints,
    RallyState,
    StrokeType,
    ShotDirection,
    ShotDepth,
    ShotType,
)

# Re-export all models
from .models import (
    # Ball
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
    # Court
    CourtCalibrationInput,
    CourtCalibrationResult,
    WallColorDetectionInput,
    WallColorResult,
    # Player
    PlayerKeypointsData,
    PlayerTrackingInput,
    PlayerDetectionResult,
    PlayerTrackingResult,
    PlayerPostprocessingInput,
    PlayerTrajectory,
    PlayerPostprocessingResult,
    # Rally
    PlayerMetrics,
    AggregatedMetrics,
    RallyStateInput,
    RallyStateResult,
    RallyStateSequence,
    Rally,
    RallySegmentation,
    # Stroke
    StrokeDetectionInput,
    StrokeResult,
    StrokeDetectionResult,
    StrokeEvent,
    StrokeSequence,
    # Shot
    ShotClassificationInput,
    ShotResult,
    ShotClassificationResult,
    ShotStatistics,
)

__all__ = [
    # Utilities
    'load_config',
    'get_package_dir',
    # Constants
    'BODY_KEYPOINT_INDICES',
    'KEYPOINT_NAMES',
    'SKELETON_CONNECTIONS',
    'COCO_KEYPOINT_NAMES_FULL',
    # Types
    'Frame',
    'Config',
    'Point2D',
    'BoundingBox',
    'Homography',
    'Keypoints',
    'RallyState',
    'StrokeType',
    'ShotDirection',
    'ShotDepth',
    'ShotType',
    # Ball
    'BallTrackingInput',
    'BallDetectionResult',
    'BallPostprocessingInput',
    'BallTrajectory',
    'WallHitInput',
    'WallHit',
    'WallHitDetectionResult',
    'RacketHitInput',
    'RacketHit',
    'RacketHitDetectionResult',
    # Court
    'CourtCalibrationInput',
    'CourtCalibrationResult',
    'WallColorDetectionInput',
    'WallColorResult',
    # Player
    'PlayerKeypointsData',
    'PlayerTrackingInput',
    'PlayerDetectionResult',
    'PlayerTrackingResult',
    'PlayerPostprocessingInput',
    'PlayerTrajectory',
    'PlayerPostprocessingResult',
    # Rally
    'PlayerMetrics',
    'AggregatedMetrics',
    'RallyStateInput',
    'RallyStateResult',
    'RallyStateSequence',
    'Rally',
    'RallySegmentation',
    # Stroke
    'StrokeDetectionInput',
    'StrokeResult',
    'StrokeDetectionResult',
    'StrokeEvent',
    'StrokeSequence',
    # Shot
    'ShotClassificationInput',
    'ShotResult',
    'ShotClassificationResult',
    'ShotStatistics',
]

__version__ = '0.1.0'
