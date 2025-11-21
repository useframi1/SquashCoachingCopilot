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

# Re-export all models (new DataFrame-based architecture)
from .models import (
    # Video metadata
    VideoMetadata,
    # Stage 1: Court Calibration
    CourtCalibrationInput,
    CourtCalibrationOutput,
    # Stage 2a: Player Tracking
    PlayerTrackingInput,
    PlayerTrackingOutput,
    player_tracking_output_to_dict,
    player_tracking_outputs_to_dataframe,
    PlayerPostprocessingInput,
    PlayerPostprocessingOutput,
    # Stage 2b: Ball Tracking
    BallTrackingInput,
    BallTrackingOutput,
    ball_tracking_output_to_dict,
    ball_tracking_outputs_to_dataframe,
    BallPostprocessingInput,
    BallPostprocessingOutput,
    # Stage 4: Rally Segmentation
    RallySegment,
    RallySegmentationInput,
    RallySegmentationOutput,
    # Stage 5a: Wall Hit Detection
    WallHitDetectionInput,
    WallHitDetectionOutput,
    # Stage 5b: Racket Hit Detection
    RacketHitDetectionInput,
    RacketHitDetectionOutput,
    # Stage 6a: Stroke Classification
    StrokeClassificationInput,
    StrokeClassificationOutput,
    # Stage 6b: Shot Classification
    ShotClassificationInput,
    ShotClassificationOutput,
    # Pipeline Session
    PipelineSession,
)

__all__ = [
    # Utilities
    "load_config",
    "get_package_dir",
    # Constants
    "BODY_KEYPOINT_INDICES",
    "KEYPOINT_NAMES",
    "SKELETON_CONNECTIONS",
    "COCO_KEYPOINT_NAMES_FULL",
    # Types
    "Frame",
    "Config",
    "Point2D",
    "BoundingBox",
    "Homography",
    "Keypoints",
    "RallyState",
    "StrokeType",
    "ShotDirection",
    "ShotDepth",
    "ShotType",
    # Video metadata
    "VideoMetadata",
    # Stage 1: Court Calibration
    "CourtCalibrationInput",
    "CourtCalibrationOutput",
    # Stage 2a: Player Tracking
    "PlayerTrackingInput",
    "PlayerTrackingOutput",
    "player_tracking_output_to_dict",
    "player_tracking_outputs_to_dataframe",
    "PlayerPostprocessingInput",
    "PlayerPostprocessingOutput",
    # Stage 2b: Ball Tracking
    "BallTrackingInput",
    "BallTrackingOutput",
    "ball_tracking_output_to_dict",
    "ball_tracking_outputs_to_dataframe",
    "BallPostprocessingInput",
    "BallPostprocessingOutput",
    # Stage 4: Rally Segmentation
    "RallySegment",
    "RallySegmentationInput",
    "RallySegmentationOutput",
    # Stage 5a: Wall Hit Detection
    "WallHitDetectionInput",
    "WallHitDetectionOutput",
    # Stage 5b: Racket Hit Detection
    "RacketHitDetectionInput",
    "RacketHitDetectionOutput",
    # Stage 6a: Stroke Classification
    "StrokeClassificationInput",
    "StrokeClassificationOutput",
    # Stage 6b: Shot Classification
    "ShotClassificationInput",
    "ShotClassificationOutput",
    # Pipeline Session
    "PipelineSession",
]

__version__ = "0.1.0"
