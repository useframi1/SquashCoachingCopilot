"""
SquashCopilot - AI-Powered Squash Coaching System

A comprehensive system for analyzing squash videos and providing coaching insights
through computer vision and machine learning.
"""

# Re-export common types and models
from squashcopilot.common import (
    # Utilities
    load_config,
    get_package_dir,
    # Constants
    BODY_KEYPOINT_INDICES,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    COCO_KEYPOINT_NAMES_FULL,
    # Types
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

# Import main module classes
from squashcopilot.modules.ball_tracking.ball_tracker import BallTracker
from squashcopilot.modules.hit_detection.wall_hit_detector import WallHitDetector
from squashcopilot.modules.hit_detection.racket_hit_detector import RacketHitDetector
from squashcopilot.modules.court_calibration.court_calibrator import CourtCalibrator
from squashcopilot.modules.player_tracking.player_tracker import PlayerTracker
from squashcopilot.modules.rally_state_detection.rally_state_detector import (
    RallyStateDetector,
)
from squashcopilot.modules.shot_type_classification.shot_classifier import (
    ShotClassifier,
)
from squashcopilot.modules.stroke_detection.stroke_detector import StrokeDetector

from squashcopilot.pipeline.pipeline import Pipeline

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
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
    # Module classes
    "BallTracker",
    "WallHitDetector",
    "RacketHitDetector",
    "CourtCalibrator",
    "PlayerTracker",
    "RallyStateDetector",
    "ShotClassifier",
    "StrokeDetector",
    # Pipeline
    "Pipeline",
]
