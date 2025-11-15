"""
SquashCopilot - AI-Powered Squash Coaching System

A comprehensive system for analyzing squash videos and providing coaching insights
through computer vision and machine learning.
"""

# Re-export common types and models
from squashcopilot.common import (
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
    # Ball Models
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
    # Court Models
    CourtCalibrationInput,
    CourtCalibrationResult,
    WallColorDetectionInput,
    WallColorResult,
    # Player Models
    PlayerKeypointsData,
    PlayerTrackingInput,
    PlayerDetectionResult,
    PlayerTrackingResult,
    PlayerPostprocessingInput,
    PlayerTrajectory,
    PlayerPostprocessingResult,
    # Rally Models
    PlayerMetrics,
    AggregatedMetrics,
    RallyStateInput,
    RallyStateResult,
    RallyStateSequence,
    Rally,
    RallySegmentation,
    # Stroke Models
    StrokeDetectionInput,
    StrokeResult,
    StrokeDetectionResult,
    StrokeEvent,
    StrokeSequence,
    # Shot Models
    ShotClassificationInput,
    ShotResult,
    ShotClassificationResult,
    ShotStatistics,
)

# Import main module classes
from squashcopilot.modules.ball_tracking.ball_tracker import BallTracker
from squashcopilot.modules.ball_tracking.wall_hit_detector import WallHitDetector
from squashcopilot.modules.ball_tracking.racket_hit_detector import RacketHitDetector
from squashcopilot.modules.court_calibration.court_calibrator import CourtCalibrator
from squashcopilot.modules.player_tracking.player_tracker import PlayerTracker
from squashcopilot.modules.rally_state_detection.rally_state_detector import RallyStateDetector
from squashcopilot.modules.shot_type_classification.shot_classifier import ShotClassifier
from squashcopilot.modules.stroke_detection.stroke_detector import StrokeDetector

__version__ = '0.1.0'

__all__ = [
    # Version
    '__version__',
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
    'BallTracker',
    'WallHitDetector',
    'RacketHitDetector',
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
    'CourtCalibrator',
    'CourtCalibrationInput',
    'CourtCalibrationResult',
    'WallColorDetectionInput',
    'WallColorResult',
    # Player
    'PlayerTracker',
    'PlayerKeypointsData',
    'PlayerTrackingInput',
    'PlayerDetectionResult',
    'PlayerTrackingResult',
    'PlayerPostprocessingInput',
    'PlayerTrajectory',
    'PlayerPostprocessingResult',
    # Rally
    'RallyStateDetector',
    'PlayerMetrics',
    'AggregatedMetrics',
    'RallyStateInput',
    'RallyStateResult',
    'RallyStateSequence',
    'Rally',
    'RallySegmentation',
    # Stroke
    'StrokeDetector',
    'StrokeDetectionInput',
    'StrokeResult',
    'StrokeDetectionResult',
    'StrokeEvent',
    'StrokeSequence',
    # Shot
    'ShotClassifier',
    'ShotClassificationInput',
    'ShotResult',
    'ShotClassificationResult',
    'ShotStatistics',
]
