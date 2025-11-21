"""
Data models for the squash coaching copilot pipeline.

This module exports all data models for the DataFrame-based pipeline architecture.
"""

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
