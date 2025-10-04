"""Data layer for collecting, validating and post-processing pipeline outputs.

Post-processing (smoothing and interpolation) is applied AFTER all frames are collected,
not during real-time processing. This ensures better quality and allows for bidirectional
interpolation and smoothing.
"""

from .data_models import (
    CourtData,
    PlayerData,
    BallData,
    FrameData,
)
from .data_collector import DataCollector

__all__ = [
    "CourtData",
    "PlayerData",
    "BallData",
    "FrameData",
    "DataCollector",
]
