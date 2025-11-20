"""
Pipeline module for orchestrating the complete squash video analysis pipeline.

This module provides the main Pipeline class that coordinates all 7 stages of video processing:
1. Court calibration
2. Frame-by-frame tracking (player + ball)
3. Trajectory postprocessing
4. Rally segmentation
5. Hit detection (wall + racket)
6. Stroke and shot classification
7. Export and visualization
"""

from .pipeline import Pipeline

__all__ = ["Pipeline"]
