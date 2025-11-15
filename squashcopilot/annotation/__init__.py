"""
Annotation module for squash video analysis.

This module provides the Annotator class for processing squash videos
and generating comprehensive annotations including:
- Player tracking and keypoints
- Ball tracking
- Wall and racket hit detection
- Annotated video output
"""

from .annotator import Annotator

__all__ = ['Annotator']
