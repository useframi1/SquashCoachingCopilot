"""
SquashCopilot Modules

Collection of specialized modules for squash video analysis:
- ball_tracking: Ball detection and trajectory tracking
- court_calibration: Court detection and homography calibration
- player_tracking: Player detection and tracking
- rally_state_detection: Rally state classification (start/play/end)
- shot_type_classification: Shot type classification
- stroke_detection: Stroke detection from player keypoints
"""

from . import ball_tracking
from . import court_calibration
from . import player_tracking
from . import rally_state_detection
from . import shot_type_classification
from . import stroke_detection

__all__ = [
    'ball_tracking',
    'court_calibration',
    'player_tracking',
    'rally_state_detection',
    'shot_type_classification',
    'stroke_detection',
]
