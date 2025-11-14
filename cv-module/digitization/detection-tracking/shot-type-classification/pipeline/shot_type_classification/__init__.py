"""Shot Type Classification Pipeline - A package for classifying shot types in squash videos."""

from shot_type_classification.shot_types import Shot, ShotType, ShotDirection, ShotDepth
from shot_type_classification.shot_classifier import ShotClassifier
from shot_type_classification.shot_features import ShotFeatureExtractor

__version__ = "0.1.0"
__all__ = [
    "ShotClassifier",
    "ShotFeatureExtractor",
    "Shot",
    "ShotType",
    "ShotDirection",
    "ShotDepth",
]
