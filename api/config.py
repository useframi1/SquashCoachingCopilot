"""
Pipeline configuration settings.

This module centralizes all configurable parameters for the video processing pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataCollectorConfig:
    """Configuration for DataCollector component."""

    # Smoothing options
    enable_smoothing: bool = True
    smoothing_window: int = 5

    # Validation options
    enable_validation: bool = True
    min_confidence: float = 0.3
    max_position_change: float = 25.0

    # Missing data handling
    handle_missing_data: bool = True
    max_interpolation_frames: int = 10


@dataclass
class AnalyzerConfig:
    """Configuration for CoachingAnalyzer component."""

    # No config needed - fps is extracted from video
    pass


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all component configs."""

    # Video processing options
    video_path: str = "tests/video-5.mp4"

    # Component configurations
    data_collector: DataCollectorConfig = None
    analyzer: AnalyzerConfig = None

    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.data_collector is None:
            self.data_collector = DataCollectorConfig()
        if self.analyzer is None:
            self.analyzer = AnalyzerConfig()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create config from dictionary."""
        # Extract component configs
        data_collector_dict = config_dict.get("data_collector", {})
        analyzer_dict = config_dict.get("analyzer", {})

        # Create component configs
        data_collector = (
            DataCollectorConfig(**data_collector_dict) if data_collector_dict else None
        )
        analyzer = AnalyzerConfig(**analyzer_dict) if analyzer_dict else None

        # Extract main config
        return cls(
            data_collector=data_collector,
            analyzer=analyzer,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data_collector": {
                "enable_smoothing": self.data_collector.enable_smoothing,
                "smoothing_window": self.data_collector.smoothing_window,
                "enable_validation": self.data_collector.enable_validation,
                "min_confidence": self.data_collector.min_confidence,
                "max_position_change": self.data_collector.max_position_change,
                "handle_missing_data": self.data_collector.handle_missing_data,
                "max_interpolation_frames": self.data_collector.max_interpolation_frames,
            },
            "analyzer": {},
        }


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
