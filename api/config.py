"""
Pipeline configuration settings.

This module centralizes all configurable parameters for the video processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataCollectorConfig:
    """Configuration for DataCollector component."""

    # Smoothing options
    enable_smoothing: bool = True
    smoothing_window: int = 5
    median_window: int = 5
    savgol_window: int = 11
    savgol_poly: int = 3
    min_states_duration: dict = field(
        default_factory=lambda: {
            "start": 50,
            "active": 200,
            "end": 200,
        }
    )

    # Validation options
    enable_validation: bool = True
    min_confidence: float = 0.3
    max_position_change: float = 25.0

    # Missing data handling
    handle_missing_data: bool = True
    max_interpolation_frames: int = 20

    # Ball hit options
    prominence: float = 50.0
    width: int = 3
    min_distance: int = 20


@dataclass
class VisualizerConfig:
    """Configuration for Visualizer component."""

    show_court_keypoints: bool = True
    show_player_keypoints: bool = True
    show_player_bbox: bool = True
    show_ball: bool = True
    show_rally_state: bool = True
    show_stroke_type: bool = True
    show_wall_hits: bool = True
    keypoint_confidence_threshold: float = 0.5


@dataclass
class AnalyzerConfig:
    """Configuration for CoachingAnalyzer component."""

    # No config needed - fps is extracted from video
    pass


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all component configs."""

    # Video processing options
    video_path: str = "tests/video-5.mp4"  # Default for testing, override in API
    display: bool = False
    base_output_path: Optional[str] = "output"  # Default for testing, override in API
    output_codec: str = "mp4v"  # Video codec for output (H.264)

    # Component configurations
    data_collector: DataCollectorConfig = None
    visualizer: VisualizerConfig = None
    analyzer: AnalyzerConfig = None

    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.data_collector is None:
            self.data_collector = DataCollectorConfig()
        if self.visualizer is None:
            self.visualizer = VisualizerConfig()
        if self.analyzer is None:
            self.analyzer = AnalyzerConfig()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create config from dictionary."""
        # Extract component configs
        data_collector_dict = config_dict.get("data_collector", {})
        visualizer_dict = config_dict.get("visualizer", {})
        analyzer_dict = config_dict.get("analyzer", {})

        # Create component configs
        data_collector = (
            DataCollectorConfig(**data_collector_dict) if data_collector_dict else None
        )
        visualizer = VisualizerConfig(**visualizer_dict) if visualizer_dict else None
        analyzer = AnalyzerConfig(**analyzer_dict) if analyzer_dict else None

        # Extract main config
        return cls(
            display=config_dict.get("display", True),
            output_path=config_dict.get("output_path", None),
            data_collector=data_collector,
            visualizer=visualizer,
            analyzer=analyzer,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "output_path": self.base_output_path,
            "data_collector": {
                "enable_smoothing": self.data_collector.enable_smoothing,
                "smoothing_window": self.data_collector.smoothing_window,
                "median_window": self.data_collector.median_window,
                "savgol_window": self.data_collector.savgol_window,
                "savgol_poly": self.data_collector.savgol_poly,
                "enable_validation": self.data_collector.enable_validation,
                "min_confidence": self.data_collector.min_confidence,
                "max_position_change": self.data_collector.max_position_change,
                "handle_missing_data": self.data_collector.handle_missing_data,
                "max_interpolation_frames": self.data_collector.max_interpolation_frames,
                "prominence": self.data_collector.prominence,
                "width": self.data_collector.width,
                "min_distance": self.data_collector.min_distance,
            },
            "visualizer": {
                "show_court_keypoints": self.visualizer.show_court_keypoints,
                "show_player_keypoints": self.visualizer.show_player_keypoints,
                "show_player_bbox": self.visualizer.show_player_bbox,
                "show_ball": self.visualizer.show_ball,
                "show_rally_state": self.visualizer.show_rally_state,
                "keypoint_confidence_threshold": self.visualizer.keypoint_confidence_threshold,
            },
            "analyzer": {},
        }


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
