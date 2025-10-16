"""
Integrated squash video analysis pipeline.

This pipeline combines video processing, data collection, and coaching analysis
into a single, easy-to-use interface.
"""

from typing import Optional, Union
from orchestration import PipelineOrchestrator
from video_io import VideoHandler
from data import DataCollector
from analysis import CoachingAnalyzer
from config import PipelineConfig, DEFAULT_CONFIG


class Pipeline:
    """
    Complete pipeline for squash video analysis.

    Integrates four main components:
    - VideoHandler: Handles video I/O and metadata extraction
    - PipelineOrchestrator: Processes video frames through detection pipelines
    - DataCollector: Aggregates and validates tracking data
    - CoachingAnalyzer: Generates coaching insights and analysis
    """

    def __init__(self, config: Optional[Union[PipelineConfig, dict]] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: PipelineConfig object or dict with configuration.
                   If None, uses DEFAULT_CONFIG.
        """
        # Handle configuration
        if config is None:
            self.config = DEFAULT_CONFIG
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config

        # Initialize DataCollector with its config
        self.data_collector = DataCollector(
            enable_smoothing=self.config.data_collector.enable_smoothing,
            smoothing_window=self.config.data_collector.smoothing_window,
            enable_validation=self.config.data_collector.enable_validation,
            min_confidence=self.config.data_collector.min_confidence,
            max_position_change=self.config.data_collector.max_position_change,
            handle_missing_data=self.config.data_collector.handle_missing_data,
            max_interpolation_frames=self.config.data_collector.max_interpolation_frames,
        )

        # Initialize PipelineOrchestrator
        self.orchestrator = PipelineOrchestrator()

        # VideoHandler and Analyzer will be initialized in run()
        self.video_handler = None
        self.analyzer = None

    def run(self):
        """
        Run the complete pipeline: process video, analyze, and export results.
        """
        # Use config defaults if not specified
        video_path = self.config.video_path

        # Initialize VideoHandler
        print(f"Opening video: {video_path}")
        self.video_handler = VideoHandler(input_path=video_path)

        metadata = self.video_handler.get_metadata()
        print(f"Video metadata: {metadata}")

        # Initialize analyzer with actual video fps
        self.analyzer = CoachingAnalyzer(fps=metadata.fps)

        # Read and process video frames
        raw_frame_detections = self.orchestrator.process_frames(
            frames_iterator=self.video_handler.read_video()
        )

        # Post-process collected data
        print("\nApplying post-processing to collected data...")
        frames = self.data_collector.post_process(
            raw_frame_detections=raw_frame_detections
        )
        print("Post-processing complete!")

        # Step 5: Perform analysis
        print("\nPerforming coaching analysis...")
        analysis = self.analyzer.analyze_match(frames)

        print("\nPipeline completed successfully!")

        print("Coaching Analysis Results:")
        print(analysis)

        return analysis
