"""
Integrated squash video analysis pipeline.

This pipeline combines video processing, data collection, and coaching analysis
into a single, easy-to-use interface.
"""

from typing import Optional, Union
from orchestration import PipelineOrchestrator
from visualization import Visualizer
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
            median_window=self.config.data_collector.median_window,
            savgol_window=self.config.data_collector.savgol_window,
            savgol_poly=self.config.data_collector.savgol_poly,
            enable_validation=self.config.data_collector.enable_validation,
            min_confidence=self.config.data_collector.min_confidence,
            max_position_change=self.config.data_collector.max_position_change,
            handle_missing_data=self.config.data_collector.handle_missing_data,
            max_interpolation_frames=self.config.data_collector.max_interpolation_frames,
            prominence=self.config.data_collector.prominence,
            width=self.config.data_collector.width,
            min_distance=self.config.data_collector.min_distance,
            min_states_duration=self.config.data_collector.min_states_duration,
        )

        # Initialize Visualizer with its config
        self.visualizer = Visualizer(
            show_court_keypoints=self.config.visualizer.show_court_keypoints,
            show_player_keypoints=self.config.visualizer.show_player_keypoints,
            show_player_bbox=self.config.visualizer.show_player_bbox,
            show_ball=self.config.visualizer.show_ball,
            show_rally_state=self.config.visualizer.show_rally_state,
            show_stroke_type=self.config.visualizer.show_stroke_type,
            keypoint_confidence_threshold=self.config.visualizer.keypoint_confidence_threshold,
        )

        # Initialize PipelineOrchestrator
        self.orchestrator = PipelineOrchestrator(data_collector=self.data_collector)

        # VideoHandler and Analyzer will be initialized in run()
        self.video_handler = None
        self.analyzer = None

    def run(self):
        """
        Run the complete pipeline: process video, analyze, and export results.
        """
        # Use config defaults if not specified
        video_path = self.config.video_path
        base_output_path = self.config.base_output_path
        codec = self.config.output_codec

        # Step 1: Initialize VideoHandler
        print(f"Opening video: {video_path}")
        self.video_handler = VideoHandler(
            input_path=video_path, base_output_path=base_output_path, codec=codec
        )

        metadata = self.video_handler.get_metadata()
        print(f"Video metadata: {metadata}")

        # Initialize analyzer with actual video fps
        self.analyzer = CoachingAnalyzer(fps=metadata.fps)

        self.orchestrator.process_frames(
            frames_iterator=self.video_handler.read_video()
        )

        # Post-process collected data
        print("\nApplying post-processing to collected data...")
        rallies = self.data_collector.post_process()
        print("Post-processing complete!")

        for i in range(len(rallies)):
            annotated_frames = self.visualizer.render_frames(
                frames=self.video_handler.read_video(
                    start_frame=rallies[i].start_frame,
                    end_frame=rallies[i].end_frame,
                ),
                frame_data_list=rallies[i].rally_frames,
            )
            video_name = video_path.split("/")[-1]
            video_name = video_name.split(".")[0]
            output_path = f"{video_name}/rally_{i+1}"
            self.video_handler.write_video(
                frames=annotated_frames, output_path=output_path
            )

        # Step 5: Perform analysis
        print("\nPerforming coaching analysis...")
        analysis = self.analyzer.analyze_match(rallies)

        print("\nPipeline completed successfully!")

        return analysis
