"""
Integrated squash video analysis pipeline.

This pipeline combines video processing, data collection, and coaching analysis
into a single, easy-to-use interface.
"""

from typing import Optional, Union
from orchestration import PipelineOrchestrator, Visualizer
from video_io import VideoReader, VideoWriter
from data import DataCollector
from analysis import CoachingAnalyzer
from config import PipelineConfig, DEFAULT_CONFIG


class Pipeline:
    """
    Complete pipeline for squash video analysis.

    Integrates four main components:
    - VideoReader: Handles video I/O and metadata extraction
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
        self.orchestrator = PipelineOrchestrator(
            data_collector=self.data_collector,
            visualizer=self.visualizer,
        )

        # Analyzer will be initialized after reading video metadata (to get fps)
        self.analyzer = None

    def run(self):
        """
        Run the complete pipeline: process video, analyze, and export results.
        """
        # Use config defaults if not specified
        video_path = self.config.video_path
        output_path = self.config.output_path
        display = self.config.display
        analysis_output_path = self.config.analysis_output_path

        # Step 1: Initialize video reader and extract metadata
        print(f"Opening video: {video_path}")
        with VideoReader(video_path) as video_reader:
            metadata = video_reader.get_metadata()
            print(f"Video metadata: {metadata}")

            # Initialize analyzer with actual video fps
            self.analyzer = CoachingAnalyzer(fps=metadata.fps)

            # Step 2: Initialize video writer if output path is specified
            video_writer = None
            if output_path:
                video_writer = VideoWriter(output_path, metadata)

            # Step 3: Process video frames
            metadata_dict = {
                "fps": metadata.fps,
                "width": metadata.width,
                "height": metadata.height,
                "total_frames": metadata.total_frames,
            }

            try:
                # Define callback to write frames
                def write_frame(frame_data, annotated_frame):
                    if video_writer:
                        video_writer.write(annotated_frame)

                self.orchestrator.process_frames(
                    frames_iterator=video_reader.frames(),
                    video_metadata=metadata_dict,
                    display=display,
                    on_frame_processed=write_frame,
                )
            finally:
                # Release video writer
                if video_writer:
                    video_writer.release()

        # Step 4: Get collected data
        frames = self.orchestrator.get_collected_data()
        print(f"\nCollected {len(frames)} frames of data")

        # Step 5: Perform analysis
        print("\nPerforming coaching analysis...")
        analysis = self.analyzer.analyze_match(frames)

        print(f"  Total Rallies: {analysis['rally_statistics']['total_rallies']}")
        print(
            f"  Average Rally Duration: {analysis['rally_statistics']['average_duration']:.2f}s"
        )

        # Step 6: Export analysis
        print(f"\nExporting analysis to {analysis_output_path}.*")
        self.analyzer.export_analysis(
            frames, f"{analysis_output_path}.json", format="json"
        )
        self.analyzer.export_analysis(
            frames, f"{analysis_output_path}.txt", format="summary"
        )

        print("\nPipeline completed successfully!")

        return analysis
