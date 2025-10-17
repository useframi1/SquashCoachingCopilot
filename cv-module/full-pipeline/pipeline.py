"""
Integrated squash video analysis pipeline.

This pipeline combines video processing, data collection, and coaching analysis
into a single, easy-to-use interface.
"""

from typing import Optional, Union
from orchestration import PipelineOrchestrator, Visualizer
from video_io import VideoHandler
from data import DataCollector
from analysis import CoachingAnalyzer
from data import DataPlotter
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
        self.orchestrator = PipelineOrchestrator(
            data_collector=self.data_collector,
            visualizer=self.visualizer,
        )

        # VideoHandler and Analyzer will be initialized in run()
        self.video_handler = None
        self.analyzer = None

    def run(self):
        """
        Run the complete pipeline: process video, analyze, and export results.
        """
        # Use config defaults if not specified
        video_path = self.config.video_path
        output_path = self.config.output_path
        output_codec = self.config.output_codec
        display = self.config.display
        analysis_output_path = self.config.analysis_output_path

        # Step 1: Initialize VideoHandler
        print(f"Opening video: {video_path}")
        self.video_handler = VideoHandler(
            input_path=video_path, output_path=output_path, codec=output_codec
        )

        metadata = self.video_handler.get_metadata()
        print(f"Video metadata: {metadata}")

        # Initialize analyzer with actual video fps
        self.analyzer = CoachingAnalyzer(fps=metadata.fps)

        # Step 2: Read and process video frames
        metadata_dict = {
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "total_frames": metadata.total_frames,
        }

        self.orchestrator.process_frames(
            frames_iterator=self.video_handler.read_video(),
            video_metadata=metadata_dict,
            display=display,
        )

        # Step 3: Get collected and post-processed data
        frames = self.data_collector.get_frame_history()
        print(f"\nCollected {len(frames)} frames of data")

        annotated_frames = self.visualizer.render_frames(
            frames=self.video_handler.read_video(),
            frame_data_list=frames,
        )
        self.video_handler.write_video(
            annotated_frames, output_path="output/annotated_video.mp4"
        )

        # Post-process collected data
        print("\nApplying post-processing to collected data...")
        frames = self.data_collector.post_process()
        print("Post-processing complete!")

        # Step 4: Render and write output video if output path is specified
        if output_path:
            annotated_frames = self.visualizer.render_frames(
                frames=self.video_handler.read_video(
                    start_frame=frames[0].frame_number,
                    end_frame=frames[-1].frame_number,
                ),
                frame_data_list=frames,
            )
            self.video_handler.write_video(annotated_frames)

        # Generate diagnostic plots
        plotter = DataPlotter(output_dir="output/plots")
        plotter.compare_before_after(
            raw_frames=self.data_collector.raw_frame_history,
            processed_frames=self.data_collector.processed_frame_history,
        )

        # Step 5: Perform analysis
        print("\nPerforming coaching analysis...")
        analysis = self.analyzer.analyze_match(frames)

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
