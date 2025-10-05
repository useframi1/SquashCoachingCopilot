"""Pipeline orchestrator for coordinating sub-pipelines and video processing."""

import cv2
import numpy as np
from typing import Optional, Callable
from tqdm import tqdm
from court_detection_pipeline import CourtCalibrator
from player_tracking_pipeline import PlayerTracker
from rally_state_pipeline import RallyStateDetector
from ball_detection_pipeline import BallTracker
from stroke_detection_pipeline import StrokeDetector
from data import DataCollector
from data.data_models import FrameData
from .visualizer import Visualizer


class PipelineOrchestrator:
    """
    Orchestrates the execution of all sub-pipelines and manages video processing.

    Responsibilities:
    - Initialize and manage sub-pipeline lifecycles
    - Handle video I/O (reading, writing, display)
    - Coordinate frame-by-frame processing
    - Pass data to DataCollector for aggregation and cleaning
    - Manage progress tracking and error handling
    - Does NOT: Process data, compute metrics, or handle visualization details
    """

    def __init__(
        self,
        data_collector: Optional[DataCollector] = None,
        visualizer: Optional[Visualizer] = None,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            data_collector: DataCollector instance (creates default if None)
            visualizer: Visualizer instance (creates default if None)
        """
        # Sub-pipelines
        self.court_calibrator = CourtCalibrator()
        self.player_tracker = None  # Initialized after court calibration
        self.rally_state_detector = RallyStateDetector()
        self.ball_tracker = BallTracker()

        # Data management and visualization
        self.data_collector = data_collector or DataCollector()
        self.visualizer = visualizer or Visualizer()

        # State
        self.is_calibrated = False
        self.homographies = None
        self.keypoints = None

    def process_frames(
        self,
        frames_iterator,
        video_metadata: dict,
        display: bool = True,
        on_frame_processed: Optional[Callable[[FrameData, np.ndarray], None]] = None,
    ):
        """
        Process frames from an iterator through all pipelines.

        Args:
            frames_iterator: Iterator yielding (frame_number, timestamp, frame) tuples
            video_metadata: Dictionary with video metadata (fps, width, height, total_frames)
            display: Whether to display video in real-time
            on_frame_processed: Optional callback called after each frame is processed.
                              Receives (frame_data, annotated_frame) as arguments.
        """
        # Extract video properties
        total_frames = video_metadata["total_frames"]

        frame_count = 0

        try:
            # Wrap iterator with tqdm for progress tracking
            with tqdm(
                total=total_frames, desc="Processing frames", unit="frame"
            ) as pbar:
                for frame_number, timestamp, frame in frames_iterator:
                    # Process frame through all pipelines and collect data
                    frame_data = self.process_frame(frame, frame_number, timestamp)

                    # Visualize results
                    annotated_frame = self.visualizer.render_frame(frame, frame_data)

                    # Display frame if requested
                    if display:
                        cv2.imshow("Squash Analysis Pipeline", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    # Call callback with both frame_data and annotated_frame
                    if on_frame_processed:
                        on_frame_processed(frame_data, annotated_frame)

                    frame_count += 1
                    pbar.update(1)

        finally:
            # Cleanup
            if display:
                cv2.destroyAllWindows()

        print(f"\nProcessing complete. Processed {frame_count} frames.")

        # Post-process collected data
        print("\nApplying post-processing to collected data...")
        self.data_collector.post_process()
        print("All processing complete!")

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> FrameData:
        """
        Process a single frame through all pipelines.

        Args:
            frame: Input frame
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            FrameData with processed results
        """
        # 1. Court Detection (calibrate on first frame)
        if not self.is_calibrated:
            self.homographies, self.keypoints = self.court_calibrator.process_frame(
                frame
            )
            self.player_tracker = PlayerTracker(
                homography=self.court_calibrator.get_homography("t-boxes")
            )
            self.is_calibrated = True

        # 2. Player Tracking
        player_results = self.player_tracker.process_frame(frame)

        # 3. Ball Detection
        ball_x, ball_y = self.ball_tracker.process_frame(frame)

        # 4. Rally State Detection
        real_coordinates = self._get_real_coordinates(player_results)
        rally_state = self.rally_state_detector.process_frame(real_coordinates)

        # 5. Collect and process data through DataCollector
        court_data = {
            "homographies": self.homographies,
            "keypoints": self.keypoints,
            "is_calibrated": self.is_calibrated,
        }

        frame_data = self.data_collector.collect_frame_data(
            frame_number=frame_number,
            timestamp=timestamp,
            court_data=court_data,
            player_results=player_results,
            ball_position=(ball_x, ball_y),
            rally_state=rally_state,
        )

        return frame_data

    def _get_real_coordinates(self, player_results: dict) -> dict:
        """Extract real-world coordinates from player results."""
        return {
            1: player_results[1]["real_position"],
            2: player_results[2]["real_position"],
        }

    def get_collected_data(self):
        """
        Get all collected data from the DataCollector.

        Returns:
            List of FrameData objects
        """
        return self.data_collector.get_frame_history()

    def reset(self):
        """Reset orchestrator and all sub-pipelines."""
        self.is_calibrated = False
        self.homographies = None
        self.keypoints = None
        self.player_tracker = None
        self.data_collector.reset()

        # Reset sub-pipelines if they have reset methods
        if hasattr(self.rally_state_detector, "reset"):
            self.rally_state_detector.reset()
        if hasattr(self.ball_tracker, "reset"):
            self.ball_tracker.reset()
