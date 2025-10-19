"""Pipeline orchestrator for coordinating sub-pipelines and video processing."""

import numpy as np
from typing import Optional
from court_detection_pipeline import CourtCalibrator
from player_tracking_pipeline import PlayerTracker
from rally_state_pipeline import RallyStateDetector
from ball_detection_pipeline import BallTracker
from stroke_detection_pipeline import StrokeDetector
from data import DataCollector


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

    def __init__(self, data_collector: Optional[DataCollector] = None):
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
        self.stroke_detector = StrokeDetector()

        # Data management and visualization
        self.data_collector = data_collector or DataCollector()

        # State
        self.is_calibrated = False
        self.homographies = None
        self.keypoints = None

    def process_frames(self, frames_iterator):
        """
        Process frames from an iterator through all pipelines.
        Only processes and visualizes frames without writing.

        Args:
            frames_iterator: Iterator yielding (frame_number, timestamp, frame) tuples
        """
        for frame_number, timestamp, frame in frames_iterator:
            # Process frame through all pipelines and collect data
            raw_frame_data = self.process_frame(frame, frame_number, timestamp)

            # Pass data to DataCollector for aggregation and cleaning
            self.data_collector.collect_frame_data(**raw_frame_data)

            if frame_number % 10 == 0:
                print(f"Processed frame {frame_number}")

    def process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> dict:
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

        # 5. Stroke Detection
        player_keypoints = self._get_player_keypoints(player_results)
        stroke_data = self.stroke_detector.process_frame(player_keypoints)

        # 5. Collect and process data through DataCollector
        court_data = {
            "homographies": self.homographies,
            "keypoints": self.keypoints,
            "is_calibrated": self.is_calibrated,
        }

        raw_frame_data = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "court_data": court_data,
            "player_results": player_results,
            "ball_position": (ball_x, ball_y),
            "rally_state": rally_state,
            "stroke_data": stroke_data,
        }

        return raw_frame_data

    def _get_real_coordinates(self, player_results: dict) -> dict:
        """Extract real-world coordinates from player results."""
        return {
            1: player_results[1]["real_position"],
            2: player_results[2]["real_position"],
        }

    def _get_player_keypoints(self, player_results: dict) -> dict:
        """Extract player keypoints from player results."""
        return {
            1: player_results[1]["keypoints"]["xy"],
            2: player_results[2]["keypoints"]["xy"],
        }

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
