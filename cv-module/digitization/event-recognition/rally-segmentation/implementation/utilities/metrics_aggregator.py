"""
Metrics Aggregator Module
Handles player tracking, court calibration, and metric calculation with windowing.
Shared by annotation pipeline and visualizer.
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

from .court_calibrator import CourtCalibrator
from .player_tracker import PlayerTracker


class MetricsAggregator:
    """
    Aggregates player tracking metrics over a sliding window.
    Handles court calibration, player tracking, and distance calculation.
    """

    def __init__(self, window_size: int = 50):
        """
        Initialize the metrics aggregator.

        Args:
            window_size: Number of frames to aggregate for statistics
            calibrator: Optional pre-initialized CourtCalibrator
            tracker: Optional pre-initialized PlayerTracker
        """
        self.window_size = window_size

        # Court calibration and player tracking
        self.calibrator = CourtCalibrator()
        self.tracker = None
        self.homography = None
        self.court_calibrated = False

        # Metrics storage
        self.metrics_history = deque(maxlen=window_size * 2)

        self.last_player_bboxes = {}

    def calibrate_court(self, frame: np.ndarray) -> bool:
        """
        Calibrate court using the provided frame.

        Args:
            frame: Video frame to use for calibration

        Returns:
            True if calibration successful, False otherwise
        """
        if self.court_calibrated:
            return True

        try:
            print("Attempting court calibration...")
            self.homography = self.calibrator.compute_homography(frame)
            self.court_calibrated = True
            print("Court calibration successful!")
            return True
        except Exception as e:
            print(f"Court calibration failed: {e}")
            print("Using identity matrix as fallback")
            self.homography = np.eye(3, dtype=np.float32)
            self.court_calibrated = True
            return False

    def initialize_tracker(self, frame: np.ndarray) -> None:
        """
        Initialize player tracker with calibrated homography.

        Args:
            frame: Video frame to use for initial calibration
        """
        if self.tracker is None:
            # Ensure court is calibrated first
            self.calibrate_court(frame)

            # Initialize tracker with homography
            self.tracker = PlayerTracker(homography=self.homography)
            print("Player tracker initialized with homography")

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            pos1: (x, y) position of player 1
            pos2: (x, y) position of player 2

        Returns:
            Distance in meters
        """
        if pos1 is None or pos2 is None:
            return 0.0

        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx**2 + dy**2)

    def calculate_frame_metrics(
        self, frame: np.ndarray, frame_num: int
    ) -> Dict[str, any]:
        """
        Calculate distance and position metrics for current frame.
        """
        if self.tracker is None:
            self.initialize_tracker(frame)

        # Process frame to get player positions
        player_coords, player_real_coords = self.tracker.process_frame(frame)

        # Store pixel bounding boxes for drawing (make a copy to avoid reference issues)
        self.last_player_bboxes = dict(player_coords)  # Store as regular dict

        metrics = {
            "frame_number": frame_num,
            "player_distance": None,
            "player1_x": None,
            "player1_y": None,
            "player2_x": None,
            "player2_y": None,
        }

        # Store current player positions
        player_ids = list(player_real_coords.keys())[:2]
        for i, player_id in enumerate(player_ids, 1):
            if len(player_real_coords[player_id]) > 0:
                pos = player_real_coords[player_id][-1]
                metrics[f"player{i}_x"] = pos[0]
                metrics[f"player{i}_y"] = pos[1]

        # Calculate distance between players if both detected
        if len(player_real_coords) >= 2:
            if (
                len(player_real_coords[player_ids[0]]) > 0
                and len(player_real_coords[player_ids[1]]) > 0
            ):
                pos1 = player_real_coords[player_ids[0]][-1]
                pos2 = player_real_coords[player_ids[1]][-1]

                distance = self.calculate_distance(pos1, pos2)
                metrics["player_distance"] = distance

        return metrics

    def update_metrics(self, frame: np.ndarray, frame_num: int) -> Dict[str, any]:
        """
        Calculate metrics for current frame and add to history.

        Args:
            frame: Current video frame
            frame_num: Current frame number

        Returns:
            Frame metrics dictionary
        """
        metrics = self.calculate_frame_metrics(frame, frame_num)
        self.metrics_history.append(metrics)
        return metrics

    def get_aggregated_metrics(
        self, additional_data: Optional[Dict] = None
    ) -> Optional[Dict[str, any]]:
        """
        Calculate aggregated statistics over the current window.

        Args:
            additional_data: Optional additional fields to include (e.g., video_name, state)

        Returns:
            Dictionary with aggregated statistics, or None if insufficient data
        """
        if len(self.metrics_history) < self.window_size:
            return None

        # Get the last window_size frames for analysis
        window_metrics = list(self.metrics_history)[-self.window_size :]

        # Extract metrics for calculation
        distances = [
            m["player_distance"]
            for m in window_metrics
            if m["player_distance"] is not None
        ]

        # Extract player positions
        player1_x = [
            m["player1_x"] for m in window_metrics if m["player1_x"] is not None
        ]
        player1_y = [
            m["player1_y"] for m in window_metrics if m["player1_y"] is not None
        ]
        player2_x = [
            m["player2_x"] for m in window_metrics if m["player2_x"] is not None
        ]
        player2_y = [
            m["player2_y"] for m in window_metrics if m["player2_y"] is not None
        ]

        # Build statistics dictionary
        stats = {
            "frame_number": window_metrics[-1]["frame_number"],
            "window_size": len(window_metrics),
            "mean_distance": np.mean(distances) if distances else None,
            "median_player1_x": np.median(player1_x) if player1_x else None,
            "median_player1_y": np.median(player1_y) if player1_y else None,
            "median_player2_x": np.median(player2_x) if player2_x else None,
            "median_player2_y": np.median(player2_y) if player2_y else None,
        }

        # Add any additional data
        if additional_data:
            stats.update(additional_data)

        return stats

    def get_mean_distance(self) -> float:
        """
        Get mean distance over the current window.

        Returns:
            Mean distance in meters, or 0.0 if insufficient data
        """
        if len(self.metrics_history) < self.window_size:
            return 0.0

        window_metrics = list(self.metrics_history)[-self.window_size :]
        distances = [
            m["player_distance"]
            for m in window_metrics
            if m["player_distance"] is not None
        ]

        return np.mean(distances) if distances else 0.0

    def get_player_positions(self) -> Dict[int, Dict[str, any]]:
        """
        Get current player positions from tracker.

        Returns:
            Dictionary with player positions (pixel and real-world coordinates)
        """
        if self.tracker is None:
            return {}

        return {
            "pixel": self.tracker.player_positions,
            "real": self.tracker.player_real_positions,
        }

    def has_full_window(self) -> bool:
        """
        Check if we have enough frames for a full window.

        Returns:
            True if window is full
        """
        return len(self.metrics_history) >= self.window_size

    def reset(self) -> None:
        """Reset all metrics history."""
        self.metrics_history.clear()
