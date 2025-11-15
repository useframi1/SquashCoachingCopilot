import cv2
import numpy as np
from typing import List, Tuple, Optional
from squashcopilot.common.utils import load_config
from .model.tracknet_tracker import TrackNetTracker

from squashcopilot.common import (
    Point2D,
    BallTrackingInput,
    BallDetectionResult,
    BallPostprocessingInput,
    BallTrajectory,
)


class BallTracker:
    """Ball tracking using TrackNet model.

    This class provides ball detection and tracking using the TrackNet deep learning model,
    along with preprocessing and postprocessing capabilities.
    """

    def __init__(self, config: dict = None):
        """Initialize the ball tracker.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        self.is_black_ball = False

        # Load configuration
        if config is None:
            config = load_config(config_name='ball_tracking')

        self.config = config

        # Initialize TrackNet tracker
        self.tracker = TrackNetTracker(config=config)

        # Expose the device attribute for compatibility with evaluator
        self.device = getattr(self.tracker, "device", "N/A")

    def set_is_black_ball(self, is_black: bool):
        """Set whether the ball is black for preprocessing.

        Args:
            is_black: True if the ball is black, False otherwise.
        """
        self.is_black_ball = is_black

    def preprocess_frame(self, frame):
        """Preprocess the input frame if needed.

        Args:
            frame: Input frame (BGR format, any resolution)

        Returns:
            Preprocessed frame (BGR format).
        """
        frame = cv2.bitwise_not(frame)

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE only for black ball
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        # Apply dilation in both cases
        l_dilated = cv2.dilate(
            l_channel,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        # Merge and convert back to BGR for color-enhanced frame
        enhanced_lab = cv2.merge([l_dilated, a_channel, b_channel])
        enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Return the color-enhanced frame to the tracker
        return enhanced_color

    def reset(self):
        """Reset the tracker state."""
        self.tracker.reset()

    def process_frame(self, input_data: BallTrackingInput) -> BallDetectionResult:
        """Process a single frame and return ball detection result.

        Args:
            input_data: BallTrackingInput containing frame and metadata

        Returns:
            BallDetectionResult with position and metadata
        """
        frame = input_data.frame.image
        frame_number = input_data.frame.frame_number

        # Preprocess if needed
        if self.is_black_ball:
            frame = self.preprocess_frame(frame)

        # Get detection from tracker
        x, y = self.tracker.process_frame(frame)

        # Create result
        if x is not None and y is not None:
            position = Point2D(x=float(x), y=float(y))
            return BallDetectionResult(
                position=position,
                confidence=1.0,  # TrackNet doesn't provide confidence
                frame_number=frame_number,
                detected=True,
            )
        else:
            return BallDetectionResult.not_detected(frame_number)

    def postprocess(self, input_data: BallPostprocessingInput) -> BallTrajectory:
        """Apply postprocessing pipeline to ball positions.

        Steps:
        1. Remove outliers using rolling window distance check
        2. Fill missing values using linear interpolation

        Args:
            input_data: BallPostprocessingInput with positions and config

        Returns:
            BallTrajectory with smoothed positions and statistics
        """
        # Convert Point2D to tuples for processing
        positions_tuples = [
            (p.x, p.y) if p is not None else (None, None) for p in input_data.positions
        ]

        # Get config
        if input_data.config:
            window = input_data.config.get("postprocessing.window", 10)
            threshold = input_data.config.get("postprocessing.threshold", 100)
        else:
            config = self.config.get("postprocessing", {})
            window = config.get("window", 10)
            threshold = config.get("threshold", 100)

        # Step 1: Remove outliers (track how many)
        cleaned = self._remove_outliers(
            positions_tuples, window=window, threshold=threshold
        )
        outliers_removed = sum(
            1
            for orig, clean in zip(positions_tuples, cleaned)
            if orig[0] is not None and clean[0] is None
        )

        # Step 2: Fill missing values with interpolation (track how many)
        gaps_filled = sum(1 for p in cleaned if p[0] is None)
        imputed = self._impute_missing(cleaned)

        # Convert back to Point2D
        smoothed_positions = [Point2D(x=float(x), y=float(y)) for x, y in imputed]

        return BallTrajectory(
            positions=smoothed_positions,
            original_positions=input_data.positions,
            outliers_removed=outliers_removed,
            gaps_filled=gaps_filled,
        )

    def _remove_outliers(
        self,
        positions: List[Tuple[Optional[float], Optional[float]]],
        window: int = 10,
        threshold: float = 100,
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Remove outlier positions using rolling window distance check.

        Detects positions that are too far from the average position of neighbors.

        Args:
            positions: List of (x, y) tuples
            window: Size of rolling window (default: 10 frames)
            threshold: Maximum distance in pixels from average of neighbors (default: 100 pixels)

        Returns:
            Positions with outliers marked as (None, None)
        """
        n = len(positions)
        result = list(positions)

        for i in range(n):
            if positions[i][0] is None or positions[i][1] is None:
                continue

            current_x, current_y = positions[i]

            # Get neighboring positions in the window
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)

            neighbors = []
            for j in range(start, end):
                if (
                    j != i
                    and positions[j][0] is not None
                    and positions[j][1] is not None
                ):
                    neighbors.append(positions[j])

            if len(neighbors) < 2:
                # Need at least 2 neighbors to calculate average
                continue

            # Calculate average position of neighbors
            avg_x = np.mean([nx for nx, ny in neighbors])
            avg_y = np.mean([ny for nx, ny in neighbors])

            # Calculate distance from current position to average of neighbors
            distance = np.sqrt((current_x - avg_x) ** 2 + (current_y - avg_y) ** 2)

            # If distance from average is >= threshold, mark as outlier
            if distance >= threshold:
                result[i] = (None, None)

        return result

    def _impute_missing(
        self,
        positions: List[Tuple[Optional[float], Optional[float]]],
    ) -> List[Tuple[float, float]]:
        """Fill missing values using linear interpolation.

        Args:
            positions: List of (x, y) tuples with possible None values

        Returns:
            List of (x, y) tuples with all values filled
        """
        n = len(positions)
        x_coords = np.array([p[0] if p[0] is not None else np.nan for p in positions])
        y_coords = np.array([p[1] if p[1] is not None else np.nan for p in positions])

        # Interpolate each axis
        for coords in [x_coords, y_coords]:
            valid_mask = ~np.isnan(coords)
            if np.sum(valid_mask) < 2:
                # Not enough valid points
                coords[:] = np.nanmean(coords) if np.sum(valid_mask) > 0 else 0
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_values = coords[valid_mask]

            # Linear interpolation
            coords[:] = np.interp(
                np.arange(n),
                valid_indices,
                valid_values,
                left=valid_values[0],
                right=valid_values[-1],
            )

        return [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
