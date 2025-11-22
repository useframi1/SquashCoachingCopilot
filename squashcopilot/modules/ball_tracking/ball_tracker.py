"""
Ball tracking module for the squash coaching copilot.

Uses TrackNet model for ball detection.
Updates TrackingOutput with ball detection data.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from squashcopilot.common.utils import load_config
from squashcopilot.common.types.geometry import Point2D
from squashcopilot.common.types.base import Frame
from squashcopilot.common.models import (
    BallTrackingInput,
    BallTrackingOutput,
    BallPostprocessingInput,
    BallPostprocessingOutput,
)
from .model.tracknet_tracker import TrackNetTracker


class BallTracker:
    """Ball tracking using TrackNet model.

    This class provides ball detection using the TrackNet deep learning model,
    along with preprocessing capabilities for black ball detection.
    """

    def __init__(self, config: dict = None):
        """Initialize the ball tracker.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        self.is_black_ball = False

        # Load configuration
        if config is None:
            config = load_config(config_name="ball_tracking")

        self.config = config

        # Initialize TrackNet tracker
        self.tracker = TrackNetTracker(config=config)

        # Expose the device attribute for compatibility
        self.device = getattr(self.tracker, "device", "N/A")

    def set_is_black_ball(self, is_black: bool):
        """Set whether the ball is black for preprocessing.

        Args:
            is_black: True if the ball is black, False otherwise.
        """
        self.is_black_ball = is_black

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the input frame for black ball detection.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Preprocessed frame (BGR format).
        """
        frame = cv2.bitwise_not(frame)

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        # Apply dilation
        l_dilated = cv2.dilate(
            l_channel,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        # Merge and convert back to BGR
        enhanced_lab = cv2.merge([l_dilated, a_channel, b_channel])
        enhanced_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_color

    def reset(self):
        """Reset the tracker state."""
        self.tracker.reset()

    def process_frame(self, input_data: BallTrackingInput) -> BallTrackingOutput:
        """Process a single frame and return ball tracking output.

        Args:
            input_data: BallTrackingInput with frame

        Returns:
            BallTrackingOutput with ball detection data
        """
        frame = input_data.frame
        image = frame.image
        frame_number = frame.frame_number
        timestamp = (
            frame.timestamp if hasattr(frame, "timestamp") else frame_number / 30.0
        )

        # Preprocess if black ball
        if self.is_black_ball:
            image = self.preprocess_frame(image)

        # Get detection from tracker
        x, y = self.tracker.process_frame(image)

        # Create output
        if x is not None and y is not None:
            return BallTrackingOutput(
                frame_number=frame_number,
                timestamp=timestamp,
                ball_detected=True,
                ball_x=float(x),
                ball_y=float(y),
                ball_confidence=1.0,  # TrackNet doesn't provide confidence
            )
        else:
            return BallTrackingOutput(
                frame_number=frame_number,
                timestamp=timestamp,
                ball_detected=False,
                ball_x=None,
                ball_y=None,
                ball_confidence=0.0,
            )

    def detect(self, frame: np.ndarray) -> Optional[Point2D]:
        """Simple detection method returning ball position.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Point2D with ball position or None if not detected
        """
        if self.is_black_ball:
            frame = self.preprocess_frame(frame)

        x, y = self.tracker.process_frame(frame)

        if x is not None and y is not None:
            return Point2D(x=float(x), y=float(y))
        return None

    def process_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        timestamps: List[float],
        batch_size: int = 32,
        carryover_frames: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[BallTrackingOutput], List[np.ndarray]]:
        """
        Process a batch of frames and return ball tracking outputs.

        This method processes multiple frames efficiently by batching them
        through the TrackNet model on GPU.

        Args:
            frames: List of input frames (BGR format).
            frame_numbers: List of frame numbers corresponding to each frame.
            timestamps: List of timestamps corresponding to each frame.
            batch_size: Number of sliding windows to process in parallel on GPU.
            carryover_frames: Optional frames from previous batch for continuity.
                             Should be the last 2 frames from the previous batch.

        Returns:
            Tuple of:
                - List of BallTrackingOutput for each input frame.
                - List of last 2 frames for carryover to next batch.
        """
        if len(frames) == 0:
            return [], []

        # Preprocess frames if black ball
        if self.is_black_ball:
            processed_frames = [self.preprocess_frame(frame) for frame in frames]
        else:
            processed_frames = frames

        # Prepend carryover frames for cross-batch continuity
        if carryover_frames and len(carryover_frames) > 0:
            # Carryover frames are already preprocessed from previous batch
            full_frames = carryover_frames + processed_frames
            # Offset for results: skip carryover frame results
            carryover_offset = len(carryover_frames)
        else:
            full_frames = processed_frames
            carryover_offset = 0

        # Get ball coordinates from tracker
        all_coords = self.tracker.process_batch(full_frames, batch_size=batch_size)

        # Extract results for current batch (skip carryover results)
        current_coords = all_coords[carryover_offset:]

        # Build outputs
        outputs = []
        for i, (x, y) in enumerate(current_coords):
            frame_number = frame_numbers[i]
            timestamp = timestamps[i]

            if x is not None and y is not None:
                output = BallTrackingOutput(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    ball_detected=True,
                    ball_x=float(x),
                    ball_y=float(y),
                    ball_confidence=1.0,
                )
            else:
                output = BallTrackingOutput(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    ball_detected=False,
                    ball_x=None,
                    ball_y=None,
                    ball_confidence=0.0,
                )
            outputs.append(output)

        # Prepare carryover: last 2 processed frames for next batch
        if len(processed_frames) >= 2:
            next_carryover = processed_frames[-2:]
        elif len(processed_frames) == 1:
            # If only 1 frame, carry it plus last from previous carryover if available
            if carryover_frames and len(carryover_frames) >= 1:
                next_carryover = [carryover_frames[-1], processed_frames[0]]
            else:
                next_carryover = processed_frames
        else:
            next_carryover = []

        return outputs, next_carryover

    def postprocess(
        self,
        input_data: BallPostprocessingInput,
    ) -> BallPostprocessingOutput:
        """Apply postprocessing pipeline to ball positions.

        Steps:
        1. Remove outliers using rolling window distance check
        2. Fill missing values using linear interpolation

        Args:
            input_data: BallPostprocessingInput with df

        Returns:
            BallPostprocessingOutput with processed df and metadata
        """
        df = input_data.df.copy()

        # Get postprocessing config
        postprocess_config = self.config.get("postprocessing", {})
        window = postprocess_config.get("window", 10)
        threshold = postprocess_config.get("threshold", 100)

        # Convert ball_x, ball_y to positions list
        positions: List[Tuple[Optional[float], Optional[float]]] = []
        for _, row in df.iterrows():
            x = row.get("ball_x")
            y = row.get("ball_y")
            if pd.notna(x) and pd.notna(y):
                positions.append((float(x), float(y)))
            else:
                positions.append((None, None))

        # Step 1: Remove outliers (track how many)
        cleaned = self._remove_outliers(positions, window=window, threshold=threshold)
        outliers_removed = sum(
            1
            for orig, clean in zip(positions, cleaned)
            if orig[0] is not None and clean[0] is None
        )

        # Step 2: Fill missing values with interpolation (track how many)
        gaps_filled = sum(1 for p in cleaned if p[0] is None)
        imputed = self._impute_missing(cleaned)

        # Update DataFrame with processed positions
        df["ball_x"] = [float(x) for x, y in imputed]
        df["ball_y"] = [float(y) for x, y in imputed]

        return BallPostprocessingOutput(
            df=df,
            num_ball_outliers=outliers_removed,
            num_ball_gaps_filled=gaps_filled,
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

        return [(float(x), float(y)) for x, y in zip(x_coords, y_coords)]
