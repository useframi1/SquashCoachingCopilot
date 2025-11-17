"""
Rally State Detector Module

Detects rally segments from ball trajectory using FFT-based analysis.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import List

from squashcopilot.common.utils import load_config
from squashcopilot.common.models.rally import (
    RallySegmentationInput,
    RallySegmentationResult,
    RallySegment,
)


class RallyStateDetector:
    """
    Detects rally segments from ball trajectory using frequency analysis.

    The detector uses FFT to identify consistent oscillation patterns in the ball
    trajectory that indicate active rally periods, as opposed to random/erratic
    movement during inactive periods.
    """

    def __init__(self, config: dict = None):
        """
        Initialize rally state detector.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        self.config = (
            config if config else load_config(config_name="rally_state_detection")
        )

        # Load preprocessing parameters
        preprocess_config = self.config["preprocessing"]
        self.savgol_window = preprocess_config["savgol_window_length"]
        self.savgol_poly = preprocess_config["savgol_polyorder"]

        # Load segmentation parameters
        seg_config = self.config["segmentation"]
        self.fft_window_size = seg_config["fft_window_size"]
        self.frequency_threshold = seg_config["frequency_threshold"]
        self.min_rally_duration = seg_config["min_rally_duration"]
        self.merge_gap_threshold = seg_config["merge_gap_threshold"]

    def preprocess(self, ball_positions: List[float]) -> np.ndarray:
        """
        Preprocess ball trajectory using Savitzky-Golay filter.

        Args:
            ball_positions: List of ball y-coordinates

        Returns:
            Smoothed trajectory as numpy array
        """
        # Convert to numpy array
        trajectory = np.array(ball_positions)

        # Check if we have enough data points
        if len(trajectory) < self.savgol_window:
            print(
                f"Warning: Not enough data points ({len(trajectory)}) for "
                f"Savgol filter (window={self.savgol_window}). Using original values."
            )
            return trajectory

        # Apply Savitzky-Golay filter
        smoothed = savgol_filter(
            trajectory, window_length=self.savgol_window, polyorder=self.savgol_poly
        )

        return smoothed

    def _compute_local_frequency_content(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute local frequency content using sliding window FFT.

        During active rallies, the ball trajectory has consistent oscillations
        (regular peaks/valleys). During inactive periods, the trajectory is
        erratic with high-frequency noise.

        Args:
            trajectory: Smoothed ball trajectory

        Returns:
            Array of frequency content scores (higher = more consistent oscillation)
        """
        n_frames = len(trajectory)
        frequency_scores = np.zeros(n_frames)
        half_window = self.fft_window_size // 2

        for i in range(n_frames):
            # Define window boundaries
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window)
            window = trajectory[start:end]

            if len(window) < self.fft_window_size // 2:
                # Skip if window too small
                frequency_scores[i] = 0.0
                continue

            # Compute FFT
            fft = np.fft.fft(window)
            freqs = np.fft.fftfreq(len(window))

            # Get power spectrum (magnitude)
            power = np.abs(fft) ** 2

            # Focus on low-to-mid frequencies (consistent oscillations)
            # High frequencies indicate noise/erratic movement
            low_freq_mask = (np.abs(freqs) > 0.05) & (np.abs(freqs) < 0.3)
            high_freq_mask = np.abs(freqs) > 0.3

            low_freq_power = np.sum(power[low_freq_mask])
            high_freq_power = np.sum(power[high_freq_mask])

            # Score: ratio of low-freq to high-freq power
            # High score = consistent oscillation (active rally)
            # Low score = erratic movement (inactive)
            if high_freq_power > 0:
                frequency_scores[i] = low_freq_power / (high_freq_power + 1e-10)
            else:
                frequency_scores[i] = low_freq_power

        return frequency_scores

    def segment_rallies(
        self, input_data: RallySegmentationInput
    ) -> RallySegmentationResult:
        """
        Segment rallies from ball trajectory using FFT-based analysis.

        Args:
            input_data: RallySegmentationInput with ball positions and frame numbers

        Returns:
            RallySegmentationResult with detected rally segments
        """
        # Step 1: Preprocess trajectory
        smoothed_trajectory = self.preprocess(input_data.ball_positions)

        # Step 2: Compute frequency content
        frequency_scores = self._compute_local_frequency_content(smoothed_trajectory)

        # Step 3: Threshold to identify rally periods
        # High frequency score = active rally (consistent oscillation)
        rally_mask = frequency_scores > self.frequency_threshold

        # Step 4: Extract segments from binary mask
        segments = []
        in_rally = False
        rally_start = None
        rally_id = 0

        for i, frame_num in enumerate(input_data.frame_numbers):
            if rally_mask[i] and not in_rally:
                # Start of new rally
                rally_start = frame_num
                in_rally = True

            elif not rally_mask[i] and in_rally:
                # End of rally
                rally_end = input_data.frame_numbers[i - 1]
                duration = rally_end - rally_start + 1

                # Only keep rallies that meet minimum duration
                if duration >= self.min_rally_duration:
                    segments.append(
                        RallySegment(
                            rally_id=rally_id,
                            start_frame=rally_start,
                            end_frame=rally_end,
                        )
                    )
                    rally_id += 1

                in_rally = False
                rally_start = None

        # Handle case where trajectory ends during a rally
        if in_rally and rally_start is not None:
            rally_end = input_data.frame_numbers[-1]
            duration = rally_end - rally_start + 1
            if duration >= self.min_rally_duration:
                segments.append(
                    RallySegment(
                        rally_id=rally_id, start_frame=rally_start, end_frame=rally_end
                    )
                )

        # Step 5: Merge nearby segments
        segments = self._merge_nearby_segments(segments)

        # Re-assign rally IDs after merging
        for i, segment in enumerate(segments):
            segment.rally_id = i

        return RallySegmentationResult(
            segments=segments,
            total_frames=len(input_data.frame_numbers),
            preprocessed_trajectory=smoothed_trajectory.tolist(),
        )

    def _merge_nearby_segments(self, segments: List[RallySegment]) -> List[RallySegment]:
        """
        Merge rally segments that are close together.

        Args:
            segments: List of rally segments

        Returns:
            Merged list of segments
        """
        if len(segments) <= 1:
            return segments

        merged = []
        current = segments[0]

        for next_segment in segments[1:]:
            gap = next_segment.start_frame - current.end_frame

            if gap <= self.merge_gap_threshold:
                # Merge segments
                current = RallySegment(
                    rally_id=current.rally_id,
                    start_frame=current.start_frame,
                    end_frame=next_segment.end_frame,
                )
            else:
                # Keep current and move to next
                merged.append(current)
                current = next_segment

        # Add the last segment
        merged.append(current)

        return merged
