"""
Wall hit detection for squash ball tracking.

Detects front wall hits by finding local minima in the Y-coordinate curve.
Uses DataFrame-based pipeline architecture.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Dict

from squashcopilot.common.utils import load_config
from squashcopilot.common.models import (
    WallHitDetectionInput,
    WallHitDetectionOutput,
    RallySegment,
)
from squashcopilot.common.types.geometry import Point2D


class WallHitDetector:
    """Detects front wall hits using local minima in Y-coordinate.

    Front wall hits appear as valleys (local minima) in the Y-coordinate curve.
    The algorithm finds these minima and validates them based on prominence and width.
    """

    def __init__(self, config: dict = None):
        """Initialize the wall hit detector.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        if config is None:
            config = load_config(config_name="hit_detection")

        wall_config = config.get("wall_hit_detection", {})
        self.prominence = wall_config.get("prominence", 50.0)
        self.width = wall_config.get("width", 3)
        self.min_distance = wall_config.get("min_distance", 20)

    def detect_wall_hits(
        self,
        input_data: WallHitDetectionInput,
    ) -> WallHitDetectionOutput:
        """Detect front wall hits and add columns to DataFrame.

        Args:
            input_data: WallHitDetectionInput with df, segments, and calibration

        Returns:
            WallHitDetectionOutput with df containing added columns:
            is_wall_hit, wall_hit_x_pixel, wall_hit_y_pixel, wall_hit_x_meter, wall_hit_y_meter
        """
        df = input_data.df
        segments = input_data.segments
        calibration = input_data.calibration

        # Initialize new columns
        df = df.copy()
        df["is_wall_hit"] = False
        df["wall_hit_x_pixel"] = np.nan
        df["wall_hit_y_pixel"] = np.nan
        df["wall_hit_x_meter"] = np.nan
        df["wall_hit_y_meter"] = np.nan

        # Process each rally segment
        for segment in segments:
            rally_df = df.loc[segment.start_frame : segment.end_frame]

            # Get ball coordinates for this rally
            x_coords = rally_df["ball_x"].values
            y_coords = rally_df["ball_y"].values
            frame_numbers = rally_df.index.tolist()

            if len(y_coords) < self.width:
                continue

            # Skip if too many NaN values
            valid_mask = ~np.isnan(y_coords)
            if np.sum(valid_mask) < self.width:
                continue

            # Interpolate NaN values for detection
            y_interp = np.interp(
                np.arange(len(y_coords)), np.where(valid_mask)[0], y_coords[valid_mask]
            )

            # Also interpolate X coordinates
            valid_x_mask = ~np.isnan(x_coords)
            if np.sum(valid_x_mask) > 0:
                x_interp = np.interp(
                    np.arange(len(x_coords)),
                    np.where(valid_x_mask)[0],
                    x_coords[valid_x_mask],
                )
            else:
                x_interp = np.zeros_like(y_interp)

            # Find peaks in inverted signal (= minima in original signal)
            inverted_y = -y_interp
            peaks, properties = find_peaks(
                inverted_y,
                prominence=self.prominence,
                width=self.width,
                distance=self.min_distance,
            )

            # Mark wall hits in DataFrame
            for i, peak_idx in enumerate(peaks):
                frame_num = frame_numbers[peak_idx]
                hit_x_pixel = float(x_interp[peak_idx])
                hit_y_pixel = float(y_interp[peak_idx])

                df.loc[frame_num, "is_wall_hit"] = True
                df.loc[frame_num, "wall_hit_x_pixel"] = hit_x_pixel
                df.loc[frame_num, "wall_hit_y_pixel"] = hit_y_pixel

                # Convert to meter coordinates using wall homography
                if calibration is not None and calibration.wall_homography is not None:
                    pixel_point = Point2D(x=hit_x_pixel, y=hit_y_pixel)
                    meter_point = calibration.pixel_to_wall(pixel_point)
                    df.loc[frame_num, "wall_hit_x_meter"] = meter_point.x
                    df.loc[frame_num, "wall_hit_y_meter"] = meter_point.y

        # Compute stats
        stats = self.get_wall_hit_stats(df, segments)

        return WallHitDetectionOutput(
            df=df,
            num_wall_hits=stats["num_wall_hits"],
            wall_hits_per_rally=stats["wall_hits_per_rally"],
        )

    def get_wall_hit_stats(
        self, df: pd.DataFrame, segments: List[RallySegment]
    ) -> Dict:
        """Get wall hit statistics.

        Args:
            df: DataFrame with is_wall_hit column
            segments: List of rally segments

        Returns:
            Dictionary with stats: num_wall_hits, wall_hits_per_rally
        """
        num_wall_hits = int(df["is_wall_hit"].sum())

        wall_hits_per_rally = {}
        for segment in segments:
            rally_df = df.loc[segment.start_frame : segment.end_frame]
            wall_hits_per_rally[segment.rally_id] = int(rally_df["is_wall_hit"].sum())

        return {
            "num_wall_hits": num_wall_hits,
            "wall_hits_per_rally": wall_hits_per_rally,
        }
