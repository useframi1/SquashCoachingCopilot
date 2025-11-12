"""
Metrics Aggregator Module
Handles player tracking, court calibration, metric calculation with windowing, and feature engineering.
Shared by annotation pipeline and visualizer.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List


class MetricsAggregator:
    """
    Aggregates player tracking metrics over a sliding window and engineers features for ML models.
    Handles court calibration, player tracking, distance calculation, and feature engineering.
    """

    def __init__(self, window_size: int, config: Optional[Dict] = None):
        """
        Initialize the metrics aggregator.

        Args:
            window_size: Number of frames to aggregate for statistics
            config: Configuration dictionary with feature engineering parameters
        """
        self.window_size = window_size
        self.config = config or {}

        # Feature engineering parameters
        fe_config = self.config.get("feature_engineering", {})
        self.lookback_frames = fe_config.get("lookback_frames", 3)
        self.court_center_x = fe_config.get("court_center_x", 3.2)
        self.service_line_y = fe_config.get("service_line_y", 5.44)

        # Metrics storage
        self.metrics_history = []
        self.current_stats = None

        # Store aggregated metrics for feature engineering
        self.aggregated_history = []

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

    def calculate_frame_metrics(self, player_real_coords) -> Dict[str, any]:
        """
        Calculate distance and position metrics for current frame.
        """
        metrics = {
            "player_distance": None,
            "player1_x": None,
            "player1_y": None,
            "player2_x": None,
            "player2_y": None,
        }

        # Store current player positions
        player_ids = list(player_real_coords.keys())[:2]
        for i, player_id in enumerate(player_ids, 1):
            pos = player_real_coords[player_id]
            metrics[f"player{i}_x"] = pos[0]
            metrics[f"player{i}_y"] = pos[1]

        # Calculate distance between players if both detected
        if len(player_real_coords) >= 2:
            pos1 = player_real_coords[player_ids[0]]
            pos2 = player_real_coords[player_ids[1]]

            distance = self.calculate_distance(pos1, pos2)
            metrics["player_distance"] = distance

        return metrics

    def update_metrics(self, player_real_coords) -> Dict[str, any]:
        """
        Calculate metrics for current frame and add to history.

        Args:
            player_real_coords: Dictionary of player real-world coordinates

        Returns:
            Frame metrics dictionary
        """
        metrics = self.calculate_frame_metrics(player_real_coords)
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
            return self.current_stats

        # Get the last window_size frames for analysis
        window_metrics = self.metrics_history

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
            "mean_distance": np.mean(distances) if distances else None,
            "median_player1_x": np.median(player1_x) if player1_x else None,
            "median_player1_y": np.median(player1_y) if player1_y else None,
            "median_player2_x": np.median(player2_x) if player2_x else None,
            "median_player2_y": np.median(player2_y) if player2_y else None,
        }

        # Add any additional data
        if additional_data:
            stats.update(additional_data)

        # Store in aggregated history for feature engineering
        self.aggregated_history.append(stats)

        # Clear history after aggregation
        self.metrics_history.clear()

        self.current_stats = stats
        return stats

    def has_full_window(self) -> bool:
        """
        Check if we have enough frames for a full window.

        Returns:
            True if window is full
        """
        return len(self.metrics_history) == self.window_size

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from base metrics.

        Args:
            df: DataFrame with base metrics (mean_distance, median_player1_x, etc.)

        Returns:
            DataFrame with engineered features
        """
        # Initialize result DataFrame
        result_df = df.copy()
        n_rows = len(df)

        # Distance lag features
        for lag in range(1, self.lookback_frames + 1):
            col_name = f"distance_lag_{lag}"
            result_df[col_name] = (
                df["mean_distance"].shift(lag).fillna(df["mean_distance"].iloc[0])
            )

        # Distance change and acceleration
        result_df["distance_change"] = df["mean_distance"].diff().fillna(0)
        result_df["distance_acceleration"] = (
            result_df["distance_change"].diff().fillna(0)
        )

        # Rolling statistics (using window of lookback_frames)
        window = min(self.lookback_frames, n_rows)
        result_df["distance_rolling_mean"] = (
            df["mean_distance"].rolling(window=window, min_periods=1).mean()
        )
        result_df["distance_rolling_std"] = (
            df["mean_distance"].rolling(window=window, min_periods=1).std().fillna(0)
        )
        result_df["distance_rolling_min"] = (
            df["mean_distance"].rolling(window=window, min_periods=1).min()
        )
        result_df["distance_rolling_max"] = (
            df["mean_distance"].rolling(window=window, min_periods=1).max()
        )

        # Player movement features
        for player in ["player1", "player2"]:
            x_col = f"median_{player}_x"
            y_col = f"median_{player}_y"

            # Calculate movement as distance between consecutive positions
            x_diff = df[x_col].diff().fillna(0)
            y_diff = df[y_col].diff().fillna(0)
            result_df[f"{player}_movement"] = np.sqrt(x_diff**2 + y_diff**2)

        # Court position features
        result_df["player1_court_side"] = (
            df["median_player1_x"] > self.court_center_x
        ).astype(int)
        result_df["player2_court_side"] = (
            df["median_player2_x"] > self.court_center_x
        ).astype(int)
        result_df["players_same_side"] = (
            result_df["player1_court_side"] == result_df["player2_court_side"]
        ).astype(int)

        # Distance from service line
        result_df["player1_from_service_line"] = (
            df["median_player1_y"] - self.service_line_y
        )
        result_df["player2_from_service_line"] = (
            df["median_player2_y"] - self.service_line_y
        )

        return result_df

    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        features = [
            "mean_distance",
            "distance_lag_1",
            "distance_lag_2",
            "distance_lag_3",
            "distance_change",
            "distance_acceleration",
            "distance_rolling_mean",
            "distance_rolling_std",
            "distance_rolling_min",
            "distance_rolling_max",
            "median_player1_x",
            "median_player1_y",
            "median_player2_x",
            "median_player2_y",
            "player1_movement",
            "player2_movement",
            "player1_court_side",
            "player2_court_side",
            "players_same_side",
            "player1_from_service_line",
            "player2_from_service_line",
        ]
        return features

    def process_and_engineer(
        self, metrics_list: List[Dict], aggregated: bool = False
    ) -> pd.DataFrame:
        """
        Process metrics and engineer features in one go.

        Args:
            metrics_list: List of dictionaries containing frame metrics.
                If aggregated=False (default):
                    - player_distance: Distance between players
                    - player1_x, player1_y: Player 1 coordinates
                    - player2_x, player2_y: Player 2 coordinates
                If aggregated=True:
                    - mean_distance: Mean distance (already aggregated)
                    - median_player1_x, median_player1_y: Player 1 median positions
                    - median_player2_x, median_player2_y: Player 2 median positions
            aggregated: Whether the metrics are already aggregated by window size.
                       If False, will aggregate frame-level metrics by window_size.
                       If True, assumes metrics are already aggregated.

        Returns:
            DataFrame with base metrics and engineered features
        """
        # Convert metrics list to DataFrame
        df = pd.DataFrame(metrics_list)

        if not aggregated:
            # Aggregate metrics by window size
            aggregated_metrics = []
            for i in range(0, len(df), self.window_size):
                window = df.iloc[i : i + self.window_size]

                if len(window) < self.window_size:
                    # Skip incomplete windows at the end
                    continue

                # Calculate aggregated statistics for this window
                agg_metrics = {
                    "mean_distance": window["player_distance"].mean(),
                    "median_player1_x": window["player1_x"].median(),
                    "median_player1_y": window["player1_y"].median(),
                    "median_player2_x": window["player2_x"].median(),
                    "median_player2_y": window["player2_y"].median(),
                }

                # Preserve any additional columns from the first row
                for col in window.columns:
                    if col not in ["player_distance", "player1_x", "player1_y",
                                  "player2_x", "player2_y"] and col not in agg_metrics:
                        agg_metrics[col] = window[col].iloc[-1]  # Use last frame's value

                aggregated_metrics.append(agg_metrics)

            # Create DataFrame from aggregated metrics
            df_aggregated = pd.DataFrame(aggregated_metrics)
        else:
            # Metrics are already aggregated, use as-is
            df_aggregated = df

        # Engineer features
        df_features = self.engineer_features(df_aggregated)

        return df_features
