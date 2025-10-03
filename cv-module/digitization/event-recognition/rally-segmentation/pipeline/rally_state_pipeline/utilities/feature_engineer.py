"""
Feature Engineer - Converts base metrics to ML features
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """Handles feature engineering for rally state prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.lookback_frames = self.config["feature_engineering"]["lookback_frames"]
        self.court_center_x = self.config["feature_engineering"]["court_center_x"]
        self.service_line_y = self.config["feature_engineering"]["service_line_y"]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from base metrics.

        Args:
            df: DataFrame with base metrics (mean_distance, median_player1_x, etc.)
            group_col: Column to group by (e.g., 'video_name') for temporal features

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
