"""
Feature Engineer - Converts base metrics to ML features
"""

import pandas as pd
import numpy as np
from typing import Optional
from config import CONFIG


class FeatureEngineer:
    """Handles feature engineering for rally state prediction."""

    def __init__(self):
        self.lookback_frames = CONFIG["lookback_frames"]
        self.court_center_x = CONFIG["court_center_x"]
        self.service_line_y = CONFIG["service_line_y"]

    def engineer_features(
        self, df: pd.DataFrame, group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Engineer features from base metrics.

        Args:
            df: DataFrame with base metrics (mean_distance, median_player1_x, etc.)
            group_col: Column to group by (e.g., 'video_name') for temporal features

        Returns:
            DataFrame with engineered features
        """
        if group_col and group_col in df.columns:
            # Sort by group and frame number
            df_sorted = df.sort_values([group_col, "frame_number"]).copy()
        else:
            # Sort by frame number only
            df_sorted = df.sort_values("frame_number").copy()

        # Initialize result DataFrame
        result_df = df_sorted.copy()

        # Engineer features group by group
        if group_col and group_col in df.columns:
            result_rows = []
            for group_name, group_df in df_sorted.groupby(group_col):
                group_features = self._engineer_group_features(
                    group_df.reset_index(drop=True)
                )
                result_rows.append(group_features)
            result_df = pd.concat(result_rows, ignore_index=True)
        else:
            result_df = self._engineer_group_features(df_sorted.reset_index(drop=True))

        return result_df

    def _engineer_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for a single group (video)."""
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

    def get_feature_names(self) -> list:
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
