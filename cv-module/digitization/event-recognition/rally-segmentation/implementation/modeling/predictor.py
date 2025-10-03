"""
Predictor - Model-agnostic inference pipeline for rally state prediction
"""

import pandas as pd
from typing import Dict
from collections import deque

from modeling.ml.ml_based_model import MLBasedModel
from modeling.rule.rule_based_model import RuleBasedModel
from config import CONFIG


class StatePredictor:
    """Model-agnostic predictor that works with both rule-based and ML-based models."""

    def __init__(self):
        """Initialize predictor with the active model from config."""
        active_model_type = CONFIG["active_model"]

        if active_model_type == "ml_based":
            self.model = MLBasedModel()
        elif active_model_type == "rule_based":
            self.model = RuleBasedModel()
        else:
            raise ValueError(f"Unknown model type: {active_model_type}")

        print(f"Initialized StatePredictor with {active_model_type} model")

    def reset_state(self):
        """Reset predictor state for new sequence."""
        max_length = CONFIG["feature_engineering"]["lookback_frames"]
        self.metrics_history = deque(maxlen=max_length + 1)
        self.model.reset_state()

    def predict_single(self, base_metrics: Dict) -> str:
        """
        Predict state for single set of base metrics.

        Args:
            base_metrics: Dict with keys: mean_distance, median_player1_x, median_player1_y,
                         median_player2_x, median_player2_y

        Returns:
            Predicted state: "start", "active", or "end"
        """
        # Store current metrics
        self.metrics_history.append(base_metrics.copy())

        # Create single-row DataFrame
        df_temp = pd.DataFrame(
            [base_metrics]
            if CONFIG["active_model"] == "rule_based"
            else self.metrics_history
        )

        # Use model's predict method
        result_df = self.model.predict(df_temp)

        return result_df["predicted_state"].iloc[-1]

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict states for batch of base metrics.

        Args:
            df: DataFrame with base metrics columns

        Returns:
            DataFrame with predicted_state column added
        """
        return self.model.predict(df)
