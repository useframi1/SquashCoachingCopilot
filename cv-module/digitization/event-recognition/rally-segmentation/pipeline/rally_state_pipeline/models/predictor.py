"""
Predictor - Model-agnostic inference pipeline for rally state prediction
"""

import pandas as pd
from typing import Dict
from collections import deque

from rally_state_pipeline.models.ml.ml_based_model import MLBasedModel
from rally_state_pipeline.models.rule.rule_based_model import RuleBasedModel


class StatePredictor:
    """Model-agnostic predictor that works with both rule-based and ML-based models."""

    def __init__(self, config: dict):
        """Initialize predictor with the active model from config."""
        self.config = config

        active_model_type = self.config["active_model"]

        if active_model_type == "ml_based":
            self.model = MLBasedModel()
        elif active_model_type == "rule_based":
            self.model = RuleBasedModel()
        else:
            raise ValueError(f"Unknown model type: {active_model_type}")

        print(f"Initialized StatePredictor with {active_model_type} model")

    def set_state(self, state: str):
        """Reset predictor state for new sequence."""
        self.model.set_state(state)

    def predict(self, base_metrics: Dict) -> str:
        """
        Predict state for single set of base metrics.

        Args:
            base_metrics: Dict with keys: mean_distance, median_player1_x, median_player1_y,
                         median_player2_x, median_player2_y

        Returns:
            Predicted state: "start", "active", or "end"
        """
        # Use model's predict method
        result = self.model.predict(base_metrics)

        return result
