"""
Base model interface for rally state prediction.
Provides a common interface for both rule-based and ML-based models.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for rally state prediction models."""

    @abstractmethod
    def predict(self, base_metrics: dict) -> str:
        """
        Predict rally states for the given data.

        Args:
            df: DataFrame with required columns (mean_distance, median_player1_x,
                median_player1_y, median_player2_x, median_player2_y)

        Returns:
            DataFrame with 'predicted_state' column added
        """
        pass

    @abstractmethod
    def set_state(self, state: str):
        """Reset any internal state of the model."""
        pass
