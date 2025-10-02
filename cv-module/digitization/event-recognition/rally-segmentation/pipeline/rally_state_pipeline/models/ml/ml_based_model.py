"""
Model Trainer - Trains ML models for rally state prediction with state transition logic
"""

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
from collections import deque

from rally_state_pipeline.utilities.feature_engineer import FeatureEngineer
from rally_state_pipeline.models.base_model import BaseModel


class MLBasedModel(BaseModel):
    """Handles training of rally state prediction models with state transition logic."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_engineer = FeatureEngineer(config=self.config)
        self.metrics_history = deque(
            maxlen=self.config["feature_engineering"]["lookback_frames"] + 1
        )
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.load_trained_model()

    def load_trained_model(self):
        """Load a pre-trained model for inference."""
        model_path = self.config["models"]["ml_based"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")

        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]

    def set_state(self, state: str):
        """Reset any internal state of the model."""
        pass

    def predict(self, base_metrics: dict) -> str:
        """
        Predict rally states for the given data.

        Args:
            df: DataFrame with required columns (mean_distance, median_player1_x,
                median_player1_y, median_player2_x, median_player2_y)

        Returns:
            DataFrame with 'predicted_state' column added
        """
        self.metrics_history.append(base_metrics.copy())

        df = pd.DataFrame(self.metrics_history)

        # Engineer features for the entire history and take the last row
        df_features = self.feature_engineer.engineer_features(df).iloc[-1]

        # Prepare feature matrix
        X = df_features[self.feature_names].values.reshape(1, -1)

        # Make predictions
        y_pred = self.model.predict(X)
        prediction = self.label_encoder.inverse_transform(y_pred)

        return prediction[0]
