"""
Predictor - Inference pipeline for rally state prediction
"""

import pandas as pd
import numpy as np
import joblib
from collections import deque
from typing import Dict

from utilities.feature_engineer import FeatureEngineer
from config import CONFIG


class StatePredictor:
    """Handles inference for rally state prediction."""

    def __init__(self):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model. If None, uses CONFIG["model_path"]
        """
        model_path = CONFIG["modeling"][CONFIG["active_model"]]["model_path"]

        # Load trained model
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()

        # Initialize inference state
        self.reset_state()

        print(f"Loaded {self.model_type} model from {model_path}")
        print(f"Features: {len(self.feature_names)}")

    def reset_state(self):
        """Reset predictor state for new sequence."""
        # Store recent base metrics for temporal features
        self.metrics_history = deque(
            maxlen=CONFIG["feature_engineering"]["lookback_frames"] + 1
        )

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

        # Create temporary dataframe for feature engineering
        df_temp = pd.DataFrame(list(self.metrics_history))

        # Add frame numbers (not used in features but needed for processing)
        df_temp["frame_number"] = range(len(df_temp))

        # Engineer features
        df_features = self.feature_engineer.engineer_features(df_temp)

        # Get features for the latest (current) row
        current_features = df_features.iloc[-1]

        # Prepare feature vector
        X = current_features[self.feature_names].values.reshape(1, -1)

        # Make prediction
        y_pred = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform([y_pred])[0]

        return prediction

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict states for batch of base metrics.

        Args:
            df: DataFrame with base metrics columns

        Returns:
            DataFrame with predicted_state column added
        """
        # Reset state for batch processing
        self.reset_state()

        # Engineer features for entire batch
        df_features = self.feature_engineer.engineer_features(df)

        # Prepare feature matrix
        X = df_features[self.feature_names].values

        # Make predictions
        y_pred = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(y_pred)

        # Add predictions to dataframe
        result_df = df.copy()
        result_df["predicted_state"] = predictions

        return result_df
