"""
Fixed Unified Predictor Interface
Ensures single source of truth for temporal state tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List
from utilities.general import load_config
from feature_engineer import FeatureEngineer


class UnifiedPredictor:
    """
    Fixed unified interface - single temporal state tracking source.
    """

    def __init__(self):
        """Initialize the predictor based on config."""
        self.config = load_config()
        self.model_config = self.config["rally_segmenter"]
        self.model_type = self.model_config.get("active_model", "rule_based")

        # Initialize feature engineer (SINGLE source of temporal state)
        self.feature_engineer = FeatureEngineer()

        # Load the appropriate model
        self._load_model()

        print(f"Initialized {self.model_type} predictor")

    def _load_model(self):
        """Load the appropriate model based on config."""
        if self.model_type == "ml_based":
            model_path = self.model_config["ml_based"]["model_path"]

            # Load the pre-trained model components
            import joblib

            model_data = joblib.load(model_path)

            # Only load the core model - NO internal state tracking
            self.model = model_data["model"]
            self.label_encoder = model_data["label_encoder"]
            self.prev_state_encoder = model_data["prev_state_encoder"]
            self.feature_names = model_data["feature_names"]

            print(f"Loaded ML model from: {model_path}")

        elif self.model_type == "rule_based":
            from modeling.rule_based_model import RallyStateSegmenter

            self.model = RallyStateSegmenter()
            print("Loaded rule-based model")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _prepare_ml_features(self, features: Dict) -> pd.DataFrame:
        """Prepare features for ML model prediction - stateless."""
        # Create DataFrame from features
        feature_df = pd.DataFrame([features])

        # Encode prev_state using the features from FeatureEngineer
        try:
            prev_state_encoded = self.prev_state_encoder.transform(
                [features["prev_state"]]
            )[0]
        except:
            prev_state_encoded = 0

        feature_df["prev_state_encoded"] = prev_state_encoded

        # Ensure all required features are present
        for feature_name in self.feature_names:
            if feature_name not in feature_df.columns:
                if "distance" in feature_name:
                    feature_df[feature_name] = 0.0
                elif "player" in feature_name and (
                    "x" in feature_name or "y" in feature_name
                ):
                    feature_df[feature_name] = 0.0
                elif "movement" in feature_name:
                    feature_df[feature_name] = 0.0
                elif "side" in feature_name:
                    feature_df[feature_name] = 0
                else:
                    feature_df[feature_name] = 0

        # Select only required features in correct order
        return feature_df[self.feature_names]

    def predict(self, base_metrics: Union[Dict, pd.DataFrame]) -> Union[str, List[str]]:
        """
        Predict state(s) from base metrics.
        Single temporal state source via FeatureEngineer.
        """
        if isinstance(base_metrics, dict):
            return self._predict_single(base_metrics)
        else:
            return self._predict_batch(base_metrics)

    def _predict_single(self, base_metrics: Dict) -> str:
        """Predict single sample with unified state tracking."""
        # Engineer features (FeatureEngineer handles temporal state)
        features = self.feature_engineer.compute_features(base_metrics)

        if self.model_type == "ml_based":
            # Prepare features for ML model (stateless)
            feature_df = self._prepare_ml_features(features)

            # Make prediction (model has NO internal state)
            prediction_encoded = self.model.predict(feature_df)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

        else:  # rule_based
            # Create DataFrame for rule-based model
            df = pd.DataFrame([features])
            result_df = self.model.predict(df, apply_smoothing=False)
            prediction = result_df.iloc[0]["predicted_state"]

        # Update ONLY FeatureEngineer state (single source of truth)
        self.feature_engineer.update_state(prediction)

        return prediction

    def _predict_batch(self, df: pd.DataFrame) -> List[str]:
        """Predict batch - processes sequentially to maintain temporal consistency."""
        predictions = []
        current_group = None

        # Determine if we have groups
        group_col = "video_name" if "video_name" in df.columns else None

        # Sort to ensure correct order
        if group_col:
            df_sorted = df.sort_values([group_col, "frame_number"]).reset_index(
                drop=True
            )
        else:
            df_sorted = df.sort_values("frame_number").reset_index(drop=True)

        # Reset state for batch processing
        self.feature_engineer.reset_state()

        # Process each row sequentially (same as single prediction)
        for idx, row in df_sorted.iterrows():
            # Reset for new group
            if group_col and row[group_col] != current_group:
                self.feature_engineer.reset_state()
                current_group = row[group_col]

            # Use the SAME logic as single prediction
            base_metrics = row.to_dict()
            prediction = self._predict_single(base_metrics)
            predictions.append(prediction)

        return predictions

    def reset_state(self):
        """Reset predictor state for new sequence."""
        self.feature_engineer.reset_state()

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        info = {"model_type": self.model_type, "model_class": type(self.model).__name__}

        if self.model_type == "ml_based":
            info.update(
                {
                    "ml_model_type": self.model_config["ml_based"]["model_type"],
                    "num_features": len(self.feature_names),
                }
            )

        return info
