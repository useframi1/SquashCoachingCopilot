"""
Unified State Predictor Interface
Provides a clean, model-agnostic interface for rally state prediction.
"""

from typing import Dict, Union, List
import pandas as pd
from utilities.general import load_config
from modeling.rule_based_model import RallyStateSegmenter
from modeling.rally_state_predictor import RallyStatePredictor


class StatePredictor:
    """
    Unified interface for rally state prediction.
    Supports both rule-based and ML models transparently.
    """
    
    def __init__(self):
        """Initialize the state predictor based on config."""
        self.config = load_config()
        model_config = self.config["rally_segmenter"]
        
        # Determine which model to use from config
        self.model_type = model_config.get("active_model", "rule_based")  # default to rule_based
        
        if self.model_type == "ml_based":
            model_path = model_config["ml_based"]["model_path"]
            self.model = RallyStatePredictor.load_model(model_path)
            print(f"Loaded ML model from: {model_path}")
        elif self.model_type == "rule_based":
            self.model = RallyStateSegmenter()
            print("Initialized rule-based model")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict_single(self, base_metrics: Dict[str, any]) -> str:
        """
        Predict state for a single set of base metrics.
        
        Args:
            base_metrics: Dictionary containing aggregated metrics from MetricsAggregator
            
        Returns:
            Predicted state: "start", "active", or "end"
        """
        if self.model_type == "ml_based":
            return self.model.predict(base_metrics)
        else:
            # For rule-based model, create a single-row DataFrame
            df = pd.DataFrame([base_metrics])
            result_df = self.model.predict(df, apply_smoothing=False)
            return result_df.iloc[0]["predicted_state"]
    
    def predict_batch(self, df: pd.DataFrame) -> List[str]:
        """
        Predict states for a batch of data (DataFrame).
        Processes sequentially to maintain temporal consistency.
        
        Args:
            df: DataFrame where each row contains aggregated metrics
            
        Returns:
            List of predicted states
        """
        if self.model_type == "ml_based":
            return self.model.predict(df)
        else:
            result_df = self.model.predict(df, apply_smoothing=True)
            return result_df["predicted_state"].tolist()
    
    def reset_state(self):
        """Reset internal state for processing new sequence."""
        if self.model_type == "ml_based":
            self.model.reset_inference_state()
        # Rule-based model is stateless, no reset needed
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        info = {
            "model_type": self.model_type,
            "model_class": type(self.model).__name__
        }
        
        if self.model_type == "ml_based":
            info.update({
                "ml_model_type": self.model.model_type,
                "num_features": len(self.model.feature_names) if self.model.feature_names else 0
            })
        
        return info