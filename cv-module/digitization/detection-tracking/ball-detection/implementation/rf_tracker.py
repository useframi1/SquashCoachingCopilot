from inference import get_model
import supervision as sv
import cv2
import numpy as np
from general import load_config


class RFTracker:
    """Ball tracker using Roboflow model for ball detection.

    This class uses a Roboflow inference model to detect the ball position
    in each frame. It keeps only the highest confidence detection per frame.
    """

    def __init__(self, config: dict = None):
        """Initialize the RF tracker.

        Args:
            config: Configuration dictionary. If None, loads from config.json
        """
        # Load configuration
        if config is None:
            config = load_config("config.json")

        self.config = config

        # Get RF model configuration
        rf_config = self.config.get("rf_model", {})
        model_id = rf_config.get("model_id", "squash-ball-detection-1lbti/1")
        api_key = rf_config.get("api_key", "56e6SM7rfayFBoIrsIFz")

        # Load the Roboflow model
        self.model = get_model(model_id=model_id, api_key=api_key)

    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0

    def process_frame(self, frame):
        """Process a single frame and return ball coordinates.

        Args:
            frame: Input frame (BGR format, any resolution)

        Returns:
            tuple: (x, y) coordinates of the ball center, or (None, None) if not detected.
                   Coordinates are in the original frame's coordinate system.
        """
        # Run inference
        results = self.model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        # Keep only the detection with highest confidence
        if len(detections) > 0:
            max_confidence_idx = np.argmax(detections.confidence)
            detection = detections[[max_confidence_idx]]

            # Get bounding box coordinates [x1, y1, x2, y2]
            bbox = detection.xyxy[0]  # First element is the bounding box
            x1, y1, x2, y2 = bbox

            # Calculate center of bounding box
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            return x_center, y_center

        return None, None
