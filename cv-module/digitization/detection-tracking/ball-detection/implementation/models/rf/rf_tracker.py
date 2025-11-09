import torch
from ultralytics import YOLO
import numpy as np
from utilities.general import load_config
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from collections import deque


class RFTracker:
    """Ball tracker using trained YOLO11 model for ball detection.

    This class uses a locally trained YOLO11 model (best.pt) to detect the ball position
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
        self.confidence_threshold = rf_config.get("confidence_threshold", 0.75)
        self.use_sahi = rf_config.get("use_sahi", True)
        self.use_motion_filter = rf_config.get("use_motion_filter", True)

        # Motion filtering parameters
        self.history_size = rf_config.get("history_size", 5)
        self.max_velocity = rf_config.get("max_velocity", 2000)
        self.distance_weight = rf_config.get("distance_weight", 0.7)
        self.confidence_weight = rf_config.get("confidence_weight", 0.3)

        # Path to the trained best.pt model
        model_path = os.path.join(
            os.path.dirname(__file__), "runs/detect/train2/weights/best.pt"
        )

        # Load the YOLO model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

        # Initialize SAHI model wrapper if enabled
        if self.use_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolo11",
                model_path=model_path,
                device=self.device,
            )

        # Initialize motion tracking state
        self.position_history = deque(maxlen=self.history_size)

    def reset(self):
        """Reset the tracker state."""
        self.frame_count = 0
        self.position_history.clear()

    def process_frame(self, frame):
        """Process a single frame and return ball coordinates.

        Args:
            frame: Input frame (BGR format, any resolution)

        Returns:
            tuple: (x, y) coordinates of the ball center, or (None, None) if not detected.
                   Coordinates are in the original frame's coordinate system.
        """
        # Get all detections from SAHI or standard YOLO
        detections = []  # List of (x_center, y_center, width, height, confidence)

        if self.use_sahi:
            # Use SAHI sliced inference
            result = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=320,
                slice_width=320,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )

            # Convert SAHI detections to standard format
            for det in result.object_prediction_list:
                bbox = det.bbox
                x_center = (bbox.minx + bbox.maxx) / 2
                y_center = (bbox.miny + bbox.maxy) / 2
                width = bbox.maxx - bbox.minx
                height = bbox.maxy - bbox.miny
                confidence = det.score.value
                detections.append((x_center, y_center, width, height, confidence))

        else:
            # Use standard YOLO inference
            results = self.model.predict(frame, verbose=False)[0]

            # Convert YOLO detections to standard format
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    detections.append((x_center, y_center, width, height, conf))

        # Use motion filter for detection selection if enabled
        if self.use_motion_filter:
            return self._process_with_motion_filter(detections)
        else:
            # Fallback to highest confidence detection
            return self._process_without_motion_filter(detections)

    def _process_without_motion_filter(self, detections):
        """Process detections without motion filter (highest confidence).

        Args:
            detections: List of (x_center, y_center, width, height, confidence)

        Returns:
            tuple: (x, y) coordinates or (None, None)
        """
        if not detections:
            return None, None

        # Find detection with highest confidence
        best_detection = max(detections, key=lambda x: x[4])
        x_center, y_center, _, _, confidence = best_detection

        # Check confidence threshold
        if confidence >= self.confidence_threshold:
            return int(x_center), int(y_center)

        return None, None

    def _process_with_motion_filter(self, detections):
        """Process detections with simple motion filtering.

        Args:
            detections: List of (x_center, y_center, width, height, confidence)

        Returns:
            tuple: (x, y) coordinates or (None, None)
        """
        # If not enough history, use highest confidence detection to build history
        if len(self.position_history) < 2:
            if detections:
                best_det = max(detections, key=lambda x: x[4])
                x, y, _, _, conf = best_det

                if conf >= self.confidence_threshold:
                    self.position_history.append((x, y))
                    return int(x), int(y)

            return None, None

        # Calculate predicted position based on velocity
        recent_positions = list(self.position_history)
        last_pos = np.array(recent_positions[-1])
        prev_pos = np.array(recent_positions[-2])
        velocity = last_pos - prev_pos
        predicted_pos = last_pos + velocity

        # If no detections, return None
        if not detections:
            return None, None

        # Score each detection
        best_score = -1
        best_detection = None

        for det in detections:
            x, y, w, h, conf = det

            # Calculate distance to predicted position
            distance = np.sqrt(
                (x - predicted_pos[0]) ** 2 + (y - predicted_pos[1]) ** 2
            )

            # Check if velocity is plausible
            # Skip detections that are too far (impossible velocity)
            if distance > self.max_velocity:
                continue

            # Distance score (closer to prediction is better)
            # Normalize distance to 0-1 range where 1 is best
            distance_score = 1.0 / (1.0 + distance / 100.0)

            # Combined score: weighted sum of confidence and distance
            score = (
                self.confidence_weight * conf + self.distance_weight * distance_score
            )

            if score > best_score:
                best_score = score
                best_detection = det

        # Update position history with best detection
        if best_detection is not None:
            x, y, _, _, conf = best_detection
            self.position_history.append((x, y))
            return int(x), int(y)
        else:
            # No good detection found
            return None, None
