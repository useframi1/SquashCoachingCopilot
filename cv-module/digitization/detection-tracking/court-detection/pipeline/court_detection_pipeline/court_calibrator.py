from inference import get_model
import cv2
import numpy as np

from court_detection_pipeline.utils import load_config


class CourtCalibrator:
    def __init__(self, config: dict = None):
        """Initialize the court calibrator with Roboflow model."""
        self.config = config if config else load_config()
        self.model = get_model(
            model_id=self.config["model_id"],
            api_key=self.config["api_key"]
        )
        self.homographies = {}

    @staticmethod
    def _get_quadrilateral_corners(points, epsilon_factor=0.02):
        """
        Approximate segmentation to exactly 4 corners using polygon approximation

        Args:
            points: List of Point objects from Roboflow prediction
            epsilon_factor: Controls approximation accuracy (lower = more precise to original shape)

        Returns:
            numpy array of shape (4, 2) containing [x, y] coordinates of corners
        """
        # Convert to numpy array
        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)

        # Calculate perimeter
        perimeter = cv2.arcLength(coords, True)

        # Approximate polygon - try to get exactly 4 points
        # Start with a reasonable epsilon
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(coords, epsilon, True)

        # If we don't get 4 points, adjust epsilon
        attempts = 0
        while len(approx) != 4 and attempts < 20:
            if len(approx) > 4:
                # Too many points, increase epsilon (more aggressive approximation)
                epsilon_factor *= 1.2
            else:
                # Too few points, decrease epsilon (less aggressive approximation)
                epsilon_factor *= 0.8

            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(coords, epsilon, True)
            attempts += 1

        # If we still don't have 4 points, fall back to convex hull or rotated rect
        if len(approx) != 4:
            # Try convex hull first
            hull = cv2.convexHull(coords)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

            # If still not 4 points, use rotated rectangle as fallback
            if len(approx) != 4:
                rect = cv2.minAreaRect(coords)
                approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.float32)

        return approx.reshape(4, 2).astype(np.int32)

    @staticmethod
    def _order_corners(corners):
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left
        """
        # Sort by y-coordinate
        corners = sorted(corners, key=lambda x: x[1])

        # Top two points (sorted left to right)
        top = sorted(corners[:2], key=lambda x: x[0])
        # Bottom two points (sorted left to right)
        bottom = sorted(corners[2:], key=lambda x: x[0])

        return np.array([top[0], top[1], bottom[1], bottom[0]])

    def _get_real_coords(self, class_name):
        """Return real-world coordinates for court elements in meters."""
        # Map class names to config keys
        class_mapping = self.config.get("class_mapping", {})

        if class_name in class_mapping:
            config_key = class_mapping[class_name]
            if config_key in self.config:
                return np.array(self.config[config_key], dtype=np.float32)

        raise ValueError(f"Unknown class '{class_name}' or missing real coordinates")

    def detect_keypoints(self, frame):
        """
        Detect keypoints from frame using Roboflow segmentation model.

        Args:
            frame: Input frame (BGR format from cv2)

        Returns:
            Dictionary mapping class_name to numpy array of keypoints (4, 2)
        """
        # Run inference with Roboflow model
        results = self.model.infer(frame)

        if not results or len(results) == 0:
            raise ValueError("No predictions returned from model")

        predictions = results[0].predictions

        if not predictions:
            raise ValueError("No court elements detected")

        keypoints_per_class = {}
        epsilon_factor = self.config.get("epsilon_factor", 0.02)
        conf_threshold = self.config.get("conf", 0.5)
        target_classes = self.config.get("target_classes", [])

        for prediction in predictions:
            class_name = prediction.class_name

            # Filter by confidence threshold
            if prediction.confidence < conf_threshold:
                continue

            # Filter for target classes if specified
            if target_classes and class_name not in target_classes:
                continue

            # Check if prediction has polygon points
            if not hasattr(prediction, 'points') or not prediction.points:
                continue

            # Get the 4 corners using polygon approximation
            corners = self._get_quadrilateral_corners(
                prediction.points,
                epsilon_factor=epsilon_factor
            )

            # Order corners consistently (TL, TR, BR, BL)
            ordered_corners = self._order_corners(corners)

            keypoints_per_class[class_name] = ordered_corners.astype(np.float32)

        if not keypoints_per_class:
            raise ValueError("No valid court elements detected after filtering")

        return keypoints_per_class

    def process_frame(self, frame):
        """
        Compute homography matrices for all detected classes.

        Args:
            frame: Input frame

        Returns:
            homographies: Dict mapping class_name to homography matrix
            keypoints_per_class: Dict mapping class_name to detected keypoints
        """
        keypoints_per_class = self.detect_keypoints(frame)

        for class_name, pixel_coords in keypoints_per_class.items():
            real_coords = self._get_real_coords(class_name)
            H, _ = cv2.findHomography(pixel_coords, real_coords)
            self.homographies[class_name] = H

        return self.homographies, keypoints_per_class

    def get_homography(self, class_name):
        """Get homography matrix for specific class."""
        if class_name not in self.homographies:
            raise ValueError(
                f"No homography found for '{class_name}'. Run process_frame first."
            )
        return self.homographies[class_name]
