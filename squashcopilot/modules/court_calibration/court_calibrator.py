"""
Court calibration module for the squash coaching copilot.

This module detects court elements and computes homography matrices
for coordinate transformation between pixel and real-world coordinates.
"""

from inference import get_model
import cv2
import numpy as np

from squashcopilot.common.utils import load_config
from squashcopilot.common.types.geometry import Point2D
from squashcopilot.common.models import CourtCalibrationInput, CourtCalibrationOutput


class CourtCalibrator:
    """Court calibration using Roboflow segmentation model."""

    def __init__(self, config: dict = None):
        """Initialize the court calibrator with Roboflow model."""
        self.config = config if config else load_config(config_name="court_calibration")
        self.model = get_model(
            model_id=self.config["model_id"], api_key=self.config["api_key"]
        )
        self.homographies = {}

    @staticmethod
    def _get_quadrilateral_corners(points, epsilon_factor=0.02):
        """
        Approximate segmentation to exactly 4 corners using polygon approximation.

        Args:
            points: List of Point objects from Roboflow prediction
            epsilon_factor: Controls approximation accuracy

        Returns:
            numpy array of shape (4, 2) containing [x, y] coordinates of corners
        """
        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        perimeter = cv2.arcLength(coords, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(coords, epsilon, True)

        attempts = 0
        while len(approx) != 4 and attempts < 20:
            if len(approx) > 4:
                epsilon_factor *= 1.2
            else:
                epsilon_factor *= 0.8
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(coords, epsilon, True)
            attempts += 1

        if len(approx) != 4:
            hull = cv2.convexHull(coords)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            if len(approx) != 4:
                rect = cv2.minAreaRect(coords)
                approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.float32)

        return approx.reshape(4, 2).astype(np.int32)

    @staticmethod
    def _order_corners(corners):
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        corners = sorted(corners, key=lambda x: x[1])
        top = sorted(corners[:2], key=lambda x: x[0])
        bottom = sorted(corners[2:], key=lambda x: x[0])
        return np.array([top[0], top[1], bottom[1], bottom[0]])

    def _get_real_coords(self, class_name):
        """Return real-world coordinates for court elements in meters."""
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

            if prediction.confidence < conf_threshold:
                continue

            if target_classes and class_name not in target_classes:
                continue

            if not hasattr(prediction, "points") or not prediction.points:
                continue

            corners = self._get_quadrilateral_corners(
                prediction.points, epsilon_factor=epsilon_factor
            )
            ordered_corners = self._order_corners(corners)
            keypoints_per_class[class_name] = ordered_corners.astype(np.float32)

        if not keypoints_per_class:
            raise ValueError("No valid court elements detected after filtering")

        return keypoints_per_class

    def _compute_floor_homography(self, keypoints_per_class):
        """
        Compute floor homography using 10 points.

        Uses:
        - 2 bottom points of tin
        - 4 points of left service box
        - 4 points of right service box
        """
        pixel_points = []
        real_points = []

        if "tin" in keypoints_per_class:
            tin_pixels = keypoints_per_class["tin"]
            tin_real = self._get_real_coords("tin")
            pixel_points.extend([tin_pixels[2], tin_pixels[3]])
            real_points.extend([tin_real[2], tin_real[3]])
        else:
            raise ValueError("Tin not detected - required for floor homography")

        if "left-square" in keypoints_per_class:
            left_pixels = keypoints_per_class["left-square"]
            left_real = self._get_real_coords("left-square")
            pixel_points.extend(left_pixels)
            real_points.extend(left_real)
        else:
            raise ValueError("Left service box not detected")

        if "right-square" in keypoints_per_class:
            right_pixels = keypoints_per_class["right-square"]
            right_real = self._get_real_coords("right-square")
            pixel_points.extend(right_pixels)
            real_points.extend(right_real)
        else:
            raise ValueError("Right service box not detected")

        pixel_points = np.array(pixel_points, dtype=np.float32)
        real_points = np.array(real_points, dtype=np.float32)

        H, _ = cv2.findHomography(pixel_points, real_points)
        return H

    def _compute_wall_homography(self, keypoints_per_class):
        """
        Compute wall homography using 4 points.

        Uses:
        - 2 bottom points of tin
        - 2 top points of front wall
        """
        pixel_points = []
        real_points = []

        if "tin" in keypoints_per_class:
            tin_pixels = keypoints_per_class["tin"]
            tin_real = self._get_real_coords("tin")
            pixel_points.extend([tin_pixels[2], tin_pixels[3]])
            real_points.extend([tin_real[2], tin_real[3]])
        else:
            raise ValueError("Tin not detected - required for wall homography")

        if "front-wall-down" in keypoints_per_class:
            wall_pixels = keypoints_per_class["front-wall-down"]
            wall_real = self._get_real_coords("front-wall-down")
            pixel_points.extend([wall_pixels[0], wall_pixels[1]])
            real_points.extend([wall_real[0], wall_real[1]])
        else:
            raise ValueError("Front wall not detected")

        pixel_points = np.array(pixel_points, dtype=np.float32)
        real_points = np.array(real_points, dtype=np.float32)

        H = cv2.getPerspectiveTransform(pixel_points, real_points)
        return H

    def _detect_wall_color(self, frame, keypoints_per_class_np):
        """
        Detect if the front wall is white (indicating black ball usage).

        Returns:
            bool: True if wall is white (use black ball)
        """
        if "front-wall-down" not in keypoints_per_class_np:
            return False  # Default to white ball if wall not detected

        wall_points = keypoints_per_class_np["front-wall-down"]

        # Create mask for the front wall region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = wall_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        wall_pixels = frame[mask > 0]

        if len(wall_pixels) == 0:
            return False

        # Calculate brightness
        mean_rgb = np.mean(wall_pixels, axis=0)
        mean_color_bgr = mean_rgb.reshape(1, 1, 3).astype(np.uint8)
        mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        mean_brightness = float(mean_color_hsv[2])

        brightness_threshold = self.config.get("wall_detection", {}).get(
            "brightness_threshold", 180
        )

        return mean_brightness > brightness_threshold

    def process_frame(
        self, input_data: CourtCalibrationInput
    ) -> CourtCalibrationOutput:
        """
        Compute floor and wall homography matrices.

        Args:
            input_data: CourtCalibrationInput with frame

        Returns:
            CourtCalibrationOutput with homographies and keypoints
        """
        frame = input_data.frame.image
        frame_number = input_data.frame.frame_number

        try:
            # Detect keypoints
            keypoints_per_class_np = self.detect_keypoints(frame)

            # Compute homographies
            floor_H = self._compute_floor_homography(keypoints_per_class_np)
            wall_H = self._compute_wall_homography(keypoints_per_class_np)

            # Store in instance variable
            self.homographies["floor"] = floor_H
            self.homographies["wall"] = wall_H

            # Convert numpy keypoints to flat Dict[str, Point2D]
            court_keypoints = {}
            for class_name, kpts_array in keypoints_per_class_np.items():
                # Add all 4 corners with class prefix
                court_keypoints[f"{class_name}_top_left"] = Point2D(
                    x=float(kpts_array[0][0]), y=float(kpts_array[0][1])
                )
                court_keypoints[f"{class_name}_top_right"] = Point2D(
                    x=float(kpts_array[1][0]), y=float(kpts_array[1][1])
                )
                court_keypoints[f"{class_name}_bottom_right"] = Point2D(
                    x=float(kpts_array[2][0]), y=float(kpts_array[2][1])
                )
                court_keypoints[f"{class_name}_bottom_left"] = Point2D(
                    x=float(kpts_array[3][0]), y=float(kpts_array[3][1])
                )

            # Detect wall color
            is_black_ball = self._detect_wall_color(frame, keypoints_per_class_np)

            return CourtCalibrationOutput(
                frame_number=frame_number,
                calibration_success=True,
                floor_homography=floor_H,
                wall_homography=wall_H,
                court_keypoints=court_keypoints,
                is_black_ball=is_black_ball,
            )

        except (ValueError, Exception) as e:
            # Calibration failed - return failure output
            return CourtCalibrationOutput(
                frame_number=frame_number,
                calibration_success=False,
                floor_homography=np.eye(3),
                wall_homography=np.eye(3),
                court_keypoints={},
                is_black_ball=False,
            )

    def get_homography(self, class_name):
        """Get homography matrix for specific class."""
        if class_name not in self.homographies:
            raise ValueError(
                f"No homography found for '{class_name}'. Run process_frame first."
            )
        return self.homographies[class_name]
