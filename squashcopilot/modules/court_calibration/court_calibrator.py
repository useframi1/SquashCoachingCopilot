from inference import get_model
import cv2
import numpy as np

from squashcopilot.common.utils import load_config

from squashcopilot.common import (
    Point2D,
    Homography,
    Keypoints,
    CourtCalibrationInput,
    CourtCalibrationResult,
    WallColorDetectionInput,
    WallColorResult,
)


class CourtCalibrator:
    def __init__(self, config: dict = None):
        """Initialize the court calibrator with Roboflow model."""
        self.config = config if config else load_config(config_name='court_calibration')
        self.model = get_model(
            model_id=self.config["model_id"], api_key=self.config["api_key"]
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
            if not hasattr(prediction, "points") or not prediction.points:
                continue

            # Get the 4 corners using polygon approximation
            corners = self._get_quadrilateral_corners(
                prediction.points, epsilon_factor=epsilon_factor
            )

            # Order corners consistently (TL, TR, BR, BL)
            ordered_corners = self._order_corners(corners)

            keypoints_per_class[class_name] = ordered_corners.astype(np.float32)

        if not keypoints_per_class:
            raise ValueError("No valid court elements detected after filtering")

        return keypoints_per_class

    def _compute_floor_homography(self, keypoints_per_class):
        """
        Compute floor homography using 10 points:
        - 2 bottom points of tin
        - 4 points of left service box
        - 4 points of right service box

        Args:
            keypoints_per_class: Dict mapping class_name to detected keypoints

        Returns:
            Homography matrix for floor plane
        """
        pixel_points = []
        real_points = []

        # Get tin bottom points (indices 2, 3: bottom-right, bottom-left)
        if "tin" in keypoints_per_class:
            tin_pixels = keypoints_per_class["tin"]
            tin_real = self._get_real_coords("tin")
            pixel_points.extend([tin_pixels[2], tin_pixels[3]])  # BR, BL
            real_points.extend([tin_real[2], tin_real[3]])
        else:
            raise ValueError("Tin not detected - required for floor homography")

        # Get left service box points (all 4 corners)
        if "left-square" in keypoints_per_class:
            left_pixels = keypoints_per_class["left-square"]
            left_real = self._get_real_coords("left-square")
            pixel_points.extend(left_pixels)
            real_points.extend(left_real)
        else:
            raise ValueError(
                "Left service box not detected - required for floor homography"
            )

        # Get right service box points (all 4 corners)
        if "right-square" in keypoints_per_class:
            right_pixels = keypoints_per_class["right-square"]
            right_real = self._get_real_coords("right-square")
            pixel_points.extend(right_pixels)
            real_points.extend(right_real)
        else:
            raise ValueError(
                "Right service box not detected - required for floor homography"
            )

        pixel_points = np.array(pixel_points, dtype=np.float32)
        real_points = np.array(real_points, dtype=np.float32)

        H, _ = cv2.findHomography(pixel_points, real_points)
        return H

    def _compute_wall_homography(self, keypoints_per_class):
        """
        Compute wall homography using 4 points:
        - 2 bottom points of tin
        - 2 top points of front wall

        Args:
            keypoints_per_class: Dict mapping class_name to detected keypoints

        Returns:
            Homography matrix for wall plane
        """
        pixel_points = []
        real_points = []

        # Get tin bottom points (indices 2, 3: bottom-right, bottom-left)
        if "tin" in keypoints_per_class:
            tin_pixels = keypoints_per_class["tin"]
            tin_real = self._get_real_coords("tin")
            pixel_points.extend([tin_pixels[2], tin_pixels[3]])  # BR, BL
            real_points.extend([tin_real[2], tin_real[3]])
        else:
            raise ValueError("Tin not detected - required for wall homography")

        # Get front wall top points (indices 0, 1: top-left, top-right)
        if "front-wall-down" in keypoints_per_class:
            wall_pixels = keypoints_per_class["front-wall-down"]
            wall_real = self._get_real_coords("front-wall-down")
            pixel_points.extend([wall_pixels[0], wall_pixels[1]])  # TL, TR
            real_points.extend([wall_real[0], wall_real[1]])
        else:
            raise ValueError("Front wall not detected - required for wall homography")

        pixel_points = np.array(pixel_points, dtype=np.float32)
        real_points = np.array(real_points, dtype=np.float32)

        H = cv2.getPerspectiveTransform(pixel_points, real_points)
        return H

    def process_frame(
        self, input_data: CourtCalibrationInput
    ) -> CourtCalibrationResult:
        """
        Compute floor and wall homography matrices.

        Args:
            input_data: CourtCalibrationInput with frame and metadata

        Returns:
            CourtCalibrationResult with homographies and keypoints
        """
        frame = input_data.frame.image
        frame_number = input_data.frame.frame_number

        try:
            # Detect keypoints
            keypoints_per_class_np = self.detect_keypoints(frame)

            # Compute homographies
            floor_H = self._compute_floor_homography(keypoints_per_class_np)
            wall_H = self._compute_wall_homography(keypoints_per_class_np)

            # Store in instance variable for get_homography compatibility
            self.homographies["floor"] = floor_H
            self.homographies["wall"] = wall_H

            # Convert numpy keypoints to Keypoints objects
            keypoints_per_class = {}
            for class_name, kpts_array in keypoints_per_class_np.items():
                # Create Point2D for each corner
                points_dict = {
                    "top_left": Point2D(
                        x=float(kpts_array[0][0]), y=float(kpts_array[0][1])
                    ),
                    "top_right": Point2D(
                        x=float(kpts_array[1][0]), y=float(kpts_array[1][1])
                    ),
                    "bottom_right": Point2D(
                        x=float(kpts_array[2][0]), y=float(kpts_array[2][1])
                    ),
                    "bottom_left": Point2D(
                        x=float(kpts_array[3][0]), y=float(kpts_array[3][1])
                    ),
                }
                keypoints_per_class[class_name] = Keypoints(points=points_dict)

            # Create Homography objects
            homographies = {
                "floor": Homography(matrix=floor_H, source_plane="floor"),
                "wall": Homography(matrix=wall_H, source_plane="wall"),
            }

            return CourtCalibrationResult(
                homographies=homographies,
                keypoints_per_class=keypoints_per_class,
                frame_number=frame_number,
                calibrated=True,
            )

        except (ValueError, Exception) as e:
            # Calibration failed
            return CourtCalibrationResult.not_calibrated(frame_number)

    def detect_wall_color(self, input_data: WallColorDetectionInput) -> WallColorResult:
        """
        Detect if the front wall is white (indicating black ball usage).

        Args:
            input_data: WallColorDetectionInput with frame and optional keypoints

        Returns:
            WallColorResult with wall color information
        """
        frame = input_data.frame.image

        # Get keypoints - either from input or detect them
        if input_data.keypoints_per_class is not None:
            # Convert Keypoints objects back to numpy arrays for processing
            keypoints_per_class_np = {}
            for class_name, kpts in input_data.keypoints_per_class.items():
                # Extract points in order: TL, TR, BR, BL
                points_list = [
                    kpts.get_point("top_left"),
                    kpts.get_point("top_right"),
                    kpts.get_point("bottom_right"),
                    kpts.get_point("bottom_left"),
                ]
                keypoints_per_class_np[class_name] = np.array(
                    [[p.x, p.y] for p in points_list], dtype=np.float32
                )
        else:
            keypoints_per_class_np = self.detect_keypoints(frame)

        if "front-wall-down" not in keypoints_per_class_np:
            raise ValueError("Front wall not detected - cannot determine wall color")

        # Get front wall polygon points
        wall_points = keypoints_per_class_np["front-wall-down"]

        # Create mask for the front wall region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = wall_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # Sample pixels within the front wall region
        wall_pixels = frame[mask > 0]

        if len(wall_pixels) == 0:
            raise ValueError("No pixels found in front wall region")

        # Calculate color statistics
        mean_rgb = np.mean(wall_pixels, axis=0)  # Mean RGB values

        # Convert to HSV for better color analysis
        mean_color_bgr = mean_rgb.reshape(1, 1, 3).astype(np.uint8)
        mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)[0, 0]

        # Extract brightness (V in HSV)
        mean_brightness = float(mean_color_hsv[2])  # V channel (0-255)

        # Determine if wall is white
        brightness_threshold = self.config.get("wall_detection", {}).get(
            "brightness_threshold", 180
        )

        is_white = mean_brightness > brightness_threshold

        # Recommend ball color
        recommended_ball = "black" if is_white else "white"

        # Convert to WallColorResult
        return WallColorResult(
            is_white=is_white,
            mean_brightness=mean_brightness,
            wall_color_rgb=tuple(mean_rgb[::-1]),  # Convert BGR to RGB
            wall_color_bgr=tuple(mean_rgb),
            recommended_ball_color=recommended_ball,
        )

    def get_homography(self, class_name):
        """Get homography matrix for specific class."""
        if class_name not in self.homographies:
            raise ValueError(
                f"No homography found for '{class_name}'. Run process_frame first."
            )
        return self.homographies[class_name]
