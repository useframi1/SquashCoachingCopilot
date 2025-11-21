"""
Court Calibration Evaluator

Evaluates court calibration performance and generates comprehensive visualizations
for verifying homography accuracy.
"""

import cv2
import numpy as np
import os
from typing import Dict, Optional

from squashcopilot.modules.court_calibration import CourtCalibrator
from squashcopilot.common.utils import load_config
from squashcopilot.common import (
    Frame,
    CourtCalibrationInput,
    CourtCalibrationOutput,
)


class CourtCalibrationEvaluator:
    """Evaluates court calibration performance on test images."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize evaluator with configuration."""
        if config is None:
            # Load the tests section from the court_calibration config
            full_config = load_config(config_name="court_calibration")
            config = full_config["tests"]
        self.config = config

        # Get test directory path
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve paths relative to test directory
        self.test_image_path = os.path.join(
            self.test_dir, self.config["paths"]["test_image"]
        )

        # Generate output path based on input image name
        image_name = os.path.basename(self.test_image_path)
        self.output_image_path = os.path.join(self.test_dir, "outputs", image_name)

        # Initialize court calibrator
        print("Initializing court calibrator...")
        self.calibrator = CourtCalibrator()

        # Load test image
        if not os.path.exists(self.test_image_path):
            raise FileNotFoundError(f"Test image not found: {self.test_image_path}")

        test_image = cv2.imread(self.test_image_path)
        if test_image is None:
            raise ValueError(f"Could not load test image: {self.test_image_path}")

        print(f"Test image loaded: {test_image.shape}")

        # Create Frame object for the test image
        self.test_frame = Frame(image=test_image, frame_number=0, timestamp=0.0)

        # Compute homographies using CourtCalibrationInput
        print("Computing homography matrices...")
        calibration_input = CourtCalibrationInput(frame=self.test_frame)
        self.calibration: CourtCalibrationOutput = self.calibrator.process_frame(
            calibration_input
        )

        if not self.calibration.calibration_success:
            raise ValueError("Failed to compute floor and wall homographies")

        print("✓ Floor homography computed")
        print("✓ Wall homography computed")

        # Print ball color recommendation
        print("\nDetecting wall color...")
        ball_color = "Black" if self.calibration.is_black_ball else "White"
        print(f"✓ Recommended ball: {ball_color.upper()}")

    def visualize_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected keypoints on the frame."""
        vis_frame = frame.copy()

        if not self.config["visualization"]["show_keypoints"]:
            return vis_frame

        color = tuple(self.config["visualization"]["keypoint_color"])
        radius = self.config["visualization"]["keypoint_radius"]

        # Group keypoints by class
        keypoints_by_class = {}
        for key, point in self.calibration.court_keypoints.items():
            # Parse class name from key (e.g., "tin_top_left" -> "tin")
            parts = key.rsplit("_", 2)  # Split from right to get class name
            if len(parts) >= 3:
                class_name = parts[0]
                corner_type = f"{parts[-2]}_{parts[-1]}"
            else:
                continue

            if class_name not in keypoints_by_class:
                keypoints_by_class[class_name] = {}
            keypoints_by_class[class_name][corner_type] = point

        # Draw keypoints for each class
        for class_name, corners in keypoints_by_class.items():
            points = []
            for corner_type in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                if corner_type in corners:
                    point = corners[corner_type]
                    points.append((point.x, point.y))

                    # Draw point
                    x, y = int(point.x), int(point.y)
                    cv2.circle(vis_frame, (x, y), radius, color, -1)

                    # Add label
                    label = f"{class_name}_{corner_type}"
                    cv2.putText(
                        vis_frame,
                        label,
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

            # Draw polygon connecting the points
            if len(points) == 4:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], True, color, 2)

        # Add ball color info overlay
        self._draw_ball_color_info(vis_frame)

        return vis_frame

    def _draw_ball_color_info(self, frame: np.ndarray) -> None:
        """Draw ball color information overlay on the frame."""
        # Position for the info box (top-right corner)
        margin = 20
        box_width = 300
        box_height = 80
        x = frame.shape[1] - box_width - margin
        y = margin

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(
            frame, (x, y), (x + box_width, y + box_height), (255, 255, 255), 2
        )

        # Add text
        text_x = x + 15
        text_y = y + 30

        # Wall type
        wall_type = "WHITE WALL" if self.calibration.is_black_ball else "DARK/COLORED WALL"
        cv2.putText(
            frame,
            wall_type,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Recommended ball
        ball_color = "BLACK" if self.calibration.is_black_ball else "WHITE"
        ball_text = f"Recommended Ball: {ball_color}"
        cv2.putText(
            frame,
            ball_text,
            (text_x, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def test_inverse_transform(
        self, real_point: list, homography_type: str = "floor"
    ) -> np.ndarray:
        """
        Apply inverse homography transform to convert real-world coordinates to pixel coordinates.

        Args:
            real_point: [x, y] in meters
            homography_type: 'floor' or 'wall'

        Returns:
            Pixel coordinates [x, y]
        """
        if homography_type == "floor":
            H = self.calibration.floor_homography
        elif homography_type == "wall":
            H = self.calibration.wall_homography
        else:
            raise ValueError(f"Unknown homography type: {homography_type}")

        # Compute inverse homography
        H_inv = np.linalg.inv(H)

        # Apply inverse transform
        real_pt = np.array([[real_point]], dtype=np.float32)
        pixel_pt = cv2.perspectiveTransform(real_pt, H_inv)

        return pixel_pt[0][0]

    def visualize_test_points(self, frame: np.ndarray) -> np.ndarray:
        """Visualize test points by applying inverse homography."""
        vis_frame = frame.copy()

        font_scale = self.config["visualization"]["font_scale"]
        font_thickness = self.config["visualization"]["font_thickness"]
        point_radius = self.config["visualization"]["point_radius"]
        line_thickness = self.config["visualization"]["line_thickness"]

        # Test floor points
        print("\n" + "=" * 60)
        print("FLOOR HOMOGRAPHY VERIFICATION")
        print("=" * 60)

        for idx, test_point in enumerate(self.config["test_points"]["floor"]):
            real_coord = test_point["real"]
            description = test_point["description"]
            color = tuple(test_point["color"])

            # Apply inverse transform
            pixel_coord = self.test_inverse_transform(
                real_coord, homography_type="floor"
            )

            print(f"\nFloor Test Point {idx + 1}:")
            print(f"   Description: {description}")
            print(f"   Real coordinates: {real_coord} meters")
            print(f"   Pixel coordinates: [{pixel_coord[0]:.1f}, {pixel_coord[1]:.1f}]")

            # Draw on frame
            x, y = int(pixel_coord[0]), int(pixel_coord[1])

            # Draw crosshair
            cv2.drawMarker(
                vis_frame,
                (x, y),
                color,
                cv2.MARKER_CROSS,
                markerSize=20,
                thickness=line_thickness,
            )

            # Draw circle
            cv2.circle(vis_frame, (x, y), point_radius, color, -1)

            # Add label with real coordinates
            label = f"F{idx + 1}: ({real_coord[0]}, {real_coord[1]})m"

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Draw background rectangle
            cv2.rectangle(
                vis_frame,
                (x + 15, y - text_height - 10),
                (x + 15 + text_width + 10, y + 5),
                (255, 255, 255),
                -1,
            )

            # Draw text
            cv2.putText(
                vis_frame,
                label,
                (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
            )

        # Test wall points
        print("\n" + "=" * 60)
        print("WALL HOMOGRAPHY VERIFICATION")
        print("=" * 60)

        for idx, test_point in enumerate(self.config["test_points"]["wall"]):
            real_coord = test_point["real"]
            description = test_point["description"]
            color = tuple(test_point["color"])

            # Apply inverse transform
            pixel_coord = self.test_inverse_transform(
                real_coord, homography_type="wall"
            )

            print(f"\nWall Test Point {idx + 1}:")
            print(f"   Description: {description}")
            print(f"   Real coordinates: {real_coord} meters")
            print(f"   Pixel coordinates: [{pixel_coord[0]:.1f}, {pixel_coord[1]:.1f}]")

            # Draw on frame
            x, y = int(pixel_coord[0]), int(pixel_coord[1])

            # Draw crosshair
            cv2.drawMarker(
                vis_frame,
                (x, y),
                color,
                cv2.MARKER_CROSS,
                markerSize=20,
                thickness=line_thickness,
            )

            # Draw circle
            cv2.circle(vis_frame, (x, y), point_radius, color, -1)

            # Add label with real coordinates
            label = f"W{idx + 1}: ({real_coord[0]}, {real_coord[1]})m"

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Draw background rectangle
            cv2.rectangle(
                vis_frame,
                (x + 15, y - text_height - 10),
                (x + 15 + text_width + 10, y + 5),
                (255, 255, 255),
                -1,
            )

            # Draw text
            cv2.putText(
                vis_frame,
                label,
                (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
            )

        return vis_frame

    def print_homography_info(self) -> None:
        """Print information about the computed homographies."""
        print("\n" + "=" * 60)
        print("HOMOGRAPHY MATRICES")
        print("=" * 60)

        print("\nFloor Homography Matrix:")
        print(self.calibration.floor_homography)

        print("\nWall Homography Matrix:")
        print(self.calibration.wall_homography)

        print("\n" + "=" * 60)
        print("BALL COLOR RECOMMENDATION")
        print("=" * 60)

        ball_color = "BLACK" if self.calibration.is_black_ball else "WHITE"
        wall_type = "WHITE" if self.calibration.is_black_ball else "DARK/COLORED"
        print(f"\nWall Type: {wall_type}")
        print(f"Recommended Ball Color: {ball_color}")

        print("\n" + "=" * 60)
        print("DETECTED KEYPOINTS")
        print("=" * 60)

        # Group keypoints by class for display
        keypoints_by_class = {}
        for key, point in self.calibration.court_keypoints.items():
            parts = key.rsplit("_", 2)
            if len(parts) >= 3:
                class_name = parts[0]
                corner_type = f"{parts[-2]}_{parts[-1]}"
            else:
                continue

            if class_name not in keypoints_by_class:
                keypoints_by_class[class_name] = {}
            keypoints_by_class[class_name][corner_type] = point

        for class_name, corners in keypoints_by_class.items():
            print(f"\n{class_name}:")
            for corner_type in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                if corner_type in corners:
                    point = corners[corner_type]
                    print(f"   {corner_type}: [{point.x:.1f}, {point.y:.1f}]")

    def save_visualization(self, output_path: Optional[str] = None) -> np.ndarray:
        """Save visualization with detected keypoints and test points."""
        if output_path is None:
            output_path = self.output_image_path

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create visualization - use the raw image from Frame object
        vis_frame = self.visualize_keypoints(self.test_frame.image)
        vis_frame = self.visualize_test_points(vis_frame)

        # Save image
        cv2.imwrite(output_path, vis_frame)

        print(f"\n✓ Visualization saved: {output_path}")

        return vis_frame

    def run_evaluation(self) -> np.ndarray:
        """Run complete evaluation pipeline."""
        print("\n" + "=" * 60)
        print("COURT CALIBRATION EVALUATION")
        print("=" * 60)

        # Print homography information
        self.print_homography_info()

        # Visualize and test inverse transforms
        vis_frame = self.save_visualization()

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nReview the output image to verify:")
        print(f"   1. All court elements were detected correctly (magenta polygons)")
        print(f"   2. Floor test points (green/red) land on expected locations")
        print(f"   3. Wall test points (blue/yellow) land on expected locations")
        print(f"\nIf points don't align correctly, the homography may need adjustment.")

        return vis_frame


if __name__ == "__main__":
    evaluator = CourtCalibrationEvaluator()
    evaluator.run_evaluation()
