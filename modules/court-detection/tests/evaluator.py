import json
import cv2
import numpy as np
import os
import sys

from court_calibration import CourtCalibrator


def load_test_config(config_path="config.json"):
    """Load test configuration file"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(config_dir, config_path)
    with open(full_path, "r") as f:
        config = json.load(f)
    return config


class CourtCalibrationEvaluator:
    def __init__(self, config: dict = None):
        self.config = config if config else load_test_config()

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

        self.test_frame = cv2.imread(self.test_image_path)
        if self.test_frame is None:
            raise ValueError(f"Could not load test image: {self.test_image_path}")

        print(f"Test image loaded: {self.test_frame.shape}")

        # Compute homographies
        print("Computing homography matrices...")
        self.homographies, self.keypoints = self.calibrator.process_frame(
            self.test_frame
        )

        if "floor" not in self.homographies or "wall" not in self.homographies:
            raise ValueError("Failed to compute floor and wall homographies")

        print("✓ Floor homography computed")
        print("✓ Wall homography computed")

        # Detect wall color
        print("\nDetecting wall color...")
        self.wall_color_info = self.calibrator.detect_wall_color(
            self.test_frame, self.keypoints
        )
        print(
            f"✓ Wall color detected: {'White' if self.wall_color_info['is_white'] else 'Dark/Colored'}"
        )
        print(f"  Recommended ball: {self.wall_color_info['recommended_ball'].upper()}")

    def visualize_keypoints(self, frame):
        """Draw detected keypoints on the frame"""
        vis_frame = frame.copy()

        if not self.config["visualization"]["show_keypoints"]:
            return vis_frame

        color = tuple(self.config["visualization"]["keypoint_color"])
        radius = self.config["visualization"]["keypoint_radius"]

        for class_name, points in self.keypoints.items():
            # Draw all 4 corners
            for i, point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_frame, (x, y), radius, color, -1)

                # Add label
                label = f"{class_name}_{i}"
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
            pts = points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [pts], True, color, 2)

        # Add wall color info overlay
        self._draw_wall_color_info(vis_frame)

        return vis_frame

    def _draw_wall_color_info(self, frame):
        """Draw wall color information overlay on the frame"""
        # Position for the info box (top-right corner)
        margin = 20
        box_width = 350
        box_height = 120
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
        wall_type = (
            "WHITE WALL" if self.wall_color_info["is_white"] else "DARK/COLORED WALL"
        )
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
        ball_text = f"Ball: {self.wall_color_info['recommended_ball'].upper()}"
        ball_color = (
            (0, 0, 0)
            if self.wall_color_info["recommended_ball"] == "black"
            else (255, 255, 255)
        )
        cv2.putText(
            frame,
            ball_text,
            (text_x, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ball_color,
            2,
        )

        # Color swatch
        swatch_size = 30
        swatch_x = text_x
        swatch_y = text_y + 50
        wall_color_bgr = tuple(int(c) for c in self.wall_color_info["wall_color_bgr"])
        cv2.rectangle(
            frame,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            wall_color_bgr,
            -1,
        )
        cv2.rectangle(
            frame,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            (255, 255, 255),
            1,
        )

        # Stats
        stats_text = f"B:{self.wall_color_info['mean_brightness']:.0f}"
        cv2.putText(
            frame,
            stats_text,
            (swatch_x + swatch_size + 10, swatch_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def test_inverse_transform(self, real_point, homography_type="floor"):
        """
        Apply inverse homography transform to convert real-world coordinates to pixel coordinates

        Args:
            real_point: [x, y] in meters
            homography_type: 'floor' or 'wall'

        Returns:
            Pixel coordinates [x, y]
        """
        H = self.homographies[homography_type]

        # Compute inverse homography
        H_inv = np.linalg.inv(H)

        # Apply inverse transform
        real_pt = np.array([[real_point]], dtype=np.float32)
        pixel_pt = cv2.perspectiveTransform(real_pt, H_inv)

        return pixel_pt[0][0]

    def visualize_test_points(self, frame):
        """Visualize test points by applying inverse homography"""
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

    def print_homography_info(self):
        """Print information about the computed homographies"""
        print("\n" + "=" * 60)
        print("HOMOGRAPHY MATRICES")
        print("=" * 60)

        print("\nFloor Homography Matrix:")
        print(self.homographies["floor"])

        print("\nWall Homography Matrix:")
        print(self.homographies["wall"])

        print("\n" + "=" * 60)
        print("WALL COLOR ANALYSIS")
        print("=" * 60)

        print(
            f"\nWall Type: {'WHITE' if self.wall_color_info['is_white'] else 'DARK/COLORED'}"
        )
        print(
            f"Recommended Ball Color: {self.wall_color_info['recommended_ball'].upper()}"
        )
        print(f"\nColor Statistics:")
        print(f"   Mean RGB: {self.wall_color_info['wall_color_rgb']}")
        print(
            f"   Mean Brightness (V): {self.wall_color_info['mean_brightness']:.1f} / 255"
        )

        print("\n" + "=" * 60)
        print("DETECTED KEYPOINTS")
        print("=" * 60)

        for class_name, points in self.keypoints.items():
            print(f"\n{class_name}:")
            for i, point in enumerate(points):
                print(f"   Point {i}: [{point[0]:.1f}, {point[1]:.1f}]")

    def save_visualization(self, output_path=None):
        """Save visualization with detected keypoints and test points"""
        if output_path is None:
            output_path = self.output_image_path

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create visualization
        vis_frame = self.visualize_keypoints(self.test_frame)
        vis_frame = self.visualize_test_points(vis_frame)

        # Save image
        cv2.imwrite(output_path, vis_frame)

        print(f"\n✓ Visualization saved: {output_path}")

        return vis_frame

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
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
