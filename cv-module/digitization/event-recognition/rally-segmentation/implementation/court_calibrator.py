from ultralytics import YOLO
import cv2
import numpy as np


class CourtCalibrator:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.homography = None

    def _get_real_coords(self):
        return np.array(
            [[1.6, 5.44], [4.8, 5.44], [4.8, 7.04], [1.6, 7.04]],
            dtype=np.float32,
        )

    def detect_keypoints(self, frame):
        results = self.model(frame)[0]
        keypoints_dict = {}

        if results.keypoints is None or not hasattr(results.keypoints, "data"):
            raise ValueError("No keypoints detected")

        keypoints_array = results.keypoints.data.cpu().numpy()

        for person_keypoints in keypoints_array:
            for idx, (x, y, conf) in enumerate(person_keypoints):
                if conf > 0.5:
                    keypoints_dict[idx] = (x, y)

        if len(keypoints_dict) != 4:
            # raise ValueError(
            #     f"Did not detect all 4 keypoints (found: {keypoints_dict.keys()})"
            # )
            return None

        pixel_coords = np.array([keypoints_dict[i] for i in range(4)])
        return pixel_coords

    def compute_homography(self, frame):
        pixel_coords = self.detect_keypoints(frame)
        if pixel_coords is None:
            return None
        # self.display_keypoints(frame, pixel_coords)
        real_coords = self._get_real_coords()
        H, _ = cv2.findHomography(pixel_coords, real_coords)
        self.homography = H
        return H

    def display_keypoints(self, frame, pixel_coords):
        """
        Display the frame with detected keypoints and optionally save the visualization.

        Args:
            frame: Input frame
            pixel_coords: Detected keypoints

        Returns:
            Visualization image with keypoints
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()

        if pixel_coords is None:
            cv2.putText(
                vis_frame,
                "Could not detect all 4 court keypoints",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            # Draw keypoints
            for x, y in pixel_coords:
                color = (0, 255, 0)

                # Convert to integer coordinates
                x, y = int(x), int(y)

                # Draw circle at keypoint
                cv2.circle(vis_frame, (x, y), 5, color, -1)

            cv2.imshow("Court Calibration", vis_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
