from ultralytics import YOLO
import cv2
import numpy as np
import torch

from utils import load_config


class CourtCalibrator:
    def __init__(self, config: dict = None):
        """Initialize the court calibrator with YOLO model."""
        self.config = config if config else load_config()
        self.model = YOLO(self.config["model_path"], verbose=False)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz = self.config["imgsz"]
        self.homographies = {}

    def _get_real_coords(self, class_name):
        """Return real-world coordinates for tbox or wall in meters."""
        if class_name == "t-boxes":
            return self.config["real_t_coords"]
        elif class_name == "wall":
            return self.config["real_wall_coords"]
        else:
            raise ValueError(f"Unknown class '{class_name}'")

    def detect_keypoints(self, frame):
        """
        Detect keypoints from frame.

        Args:
            frame: Input frame (original size)

        Returns:
            Dictionary mapping class_name to numpy array of keypoints (4, 2)
        """
        # Let YOLO handle resizing internally - it maintains aspect ratio
        results = self.model(frame, imgsz=self.imgsz, verbose=False)[0]

        if results.keypoints is None or not hasattr(results.keypoints, "data"):
            raise ValueError("No keypoints detected")

        keypoints_array = results.keypoints.data.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        keypoints_per_class = {}

        for cls_id, kp_set in zip(class_ids, keypoints_array):
            class_name = self.model.names[cls_id]
            keypoints_dict = {}

            for idx, (x, y, conf) in enumerate(kp_set):
                if conf > self.config["conf"]:
                    # Keypoints are already in original frame coordinates
                    keypoints_dict[idx] = (x, y)

            if len(keypoints_dict) == 4:
                keypoints_per_class[class_name] = np.array(
                    [keypoints_dict[i] for i in range(4)], dtype=np.float32
                )
            else:
                raise ValueError(
                    f"{class_name}: Did not detect 4 keypoints (found {len(keypoints_dict)})"
                )

        return keypoints_per_class

    def process_frame(self, frame):
        """
        Compute homography matrices for all detected classes.

        Args:
            frame: Input frame

        Returns:
            Dictionary mapping class_name to homography matrix
        """
        keypoints_per_class = self.detect_keypoints(frame)

        for class_name, pixel_coords in keypoints_per_class.items():
            real_coords = self._get_real_coords(class_name)
            H, _ = cv2.findHomography(pixel_coords, real_coords)
            self.homographies[class_name] = H

        return self.homographies

    def get_homography(self, class_name):
        """Get homography matrix for specific class."""
        if class_name not in self.homographies:
            raise ValueError(
                f"No homography found for '{class_name}'. Run process_frame first."
            )
        return self.homographies[class_name]
