from ultralytics import YOLO
import cv2
import numpy as np
import torch

from config import CONFIG


class CourtCalibrator:
    def __init__(self):
        self.model = YOLO(CONFIG["court_calibrator"]["model_path"], verbose=False)
        # Force CPU to avoid GPU hangs
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.homography = None

    def _get_real_coords(self):
        return np.array(
            CONFIG["court_calibrator"]["real_coords"],
            dtype=np.float32,
        )

    def detect_keypoints(self, frame):
        import os
        from contextlib import redirect_stdout

        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                results = self.model(frame, verbose=False)[0]

        keypoints_dict = {}

        if results.keypoints is None or not hasattr(results.keypoints, "data"):
            raise ValueError("No keypoints detected")

        keypoints_array = results.keypoints.data.cpu().numpy()

        for person_keypoints in keypoints_array:
            for idx, (x, y, conf) in enumerate(person_keypoints):
                if conf > CONFIG["court_calibrator"]["confidence_threshold"]:
                    keypoints_dict[idx] = (x, y)

        if len(keypoints_dict) != 4:
            raise ValueError(
                f"Did not detect all 4 keypoints (found: {keypoints_dict.keys()})"
            )

        pixel_coords = np.array([keypoints_dict[i] for i in range(4)])
        return pixel_coords

    def compute_homography(self, frame):
        pixel_coords = self.detect_keypoints(frame)
        real_coords = self._get_real_coords()
        H, _ = cv2.findHomography(pixel_coords, real_coords)
        self.homography = H
        return H
