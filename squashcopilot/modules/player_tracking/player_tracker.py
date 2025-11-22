"""
Player tracking module for the squash coaching copilot.

Uses YOLO for detection and ResNet50 for re-identification.
Outputs flat structure compatible with DataFrame-based pipeline.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from collections import deque
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from scipy.ndimage import gaussian_filter1d
import os

from squashcopilot.common.utils import load_config, get_package_dir
from squashcopilot.common.types.geometry import BoundingBox, Point2D
from squashcopilot.common.models import (
    PlayerTrackingInput,
    PlayerTrackingOutput,
    PlayerPostprocessingInput,
    PlayerPostprocessingOutput,
    CourtCalibrationOutput,
)
from squashcopilot.common.constants import BODY_KEYPOINT_INDICES




class PlayerTracker:
    """
    Player tracker for identifying and tracking two squash players.
    Returns flat structure for DataFrame integration.
    """

    def __init__(
        self,
        config: dict = None,
        calibration: CourtCalibrationOutput = None,
    ):
        """
        Initialize the player tracker.

        Args:
            config: Configuration dictionary. If None, loads from config.json.
            calibration: Court calibration output with homographies.
        """
        self.config = config if config else load_config(config_name="player_tracking")
        self.calibration = calibration
        self.court_mask = None
        self.max_history = self.config["tracker"]["max_history"]
        self.reid_threshold = self.config["tracker"]["reid_threshold"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Player state
        self.player_positions = {
            1: deque(maxlen=self.max_history),
            2: deque(maxlen=self.max_history),
        }
        self.player_features = {1: None, 2: None}
        self.players_initialized = False

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Load YOLO detector and ResNet50 re-identification model."""
        model_path = os.path.join(
            get_package_dir(__file__), self.config["models"]["yolo_model"]
        )
        self.detector = YOLO(model_path, verbose=False)

        # Check if model supports pose estimation
        self.has_pose = (
            hasattr(self.detector.model, "kpt_shape")
            if hasattr(self.detector, "model")
            else False
        )

        # Load ResNet50 for re-identification
        self.reid_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.reid_model.fc.in_features
        feature_size = self.config["models"]["reid_feature_size"]
        self.reid_model.fc = torch.nn.Linear(in_features, feature_size)
        self.reid_model = self.reid_model.to(self.device)
        self.reid_model.eval()

    def set_calibration(self, calibration: CourtCalibrationOutput):
        """Update calibration data."""
        self.calibration = calibration
        self.court_mask = None  # Reset mask to be recreated

    def _get_floor_from_homography(self) -> np.ndarray:
        """Get full court boundaries using floor homography."""
        if self.calibration is None or self.calibration.floor_homography is None:
            return None

        # Full court in real coordinates (squash court dimensions in meters)
        court_real = np.array(
            [[0, 9.75], [6.4, 9.75], [6.4, 0], [0, 0]], dtype=np.float32
        )

        H_inv = np.linalg.inv(self.calibration.floor_homography)
        court_pixels = cv2.perspectiveTransform(court_real.reshape(-1, 1, 2), H_inv)
        return court_pixels.reshape(-1, 2)

    def _get_wall_from_homography(self) -> np.ndarray:
        """Get wall boundaries using wall homography."""
        if self.calibration is None or self.calibration.wall_homography is None:
            return None

        wall_real = np.array(
            [[0, 0], [6.4, 0], [6.4, 1.78], [0, 1.78]], dtype=np.float32
        )

        H_inv = np.linalg.inv(self.calibration.wall_homography)
        wall_pixels = cv2.perspectiveTransform(wall_real.reshape(-1, 1, 2), H_inv)
        return wall_pixels.reshape(-1, 2)

    def _create_court_mask(self, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """Create expanded court mask to filter out audience."""
        h, w = frame_shape[:2]

        floor_points = self._get_floor_from_homography()
        if floor_points is not None:
            floor_mask = np.zeros((h, w), dtype=np.uint8)
            floor_points_int = np.array(floor_points, dtype=np.int32)
            cv2.fillPoly(floor_mask, [floor_points_int], 255)
        else:
            floor_mask = np.zeros((h, w), dtype=np.uint8)

        wall_points = self._get_wall_from_homography()
        if wall_points is not None:
            wall_mask = np.zeros((h, w), dtype=np.uint8)
            wall_points_int = np.array(wall_points, dtype=np.int32)
            cv2.fillPoly(wall_mask, [wall_points_int], 255)
        else:
            wall_mask = np.zeros((h, w), dtype=np.uint8)

        return cv2.bitwise_or(floor_mask, wall_mask)

    def process_frame(self, input_data: PlayerTrackingInput) -> PlayerTrackingOutput:
        """
        Process a single frame and return player tracking information.

        Args:
            input_data: PlayerTrackingInput with frame and calibration

        Returns:
            PlayerTrackingOutput with flat player detection data
        """
        frame = input_data.frame.image
        frame_number = input_data.frame.frame_number
        timestamp = input_data.frame.timestamp if hasattr(input_data.frame, 'timestamp') else frame_number / 30.0
        frame_width = frame.shape[1]

        # Update calibration if provided
        if input_data.calibration is not None:
            self.calibration = input_data.calibration

        # Detect people in frame
        detections, keypoints = self._detect_people(frame)

        # Assign detections to player IDs
        assignments = self._assign_detections(detections, frame, frame_width)

        # Initialize output with no detections
        output = PlayerTrackingOutput(
            frame_number=frame_number,
            timestamp=timestamp,
            # Player 1
            player_1_detected=False,
            player_1_x_pixel=None,
            player_1_y_pixel=None,
            player_1_x_meter=None,
            player_1_y_meter=None,
            player_1_confidence=0.0,
            player_1_bbox=None,
            player_1_keypoints=None,
            # Player 2
            player_2_detected=False,
            player_2_x_pixel=None,
            player_2_y_pixel=None,
            player_2_x_meter=None,
            player_2_y_meter=None,
            player_2_confidence=0.0,
            player_2_bbox=None,
            player_2_keypoints=None,
        )

        # Fill in detected players
        for det_idx, player_id in assignments.items():
            bbox = detections[det_idx][0:4]
            confidence = float(detections[det_idx][4])

            # Calculate bottom-center position (player's foot position) in pixels
            center_x_pixel = (bbox[0] + bbox[2]) / 2
            bottom_y_pixel = bbox[3]

            # Convert to meter coordinates using calibration
            center_x_meter = None
            center_y_meter = None
            if self.calibration is not None and self.calibration.floor_homography is not None:
                pixel_point = Point2D(x=center_x_pixel, y=bottom_y_pixel)
                meter_point = self.calibration.pixel_to_floor(pixel_point)
                center_x_meter = meter_point.x
                center_y_meter = meter_point.y

            # Extract body keypoints (indices 5-16, which is 12 keypoints)
            kp_array = None
            if keypoints[det_idx] is not None:
                full_kp = keypoints[det_idx]["xy"]
                # Extract only body keypoints (indices 5-16)
                kp_array = full_kp[BODY_KEYPOINT_INDICES].copy()

            # Create bounding box
            bbox_obj = BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            )

            # Update position history
            self.player_positions[player_id].append((center_x_pixel, bottom_y_pixel))

            # Set output fields based on player ID
            if player_id == 1:
                output.player_1_detected = True
                output.player_1_x_pixel = float(center_x_pixel)
                output.player_1_y_pixel = float(bottom_y_pixel)
                output.player_1_x_meter = center_x_meter
                output.player_1_y_meter = center_y_meter
                output.player_1_confidence = confidence
                output.player_1_bbox = bbox_obj
                output.player_1_keypoints = kp_array
            else:  # player_id == 2
                output.player_2_detected = True
                output.player_2_x_pixel = float(center_x_pixel)
                output.player_2_y_pixel = float(bottom_y_pixel)
                output.player_2_x_meter = center_x_meter
                output.player_2_y_meter = center_y_meter
                output.player_2_confidence = confidence
                output.player_2_bbox = bbox_obj
                output.player_2_keypoints = kp_array

        return output

    def _detect_people(self, frame: np.ndarray):
        """
        Detect people in the frame using YOLO.

        Returns:
            tuple: (detections, keypoints)
        """
        # Create mask on first frame
        if self.court_mask is None and self.calibration is not None:
            self.court_mask = self._create_court_mask(frame.shape)

        # Apply mask before detection
        detection_frame = frame
        if self.court_mask is not None:
            detection_frame = cv2.bitwise_and(frame, frame, mask=self.court_mask)

        results = self.detector(detection_frame, verbose=False)[0]
        detections = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        detections = np.hstack([detections, confidences[:, None], classes[:, None]])

        # Extract keypoints if available
        keypoints_data = None
        if self.has_pose and results.keypoints is not None:
            keypoints_data = {
                "xy": results.keypoints.xy.cpu().numpy(),
                "conf": (
                    results.keypoints.conf.cpu().numpy()
                    if hasattr(results.keypoints, "conf")
                    else None
                ),
            }

        # Filter for person class (class 0) and get top 2
        person_dets = [det for det in detections if det[5] == 0]
        person_dets = sorted(person_dets, key=lambda x: x[4], reverse=True)[:2]

        # Get corresponding keypoints for top 2 detections
        person_keypoints = []
        if keypoints_data is not None:
            for i in range(len(person_dets)):
                det_idx = next(
                    j
                    for j, det in enumerate(detections)
                    if np.array_equal(det, person_dets[i])
                )
                kp = {
                    "xy": keypoints_data["xy"][det_idx],
                    "conf": (
                        keypoints_data["conf"][det_idx]
                        if keypoints_data["conf"] is not None
                        else None
                    ),
                }
                person_keypoints.append(kp)
        else:
            person_keypoints = [None] * len(person_dets)

        return person_dets, person_keypoints

    def _assign_detections(self, detections, frame, frame_width) -> Dict[int, int]:
        """Assign detections to player IDs using re-identification and position."""
        if len(detections) == 0:
            return {}

        if not self.players_initialized:
            return self._initialize_players(detections, frame)

        det_features = [
            self._extract_features(frame, det[0:4]) for det in detections[:2]
        ]

        if self.player_features[1] is None and self.player_features[2] is None:
            return self._initialize_players(detections, frame)

        matching_scores = self._calculate_matching_scores(
            detections, det_features, frame_width
        )

        return self._greedy_assignment(matching_scores, det_features)

    def _initialize_players(self, detections, frame) -> Dict[int, int]:
        """Initialize players based on left/right positioning."""
        if len(detections) < 2:
            return {}

        det_features = [
            self._extract_features(frame, det[0:4]) for det in detections[:2]
        ]

        # Sort by x-coordinate (left to right)
        detections_with_idx = [(i, det) for i, det in enumerate(detections[:2])]
        detections_with_idx.sort(key=lambda x: (x[1][0] + x[1][2]) / 2)

        # Assign: leftmost = Player 1, rightmost = Player 2
        assignments = {}
        for player_id, (det_idx, _) in enumerate(detections_with_idx, 1):
            assignments[det_idx] = player_id
            self.player_features[player_id] = det_features[det_idx]

        self.players_initialized = True
        return assignments

    def _calculate_matching_scores(self, detections, det_features, frame_width):
        """Calculate matching scores between detections and tracked players."""
        matching_scores = {}

        for i, det in enumerate(detections[:2]):
            bbox = det[0:4]
            center = ((bbox[0] + bbox[2]) / 2, bbox[3])

            for player_id in [1, 2]:
                if self.player_features[player_id] is None:
                    continue

                # Re-identification score
                reid_score = self._feature_distance(
                    det_features[i], self.player_features[player_id]
                )

                # Position score
                pos_score = 0
                if len(self.player_positions[player_id]) > 0:
                    last_pos = self.player_positions[player_id][-1]
                    distance = np.sqrt(
                        (center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2
                    )
                    pos_score = distance / frame_width

                # Combined score
                reid_weight = self.config["tracker"]["reid_weight"]
                pos_weight = self.config["tracker"]["position_weight"]
                score = reid_score * reid_weight + pos_score * pos_weight
                matching_scores[(i, player_id)] = score

        return matching_scores

    def _greedy_assignment(self, matching_scores, det_features) -> Dict[int, int]:
        """Perform greedy assignment of detections to players."""
        assignments = {}
        assigned_players = set()

        for (det_idx, player_id), score in sorted(
            matching_scores.items(), key=lambda x: x[1]
        ):
            if det_idx in assignments or player_id in assigned_players:
                continue
            if score < self.reid_threshold:
                assignments[det_idx] = player_id
                assigned_players.add(player_id)

                # Update player features with exponential moving average
                if det_features[det_idx] is not None:
                    self.player_features[player_id] = (
                        0.5 * self.player_features[player_id]
                        + 0.5 * det_features[det_idx]
                    )

        # Handle remaining unassigned detections
        unassigned_dets = [
            i for i in range(min(2, len(det_features))) if i not in assignments
        ]
        unassigned_players = [i for i in [1, 2] if i not in assigned_players]

        if len(unassigned_dets) == 1 and len(unassigned_players) == 1:
            det_idx = unassigned_dets[0]
            player_id = unassigned_players[0]
            assignments[det_idx] = player_id
            self.player_features[player_id] = det_features[det_idx]

        return assignments

    def _extract_features(self, frame, bbox) -> Optional[np.ndarray]:
        """Extract re-identification features from a bounding box."""
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None

        input_size = tuple(self.config["models"]["reid_input_size"])
        person_img = cv2.resize(person_img, input_size)
        person_tensor = torch.from_numpy(person_img).permute(2, 0, 1).float() / 255.0
        person_tensor = person_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.reid_model(person_tensor)

        return features.squeeze().cpu().numpy()

    def _feature_distance(self, feat1, feat2) -> float:
        """Calculate cosine distance between two feature vectors."""
        if feat1 is None or feat2 is None:
            return float("inf")

        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)

        return 1 - np.dot(feat1, feat2)

    def process_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        timestamps: List[float],
        batch_size: int = 32,
    ) -> List[PlayerTrackingOutput]:
        """
        Process a batch of frames with batched YOLO detection and ReID.

        This method processes multiple frames efficiently by:
        1. Batching YOLO detection across all frames
        2. Batching ReID feature extraction for all detected persons
        3. Sequential assignment (stateful, cannot be parallelized)

        Args:
            frames: List of input frames (BGR format).
            frame_numbers: List of frame numbers corresponding to each frame.
            timestamps: List of timestamps corresponding to each frame.
            batch_size: Number of frames to process in parallel for YOLO.

        Returns:
            List of PlayerTrackingOutput for each input frame.
        """
        if len(frames) == 0:
            return []

        # Create court mask on first frame if needed
        if self.court_mask is None and self.calibration is not None:
            self.court_mask = self._create_court_mask(frames[0].shape)

        # Phase 1: Batch YOLO detection
        all_detections, all_keypoints = self._detect_people_batch(frames, batch_size)

        # Phase 2: Batch ReID feature extraction for all detections
        all_features = self._extract_features_batch(frames, all_detections, batch_size)

        # Phase 3: Sequential assignment and output building
        outputs = []
        for i, (frame, frame_number, timestamp) in enumerate(
            zip(frames, frame_numbers, timestamps)
        ):
            detections = all_detections[i]
            keypoints = all_keypoints[i]
            det_features = all_features[i]
            frame_width = frame.shape[1]

            # Assign detections to player IDs (uses stateful tracking)
            assignments = self._assign_detections_with_features(
                detections, det_features, frame_width
            )

            # Build output
            output = self._build_output(
                frame_number, timestamp, detections, keypoints, assignments
            )
            outputs.append(output)

        return outputs

    def _detect_people_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32,
    ) -> Tuple[List[List], List[List]]:
        """
        Batch YOLO detection across multiple frames.

        Args:
            frames: List of input frames.
            batch_size: Number of frames to process in parallel.

        Returns:
            Tuple of (all_detections, all_keypoints) where each is a list per frame.
        """
        all_detections = []
        all_keypoints = []

        # Apply court mask to all frames if available
        if self.court_mask is not None:
            detection_frames = [
                cv2.bitwise_and(frame, frame, mask=self.court_mask)
                for frame in frames
            ]
        else:
            detection_frames = frames

        # Process frames in batches through YOLO
        for batch_start in range(0, len(detection_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(detection_frames))
            batch_frames = detection_frames[batch_start:batch_end]

            # YOLO batch inference
            results_list = self.detector(batch_frames, verbose=False)

            for results in results_list:
                detections = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                detections = np.hstack([detections, confidences[:, None], classes[:, None]])

                # Extract keypoints if available
                keypoints_data = None
                if self.has_pose and results.keypoints is not None:
                    keypoints_data = {
                        "xy": results.keypoints.xy.cpu().numpy(),
                        "conf": (
                            results.keypoints.conf.cpu().numpy()
                            if hasattr(results.keypoints, "conf")
                            else None
                        ),
                    }

                # Filter for person class (class 0) and get top 2
                person_dets = [det for det in detections if det[5] == 0]
                person_dets = sorted(person_dets, key=lambda x: x[4], reverse=True)[:2]

                # Get corresponding keypoints for top 2 detections
                person_keypoints = []
                if keypoints_data is not None:
                    for det in person_dets:
                        det_idx = next(
                            (j for j, d in enumerate(detections) if np.array_equal(d, det)),
                            None
                        )
                        if det_idx is not None:
                            kp = {
                                "xy": keypoints_data["xy"][det_idx],
                                "conf": (
                                    keypoints_data["conf"][det_idx]
                                    if keypoints_data["conf"] is not None
                                    else None
                                ),
                            }
                            person_keypoints.append(kp)
                        else:
                            person_keypoints.append(None)
                else:
                    person_keypoints = [None] * len(person_dets)

                all_detections.append(person_dets)
                all_keypoints.append(person_keypoints)

        return all_detections, all_keypoints

    def _extract_features_batch(
        self,
        frames: List[np.ndarray],
        all_detections: List[List],
        batch_size: int = 32,
    ) -> List[List[Optional[np.ndarray]]]:
        """
        Batch ReID feature extraction for all detections across frames.

        Args:
            frames: List of input frames.
            all_detections: List of detections per frame.

        Returns:
            List of feature lists per frame.
        """
        # Collect all crops and their indices
        crops = []
        crop_indices = []  # (frame_idx, det_idx)

        input_size = tuple(self.config["models"]["reid_input_size"])

        for frame_idx, (frame, detections) in enumerate(zip(frames, all_detections)):
            frame_height, frame_width = frame.shape[:2]
            for det_idx, det in enumerate(detections[:2]):
                bbox = det[0:4]
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)

                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                person_img = cv2.resize(person_img, input_size)
                crops.append(person_img)
                crop_indices.append((frame_idx, det_idx))

        # Initialize result structure
        all_features = [[None, None] for _ in range(len(frames))]

        if len(crops) == 0:
            return all_features

        # Batch through ReID model
        all_reid_features = []

        with torch.no_grad():
            for batch_start in range(0, len(crops), batch_size):
                batch_end = min(batch_start + batch_size, len(crops))
                batch_crops = crops[batch_start:batch_end]

                # Stack into tensor: (N, C, H, W)
                batch_tensor = torch.stack([
                    torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                    for crop in batch_crops
                ]).to(self.device)

                features = self.reid_model(batch_tensor)
                all_reid_features.append(features.cpu().numpy())

        # Concatenate all features
        all_reid_features = np.concatenate(all_reid_features, axis=0)

        # Map features back to frame/detection indices
        for feat_idx, (frame_idx, det_idx) in enumerate(crop_indices):
            all_features[frame_idx][det_idx] = all_reid_features[feat_idx]

        return all_features

    def _assign_detections_with_features(
        self,
        detections: List,
        det_features: List[Optional[np.ndarray]],
        frame_width: int,
    ) -> Dict[int, int]:
        """
        Assign detections to player IDs using pre-computed features.

        Args:
            detections: List of detections for this frame.
            det_features: Pre-computed ReID features for each detection.
            frame_width: Frame width for position scoring.

        Returns:
            Dictionary mapping detection index to player ID.
        """
        if len(detections) == 0:
            return {}

        if not self.players_initialized:
            return self._initialize_players_with_features(detections, det_features)

        if self.player_features[1] is None and self.player_features[2] is None:
            return self._initialize_players_with_features(detections, det_features)

        matching_scores = self._calculate_matching_scores_with_features(
            detections, det_features, frame_width
        )

        return self._greedy_assignment_with_features(
            matching_scores, det_features, len(detections)
        )

    def _initialize_players_with_features(
        self,
        detections: List,
        det_features: List[Optional[np.ndarray]],
    ) -> Dict[int, int]:
        """Initialize players based on left/right positioning with pre-computed features."""
        if len(detections) < 2:
            return {}

        # Sort by x-coordinate (left to right)
        detections_with_idx = [(i, det) for i, det in enumerate(detections[:2])]
        detections_with_idx.sort(key=lambda x: (x[1][0] + x[1][2]) / 2)

        # Assign: leftmost = Player 1, rightmost = Player 2
        assignments = {}
        for player_id, (det_idx, _) in enumerate(detections_with_idx, 1):
            assignments[det_idx] = player_id
            if det_idx < len(det_features) and det_features[det_idx] is not None:
                self.player_features[player_id] = det_features[det_idx]

        self.players_initialized = True
        return assignments

    def _calculate_matching_scores_with_features(
        self,
        detections: List,
        det_features: List[Optional[np.ndarray]],
        frame_width: int,
    ) -> Dict[Tuple[int, int], float]:
        """Calculate matching scores using pre-computed features."""
        matching_scores = {}

        for i, det in enumerate(detections[:2]):
            bbox = det[0:4]
            center = ((bbox[0] + bbox[2]) / 2, bbox[3])

            for player_id in [1, 2]:
                if self.player_features[player_id] is None:
                    continue

                # ReID score
                feat = det_features[i] if i < len(det_features) else None
                reid_score = self._feature_distance(feat, self.player_features[player_id])

                # Position score
                pos_score = 0
                if len(self.player_positions[player_id]) > 0:
                    last_pos = self.player_positions[player_id][-1]
                    distance = np.sqrt(
                        (center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2
                    )
                    pos_score = distance / frame_width

                # Combined score
                reid_weight = self.config["tracker"]["reid_weight"]
                pos_weight = self.config["tracker"]["position_weight"]
                score = reid_score * reid_weight + pos_score * pos_weight
                matching_scores[(i, player_id)] = score

        return matching_scores

    def _greedy_assignment_with_features(
        self,
        matching_scores: Dict[Tuple[int, int], float],
        det_features: List[Optional[np.ndarray]],
        num_detections: int,
    ) -> Dict[int, int]:
        """Perform greedy assignment with pre-computed features."""
        assignments = {}
        assigned_players = set()

        for (det_idx, player_id), score in sorted(
            matching_scores.items(), key=lambda x: x[1]
        ):
            if det_idx in assignments or player_id in assigned_players:
                continue
            if score < self.reid_threshold:
                assignments[det_idx] = player_id
                assigned_players.add(player_id)

                # Update player features with exponential moving average
                if det_idx < len(det_features) and det_features[det_idx] is not None:
                    self.player_features[player_id] = (
                        0.5 * self.player_features[player_id]
                        + 0.5 * det_features[det_idx]
                    )

        # Handle remaining unassigned detections
        unassigned_dets = [
            i for i in range(min(2, num_detections)) if i not in assignments
        ]
        unassigned_players = [i for i in [1, 2] if i not in assigned_players]

        if len(unassigned_dets) == 1 and len(unassigned_players) == 1:
            det_idx = unassigned_dets[0]
            player_id = unassigned_players[0]
            assignments[det_idx] = player_id
            if det_idx < len(det_features) and det_features[det_idx] is not None:
                self.player_features[player_id] = det_features[det_idx]

        return assignments

    def _build_output(
        self,
        frame_number: int,
        timestamp: float,
        detections: List,
        keypoints: List,
        assignments: Dict[int, int],
    ) -> PlayerTrackingOutput:
        """Build PlayerTrackingOutput from detections and assignments."""
        output = PlayerTrackingOutput(
            frame_number=frame_number,
            timestamp=timestamp,
            player_1_detected=False,
            player_1_x_pixel=None,
            player_1_y_pixel=None,
            player_1_x_meter=None,
            player_1_y_meter=None,
            player_1_confidence=0.0,
            player_1_bbox=None,
            player_1_keypoints=None,
            player_2_detected=False,
            player_2_x_pixel=None,
            player_2_y_pixel=None,
            player_2_x_meter=None,
            player_2_y_meter=None,
            player_2_confidence=0.0,
            player_2_bbox=None,
            player_2_keypoints=None,
        )

        for det_idx, player_id in assignments.items():
            # Bounds check to avoid index errors
            if det_idx >= len(detections):
                continue

            bbox = detections[det_idx][0:4]
            confidence = float(detections[det_idx][4])

            center_x_pixel = (bbox[0] + bbox[2]) / 2
            bottom_y_pixel = bbox[3]

            center_x_meter = None
            center_y_meter = None
            if self.calibration is not None and self.calibration.floor_homography is not None:
                pixel_point = Point2D(x=center_x_pixel, y=bottom_y_pixel)
                meter_point = self.calibration.pixel_to_floor(pixel_point)
                center_x_meter = meter_point.x
                center_y_meter = meter_point.y

            kp_array = None
            if det_idx < len(keypoints) and keypoints[det_idx] is not None:
                full_kp = keypoints[det_idx]["xy"]
                kp_array = full_kp[BODY_KEYPOINT_INDICES].copy()

            bbox_obj = BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            )

            self.player_positions[player_id].append((center_x_pixel, bottom_y_pixel))

            if player_id == 1:
                output.player_1_detected = True
                output.player_1_x_pixel = float(center_x_pixel)
                output.player_1_y_pixel = float(bottom_y_pixel)
                output.player_1_x_meter = center_x_meter
                output.player_1_y_meter = center_y_meter
                output.player_1_confidence = confidence
                output.player_1_bbox = bbox_obj
                output.player_1_keypoints = kp_array
            else:
                output.player_2_detected = True
                output.player_2_x_pixel = float(center_x_pixel)
                output.player_2_y_pixel = float(bottom_y_pixel)
                output.player_2_x_meter = center_x_meter
                output.player_2_y_meter = center_y_meter
                output.player_2_confidence = confidence
                output.player_2_bbox = bbox_obj
                output.player_2_keypoints = kp_array

        return output

    def reset(self):
        """Reset tracker state (useful for processing new videos)."""
        self.player_positions = {
            1: deque(maxlen=self.max_history),
            2: deque(maxlen=self.max_history),
        }
        self.player_features = {1: None, 2: None}
        self.players_initialized = False
        self.court_mask = None

    def postprocess(
        self,
        input_data: PlayerPostprocessingInput,
    ) -> PlayerPostprocessingOutput:
        """
        Apply interpolation for missing positions and smoothing filter.
        Also interpolates keypoints and bounding boxes (without smoothing).

        Uses vectorized numpy operations for efficiency.

        Args:
            input_data: PlayerPostprocessingInput with df, player_keypoints, player_bboxes

        Returns:
            PlayerPostprocessingOutput with processed df, keypoints, bboxes, and gap counts
        """
        df = input_data.df.copy()
        player_keypoints = input_data.player_keypoints
        player_bboxes = input_data.player_bboxes

        # Get smoothing sigma from config
        smooth_sigma = self.config["tracker"].get("smoothing_sigma", 2.0)

        gaps_filled = {1: 0, 2: 0}

        # Process each player's positions (with interpolation and smoothing)
        for player_id in [1, 2]:
            x_pixel_col = f'player_{player_id}_x_pixel'
            y_pixel_col = f'player_{player_id}_y_pixel'
            x_meter_col = f'player_{player_id}_x_meter'
            y_meter_col = f'player_{player_id}_y_meter'

            # Process pixel coordinates
            if x_pixel_col in df.columns and y_pixel_col in df.columns:
                x_values = df[x_pixel_col].values.astype(float)
                y_values = df[y_pixel_col].values.astype(float)

                # Count gaps
                gaps_filled[player_id] = int(np.isnan(x_values).sum())

                # Interpolate and smooth
                x_smooth = self._interpolate_and_smooth_1d(x_values, smooth_sigma)
                y_smooth = self._interpolate_and_smooth_1d(y_values, smooth_sigma)

                df[x_pixel_col] = x_smooth
                df[y_pixel_col] = y_smooth

            # Process meter coordinates (same interpolation and smoothing)
            if x_meter_col in df.columns and y_meter_col in df.columns:
                x_meter_values = df[x_meter_col].values.astype(float)
                y_meter_values = df[y_meter_col].values.astype(float)

                # Interpolate and smooth
                x_meter_smooth = self._interpolate_and_smooth_1d(x_meter_values, smooth_sigma)
                y_meter_smooth = self._interpolate_and_smooth_1d(y_meter_values, smooth_sigma)

                df[x_meter_col] = x_meter_smooth
                df[y_meter_col] = y_meter_smooth

        # Process keypoints (interpolation only, no smoothing)
        processed_keypoints = {}
        for player_id in [1, 2]:
            kp_list = player_keypoints.get(player_id, [])
            interpolated_kp = self._interpolate_keypoints(kp_list)
            processed_keypoints[player_id] = interpolated_kp

        # Process bounding boxes (interpolation only, no smoothing)
        processed_bboxes = {}
        for player_id in [1, 2]:
            bbox_list = player_bboxes.get(player_id, [])
            interpolated_bboxes = self._interpolate_bboxes(bbox_list)
            processed_bboxes[player_id] = interpolated_bboxes

        return PlayerPostprocessingOutput(
            df=df,
            player_keypoints=processed_keypoints,
            player_bboxes=processed_bboxes,
            num_player_1_gaps_filled=gaps_filled[1],
            num_player_2_gaps_filled=gaps_filled[2],
        )

    def _interpolate_and_smooth_1d(
        self,
        values: np.ndarray,
        smooth_sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Interpolate missing values and apply Gaussian smoothing (vectorized).

        Args:
            values: 1D array with possible NaN values
            smooth_sigma: Gaussian smoothing sigma

        Returns:
            Interpolated and smoothed array
        """
        valid_mask = ~np.isnan(values)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return values

        if n_valid < 2:
            # Fill all with the single valid value
            result = np.full_like(values, values[valid_mask][0])
        else:
            # Use np.interp for fast linear interpolation
            valid_indices = np.where(valid_mask)[0]
            valid_values = values[valid_mask]
            all_indices = np.arange(len(values))

            result = np.interp(
                all_indices,
                valid_indices,
                valid_values,
                left=valid_values[0],
                right=valid_values[-1],
            )

        # Apply Gaussian smoothing
        return gaussian_filter1d(result, sigma=smooth_sigma)

    def _interpolate_keypoints(
        self,
        keypoints_list: List[Optional[np.ndarray]],
    ) -> List[Optional[np.ndarray]]:
        """
        Interpolate missing keypoints using vectorized operations.

        Args:
            keypoints_list: List of keypoint arrays (shape: num_keypoints x 2), can contain None

        Returns:
            List of interpolated keypoint arrays
        """
        if not keypoints_list or len(keypoints_list) == 0:
            return []

        n_frames = len(keypoints_list)

        # Find valid keypoint indices
        valid_mask = np.array([kp is not None for kp in keypoints_list])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return keypoints_list

        if len(valid_indices) == 1:
            # Fill all with the single valid keypoint
            single_kp = keypoints_list[valid_indices[0]]
            return [single_kp.copy() for _ in range(n_frames)]

        # Get shape from first valid entry
        first_valid_kp = keypoints_list[valid_indices[0]]
        num_keypoints, num_coords = first_valid_kp.shape  # (num_keypoints, 2)

        # Stack valid keypoints into 3D array: (n_valid, num_keypoints, 2)
        valid_kps = np.array([keypoints_list[i] for i in valid_indices])

        # Interpolate all keypoints at once
        all_indices = np.arange(n_frames)
        interpolated = np.zeros((n_frames, num_keypoints, num_coords))

        for kp_idx in range(num_keypoints):
            for coord_idx in range(num_coords):
                valid_values = valid_kps[:, kp_idx, coord_idx]
                interpolated[:, kp_idx, coord_idx] = np.interp(
                    all_indices,
                    valid_indices,
                    valid_values,
                    left=valid_values[0],
                    right=valid_values[-1],
                )

        # Convert back to list, keeping originals where they exist
        result = []
        for i in range(n_frames):
            if keypoints_list[i] is not None:
                result.append(keypoints_list[i])
            else:
                result.append(interpolated[i])

        return result

    def _interpolate_bboxes(
        self,
        bboxes_list: List[Optional[BoundingBox]],
    ) -> List[Optional[BoundingBox]]:
        """
        Interpolate missing bounding boxes using vectorized operations.

        Args:
            bboxes_list: List of bounding boxes (can contain None)

        Returns:
            List of interpolated BoundingBox
        """
        if not bboxes_list or len(bboxes_list) == 0:
            return []

        n_frames = len(bboxes_list)

        # Find valid bbox indices
        valid_mask = np.array([bbox is not None for bbox in bboxes_list])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return bboxes_list

        if len(valid_indices) == 1:
            # If only one valid bbox, return as-is (no interpolation needed)
            return bboxes_list

        # Extract bbox coordinates into arrays
        valid_x1 = np.array([bboxes_list[i].x1 for i in valid_indices])
        valid_y1 = np.array([bboxes_list[i].y1 for i in valid_indices])
        valid_x2 = np.array([bboxes_list[i].x2 for i in valid_indices])
        valid_y2 = np.array([bboxes_list[i].y2 for i in valid_indices])

        # Interpolate all coordinates at once
        all_indices = np.arange(n_frames)
        interp_x1 = np.interp(all_indices, valid_indices, valid_x1, left=valid_x1[0], right=valid_x1[-1])
        interp_y1 = np.interp(all_indices, valid_indices, valid_y1, left=valid_y1[0], right=valid_y1[-1])
        interp_x2 = np.interp(all_indices, valid_indices, valid_x2, left=valid_x2[0], right=valid_x2[-1])
        interp_y2 = np.interp(all_indices, valid_indices, valid_y2, left=valid_y2[0], right=valid_y2[-1])

        # Build result list, keeping originals where they exist
        result = []
        for i in range(n_frames):
            if bboxes_list[i] is not None:
                result.append(bboxes_list[i])
            else:
                result.append(BoundingBox(
                    x1=float(interp_x1[i]),
                    y1=float(interp_y1[i]),
                    x2=float(interp_x2[i]),
                    y2=float(interp_y2[i]),
                ))

        return result
