import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
import os

from player_tracking_pipeline.utils import load_config, get_package_dir


class PlayerTracker:
    """
    A clean, modular tracker for identifying and tracking two squash players.
    Returns the bottom-center position of each player's bounding box.
    """

    def __init__(self, config: dict = None, homography: np.ndarray = None):
        """
        Initialize the player tracker.
        Args:
            config (dict): Configuration dictionary. If None, loads from 'config.json'.
            homography (np.ndarray): 3x3 homography matrix for court coordinate transformation.
        """
        self.config = config if config else load_config()
        self.homography = homography
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
            get_package_dir(), self.config["models"]["yolo_model"]
        )
        self.detector = YOLO(model_path, verbose=False)

        # Check if model supports pose estimation
        self.has_pose = (
            hasattr(self.detector.model, "kpt_shape")
            if hasattr(self.detector, "model")
            else False
        )

        # Load ResNet50 and modify on CPU first, then move to device
        self.reid_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.reid_model.fc.in_features
        feature_size = self.config["models"]["reid_feature_size"]
        self.reid_model.fc = torch.nn.Linear(in_features, feature_size)

        # Move entire model to device and set to eval mode
        self.reid_model = self.reid_model.to(self.device)
        self.reid_model.eval()

    def process_frame(self, frame):
        """
        Process a single frame and return player tracking information.

        Args:
            frame: BGR image (numpy array)

        Returns:
            results: Dictionary of player tracking information
        """
        frame_width = frame.shape[1]

        # Detect people in frame
        detections, keypoints = self._detect_people(frame)

        # Assign detections to player IDs
        assignments = self._assign_detections(detections, frame, frame_width)

        # Initialize results
        results = {
            1: {
                "position": None,
                "real_position": None,
                "bbox": None,
                "confidence": None,
                "keypoints": {"xy": None, "conf": None},
            },
            2: {
                "position": None,
                "real_position": None,
                "bbox": None,
                "confidence": None,
                "keypoints": {"xy": None, "conf": None},
            },
        }

        for det_idx, player_id in assignments.items():
            bbox = detections[det_idx][0:4]
            confidence = detections[det_idx][4]

            # Calculate bottom-center position
            center_x = (bbox[0] + bbox[2]) / 2
            bottom_y = bbox[3]
            position = (center_x, bottom_y)
            pixel_point = np.array([[position]], dtype=np.float32)
            real_point = cv2.perspectiveTransform(pixel_point, self.homography)

            # Get keypoints for this detection
            kp = keypoints[det_idx] if keypoints[det_idx] is not None else None

            # Store all tracking information
            results[player_id] = {
                "position": position,
                "real_position": real_point[0][0],
                "bbox": bbox.tolist(),
                "confidence": float(confidence),
                "keypoints": {
                    "xy": kp["xy"].tolist() if kp is not None else None,
                    "conf": (
                        kp["conf"].tolist()
                        if kp is not None and kp["conf"] is not None
                        else None
                    ),
                },
            }

            # Update position history
            self.player_positions[player_id].append(position)

        return results

    def _detect_people(self, frame):
        """
        Detect people in the frame using YOLO.

        Returns:
            tuple: (detections, keypoints)
                - detections: list of top 2 person detections sorted by confidence
                - keypoints: list of keypoint arrays (None if pose not supported)
        """
        results = self.detector(frame, verbose=False)[0]
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
                # Find original index in detections
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

    def _assign_detections(self, detections, frame, frame_width):
        """
        Assign detections to player IDs using re-identification and position.

        Returns:
            dict: {detection_index: player_id}
        """
        if len(detections) == 0:
            return {}

        # Initialize players on first frame
        if not self.players_initialized:
            return self._initialize_players(detections, frame)

        # Extract features for current detections
        det_features = [
            self._extract_features(frame, det[0:4]) for det in detections[:2]
        ]

        # Reinitialize if both players lost
        if self.player_features[1] is None and self.player_features[2] is None:
            return self._initialize_players(detections, frame)

        # Calculate matching scores
        matching_scores = self._calculate_matching_scores(
            detections, det_features, frame_width
        )

        # Assign detections to players
        assignments = self._greedy_assignment(matching_scores, det_features)

        return assignments

    def _initialize_players(self, detections, frame):
        """
        Initialize players based on left/right positioning.

        Returns:
            dict: Initial player assignments
        """
        if len(detections) < 2:
            print("Warning: Less than 2 players detected")
            return {}

        # Extract features
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
        """
        Calculate matching scores between detections and tracked players.

        Returns:
            dict: {(det_idx, player_id): score}
        """
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

                # Combined score (weighted)
                reid_weight = self.config["tracker"]["reid_weight"]
                pos_weight = self.config["tracker"]["position_weight"]
                score = reid_score * reid_weight + pos_score * pos_weight
                matching_scores[(i, player_id)] = score
        return matching_scores

    def _greedy_assignment(self, matching_scores, det_features):
        """
        Perform greedy assignment of detections to players.

        Returns:
            dict: {det_idx: player_id}
        """
        assignments = {}
        assigned_players = set()

        # Assign based on best scores
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

    def _extract_features(self, frame, bbox):
        """
        Extract re-identification features from a bounding box.

        Returns:
            numpy.ndarray: Feature vector or None if invalid bbox
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None

        # Prepare image for ResNet50
        input_size = tuple(self.config["models"]["reid_input_size"])
        person_img = cv2.resize(person_img, input_size)
        person_tensor = torch.from_numpy(person_img).permute(2, 0, 1).float() / 255.0
        person_tensor = person_tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.reid_model(person_tensor)

        return features.squeeze().cpu().numpy()

    def _feature_distance(self, feat1, feat2):
        """
        Calculate cosine distance between two feature vectors.

        Returns:
            float: Distance (0 = identical, 2 = opposite)
        """
        if feat1 is None or feat2 is None:
            return float("inf")

        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)

        return 1 - np.dot(feat1, feat2)

    def reset(self):
        """Reset tracker state (useful for processing new videos)."""
        self.player_positions = {
            1: deque(maxlen=self.max_history),
            2: deque(maxlen=self.max_history),
        }
        self.player_features = {1: None, 2: None}
        self.players_initialized = False
