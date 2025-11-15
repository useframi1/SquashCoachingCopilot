import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os

from squashcopilot.common.utils import load_config, get_package_dir

from squashcopilot.common import (
    Point2D,
    BoundingBox,
    PlayerTrackingInput,
    PlayerDetectionResult,
    PlayerTrackingResult,
    PlayerKeypointsData,
    PlayerPostprocessingInput,
    PlayerTrajectory,
    PlayerPostprocessingResult,
)


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
        self.config = config if config else load_config(config_name="player_tracking")
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
            get_package_dir(__file__), self.config["models"]["yolo_model"]
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

    def preprocess_frame(self, frame):
        """
        Preprocess frame before detection.

        Args:
            frame: BGR image (numpy array)

        Returns:
            numpy.ndarray: Preprocessed frame
        """
        # For now, just return the frame as-is
        return frame

    def postprocess(
        self, input_data: PlayerPostprocessingInput
    ) -> PlayerPostprocessingResult:
        """
        Apply interpolation for missing positions and smoothing filter.

        Args:
            input_data: PlayerPostprocessingInput with position histories

        Returns:
            PlayerPostprocessingResult with smoothed trajectories
        """
        # Convert Point2D to tuples for processing
        positions_history_tuples = {}
        for player_id, positions in input_data.positions_history.items():
            positions_history_tuples[player_id] = [
                (p.x, p.y) if p else None for p in positions
            ]

        smoothed_positions = {}

        for player_id, positions in positions_history_tuples.items():
            if not positions or len(positions) == 0:
                smoothed_positions[player_id] = []
                continue

            # Separate valid and invalid indices
            valid_indices = [i for i, pos in enumerate(positions) if pos is not None]

            if len(valid_indices) == 0:
                smoothed_positions[player_id] = positions
                continue

            # Extract x and y coordinates from valid positions
            valid_x = [positions[i][0] for i in valid_indices]
            valid_y = [positions[i][1] for i in valid_indices]

            # Interpolate missing positions
            if len(valid_indices) > 1:
                # Create interpolation functions
                interp_x = interp1d(
                    valid_indices,
                    valid_x,
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                interp_y = interp1d(
                    valid_indices,
                    valid_y,
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )

                # Generate complete position arrays
                all_indices = np.arange(len(positions))
                interpolated_x = interp_x(all_indices)
                interpolated_y = interp_y(all_indices)
            else:
                # If only one valid position, use it for all frames
                interpolated_x = np.full(len(positions), valid_x[0])
                interpolated_y = np.full(len(positions), valid_y[0])

            # Apply Gaussian smoothing filter
            sigma = self.config["tracker"].get("smoothing_sigma", 2.0)
            smoothed_x = gaussian_filter1d(interpolated_x, sigma=sigma)
            smoothed_y = gaussian_filter1d(interpolated_y, sigma=sigma)

            # Combine back into position tuples
            smoothed_positions[player_id] = [
                (x, y) for x, y in zip(smoothed_x, smoothed_y)
            ]

        # Convert back to new format
        trajectories = {}
        for player_id in smoothed_positions.keys():
            original_positions = input_data.positions_history[player_id]
            gaps_filled = sum(1 for p in original_positions if p is None)

            trajectories[player_id] = PlayerTrajectory(
                player_id=player_id,
                positions=[Point2D(x=x, y=y) for x, y in smoothed_positions[player_id]],
                original_positions=original_positions,
                gaps_filled=gaps_filled,
            )

        return PlayerPostprocessingResult(trajectories=trajectories)

    def process_frame(self, input_data: PlayerTrackingInput) -> PlayerTrackingResult:
        """
        Process a single frame and return player tracking information.

        Args:
            input_data: PlayerTrackingInput with frame and optional homography

        Returns:
            PlayerTrackingResult with structured player detection data
        """
        frame = input_data.frame.image
        frame_number = input_data.frame.frame_number
        frame_width = frame.shape[1]

        # Update homography if provided
        if input_data.homography:
            self.homography = input_data.homography.matrix

        # Detect people in frame
        detections, keypoints = self._detect_people(frame)

        # Assign detections to player IDs
        assignments = self._assign_detections(detections, frame, frame_width)

        # Build PlayerTrackingResult
        players = {}

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

            # Create PlayerDetectionResult
            players[player_id] = PlayerDetectionResult(
                player_id=player_id,
                position=Point2D(x=position[0], y=position[1]),
                real_position=(
                    Point2D(x=float(real_point[0][0][0]), y=float(real_point[0][0][1]))
                    if real_point is not None
                    else None
                ),
                bbox=BoundingBox.from_list(bbox.tolist()),
                confidence=float(confidence),
                keypoints=PlayerKeypointsData(
                    xy=kp["xy"].tolist() if kp is not None else None,
                    conf=(
                        kp["conf"].tolist()
                        if kp is not None and kp["conf"] is not None
                        else None
                    ),
                ),
                frame_number=frame_number,
            )

            # Update position history
            self.player_positions[player_id].append(position)

        return PlayerTrackingResult(players=players, frame_number=frame_number)

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
